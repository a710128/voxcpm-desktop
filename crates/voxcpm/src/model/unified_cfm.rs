//! Unified CFM sampler (Euler solver + classifier-free guidance).
//!
//! Reference: `VoxCPM/src/voxcpm/modules/locdit/unified_cfm.py`.

use crate::arange_cache;
use candle_core::{DType, Device, Result, Tensor};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_distr::StandardNormal;

#[cfg(feature = "cuda")]
use crate::cuda_graph::CudaGraphModule;

pub trait VelocityEstimator {
    fn forward(
        &self,
        x: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        cond: &Tensor,
        dt: &Tensor,
    ) -> Result<Tensor>;
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct CfmConfig {
    #[serde(default = "default_sigma_min")]
    pub sigma_min: f64,
    #[serde(default = "default_solver")]
    pub solver: String,
    #[serde(default = "default_t_scheduler")]
    pub t_scheduler: String,
    #[serde(default = "default_training_cfg_rate")]
    pub training_cfg_rate: f64,
    #[serde(default = "default_inference_cfg_rate")]
    pub inference_cfg_rate: f64,
    #[serde(default = "default_reg_loss_type")]
    pub reg_loss_type: String,
    #[serde(default = "default_ratio_r_neq_t_range")]
    pub ratio_r_neq_t_range: (f64, f64),
    #[serde(default = "default_noise_cond_prob_range")]
    pub noise_cond_prob_range: (f64, f64),
    #[serde(default)]
    pub noise_cond_scale: f64,
}

fn default_sigma_min() -> f64 {
    1e-6
}
fn default_solver() -> String {
    "euler".to_owned()
}
fn default_t_scheduler() -> String {
    "log-norm".to_owned()
}
fn default_training_cfg_rate() -> f64 {
    0.1
}
fn default_inference_cfg_rate() -> f64 {
    1.0
}
fn default_reg_loss_type() -> String {
    "l1".to_owned()
}
fn default_ratio_r_neq_t_range() -> (f64, f64) {
    (0.25, 0.75)
}
fn default_noise_cond_prob_range() -> (f64, f64) {
    (0.0, 0.0)
}

#[derive(Debug)]
pub struct UnifiedCfm<E: VelocityEstimator> {
    pub in_channels: usize,
    pub mean_mode: bool,
    pub estimator: E,
}

impl<E: VelocityEstimator> UnifiedCfm<E> {
    pub fn new(in_channels: usize, estimator: E, mean_mode: bool) -> Self {
        Self {
            in_channels,
            mean_mode,
            estimator,
        }
    }

    fn optimized_scale(&self, positive: &Tensor, negative: &Tensor) -> Result<Tensor> {
        // positive/negative: [B, F]
        let dot = (positive * negative)?.sum_keepdim(1)?; // [B, 1]
        let denom = negative.sqr()?.sum_keepdim(1)?;
        dot / ((denom + 1e-8)?)
    }

    fn randn_tensor(
        &self,
        shape: (usize, usize, usize),
        device: &Device,
        seed: u64,
        dtype: DType,
    ) -> Result<Tensor> {
        let (b, c, t) = shape;

        // Candle's `Device::set_seed` works for CUDA/Metal, but CPU backend cannot be seeded.
        // To keep `seed` reproducibility on CPU, we generate deterministically in Rust.
        if device.is_cpu() {
            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            let mut data = Vec::with_capacity(b * c * t);
            for _ in 0..(b * c * t) {
                let v: f32 = rng.sample(StandardNormal);
                data.push(v);
            }
            return Tensor::from_vec(data, (b, c, t), device)?.to_dtype(dtype);
        }

        // Non-CPU: rely on Candle backend RNG, but restore prior seed to avoid leaking state.
        let prev_seed = device.get_current_seed().ok();
        device.set_seed(seed)?;
        let out = Tensor::randn(0f32, 1f32, (b, c, t), device)?.to_dtype(dtype);
        if let Some(prev) = prev_seed {
            let _ = device.set_seed(prev);
        }
        out
    }

    /// Inference entry: sample a latent patch.
    ///
    /// - mu: [B, H]
    /// - cond: [B, C, T']
    /// returns: [B, C, patch_size]
    pub fn sample(
        &self,
        mu: &Tensor,
        cond: &Tensor,
        patch_size: usize,
        n_timesteps: usize,
        seed: u64,
        temperature: f64,
        cfg_value: &Tensor,
        sway_sampling_coef: f64,
        use_cfg_zero_star: bool,
    ) -> Result<Tensor> {
        let (b, _) = mu.dims2()?;
        let dtype = mu.dtype();
        let dev = mu.device();
        let z = self.randn_tensor((b, self.in_channels, patch_size), dev, seed, dtype)?;
        let z = (&z * temperature)?;

        // Build the reference t_span purely with device ops (linspace + sway warp).
        // We intentionally do NOT cache it here (per caller preference).
        let t_span = t_span_device(n_timesteps, sway_sampling_coef, dev)?;
        self.solve_euler(&z, &t_span, mu, cond, cfg_value, use_cfg_zero_star)
    }

    /// CUDA-only optimized sampler.
    ///
    /// Uses a captured CUDA graph for the CFG else-branch (`dphi_dt` computation) and
    /// replays it each Euler step.
    #[cfg(feature = "cuda")]
    pub fn sample_optimized_cuda(
        &self,
        mu: &Tensor,
        cond: &Tensor,
        patch_size: usize,
        n_timesteps: usize,
        seed: u64,
        temperature: f64,
        cfg_value: &Tensor,
        sway_sampling_coef: f64,
        use_cfg_zero_star: bool,
        cfg_graph: &CudaGraphModule,
    ) -> Result<Tensor> {
        if !mu.device().is_cuda() {
            candle_core::bail!("sample_optimized_cuda requires CUDA tensors")
        }
        if !use_cfg_zero_star {
            candle_core::bail!("sample_optimized_cuda requires use_cfg_zero_star=true")
        }
        let (b, _) = mu.dims2()?;
        if b != 1 {
            candle_core::bail!("sample_optimized_cuda currently expects batch_size==1")
        }

        let dtype = mu.dtype();
        let dev = mu.device();
        let z = self.randn_tensor((b, self.in_channels, patch_size), dev, seed, dtype)?;
        let z = (&z * temperature)?;

        let t_span = t_span_device(n_timesteps, sway_sampling_coef, dev)?;
        self.solve_euler_optimized_cuda(
            &z,
            &t_span,
            mu,
            cond,
            cfg_value,
            use_cfg_zero_star,
            cfg_graph,
        )
    }

    /// Euler solver with CFG.
    ///
    /// - x: [B, C, T]
    /// - t_span: len = n_steps+1, decreasing from 1 to 0
    pub fn solve_euler(
        &self,
        x: &Tensor,
        t_span: &Tensor,
        mu: &Tensor,
        cond: &Tensor,
        cfg_value: &Tensor,
        use_cfg_zero_star: bool,
    ) -> Result<Tensor> {
        // cfg_value must be a device scalar. Keep it as a tensor so all math stays on-device.
        let cfg = cfg_scalar_1x1x1(cfg_value, x.dtype())?;

        // The upstream builds t_span in the model dtype; we use f32 for stability.
        // `t_span_device` returns f32, but allow callers to pass other dtypes.
        let t_span = if t_span.dtype() == DType::F32 {
            t_span.clone()
        } else {
            t_span.to_dtype(DType::F32)?
        };
        let n_steps = t_span.dims1()?;
        if n_steps < 2 {
            candle_core::bail!("t_span must have len>=2")
        }
        let (b, c, t) = x.dims3()?;
        if c != self.in_channels {
            candle_core::bail!(
                "UnifiedCfm expects in_channels={}, got C={c}",
                self.in_channels
            )
        }
        let (bm, _) = mu.dims2()?;
        if bm != b {
            candle_core::bail!("mu batch mismatch: expected B={b}, got {bm}")
        }
        let (bc, cc, _) = cond.dims3()?;
        if (bc, cc) != (b, c) {
            candle_core::bail!(
                "cond shape mismatch: expected [B={b}, C={c}, T'], got [{bc}, {cc}, _]"
            )
        }

        let zero_init_steps = ((n_steps as f32) * 0.04).floor() as usize;
        let zero_init_steps = zero_init_steps.max(1);

        let mut cur = x.clone();
        for step in 1..n_steps {
            let t_prev = t_span.narrow(0, step - 1, 1)?; // [1]
            let t_next = t_span.narrow(0, step, 1)?; // [1]
            let dt_f32 = (&t_prev - &t_next)?; // [1]

            let dphi_dt = if use_cfg_zero_star && step <= zero_init_steps {
                Tensor::zeros_like(&cur)?
            } else {
                let x_in = Tensor::cat(&[&cur, &cur], 0)?; // [2B, C, T]
                let zeros_mu = Tensor::zeros_like(mu)?;
                let mu_in = Tensor::cat(&[mu, &zeros_mu], 0)?; // [2B, H]

                let t_in = t_prev.broadcast_as((2 * b,))?;
                let dt_in = if self.mean_mode {
                    dt_f32.broadcast_as((2 * b,))?
                } else {
                    Tensor::zeros((2 * b,), DType::F32, cur.device())?
                };
                let cond_in = Tensor::cat(&[cond, cond], 0)?; // [2B, C, T']

                let pred = self
                    .estimator
                    .forward(&x_in, &mu_in, &t_in, &cond_in, &dt_in)?;
                let pos = pred.narrow(0, 0, b)?;
                let neg = pred.narrow(0, b, b)?;

                let neg_scaled = if use_cfg_zero_star {
                    let pos_flat = pos.reshape((b, c * t))?;
                    let neg_flat = neg.reshape((b, c * t))?;
                    let st = self.optimized_scale(&pos_flat, &neg_flat)?; // [B, 1]
                    let st = st.reshape((b, 1, 1))?;
                    neg.broadcast_mul(&st)?
                } else {
                    neg
                };

                let diff = (pos - &neg_scaled)?;
                (neg_scaled + diff.broadcast_mul(&cfg)?)?
            };

            // dt is scalar ([1]) and must match dtype for the update.
            let dt = dt_f32.to_dtype(cur.dtype())?.reshape((1, 1, 1))?;
            let delta = dphi_dt.broadcast_mul(&dt)?;
            cur = (&cur - &delta)?;
        }

        Ok(cur)
    }

    #[cfg(feature = "cuda")]
    fn solve_euler_optimized_cuda(
        &self,
        x: &Tensor,
        t_span: &Tensor,
        mu: &Tensor,
        cond: &Tensor,
        cfg_value: &Tensor,
        use_cfg_zero_star: bool,
        cfg_graph: &CudaGraphModule,
    ) -> Result<Tensor> {
        if !use_cfg_zero_star {
            candle_core::bail!("solve_euler_optimized_cuda requires use_cfg_zero_star=true")
        }
        // cfg_value must be a device scalar. Keep it as a tensor so all math stays on-device.
        let cfg = cfg_scalar_1x1x1(cfg_value, x.dtype())?;

        let t_span = if t_span.dtype() == DType::F32 {
            t_span.clone()
        } else {
            t_span.to_dtype(DType::F32)?
        };
        let n_steps = t_span.dims1()?;
        if n_steps < 2 {
            candle_core::bail!("t_span must have len>=2")
        }
        let (b, c, _t) = x.dims3()?;
        if b != 1 {
            candle_core::bail!("solve_euler_optimized_cuda expects B==1")
        }
        if c != self.in_channels {
            candle_core::bail!(
                "UnifiedCfm expects in_channels={}, got C={c}",
                self.in_channels
            )
        }
        let (bm, _) = mu.dims2()?;
        if bm != b {
            candle_core::bail!("mu batch mismatch: expected B={b}, got {bm}")
        }
        let (bc, cc, _) = cond.dims3()?;
        if (bc, cc) != (b, c) {
            candle_core::bail!(
                "cond shape mismatch: expected [B={b}, C={c}, T'], got [{bc}, {cc}, _]"
            )
        }

        let zero_init_steps = ((n_steps as f32) * 0.04).floor() as usize;
        let zero_init_steps = zero_init_steps.max(1);

        let mut cur = x.clone();
        for step in 1..n_steps {
            let t_prev = t_span.narrow(0, step - 1, 1)?; // [1]
            let t_next = t_span.narrow(0, step, 1)?; // [1]
            let dt_f32 = (&t_prev - &t_next)?; // [1]

            let dphi_dt = if use_cfg_zero_star && step <= zero_init_steps {
                Tensor::zeros_like(&cur)?
            } else {
                // Replay the captured CFG block graph.
                let outs = cfg_graph.run(&[
                    cur.clone(),
                    mu.clone(),
                    t_prev.clone(),
                    dt_f32.clone(),
                    cond.clone(),
                    cfg.clone(),
                ])?;
                if outs.len() != 1 {
                    candle_core::bail!("cfg_graph returned {} outputs (expected 1)", outs.len())
                }
                outs[0].clone()
            };

            let dt = dt_f32.to_dtype(cur.dtype())?.reshape((1, 1, 1))?;
            let delta = dphi_dt.broadcast_mul(&dt)?;
            cur = (&cur - &delta)?;
        }

        Ok(cur)
    }
}

fn cfg_scalar_1x1x1(cfg_value: &Tensor, dtype: DType) -> Result<Tensor> {
    // Accept any shape as long as it contains exactly 1 element.
    // We reshape to [1, 1, 1] so `broadcast_mul` is unambiguous.
    let flat = cfg_value.flatten_all()?;
    let n = flat.dims1()?;
    if n != 1 {
        candle_core::bail!("cfg_value must be a scalar tensor (1 element), got {n} elements")
    }
    flat.narrow(0, 0, 1)?.to_dtype(dtype)?.reshape((1, 1, 1))
}

fn t_span_device(n_timesteps: usize, sway_sampling_coef: f64, device: &Device) -> Result<Tensor> {
    // Reference:
    // t_span = linspace(1, 0, n_timesteps+1)
    // t_span = t_span + sway * (cos(pi/2 * t_span) - 1 + t_span)
    let steps = n_timesteps + 1;
    let t = if n_timesteps == 0 {
        Tensor::ones((1usize,), DType::F32, device)?
    } else {
        // t[i] = 1 - i/n_timesteps
        let idx = arange_cache::arange_f32(steps, device)?; // [steps]
        let inv_n = 1f64 / (n_timesteps as f64);
        idx.affine(-inv_n, 1.0)?
    };

    let half_pi = std::f64::consts::PI / 2.0;
    let warp = ((&t * half_pi)?.cos()? - 1f64)?;
    let warp = (warp + &t)?;
    &t + (&warp * sway_sampling_coef)?
}

pub fn build_t_span(n_timesteps: usize, sway_sampling_coef: f64) -> Vec<f32> {
    // Matches: linspace(1, 0, n_timesteps+1) + coef*(cos(pi/2*t)-1+t)
    let steps = n_timesteps + 1;
    let mut out = Vec::with_capacity(steps);
    for i in 0..steps {
        let alpha = (i as f64) / (n_timesteps as f64);
        let t = 1.0 - alpha;
        let warped = t + sway_sampling_coef * (((std::f64::consts::PI / 2.0) * t).cos() - 1.0 + t);
        out.push(warped as f32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[derive(Debug, Clone)]
    struct DummyEstimator {
        // mode:
        // 0: return v = x
        // 1: return v_pos = 1, v_neg = 0 (implemented via batch split)
        // 2: return zeros
        mode: i32,
    }

    impl VelocityEstimator for DummyEstimator {
        fn forward(
            &self,
            x: &Tensor,
            _mu: &Tensor,
            _t: &Tensor,
            _cond: &Tensor,
            _dt: &Tensor,
        ) -> Result<Tensor> {
            let (b2, c, t) = x.dims3()?;
            match self.mode {
                0 => Ok(x.clone()),
                2 => Tensor::zeros((b2, c, t), x.dtype(), x.device()),
                1 => {
                    // First half ones, second half zeros.
                    let b = b2 / 2;
                    if b * 2 != b2 {
                        candle_core::bail!("DummyEstimator(mode=1) expects even batch")
                    }
                    let ones = Tensor::ones((b, c, t), x.dtype(), x.device())?;
                    let zeros = Tensor::zeros((b, c, t), x.dtype(), x.device())?;
                    Tensor::cat(&[&ones, &zeros], 0)
                }
                _ => candle_core::bail!("unknown dummy estimator mode"),
            }
        }
    }

    #[test]
    fn euler_matches_closed_form_for_v_eq_x() -> Result<()> {
        let dev = Device::Cpu;
        let est = DummyEstimator { mode: 0 };
        let cfm = UnifiedCfm::new(2, est, false);

        let b = 1usize;
        let c = 2usize;
        let t = 4usize;
        let x0 = Tensor::from_vec(vec![0.5f32; b * c * t], (b, c, t), &dev)?;
        let mu = Tensor::zeros((b, 8), DType::F32, &dev)?;
        let cond = Tensor::zeros((b, c, t), DType::F32, &dev)?;

        let n_steps = 10usize;
        let t_span = Tensor::from_vec(build_t_span(n_steps, 0.0), (n_steps + 1,), &dev)?;
        let cfg = Tensor::from_vec(vec![1f32], (1,), &dev)?;
        let out = cfm.solve_euler(&x0, &t_span, &mu, &cond, &cfg, false)?;

        // With v=x and constant dt=1/n, Euler gives x_n = (1-1/n)^n x0.
        let dt = 1.0 / (n_steps as f32);
        let expected = (1.0 - dt).powi(n_steps as i32) * 0.5;
        let v = out.flatten_all()?.to_vec1::<f32>()?;
        for &vv in v.iter() {
            assert!((vv - expected).abs() < 5e-4, "vv={vv} expected={expected}");
        }
        Ok(())
    }

    #[test]
    fn cfg_mixing_matches_formula_when_pos_one_neg_zero() -> Result<()> {
        let dev = Device::Cpu;
        let est = DummyEstimator { mode: 1 };
        let cfm = UnifiedCfm::new(1, est, false);

        let b = 2usize;
        let c = 1usize;
        let t = 3usize;
        let x0 = Tensor::zeros((b, c, t), DType::F32, &dev)?;
        let mu = Tensor::zeros((b, 4), DType::F32, &dev)?;
        let cond = Tensor::zeros((b, c, t), DType::F32, &dev)?;

        let t_span = Tensor::from_vec(vec![1.0f32, 0.0f32], (2,), &dev)?; // single Euler step, dt=1
        let cfg = 2.5f64;
        let cfg_t = Tensor::from_vec(vec![cfg as f32], (1,), &dev)?;
        let out = cfm.solve_euler(&x0, &t_span, &mu, &cond, &cfg_t, false)?;

        // v = neg + cfg*(pos - neg) = cfg
        // x1 = x0 - dt*v = -cfg
        let v = out.flatten_all()?.to_vec1::<f32>()?;
        for &vv in v.iter() {
            assert!((vv - (-(cfg as f32))).abs() < 1e-6, "vv={vv}");
        }
        Ok(())
    }

    #[test]
    fn sample_is_seed_reproducible() -> Result<()> {
        let dev = Device::Cpu;
        let est = DummyEstimator { mode: 2 };
        let cfm = UnifiedCfm::new(2, est, false);

        let b = 1usize;
        let mu = Tensor::zeros((b, 4), DType::F32, &dev)?;
        let cond = Tensor::zeros((b, 2, 5), DType::F32, &dev)?;

        let cfg = Tensor::from_vec(vec![1f32], (1,), &dev)?;

        let a = cfm.sample(&mu, &cond, 5, 8, 123, 1.0, &cfg, 0.0, true)?;
        let b2 = cfm.sample(&mu, &cond, 5, 8, 123, 1.0, &cfg, 0.0, true)?;
        let c2 = cfm.sample(&mu, &cond, 5, 8, 124, 1.0, &cfg, 0.0, true)?;

        let da = a.flatten_all()?.to_vec1::<f32>()?;
        let db = b2.flatten_all()?.to_vec1::<f32>()?;
        let dc = c2.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(da, db);
        assert_ne!(da, dc);
        Ok(())
    }
}
