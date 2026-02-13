//! VoxCPM local DiT estimator (diffusion transformer backbone).
//!
//! Reference: `VoxCPM/src/voxcpm/modules/locdit/local_dit.py`.

use crate::arange_cache;
use crate::model::minicpm4::{MiniCpmConfig, MiniCpmModel};
use crate::model::unified_cfm::VelocityEstimator;
use candle_core::{DType, Result, Tensor, D};
use candle_nn::{ops, Linear, Module, VarBuilder};

#[derive(Debug)]
pub struct SinusoidalPosEmb {
    dim: usize,
    scale: f64,
}

impl SinusoidalPosEmb {
    pub fn new(dim: usize, scale: f64) -> Result<Self> {
        if dim % 2 != 0 {
            candle_core::bail!("SinusoidalPosEmb expects even dim, got {dim}")
        }
        Ok(Self { dim, scale })
    }

    /// x: [N] -> [N, dim]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _n = match x.dims() {
            [n] => *n,
            ds => candle_core::bail!("SinusoidalPosEmb expects rank-1 input, got {ds:?}"),
        };

        let half = self.dim / 2;
        if half <= 1 {
            candle_core::bail!("SinusoidalPosEmb expects dim>=4, got {}", self.dim)
        }

        // Matches the Python implementation:
        // emb = exp(arange(half_dim) * -(log(10000)/(half_dim-1)))
        let step = -((10000f64).ln() / ((half - 1) as f64));
        let arange = arange_cache::arange_f32(half, x.device())?; // [half]
        let freqs = arange.affine(step, 0.0)?.exp()?; // [half]

        let x = x.to_dtype(DType::F32)?;
        let x = (&x * self.scale)?; // [N]
        let x = x.unsqueeze(1)?; // [N, 1]
        let freqs = freqs.reshape((1, half))?; // [1, half]
        let emb = x.broadcast_mul(&freqs)?; // [N, half]

        let sin = emb.sin()?;
        let cos = emb.cos()?;
        Tensor::cat(&[&sin, &cos], D::Minus1) // [N, dim]
    }
}

#[derive(Debug)]
pub struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    pub fn new(
        in_channels: usize,
        embed_dim: usize,
        out_dim: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let out = out_dim.unwrap_or(embed_dim);
        let linear_1 = candle_nn::linear(in_channels, embed_dim, vb.pp("linear_1"))?;
        let linear_2 = candle_nn::linear(embed_dim, out, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear_1.forward(x)?;
        let x = ops::silu(&x)?;
        self.linear_2.forward(&x)
    }
}

#[derive(Debug)]
pub struct VoxCpmLocDiT {
    in_channels: usize,
    hidden: usize,
    in_proj: Linear,
    cond_proj: Linear,
    out_proj: Linear,
    time_embeddings: SinusoidalPosEmb,
    time_mlp: TimestepEmbedding,
    delta_time_mlp: TimestepEmbedding,
    decoder: MiniCpmModel,
}

impl VoxCpmLocDiT {
    pub fn new(cfg: MiniCpmConfig, in_channels: usize, vb: VarBuilder) -> Result<Self> {
        if cfg.vocab_size != 0 {
            candle_core::bail!("locdit expects vocab_size == 0, got {}", cfg.vocab_size)
        }
        let hidden = cfg.hidden_size;
        let in_proj = candle_nn::linear(in_channels, hidden, vb.pp("in_proj"))?;
        let cond_proj = candle_nn::linear(in_channels, hidden, vb.pp("cond_proj"))?;
        let out_proj = candle_nn::linear(hidden, in_channels, vb.pp("out_proj"))?;
        let time_embeddings = SinusoidalPosEmb::new(hidden, 1000.0)?;
        let time_mlp = TimestepEmbedding::new(hidden, hidden, None, vb.pp("time_mlp"))?;
        let delta_time_mlp = TimestepEmbedding::new(hidden, hidden, None, vb.pp("delta_time_mlp"))?;
        let decoder = MiniCpmModel::new(cfg, vb.pp("decoder"))?;
        Ok(Self {
            in_channels,
            hidden,
            in_proj,
            cond_proj,
            out_proj,
            time_embeddings,
            time_mlp,
            delta_time_mlp,
            decoder,
        })
    }

    /// x: [N, C, T]
    /// mu: [N, hidden]
    /// t: [N]
    /// cond: [N, C, T']
    /// dt: [N]
    /// returns velocity: [N, C, T]
    pub fn forward(
        &self,
        x: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        cond: &Tensor,
        dt: &Tensor,
    ) -> Result<Tensor> {
        let (n, c, t_len) = x.dims3()?;
        if c != self.in_channels {
            candle_core::bail!("locdit expects in_channels={}; got C={c}", self.in_channels)
        }
        let (n2, h) = mu.dims2()?;
        if n2 != n || h != self.hidden {
            candle_core::bail!(
                "locdit expects mu shape [N={n}, H={}], got [{n2}, {h}]",
                self.hidden
            )
        }
        let (n3, c3, prefix) = cond.dims3()?;
        if n3 != n || c3 != c {
            candle_core::bail!(
                "locdit expects cond shape [N={n}, C={c}, T'], got [{n3}, {c3}, {prefix}]"
            )
        }

        // Tokenize x/cond by projecting channel dim into hidden.
        let x = x.transpose(1, 2)?.contiguous()?; // [N, T, C]
        let x2 = x.reshape((n * t_len, c))?;
        let x2 = self.in_proj.forward(&x2)?;
        let x_tok = x2.reshape((n, t_len, self.hidden))?; // [N, T, H]

        let cond = cond.transpose(1, 2)?.contiguous()?; // [N, T', C]
        let cond2 = cond.reshape((n * prefix, c))?;
        let cond2 = self.cond_proj.forward(&cond2)?;
        let cond_tok = cond2.reshape((n, prefix, self.hidden))?; // [N, T', H]

        // Time embeddings (computed in f32 then cast to match x dtype).
        let dtype = x_tok.dtype();
        let t_emb = self.time_embeddings.forward(t)?.to_dtype(dtype)?;
        let t_emb = self.time_mlp.forward(&t_emb)?;
        let dt_emb = self.time_embeddings.forward(dt)?.to_dtype(dtype)?;
        let dt_emb = self.delta_time_mlp.forward(&dt_emb)?;
        let t_emb = (t_emb + dt_emb)?; // [N, H]

        let cls = (mu + t_emb)?.unsqueeze(1)?; // [N, 1, H]
        let seq = Tensor::cat(&[&cls, &cond_tok, &x_tok], D::Minus2)?; // concat on seq dim

        let seq_len = 1 + prefix + t_len;
        let pos = arange_cache::arange_u32(seq_len, x.device())?;
        let hidden = self.decoder.forward(&seq, &pos, false)?; // [N, seq, H]
        let hidden = hidden.narrow(1, prefix + 1, t_len)?; // [N, T, H]

        let h2 = hidden.reshape((n * t_len, self.hidden))?;
        let out = self.out_proj.forward(&h2)?;
        let out = out.reshape((n, t_len, c))?;
        out.transpose(1, 2)?.contiguous()
    }
}

impl VelocityEstimator for VoxCpmLocDiT {
    fn forward(
        &self,
        x: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        cond: &Tensor,
        dt: &Tensor,
    ) -> Result<Tensor> {
        VoxCpmLocDiT::forward(self, x, mu, t, cond, dt)
    }
}
