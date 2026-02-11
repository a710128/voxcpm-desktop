//! AudioVAE (waveform <-> latent).
//!
//! VoxCPM-1.5 ships `audiovae.pth` with weight-norm Conv1d/ConvTranspose1d layers.
//! We support decode first (Milestone 2), focusing on exact checkpoint key alignment.

use candle_core::{DType, Result, Tensor, D};
use candle_nn::VarBuilder;
use serde::Deserialize;

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioVaeConfig {
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    #[serde(default = "default_true")]
    pub depthwise: bool,
    #[serde(default)]
    pub use_noise_block: bool,
    pub sample_rate: u32,
}

fn prod(xs: &[usize]) -> usize {
    xs.iter().copied().product()
}

#[derive(Debug, Clone)]
struct Snake1d {
    // Stored as [1, C, 1] in the checkpoint.
    alpha: Tensor,
}

impl Snake1d {
    fn from_vb(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get((1, channels, 1), "alpha")?;
        Ok(Self { alpha })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: [B, C, T]
        // Match the reference PyTorch implementation:
        // x + (alpha + eps)^-1 * sin(alpha*x)^2
        // AudioVAE is FP32-only. Run non-linear math in f32 for stability and kernel coverage.
        let x = xs.to_dtype(DType::F32)?;
        let alpha = self.alpha.to_dtype(DType::F32)?;
        let ax = x.broadcast_mul(&alpha)?;
        let sin2 = ax.sin()?.sqr()?;
        let inv = (&alpha + 1e-9)?.recip()?;
        x + sin2.broadcast_mul(&inv)?
    }
}

#[derive(Debug, Clone)]
struct WeightNorm {
    g: Tensor, // [C0, 1, 1]
    v: Tensor, // [C0, C1, K]
}

impl WeightNorm {
    fn from_vb(out_dim0: usize, in_dim1: usize, k: usize, vb: VarBuilder) -> Result<Self> {
        let g = vb.get((out_dim0, 1, 1), "weight_g")?;
        let v = vb.get((out_dim0, in_dim1, k), "weight_v")?;
        Ok(Self { g, v })
    }

    fn weight(&self, _dtype: DType) -> Result<Tensor> {
        // w = v * (g / ||v||)
        // norm over dims (1,2), keepdim => [C0, 1, 1]
        //
        // AudioVAE is FP32-only, so we always materialize weights as FP32.
        let v = self.v.to_dtype(DType::F32)?;
        let g = self.g.to_dtype(DType::F32)?;
        let v2 = v.sqr()?;
        let v2 = v2.sum_keepdim(D::Minus1)?; // [C0, C1, 1]
        let v2 = v2.sum_keepdim(1)?; // [C0, 1, 1]
        let denom = (v2 + 1e-12)?.sqrt()?;
        let scale = g.broadcast_div(&denom)?;
        Ok(v.broadcast_mul(&scale)?)
    }
}

#[derive(Debug, Clone)]
struct WnCausalConv1d {
    wn: WeightNorm,
    bias: Option<Tensor>, // [C_out]
    causal_padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
}

impl WnCausalConv1d {
    fn from_vb(
        c_out: usize,
        c_in_per_group: usize,
        k: usize,
        causal_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        has_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let wn = WeightNorm::from_vb(c_out, c_in_per_group, k, vb.clone())?;
        let bias = if has_bias {
            Some(vb.get(c_out, "bias")?)
        } else {
            None
        };
        Ok(Self {
            wn,
            bias,
            causal_padding,
            stride,
            dilation,
            groups,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w = self.wn.weight(xs.dtype())?;
        let xs = if self.causal_padding == 0 {
            xs.clone()
        } else {
            // PyTorch: F.pad(x, (padding*2, 0)) then conv1d(padding=0)
            let (b, c, _t) = xs.dims3()?;
            let z = Tensor::zeros((b, c, self.causal_padding * 2), xs.dtype(), xs.device())?;
            Tensor::cat(&[&z, xs], D::Minus1)?
        };
        let ys = xs.conv1d(&w, 0, self.stride, self.dilation, self.groups)?;
        match &self.bias {
            None => Ok(ys),
            Some(bias) => {
                let b = bias.to_dtype(ys.dtype())?;
                ys.broadcast_add(&b.reshape((1, b.dims1()?, 1))?)
            }
        }
    }
}

#[derive(Debug, Clone)]
struct WnCausalConvTranspose1d {
    wn: WeightNorm,
    bias: Tensor, // [C_out]
    causal_padding: usize,
    causal_output_padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
}

impl WnCausalConvTranspose1d {
    fn from_vb(
        c_in: usize,
        c_out: usize,
        k: usize,
        causal_padding: usize,
        causal_output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // ConvTranspose1d kernel layout in candle-core: [c_in, c_out, k]
        let wn = WeightNorm::from_vb(c_in, c_out, k, vb.clone())?;
        let bias = vb.get(c_out, "bias")?;
        Ok(Self {
            wn,
            bias,
            causal_padding,
            causal_output_padding,
            stride,
            dilation,
            groups,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w = self.wn.weight(xs.dtype())?;
        // Match the reference PyTorch implementation:
        // ConvTranspose1d is created with padding=0/output_padding=0, then we trim.
        let mut ys = xs.conv_transpose1d(&w, 0, 0, self.stride, self.dilation, self.groups)?;
        let trim = self
            .causal_padding
            .saturating_mul(2)
            .saturating_sub(self.causal_output_padding);
        if trim > 0 {
            let (_b, _c, t) = ys.dims3()?;
            if trim >= t {
                candle_core::bail!(
                    "invalid causal trim {trim} for conv_transpose output length {t}"
                )
            }
            ys = ys.narrow(D::Minus1, 0, t - trim)?;
        }
        let b = self.bias.to_dtype(ys.dtype())?;
        ys.broadcast_add(&b.reshape((1, b.dims1()?, 1))?)
    }
}

#[derive(Debug, Clone)]
struct ResUnit {
    a0: Snake1d,
    k7: WnCausalConv1d,
    a1: Snake1d,
    k1: WnCausalConv1d,
}

impl ResUnit {
    fn from_vb(ch: usize, dilation: usize, groups: usize, vb: VarBuilder) -> Result<Self> {
        // Matches keys:
        //   block.0.alpha
        //   block.1.{bias,weight_g,weight_v} (depthwise k=7)
        //   block.2.alpha
        //   block.3.{bias,weight_g,weight_v} (pointwise k=1)
        let a0 = Snake1d::from_vb(ch, vb.pp("block").pp("0"))?;
        if ch % groups != 0 {
            candle_core::bail!("invalid groups={groups} for channels={ch}")
        }
        let pad = ((7usize - 1) * dilation) / 2;
        let k7 = WnCausalConv1d::from_vb(
            ch,
            ch / groups,
            7,
            pad,
            1,
            dilation,
            groups,
            true,
            vb.pp("block").pp("1"),
        )?;
        let a1 = Snake1d::from_vb(ch, vb.pp("block").pp("2"))?;
        let k1 = WnCausalConv1d::from_vb(ch, ch, 1, 0, 1, 1, 1, true, vb.pp("block").pp("3"))?;
        Ok(Self { a0, k7, a1, k1 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut y = self.a0.forward(xs)?;
        y = self.k7.forward(&y)?;
        y = self.a1.forward(&y)?;
        y = self.k1.forward(&y)?;
        xs + y
    }
}

#[derive(Debug, Clone)]
struct EncoderStage {
    r0: ResUnit,
    r1: ResUnit,
    r2: ResUnit,
    pre: Snake1d,
    down: WnCausalConv1d,
}

impl EncoderStage {
    fn from_vb(
        c_in: usize,
        c_out: usize,
        stride: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Reference PyTorch order (CausalEncoderBlock):
        //   res(d=1), res(d=3), res(d=9), snake, strided conv
        let r0 = ResUnit::from_vb(c_in, 1, groups, vb.pp("block").pp("0"))?;
        let r1 = ResUnit::from_vb(c_in, 3, groups, vb.pp("block").pp("1"))?;
        let r2 = ResUnit::from_vb(c_in, 9, groups, vb.pp("block").pp("2"))?;
        let pre = Snake1d::from_vb(c_in, vb.pp("block").pp("3"))?;

        // Strided conv (groups=1).
        //
        // IMPORTANT: In the reference implementation, `depthwise` only affects the residual
        // units (their internal conv uses `groups`), but the downsampling conv here is always
        // a standard conv with `groups=1`.
        let k = 2 * stride;
        let pad = (stride + 1) / 2; // ceil(stride/2)
        let down = WnCausalConv1d::from_vb(
            c_out,
            c_in,
            k,
            pad,
            stride,
            1,
            1,
            true,
            vb.pp("block").pp("4"),
        )?;
        Ok(Self {
            r0,
            r1,
            r2,
            pre,
            down,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = self.r0.forward(xs)?;
        x = self.r1.forward(&x)?;
        x = self.r2.forward(&x)?;
        x = self.pre.forward(&x)?;
        self.down.forward(&x)
    }
}

#[derive(Debug, Clone)]
struct Encoder {
    in0: WnCausalConv1d,
    stages: Vec<EncoderStage>,
    fc_mu: WnCausalConv1d,
    fc_logvar: WnCausalConv1d,
}

impl Encoder {
    fn from_vb(cfg: &AudioVaeConfig, vb: VarBuilder) -> Result<Self> {
        // Checkpoint keys under `encoder.*`.
        let vbe = vb.pp("encoder");

        // encoder.block.0: conv 1 -> encoder_dim, k=7, pad=3
        let mut d_model = cfg.encoder_dim;
        let in0 =
            WnCausalConv1d::from_vb(d_model, 1, 7, 3, 1, 1, 1, true, vbe.pp("block").pp("0"))?;

        // encoder.block.1..: downsample stages
        let mut stages = Vec::with_capacity(cfg.encoder_rates.len());
        for (i, &stride) in cfg.encoder_rates.iter().enumerate() {
            let c_in = d_model;
            d_model = d_model.saturating_mul(2);
            let c_out = d_model;
            let groups = if cfg.depthwise { c_in } else { 1 };
            stages.push(EncoderStage::from_vb(
                c_in,
                c_out,
                stride,
                groups,
                vbe.pp("block").pp((1 + i).to_string()),
            )?);
        }

        // encoder.fc_mu / encoder.fc_logvar
        let fc_mu = WnCausalConv1d::from_vb(
            cfg.latent_dim,
            d_model,
            3,
            1,
            1,
            1,
            1,
            true,
            vbe.pp("fc_mu"),
        )?;
        let fc_logvar = WnCausalConv1d::from_vb(
            cfg.latent_dim,
            d_model,
            3,
            1,
            1,
            1,
            1,
            true,
            vbe.pp("fc_logvar"),
        )?;
        Ok(Self {
            in0,
            stages,
            fc_mu,
            fc_logvar,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut h = self.in0.forward(xs)?;
        for st in self.stages.iter() {
            h = st.forward(&h)?;
        }
        let mu = self.fc_mu.forward(&h)?;
        let logvar = self.fc_logvar.forward(&h)?;
        Ok((mu, logvar))
    }
}

#[derive(Debug, Clone)]
struct NoiseBlock {
    linear: WnCausalConv1d,
}

impl NoiseBlock {
    fn from_vb(dim: usize, vb: VarBuilder) -> Result<Self> {
        // Matches keys under `linear.*` (bias is absent).
        let linear = WnCausalConv1d::from_vb(dim, dim, 1, 0, 1, 1, 1, false, vb.pp("linear"))?;
        Ok(Self { linear })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: [B, C, T]
        let (b, c, t) = xs.dims3()?;
        let noise = Tensor::randn(0f32, 1f32, (b, 1usize, t), xs.device())?.to_dtype(xs.dtype())?;
        let h = self.linear.forward(xs)?;
        let n = noise.broadcast_as((b, c, t))?.broadcast_mul(&h)?;
        xs + n
    }
}

#[derive(Debug, Clone)]
struct UpsampleStage {
    pre: Snake1d,
    up: WnCausalConvTranspose1d,
    noise: Option<NoiseBlock>,
    res: [ResUnit; 3],
}

impl UpsampleStage {
    fn from_vb(
        c_in: usize,
        c_out: usize,
        stride: usize,
        groups: usize,
        use_noise_block: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Keys under e.g. decoder.model.2.block.*
        let pre = Snake1d::from_vb(c_in, vb.pp("block").pp("0"))?;

        let k = 2 * stride;
        let padding = (stride + 1) / 2; // ceil(stride/2)
        let output_padding = stride % 2;
        let up = WnCausalConvTranspose1d::from_vb(
            c_in,
            c_out,
            k,
            padding,
            output_padding,
            stride,
            1,
            1,
            vb.pp("block").pp("1"),
        )?;

        let (noise, r0_idx) = if use_noise_block {
            (
                Some(NoiseBlock::from_vb(c_out, vb.pp("block").pp("2"))?),
                3usize,
            )
        } else {
            (None, 2usize)
        };

        let r0 = ResUnit::from_vb(c_out, 1, groups, vb.pp("block").pp(r0_idx.to_string()))?;
        let r1 = ResUnit::from_vb(
            c_out,
            3,
            groups,
            vb.pp("block").pp((r0_idx + 1).to_string()),
        )?;
        let r2 = ResUnit::from_vb(
            c_out,
            9,
            groups,
            vb.pp("block").pp((r0_idx + 2).to_string()),
        )?;
        Ok(Self {
            pre,
            up,
            noise,
            res: [r0, r1, r2],
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = self.pre.forward(xs)?;
        x = self.up.forward(&x)?;
        if let Some(nb) = &self.noise {
            x = nb.forward(&x)?;
        }
        // Reference is sequential residual units (not parallel averaging).
        x = self.res[0].forward(&x)?;
        x = self.res[1].forward(&x)?;
        self.res[2].forward(&x)
    }
}

#[derive(Debug, Clone)]
struct Decoder {
    in_proj: InProj,
    stages: Vec<UpsampleStage>,
    out_act: Snake1d,
    out: WnCausalConv1d,
}

#[derive(Debug, Clone)]
enum InProj {
    Depthwise {
        dw: WnCausalConv1d,
        pw: WnCausalConv1d,
    },
    Plain {
        conv: WnCausalConv1d,
    },
}

impl Decoder {
    fn from_vb(cfg: &AudioVaeConfig, vb: VarBuilder) -> Result<Self> {
        // Checkpoint keys are expected under `decoder.model.*`.
        let vbd = vb.pp("decoder").pp("model");

        // model.0/1: initial projection (depthwise or plain) to decoder_dim.
        let (in_proj, stage0) = if cfg.depthwise {
            let dw = WnCausalConv1d::from_vb(
                cfg.latent_dim,
                1,
                7,
                3,
                1,
                1,
                cfg.latent_dim,
                true,
                vbd.pp("0"),
            )?;
            let pw = WnCausalConv1d::from_vb(
                cfg.decoder_dim,
                cfg.latent_dim,
                1,
                0,
                1,
                1,
                1,
                true,
                vbd.pp("1"),
            )?;
            (InProj::Depthwise { dw, pw }, 2usize)
        } else {
            let conv = WnCausalConv1d::from_vb(
                cfg.decoder_dim,
                cfg.latent_dim,
                7,
                3,
                1,
                1,
                1,
                true,
                vbd.pp("0"),
            )?;
            (InProj::Plain { conv }, 1usize)
        };

        // model.{stage0}..: upsample stages
        let mut stages = Vec::with_capacity(cfg.decoder_rates.len());
        let mut c_in = cfg.decoder_dim;
        for (i, &rate) in cfg.decoder_rates.iter().enumerate() {
            let c_out = (c_in / 2).max(1);
            let groups = if cfg.depthwise { c_out } else { 1 };
            stages.push(UpsampleStage::from_vb(
                c_in,
                c_out,
                rate,
                groups,
                cfg.use_noise_block,
                vbd.pp((stage0 + i).to_string()),
            )?);
            c_in = c_out;
        }

        let out_act_idx = stage0 + cfg.decoder_rates.len();
        let out_idx = out_act_idx + 1;
        // model.{out_act_idx}: snake
        let out_act = Snake1d::from_vb(c_in, vbd.pp(out_act_idx.to_string()))?;
        // model.{out_idx}: final conv -> 1
        let out =
            WnCausalConv1d::from_vb(1, c_in, 7, 3, 1, 1, 1, true, vbd.pp(out_idx.to_string()))?;

        Ok(Self {
            in_proj,
            stages,
            out_act,
            out,
        })
    }

    fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let mut x = match &self.in_proj {
            InProj::Depthwise { dw, pw } => {
                let x = dw.forward(z)?;
                pw.forward(&x)?
            }
            InProj::Plain { conv } => conv.forward(z)?,
        };
        for st in self.stages.iter() {
            x = st.forward(&x)?;
        }
        x = self.out_act.forward(&x)?;
        self.out.forward(&x)?.tanh()
    }
}

#[derive(Debug, Clone)]
pub struct AudioVae {
    pub cfg: AudioVaeConfig,
    encoder: Encoder,
    decoder: Decoder,
}

impl AudioVae {
    pub fn new(cfg: AudioVaeConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = Encoder::from_vb(&cfg, vb.clone())?;
        let decoder = Decoder::from_vb(&cfg, vb)?;
        Ok(Self {
            cfg,
            encoder,
            decoder,
        })
    }

    /// Number of waveform samples produced per latent timestep.
    ///
    /// Mirrors Python's `audio_vae.chunk_size` used to compute `patch_len = patch_size * chunk_size`.
    pub fn chunk_size(&self) -> usize {
        prod(&self.cfg.decoder_rates)
    }

    fn preprocess_audio(&self, audio: &Tensor, sample_rate: u32) -> Result<Tensor> {
        // Match reference behavior: right-pad waveform to hop_length (prod(encoder_rates)).
        if sample_rate != self.cfg.sample_rate {
            candle_core::bail!(
                "AudioVAE expects sample_rate={}, got {}",
                self.cfg.sample_rate,
                sample_rate
            )
        }

        let audio = match audio.dims() {
            [b, t] => audio.unsqueeze(1)?.reshape((*b, 1usize, *t))?,
            [_b, c, _t] => {
                if *c != 1 {
                    candle_core::bail!("AudioVAE expects mono audio with C=1, got C={c}")
                }
                audio.clone()
            }
            ds => candle_core::bail!("AudioVAE expects audio rank 2 or 3, got dims={ds:?}"),
        };

        let (b, c, t) = audio.dims3()?;
        let hop = prod(&self.cfg.encoder_rates);
        if hop == 0 {
            candle_core::bail!("invalid encoder_rates: product is 0")
        }
        let padded = ((t + hop - 1) / hop) * hop;
        let right_pad = padded - t;
        if right_pad == 0 {
            return Ok(audio);
        }
        let z = Tensor::zeros((b, c, right_pad), audio.dtype(), audio.device())?;
        Tensor::cat(&[&audio, &z], D::Minus1)
    }

    /// Encode waveform into latent codes.
    ///
    /// - input: `[B, T]` or `[B, 1, T]` (mono)
    /// - output: `[B, latent_dim, T_latent]`
    ///
    /// This matches the reference PyTorch inference path: returns `mu` only.
    pub fn encode(&self, audio: &Tensor, sample_rate: u32) -> Result<Tensor> {
        // AudioVAE is FP32-only (see loader in `crates/voxcpm/src/lib.rs`).
        // Force input activations to FP32 so all ops/params stay consistent.
        let audio = audio.to_dtype(DType::F32)?;
        let x = self.preprocess_audio(&audio, sample_rate)?;
        let (mu, _logvar) = self.encoder.forward(&x)?;
        Ok(mu)
    }

    /// Decode latent `[B, latent_dim, T]` into waveform `[B, 1, samples]`.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        // AudioVAE is FP32-only (see loader in `crates/voxcpm/src/lib.rs`).
        // Force input activations to FP32 so all ops/params stay consistent.
        let z = z.to_dtype(DType::F32)?;
        let (_b, c, _t) = z.dims3()?;
        if c != self.cfg.latent_dim {
            candle_core::bail!(
                "decode expects latent_dim={}, got {}",
                self.cfg.latent_dim,
                c
            )
        }
        self.decoder.forward(&z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn audiovae_decode_smoke_random_weights() -> Result<()> {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

        // This only validates shapes and numerics on random weights.
        // Use small dims for a fast CPU smoke test.
        let cfg = AudioVaeConfig {
            encoder_dim: 8,
            encoder_rates: vec![2, 2, 2, 2, 2],
            latent_dim: 4,
            decoder_dim: 64,
            decoder_rates: vec![2, 2, 2, 2],
            depthwise: true,
            use_noise_block: false,
            sample_rate: 44100,
        };

        let up: usize = prod(&cfg.decoder_rates);

        // Populate the varmap by requesting all parameters (so forward won't error).
        let vae = AudioVae::new(cfg, vb)?;

        let z = Tensor::randn(0f32, 1f32, (2usize, 4usize, 8usize), &dev)?;
        let y = vae.decode(&z)?;
        let (b, c, t) = y.dims3()?;
        assert_eq!((b, c), (2, 1));
        assert_eq!(t, 8 * up);
        let yv = y.flatten_all()?.to_vec1::<f32>()?;
        assert!(yv.iter().all(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn audiovae_encode_preprocess_pad() -> Result<()> {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

        let cfg = AudioVaeConfig {
            encoder_dim: 8,
            encoder_rates: vec![2, 2], // hop_length=4
            latent_dim: 4,
            decoder_dim: 32,
            decoder_rates: vec![2, 2],
            depthwise: true,
            use_noise_block: false,
            sample_rate: 16000,
        };
        let vae = AudioVae::new(cfg.clone(), vb)?;

        // [B, T] input; T=10 should pad to 12, then downsample by hop_length=4 => T_latent=3.
        let audio = Tensor::randn(0f32, 1f32, (2usize, 10usize), &dev)?;
        let z = vae.encode(&audio, 16000)?;
        let (b, c, t) = z.dims3()?;
        assert_eq!((b, c, t), (2, cfg.latent_dim, 3));
        Ok(())
    }
}
