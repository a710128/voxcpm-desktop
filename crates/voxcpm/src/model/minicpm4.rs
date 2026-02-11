//! MiniCPM4 transformer scaffold used by VoxCPM.
//!
//! Precision behavior is intentionally strict to match the reference Python implementation:
//! - RMSNorm computes variance in f32 then casts back.
//! - RoPE applies on q/k in f32 then casts back.

use crate::model::cache::StaticKvCache;
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{linear_no_bias, ops, Embedding, Linear, Module, VarBuilder};
use serde::Deserialize;
use std::sync::Arc;

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScalingConfig {
    pub r#type: String,
    pub long_factor: Vec<f64>,
    pub short_factor: Vec<f64>,
    pub original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MiniCpmConfig {
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_scaling: RopeScalingConfig,
    pub vocab_size: usize,
    pub use_mup: bool,
    pub scale_emb: f64,
    pub dim_model_base: usize,
    pub scale_depth: f64,
    pub rope_theta: f64,
    pub kv_channels: Option<usize>,
}

impl MiniCpmConfig {
    pub fn head_dim(&self) -> usize {
        self.kv_channels
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

#[derive(Debug, Clone)]
pub struct MiniCpmRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl MiniCpmRmsNorm {
    pub fn new(hidden: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(hidden, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        // Compute variance in float32 then cast back (Python behavior).
        let var = xs.to_dtype(DType::F32)?.sqr()?.mean_keepdim(D::Minus1)?;
        let inv = (var + self.eps)?.sqrt()?.recip()?.to_dtype(dtype)?;
        let xs = xs.broadcast_mul(&inv)?;
        xs.broadcast_mul(&self.weight)
    }
}

#[derive(Debug)]
pub struct RotaryLongRope {
    cos_cached: Tensor, // [max_len, head_dim] in f32
    sin_cached: Tensor, // [max_len, head_dim] in f32
    max_len: usize,
    head_dim: usize,
}

impl RotaryLongRope {
    pub fn new(cfg: &MiniCpmConfig, device: &Device) -> Result<Self> {
        let head_dim = cfg.head_dim();
        if head_dim % 2 != 0 {
            candle_core::bail!("rope head_dim must be even, got {head_dim}")
        }
        let max_len = cfg.max_position_embeddings;

        // Matches VoxCPM/src/voxcpm/modules/minicpm4/model.py:53.
        let orig_max_pos = cfg.rope_scaling.original_max_position_embeddings;
        let scale = (max_len as f64) / (orig_max_pos as f64);
        let scaling_factor = (1.0 + scale.ln() / (orig_max_pos as f64).ln()).sqrt();

        // Precompute RoPE caches in float32.
        let half = head_dim / 2;
        if cfg.rope_scaling.short_factor.len() != half || cfg.rope_scaling.long_factor.len() != half
        {
            candle_core::bail!(
                "rope_scaling factors must have length head_dim/2={half}, got short={} long={}",
                cfg.rope_scaling.short_factor.len(),
                cfg.rope_scaling.long_factor.len()
            )
        }

        let ext_factors: Vec<f32> = if max_len > orig_max_pos {
            cfg.rope_scaling
                .long_factor
                .iter()
                .map(|&v| v as f32)
                .collect()
        } else {
            cfg.rope_scaling
                .short_factor
                .iter()
                .map(|&v| v as f32)
                .collect()
        };
        let ext_factors = Tensor::from_vec(ext_factors, (half,), device)?;
        let inv_ext = ext_factors.recip()?; // [half]

        // inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| {
                let power = (2.0 * (i as f64)) / (head_dim as f64);
                (1.0 / cfg.rope_theta.powf(power)) as f32
            })
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (half,), device)?;

        let t: Vec<f32> = (0..max_len).map(|i| i as f32).collect();
        let t = Tensor::from_vec(t, (max_len, 1), device)?; // [max_len, 1]

        // freqs = outer(t, 1/ext_factors) * inv_freq
        let outer = t.broadcast_mul(&inv_ext.reshape((1, half))?)?; // [max_len, half]
        let freqs = outer.broadcast_mul(&inv_freq.reshape((1, half))?)?; // [max_len, half]

        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // [max_len, head_dim]
        let cos_cached = (emb.cos()? * scaling_factor)?;
        let sin_cached = (emb.sin()? * scaling_factor)?;

        Ok(Self {
            cos_cached,
            sin_cached,
            max_len,
            head_dim,
        })
    }

    pub fn get_cos_sin(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        // position_ids: [bs, seq] or [seq].
        let dtype = position_ids.dtype();
        if !matches!(
            dtype,
            DType::U8 | DType::U32 | DType::I64 | DType::I32 | DType::I16
        ) {
            candle_core::bail!("position_ids must be an integer tensor, got {dtype:?}")
        }
        let ids_u32 = position_ids.to_dtype(DType::U32)?;
        let flat = ids_u32.flatten_all()?;

        let flat_vec = flat.to_vec1::<u32>()?;
        let max_pos = flat_vec.iter().copied().max().unwrap_or(0) as usize;
        if max_pos >= self.max_len {
            candle_core::bail!(
                "position_id {max_pos} exceeds rope cache max_len {}",
                self.max_len
            )
        }
        let cos = self.cos_cached.index_select(&flat, 0)?;
        let sin = self.sin_cached.index_select(&flat, 0)?;

        // Restore the original shape and broadcast to [bs, 1, seq, head_dim].
        let final_dims = position_ids.dims();
        let (bs, seq) = match final_dims {
            [s] => (1usize, *s),
            [b, s] => (*b, *s),
            _ => candle_core::bail!("position_ids must have rank 1 or 2, got {:?}", final_dims),
        };
        let cos = cos.reshape((bs, seq, self.head_dim))?.unsqueeze(1)?;
        let sin = sin.reshape((bs, seq, self.head_dim))?.unsqueeze(1)?;
        Ok((cos, sin))
    }
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    // xs: [..., head_dim]
    let dims = xs.dims();
    let hd = *dims
        .last()
        .ok_or_else(|| candle_core::Error::Msg("rotate_half expects rank>=1".into()))?;
    if hd % 2 != 0 {
        candle_core::bail!("rotate_half expects even head_dim, got {hd}")
    }
    // Python reference uses chunk(2, dim=-1): split the last dim into two contiguous halves.
    let half = hd / 2;
    let x1 = xs.narrow(D::Minus1, 0, half)?;
    let x2 = xs.narrow(D::Minus1, half, half)?;
    let x2n = x2.neg()?;
    Tensor::cat(&[&x2n, &x1], D::Minus1)
}

fn apply_rope(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
    // Python behavior: cast q/k to float32 for RoPE then cast back.
    let q_dtype = q.dtype();
    let k_dtype = k.dtype();
    let qf = q.to_dtype(DType::F32)?;
    let kf = k.to_dtype(DType::F32)?;

    let q_rot = rotate_half(&qf)?;
    let k_rot = rotate_half(&kf)?;

    let q_out = (qf.broadcast_mul(cos)? + q_rot.broadcast_mul(sin)?)?.to_dtype(q_dtype)?;
    let k_out = (kf.broadcast_mul(cos)? + k_rot.broadcast_mul(sin)?)?.to_dtype(k_dtype)?;
    Ok((q_out, k_out))
}

fn repeat_kv(xs: &Tensor, repeat: usize) -> Result<Tensor> {
    if repeat == 1 {
        return Ok(xs.clone());
    }
    // xs: [bs, kv_heads, seq, head_dim] -> [bs, kv_heads*repeat, seq, head_dim]
    let (bs, kvh, seq, hd) = xs.dims4()?;
    let xs = xs.unsqueeze(2)?; // [bs, kvh, 1, seq, hd]
    let xs = xs.broadcast_as((bs, kvh, repeat, seq, hd))?;
    xs.reshape((bs, kvh * repeat, seq, hd))
}

#[derive(Debug)]
pub struct MiniCpmAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: Arc<RotaryLongRope>,
    cache: Option<StaticKvCache>,
    cache_max_len: usize,
    cache_dtype: DType,
    device: Device,
}

impl MiniCpmAttention {
    pub fn new(cfg: &MiniCpmConfig, rope: Arc<RotaryLongRope>, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        if num_heads % num_kv_heads != 0 {
            candle_core::bail!(
                "GQA requires num_attention_heads % num_key_value_heads == 0 (got {num_heads} and {num_kv_heads})"
            )
        }
        let head_dim = cfg.head_dim();
        let hidden = cfg.hidden_size;
        let q_proj = linear_no_bias(hidden, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden, vb.pp("o_proj"))?;

        let cache_dtype = vb.dtype;
        let device = vb.device().clone();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            rope,
            cache: None,
            cache_max_len: cfg.max_position_embeddings,
            cache_dtype,
            device,
        })
    }

    pub fn forward(&self, xs: &Tensor, position_ids: &Tensor, is_causal: bool) -> Result<Tensor> {
        Ok(self.forward_with_cache(xs, position_ids, is_causal)?.0)
    }

    /// Forward pass returning per-layer KV tensors (for cache prefill).
    ///
    /// Matches the Python behavior where `forward()` returns `(hidden, (k, v))` per layer.
    /// The returned `k/v` have shape `[bs, kv_heads, seq, head_dim]` (pre-GQA repeat).
    pub fn forward_with_cache(
        &self,
        xs: &Tensor,
        position_ids: &Tensor,
        is_causal: bool,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        let (bs, seq, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((bs, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?; // [bs, h, seq, hd]
        let k = k
            .reshape((bs, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?; // [bs, kvh, seq, hd]
        let v = v
            .reshape((bs, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (cos, sin) = self.rope.get_cos_sin(position_ids)?;
        let (q, k) = apply_rope(&q, &k, &cos, &sin)?;

        // Python reference calls contiguous() before SDPA.
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        // Return caches as kv-head tensors (before GQA repeat).
        let k_cache = k.clone();
        let v_cache = v.clone();

        let repeat = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, repeat)?;
        let v = repeat_kv(&v, repeat)?;

        // scores: [bs, h, seq, seq]
        let k_t = k.transpose(D::Minus2, D::Minus1)?;
        let scores = (q.matmul(&k_t)? / (self.head_dim as f64).sqrt())?;
        let scores = if is_causal {
            // Use -inf for masked (future) positions.
            // `allow` is 1 on/below diagonal and 0 above; log(allow) is 0 or -inf.
            let allow = Tensor::tril2(seq, DType::F32, xs.device())?;
            let m = allow.log()?.to_dtype(scores.dtype())?;
            scores.broadcast_add(&m.unsqueeze(0)?.unsqueeze(0)?)?
        } else {
            scores
        };

        let attn = ops::softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v)?; // [bs, h, seq, hd]
        let out = out
            .transpose(1, 2)?
            .reshape((bs, seq, self.num_heads * self.head_dim))?;
        let out = self.o_proj.forward(&out)?;
        Ok((out, (k_cache, v_cache)))
    }

    pub fn setup_cache(&mut self, bs: usize, max_len: usize) -> Result<()> {
        if max_len > self.cache_max_len {
            candle_core::bail!(
                "requested cache max_len={max_len} exceeds model max_position_embeddings={}",
                self.cache_max_len
            )
        }
        self.cache_max_len = max_len;
        self.cache = Some(StaticKvCache::new(
            &self.device,
            self.cache_dtype,
            bs,
            self.num_kv_heads,
            self.cache_max_len,
            self.head_dim,
        )?);
        Ok(())
    }

    pub fn fill_cache_prefix(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let (bs, kvh, seq, hd) = k.dims4()?;
        if v.dims4()? != (bs, kvh, seq, hd) {
            candle_core::bail!("k/v dims mismatch for cache fill")
        }
        if kvh != self.num_kv_heads || hd != self.head_dim {
            candle_core::bail!(
                "cache fill shape mismatch: expected kv_heads={} head_dim={}, got kv_heads={kvh} head_dim={hd}",
                self.num_kv_heads,
                self.head_dim
            )
        }
        if seq > self.cache_max_len {
            candle_core::bail!(
                "cache fill seq_len={seq} exceeds cache max_len={}",
                self.cache_max_len
            )
        }
        if self.cache.is_none() {
            // Allocate on demand using the current cache_max_len.
            self.setup_cache(bs, self.cache_max_len)?;
        }
        let cache = self.cache.as_mut().unwrap();
        cache.fill_prefix(k, v)
    }

    pub fn forward_step(&mut self, x: &Tensor, position_id: &Tensor) -> Result<Tensor> {
        // x: [bs, 1, hidden]
        let (bs, seq, _) = x.dims3()?;
        if seq != 1 {
            candle_core::bail!("forward_step expects seq==1, got {seq}")
        }
        let position_id = match position_id.dims() {
            [b] if *b == bs => position_id.reshape((bs, 1))?,
            [b, s] if *b == bs && *s == 1 => position_id.clone(),
            ds => candle_core::bail!("position_id must have shape [bs] or [bs,1], got {ds:?}"),
        };
        let pos = position_id.flatten_all()?.to_vec1::<u32>()?[0] as usize;
        if pos >= self.cache_max_len {
            candle_core::bail!(
                "position_id {pos} out of cache range (max_len={})",
                self.cache_max_len
            )
        }

        let q = self.q_proj.forward(x)?; // [bs, 1, h*hd]
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((bs, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?; // [bs, h, 1, hd]
        let k = k
            .reshape((bs, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?; // [bs, kvh, 1, hd]
        let v = v
            .reshape((bs, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (cos, sin) = self.rope.get_cos_sin(&position_id)?;
        let (q, k) = apply_rope(&q, &k, &cos, &sin)?;

        if self.cache.is_none() {
            self.cache = Some(StaticKvCache::new(
                &self.device,
                self.cache_dtype,
                bs,
                self.num_kv_heads,
                self.cache_max_len,
                self.head_dim,
            )?);
        }
        let cache = self.cache.as_mut().unwrap();
        cache.set(pos, &k.contiguous()?, &v.contiguous()?)?;
        let (k_all, v_all) = cache.slice(pos)?; // [bs, kvh, t, hd]

        let repeat = self.num_heads / self.num_kv_heads;
        let k_all = repeat_kv(&k_all, repeat)?;
        let v_all = repeat_kv(&v_all, repeat)?;

        // Attention over [0..pos+1] without explicit masking.
        let k_t = k_all.transpose(D::Minus2, D::Minus1)?; // [bs, h, hd, t]
        let scores = (q.matmul(&k_t)? / (self.head_dim as f64).sqrt())?; // [bs, h, 1, t]
        let attn = ops::softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v_all)?; // [bs, h, 1, hd]
        let out = out
            .transpose(1, 2)?
            .reshape((bs, 1, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }

    /// Debug helper for parity tests.
    ///
    /// Returns intermediate K-projection tensors matching the upstream layout:
    /// - `k_lin`: [bs, seq, kv_heads*head_dim]
    /// - `k_pre`: [bs, kv_heads, seq, head_dim] (pre-RoPE)
    /// - `k_post`: [bs, kv_heads, seq, head_dim] (post-RoPE)
    #[doc(hidden)]
    pub fn debug_kproj_tensors(
        &self,
        xs: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (bs, seq, _) = xs.dims3()?;
        let k_lin = self.k_proj.forward(xs)?;
        let k_pre = k_lin
            .contiguous()?
            .reshape((bs, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (cos, sin) = self.rope.get_cos_sin(position_ids)?;
        let q_dummy = Tensor::zeros(
            (bs, self.num_heads, seq, self.head_dim),
            k_pre.dtype(),
            xs.device(),
        )?;
        let (_q_ignored, k_post) = apply_rope(&q_dummy, &k_pre, &cos, &sin)?;
        Ok((k_lin, k_pre, k_post))
    }
}

#[derive(Debug)]
pub struct MiniCpmMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MiniCpmMlp {
    pub fn new(cfg: &MiniCpmConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let gate_proj = linear_no_bias(h, i, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(h, i, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(i, h, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let up = self.up_proj.forward(xs)?;
        let act = ops::silu(&gate)?;
        let fused = (&act * &up)?;
        self.down_proj.forward(&fused)
    }
}

#[derive(Debug)]
pub struct MiniCpmDecoderLayer {
    input_norm: MiniCpmRmsNorm,
    post_attn_norm: MiniCpmRmsNorm,
    self_attn: MiniCpmAttention,
    mlp: MiniCpmMlp,
    #[allow(dead_code)]
    use_mup: bool,
    #[allow(dead_code)]
    scale_depth: f64,

    // Precomputed residual scaling factor for MuP.
    resid_scale: f64,
}

impl MiniCpmDecoderLayer {
    pub fn new(cfg: &MiniCpmConfig, rope: Arc<RotaryLongRope>, vb: VarBuilder) -> Result<Self> {
        let input_norm =
            MiniCpmRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attn_norm = MiniCpmRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let self_attn = MiniCpmAttention::new(cfg, rope, vb.pp("self_attn"))?;
        let mlp = MiniCpmMlp::new(cfg, vb.pp("mlp"))?;
        let resid_scale = if cfg.use_mup {
            cfg.scale_depth / (cfg.num_hidden_layers as f64).sqrt()
        } else {
            1.0
        };
        Ok(Self {
            input_norm,
            post_attn_norm,
            self_attn,
            mlp,
            use_mup: cfg.use_mup,
            scale_depth: cfg.scale_depth,
            resid_scale,
        })
    }

    pub fn forward(&self, xs: &Tensor, position_ids: &Tensor, is_causal: bool) -> Result<Tensor> {
        let h = self.input_norm.forward(xs)?;
        let attn_out = self.self_attn.forward(&h, position_ids, is_causal)?;
        let attn_out = if self.use_mup {
            (&attn_out * self.resid_scale)?
        } else {
            attn_out
        };
        let xs = (xs + attn_out)?;
        let h = self.post_attn_norm.forward(&xs)?;
        let mlp_out = self.mlp.forward(&h)?;
        let mlp_out = if self.use_mup {
            (&mlp_out * self.resid_scale)?
        } else {
            mlp_out
        };
        xs + mlp_out
    }

    pub fn forward_with_cache(
        &self,
        xs: &Tensor,
        position_ids: &Tensor,
        is_causal: bool,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        let h = self.input_norm.forward(xs)?;
        let (attn_out, kv) = self
            .self_attn
            .forward_with_cache(&h, position_ids, is_causal)?;
        let attn_out = if self.use_mup {
            (&attn_out * self.resid_scale)?
        } else {
            attn_out
        };
        let xs = (xs + attn_out)?;
        let h = self.post_attn_norm.forward(&xs)?;
        let mlp_out = self.mlp.forward(&h)?;
        let mlp_out = if self.use_mup {
            (&mlp_out * self.resid_scale)?
        } else {
            mlp_out
        };
        Ok(((xs + mlp_out)?, kv))
    }

    pub fn forward_step(&mut self, x: &Tensor, position_id: &Tensor) -> Result<Tensor> {
        let h = self.input_norm.forward(x)?;
        let attn_out = self.self_attn.forward_step(&h, position_id)?;
        let attn_out = if self.use_mup {
            (&attn_out * self.resid_scale)?
        } else {
            attn_out
        };
        let x = (x + attn_out)?;
        let h = self.post_attn_norm.forward(&x)?;
        let mlp_out = self.mlp.forward(&h)?;
        let mlp_out = if self.use_mup {
            (&mlp_out * self.resid_scale)?
        } else {
            mlp_out
        };
        x + mlp_out
    }
}

#[derive(Debug)]
pub struct MiniCpmModel {
    #[allow(dead_code)]
    embed_tokens: Option<Embedding>,
    layers: Vec<MiniCpmDecoderLayer>,
    norm: MiniCpmRmsNorm,
    #[allow(dead_code)]
    cfg: MiniCpmConfig,

    // Cache position tracking (PyTorch StaticKVCache.current_length equivalent).
    kv_current_length: usize,
    kv_max_length: usize,
}

impl MiniCpmModel {
    pub fn new(cfg: MiniCpmConfig, vb: VarBuilder) -> Result<Self> {
        let rope = Arc::new(RotaryLongRope::new(&cfg, vb.device())?);
        let embed_tokens = if cfg.vocab_size == 0 {
            None
        } else {
            Some(candle_nn::embedding(
                cfg.vocab_size,
                cfg.hidden_size,
                vb.pp("embed_tokens"),
            )?)
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            layers.push(MiniCpmDecoderLayer::new(
                &cfg,
                rope.clone(),
                vb.pp("layers").pp(idx.to_string()),
            )?);
        }
        let norm = MiniCpmRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            cfg,
            kv_current_length: 0,
            kv_max_length: 0,
        })
    }

    /// Pre-allocate KV caches for incremental decoding.
    ///
    /// Mirrors `MiniCPMModel.setup_cache()` in the reference implementation.
    pub fn setup_cache(&mut self, batch_size: usize, max_length: usize) -> Result<()> {
        for layer in self.layers.iter_mut() {
            layer.self_attn.setup_cache(batch_size, max_length)?;
        }
        self.kv_current_length = 0;
        self.kv_max_length = max_length;
        Ok(())
    }

    /// Advance the internal cache position and return the previous index.
    pub fn step(&mut self) -> Result<u32> {
        if self.kv_max_length == 0 {
            candle_core::bail!("KV cache is not setup")
        }
        if self.kv_current_length >= self.kv_max_length {
            candle_core::bail!("KV cache is full")
        }
        let ret = self.kv_current_length as u32;
        self.kv_current_length += 1;
        Ok(ret)
    }

    /// Fill per-layer caches from a full forward pass.
    ///
    /// Mirrors `StaticKVCache.fill_caches()` in Python.
    pub fn fill_caches(&mut self, kv_caches: &[(Tensor, Tensor)]) -> Result<()> {
        if self.kv_max_length == 0 {
            candle_core::bail!("KV cache is not setup")
        }
        if kv_caches.len() != self.layers.len() {
            candle_core::bail!(
                "kv_caches length mismatch: expected {}, got {}",
                self.layers.len(),
                kv_caches.len()
            )
        }
        let (_, _, seq, _) = kv_caches
            .first()
            .ok_or_else(|| candle_core::Error::Msg("kv_caches is empty".into()))?
            .0
            .dims4()?;
        if seq > self.kv_max_length {
            candle_core::bail!(
                "cache fill seq_len={seq} exceeds configured max_length={}",
                self.kv_max_length
            )
        }
        self.kv_current_length = seq;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let (k, v) = &kv_caches[i];
            layer.self_attn.fill_cache_prefix(k, v)?;
        }
        Ok(())
    }

    /// Convenience helper for incremental decoding using the internal cache position.
    ///
    /// Equivalent to calling `step()` then `forward_step()` with the returned position id.
    pub fn forward_step_cached(&mut self, x_embed: &Tensor) -> Result<Tensor> {
        let (bs, seq, _) = x_embed.dims3()?;
        if seq != 1 {
            candle_core::bail!("forward_step_cached expects seq==1, got {seq}")
        }
        let pos = self.step()?;
        let pos_ids = Tensor::from_vec(vec![pos; bs], (bs,), x_embed.device())?;
        self.forward_step(x_embed, &pos_ids)
    }

    pub fn forward(
        &self,
        xs_embeds: &Tensor,
        position_ids: &Tensor,
        is_causal: bool,
    ) -> Result<Tensor> {
        let mut xs = xs_embeds.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, position_ids, is_causal)?;
        }
        self.norm.forward(&xs)
    }

    pub fn forward_with_cache(
        &self,
        xs_embeds: &Tensor,
        position_ids: &Tensor,
        is_causal: bool,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let mut xs = xs_embeds.clone();
        let mut caches = Vec::with_capacity(self.layers.len());
        for layer in self.layers.iter() {
            let (y, kv) = layer.forward_with_cache(&xs, position_ids, is_causal)?;
            xs = y;
            caches.push(kv);
        }
        let xs = self.norm.forward(&xs)?;
        Ok((xs, caches))
    }

    pub fn forward_step(&mut self, x_embed: &Tensor, position_id: &Tensor) -> Result<Tensor> {
        let mut x = x_embed.clone();
        for layer in self.layers.iter_mut() {
            x = layer.forward_step(&x, position_id)?;
        }
        self.norm.forward(&x)
    }

    pub fn cfg(&self) -> &MiniCpmConfig {
        &self.cfg
    }

    /// Token embedding lookup.
    ///
    /// Only available when `vocab_size != 0` (i.e. the text-semantic LM).
    pub fn embed_tokens(&self, token_ids: &Tensor) -> Result<Tensor> {
        let Some(emb) = &self.embed_tokens else {
            candle_core::bail!("embed_tokens() called on a model without vocab (vocab_size=0)")
        };
        emb.forward(token_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn minicpm4_forward_step_smoke_cpu() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = MiniCpmConfig {
            bos_token_id: 1,
            eos_token_id: 2,
            hidden_size: 16,
            intermediate_size: 32,
            max_position_embeddings: 8,
            num_attention_heads: 4,
            num_hidden_layers: 2,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-5,
            rope_scaling: RopeScalingConfig {
                r#type: "longrope".to_owned(),
                // head_dim = 4 => half = 2
                long_factor: vec![1.0, 1.0],
                short_factor: vec![1.0, 1.0],
                original_max_position_embeddings: 8,
            },
            vocab_size: 0,
            use_mup: false,
            scale_emb: 1.0,
            dim_model_base: 1,
            scale_depth: 1.0,
            rope_theta: 10_000.0,
            kv_channels: None,
        };
        let varmap = VarMap::new();
        // Use fp32 on CPU for broad op coverage.
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let mut model = MiniCpmModel::new(cfg, vb)?;

        let bs = 2usize;
        let x0 = Tensor::randn(0f32, 1f32, (bs, 1, 16), &dev)?;
        let p0 = Tensor::from_vec(vec![0u32; bs], (bs,), &dev)?;
        let y0 = model.forward_step(&x0, &p0)?;
        assert_eq!(y0.dims3()?, (bs, 1, 16));

        let x1 = Tensor::randn(0f32, 1f32, (bs, 1, 16), &dev)?;
        let p1 = Tensor::from_vec(vec![1u32; bs], (bs,), &dev)?;
        let y1 = model.forward_step(&x1, &p1)?;
        assert_eq!(y1.dims3()?, (bs, 1, 16));

        Ok(())
    }

    #[test]
    fn minicpm4_cache_fill_matches_forward() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = MiniCpmConfig {
            bos_token_id: 1,
            eos_token_id: 2,
            hidden_size: 16,
            intermediate_size: 32,
            max_position_embeddings: 32,
            num_attention_heads: 4,
            num_hidden_layers: 2,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-5,
            rope_scaling: RopeScalingConfig {
                r#type: "longrope".to_owned(),
                // head_dim = 4 => half = 2
                long_factor: vec![1.0, 1.0],
                short_factor: vec![1.0, 1.0],
                original_max_position_embeddings: 32,
            },
            vocab_size: 0,
            use_mup: true,
            scale_emb: 1.0,
            dim_model_base: 1,
            scale_depth: 1.0,
            rope_theta: 10_000.0,
            kv_channels: None,
        };

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let mut model = MiniCpmModel::new(cfg, vb)?;

        let bs = 1usize;
        let seq = 4usize;
        let xs = Tensor::randn(0f32, 1f32, (bs, seq, 16), &dev)?;
        let pos: Vec<u32> = (0..seq as u32).collect();
        let pos = Tensor::from_vec(pos, (seq,), &dev)?;

        let (_h, caches) = model.forward_with_cache(&xs, &pos, true)?;

        // Prefill cache from the forward pass.
        model.setup_cache(bs, seq + 1)?;
        model.fill_caches(&caches)?;
        assert_eq!(model.kv_current_length, seq);

        // Next token.
        let x_next = Tensor::randn(0f32, 1f32, (bs, 1, 16), &dev)?;
        let full = Tensor::cat(&[&xs, &x_next], D::Minus2)?;
        let pos2: Vec<u32> = (0..(seq as u32 + 1)).collect();
        let pos2 = Tensor::from_vec(pos2, (seq + 1,), &dev)?;
        let h_full = model.forward(&full, &pos2, true)?;
        let h_full_last = h_full.narrow(1, seq, 1)?;

        let p_next = Tensor::from_vec(vec![seq as u32; bs], (bs,), &dev)?;
        let h_step = model.forward_step(&x_next, &p_next)?;

        let diff = (&h_full_last - &h_step)?
            .abs()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let max_abs = diff
            .into_iter()
            .fold(0f32, |acc, v| if v > acc { v } else { acc });
        assert!(max_abs < 1e-4, "max_abs_diff={max_abs}");

        Ok(())
    }
}
