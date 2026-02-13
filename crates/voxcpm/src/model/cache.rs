use candle_core::{DType, Device, Result, Tensor, D};

/// A fixed-size KV cache pre-allocated for a single attention layer.
///
/// Layout matches the attention implementation in this crate:
/// - key/value: [bs, kv_heads, max_len, head_dim]
#[derive(Debug, Clone)]
pub struct StaticKvCache {
    pub k: Tensor,
    pub v: Tensor,
}

impl StaticKvCache {
    pub fn new(
        device: &Device,
        dtype: DType,
        bs: usize,
        kv_heads: usize,
        max_len: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let k = Tensor::zeros((bs, kv_heads, max_len, head_dim), dtype, device)?;
        let v = Tensor::zeros((bs, kv_heads, max_len, head_dim), dtype, device)?;
        Ok(Self { k, v })
    }

    /// Set a single position `pos` (0-based) with k/v shaped [bs, kv_heads, 1, head_dim].
    pub fn set(&mut self, pos: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        // In-place update (not compatible with backprop; fine for inference cache).
        self.k.slice_set(k, D::Minus2, pos)?;
        self.v.slice_set(v, D::Minus2, pos)?;
        Ok(())
    }

    /// Set per-batch positions using an index tensor (device-side).
    ///
    /// `position_id` must have shape `[bs]` or `[bs, 1]` and integer dtype.
    /// `k/v` must have shape `[bs, kv_heads, 1, head_dim]`.
    ///
    /// This avoids any device->host sync by using `scatter_set` along the
    /// sequence dimension (D::Minus2).
    pub fn set_at(&mut self, position_id: &Tensor, k: &Tensor, v: &Tensor) -> Result<()> {
        let (bs, kvh, _max_len, hd) = self.k.dims4()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        if k.dims4()? != (bs, kvh, 1, hd) {
            candle_core::bail!(
                "k dims mismatch: expected [bs={bs}, kvh={kvh}, 1, hd={hd}], got {:?}",
                k.dims()
            )
        }
        if v.dims4()? != (bs, kvh, 1, hd) {
            candle_core::bail!(
                "v dims mismatch: expected [bs={bs}, kvh={kvh}, 1, hd={hd}], got {:?}",
                v.dims()
            )
        }

        let pos = match position_id.dims() {
            [b] if *b == bs => position_id.reshape((bs, 1))?,
            [b, s] if *b == bs && *s == 1 => position_id.clone(),
            ds => candle_core::bail!("position_id must have shape [bs] or [bs,1], got {ds:?}"),
        };
        // scatter_set expects integer indexes; use U32 consistently.
        let pos_u32 = pos.to_dtype(DType::U32)?;
        // Broadcast to match source shape [bs, kvh, 1, hd].
        // scatter_set requires the index tensor to be contiguous on CUDA/Metal.
        let idx = pos_u32
            .reshape((bs, 1, 1, 1))?
            .broadcast_as((bs, kvh, 1, hd))?
            .contiguous()?;

        // In-place update (not compatible with backprop; fine for inference cache).
        self.k.scatter_set(&idx, &k, D::Minus2)?;
        self.v.scatter_set(&idx, &v, D::Minus2)?;
        Ok(())
    }

    /// Returns a slice of cached k/v up to and including `pos`.
    pub fn slice(&self, pos: usize) -> Result<(Tensor, Tensor)> {
        let len = pos + 1;
        Ok((
            self.k.narrow(D::Minus2, 0, len)?,
            self.v.narrow(D::Minus2, 0, len)?,
        ))
    }

    /// Clear the whole cache to zeros.
    pub fn clear(&mut self) -> Result<()> {
        let (bs, kvh, max_len, hd) = self.k.dims4()?;
        let dtype = self.k.dtype();
        let dev = self.k.device().clone();
        self.k = Tensor::zeros((bs, kvh, max_len, hd), dtype, &dev)?;
        self.v = Tensor::zeros((bs, kvh, max_len, hd), dtype, &dev)?;
        Ok(())
    }

    /// Fill the cache prefix with provided tensors.
    ///
    /// `k/v` must have shape `[bs, kv_heads, seq, head_dim]` and `seq <= max_len`.
    pub fn fill_prefix(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let (bs, kvh, seq, hd) = k.dims4()?;
        if v.dims4()? != (bs, kvh, seq, hd) {
            candle_core::bail!("k/v dims mismatch")
        }
        let (cbs, ckvh, max_len, chd) = self.k.dims4()?;
        if (bs, kvh, hd) != (cbs, ckvh, chd) {
            candle_core::bail!(
                "cache shape mismatch: expected [bs={cbs}, kvh={ckvh}, hd={chd}], got [bs={bs}, kvh={kvh}, hd={hd}]"
            )
        }
        if seq > max_len {
            candle_core::bail!("cache fill seq_len={seq} exceeds max_len={max_len}")
        }

        self.clear()?;
        self.k.slice_set(k, D::Minus2, 0)?;
        self.v.slice_set(v, D::Minus2, 0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_kv_cache_set_at_cpu_smoke() -> Result<()> {
        let dev = Device::Cpu;
        let bs = 2usize;
        let kvh = 3usize;
        let max_len = 5usize;
        let hd = 4usize;

        let mut cache = StaticKvCache::new(&dev, DType::F32, bs, kvh, max_len, hd)?;
        // Per-batch positions.
        let pos = Tensor::from_vec(vec![1u32, 3u32], (bs,), &dev)?;

        let k = Tensor::randn(0f32, 1f32, (bs, kvh, 1, hd), &dev)?;
        let v = Tensor::randn(0f32, 1f32, (bs, kvh, 1, hd), &dev)?;
        cache.set_at(&pos, &k, &v)?;
        Ok(())
    }
}
