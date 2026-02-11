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
