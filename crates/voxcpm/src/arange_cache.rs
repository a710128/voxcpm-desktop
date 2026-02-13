//! Cached arange helpers.
//!
//! Candle's `Tensor::arange` currently materializes on the host and uploads to the device.
//! For inference, we often re-create the same `[0..len)` tensors many times, so we cache
//! the largest requested arange per device and dtype and return narrow views for shorter
//! lengths.
//!
//! Thread-safety is intentionally not considered (per project preference). We use
//! `thread_local!` caches so callers don't need `&mut` access.

use candle_core::{Device, DeviceLocation, Result, Tensor};
use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum Kind {
    U32,
    F32,
}

#[derive(Debug, Clone)]
struct Entry {
    len: usize,
    t: Tensor, // 1D contiguous tensor [len]
}

thread_local! {
    static CACHE: RefCell<HashMap<(DeviceLocation, Kind), Entry>> = RefCell::new(HashMap::new());
}

fn get_or_build(kind: Kind, len: usize, device: &Device) -> Result<Tensor> {
    if len > (u32::MAX as usize) {
        candle_core::bail!("arange length too large: len={len}")
    }
    if len == 0 {
        // Keep this simple and allocation-free for callers: return an empty view.
        return match kind {
            Kind::U32 => Ok(Tensor::zeros((0usize,), candle_core::DType::U32, device)?),
            Kind::F32 => Ok(Tensor::zeros((0usize,), candle_core::DType::F32, device)?),
        };
    }

    let key = (device.location(), kind);
    CACHE.with(|cell| {
        let mut map = cell.borrow_mut();
        if let Some(e) = map.get(&key) {
            // `DeviceLocation` is not enough for Metal; guard with `same_device`.
            if e.len >= len && e.t.device().same_device(device) {
                return e.t.narrow(0, 0, len);
            }
        }

        let t = match kind {
            Kind::U32 => Tensor::arange(0u32, len as u32, device)?,
            Kind::F32 => Tensor::arange(0f32, len as f32, device)?,
        };
        map.insert(key, Entry { len, t: t.clone() });
        Ok(t)
    })
}

/// Returns a 1D `U32` tensor containing `[0, 1, ..., len-1]` on `device`.
pub fn arange_u32(len: usize, device: &Device) -> Result<Tensor> {
    get_or_build(Kind::U32, len, device)
}

/// Returns a 1D `F32` tensor containing `[0, 1, ..., len-1]` on `device`.
pub fn arange_f32(len: usize, device: &Device) -> Result<Tensor> {
    get_or_build(Kind::F32, len, device)
}

/// Ensure the cached `U32` arange is at least `len` long.
pub fn warm_u32(len: usize, device: &Device) -> Result<()> {
    let _ = arange_u32(len, device)?;
    Ok(())
}

/// Ensure the cached `F32` arange is at least `len` long.
#[allow(dead_code)]
pub fn warm_f32(len: usize, device: &Device) -> Result<()> {
    let _ = arange_f32(len, device)?;
    Ok(())
}
