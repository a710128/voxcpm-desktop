//! CUDA graph capture/replay helpers.
//!
//! This module is CUDA-only and intended for smoke testing / experimentation.

use std::any::Any;
use std::ffi::c_void;

use crate::{DType, InplaceOp2, Layout, Result, Tensor};

use crate::backend::BackendStorage;
use crate::cuda_backend::cudarc::driver::safe::CudaGraph;
use crate::cuda_backend::cudarc::driver::sys;
use crate::cuda_backend::{CudaStorage, CudaStorageSlice, GlobalCaptureGuard};
use crate::CpuStorage;
use crate::Device;

fn cudarc_err<E: std::error::Error + Send + Sync + 'static>(e: E) -> crate::Error {
    crate::Error::Cuda(Box::new(e))
}

fn set_mempool_release_threshold(device: &Device, threshold: u64) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    let cuda = device.as_cuda_device()?;
    let stream = cuda.cuda_stream();
    let ctx = stream.context();
    ctx.bind_to_thread().map_err(cudarc_err)?;

    unsafe {
        let mut dev: sys::CUdevice = 0;
        sys::cuCtxGetDevice(&mut dev as *mut sys::CUdevice)
            .result()
            .map_err(cudarc_err)?;

        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        sys::cuDeviceGetDefaultMemPool(&mut pool as *mut sys::CUmemoryPool, dev)
            .result()
            .map_err(cudarc_err)?;

        let mut threshold = threshold;
        sys::cuMemPoolSetAttribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&mut threshold as *mut u64).cast::<c_void>(),
        )
        .result()
        .map_err(cudarc_err)?;
    }

    Ok(())
}

/// Configure CUDA async allocator's default memory pool to never release memory.
///
/// This prevents the pool from returning memory to the OS/driver, which is important
/// for stability when capturing CUDA graphs that rely on stable device pointers.
pub fn set_mempool_release_threshold_max(device: &Device) -> Result<()> {
    set_mempool_release_threshold(device, u64::MAX)
}

struct CopyDtod;

impl InplaceOp2 for CopyDtod {
    fn name(&self) -> &'static str {
        "copy_dtod"
    }

    fn cpu_fwd(
        &self,
        _dst: &mut CpuStorage,
        _dst_layout: &Layout,
        _src: &CpuStorage,
        _src_layout: &Layout,
    ) -> Result<()> {
        crate::bail!("copy_dtod: cpu_fwd not supported")
    }

    fn cuda_fwd(
        &self,
        dst: &mut CudaStorage,
        dst_layout: &Layout,
        src: &CudaStorage,
        src_layout: &Layout,
    ) -> Result<()> {
        if dst.dtype() != src.dtype() {
            crate::bail!(
                "copy_dtod: dtype mismatch {:?} != {:?}",
                dst.dtype(),
                src.dtype()
            )
        }
        if !dst_layout.is_contiguous() || !src_layout.is_contiguous() {
            crate::bail!("copy_dtod: expected contiguous layouts")
        }

        let n = dst_layout.shape().elem_count();
        if n != src_layout.shape().elem_count() {
            crate::bail!(
                "copy_dtod: elem_count mismatch {} != {}",
                n,
                src_layout.shape().elem_count()
            )
        }

        let ds = dst_layout.start_offset();
        let ss = src_layout.start_offset();

        // Borrow the device first to avoid aliasing the mutable storage borrow.
        let dev = dst.device.clone();

        macro_rules! dtod {
            ($src:expr, $dst:expr, $tag:literal) => {{
                if ss + n > $src.len() {
                    crate::bail!(
                        "copy_dtod: src out of bounds {} > {} ({})",
                        ss + n,
                        $src.len(),
                        $tag
                    )
                }
                if ds + n > $dst.len() {
                    crate::bail!(
                        "copy_dtod: dst out of bounds {} > {} ({})",
                        ds + n,
                        $dst.len(),
                        $tag
                    )
                }
                let src_view = $src.slice(ss..ss + n);
                let mut dst_view = $dst.slice_mut(ds..ds + n);
                dev.cuda_stream()
                    .memcpy_dtod(&src_view, &mut dst_view)
                    .map_err(cudarc_err)?;
                Ok(())
            }};
        }

        match (&src.slice, &mut dst.slice) {
            (CudaStorageSlice::U8(s), CudaStorageSlice::U8(d)) => dtod!(s, d, "u8"),
            (CudaStorageSlice::U32(s), CudaStorageSlice::U32(d)) => dtod!(s, d, "u32"),
            (CudaStorageSlice::I16(s), CudaStorageSlice::I16(d)) => dtod!(s, d, "i16"),
            (CudaStorageSlice::I32(s), CudaStorageSlice::I32(d)) => dtod!(s, d, "i32"),
            (CudaStorageSlice::I64(s), CudaStorageSlice::I64(d)) => dtod!(s, d, "i64"),
            (CudaStorageSlice::BF16(s), CudaStorageSlice::BF16(d)) => dtod!(s, d, "bf16"),
            (CudaStorageSlice::F16(s), CudaStorageSlice::F16(d)) => dtod!(s, d, "f16"),
            (CudaStorageSlice::F32(s), CudaStorageSlice::F32(d)) => dtod!(s, d, "f32"),
            (CudaStorageSlice::F64(s), CudaStorageSlice::F64(d)) => dtod!(s, d, "f64"),
            (CudaStorageSlice::F8E4M3(s), CudaStorageSlice::F8E4M3(d)) => dtod!(s, d, "f8e4m3"),
            (CudaStorageSlice::F6E2M3(s), CudaStorageSlice::F6E2M3(d)) => dtod!(s, d, "f6e2m3"),
            (CudaStorageSlice::F6E3M2(s), CudaStorageSlice::F6E3M2(d)) => dtod!(s, d, "f6e3m2"),
            (CudaStorageSlice::F4(s), CudaStorageSlice::F4(d)) => dtod!(s, d, "f4"),
            (CudaStorageSlice::F8E8M0(s), CudaStorageSlice::F8E8M0(d)) => dtod!(s, d, "f8e8m0"),
            _ => crate::bail!(
                "copy_dtod: unexpected storage variants (dtype={:?})",
                dst.dtype()
            ),
        }
    }
}

/// A small helper that captures a CUDA graph from a build closure, then replays it.
///
/// - Capture allocates internal input slots (same dtype/shape as `example_inputs`).
/// - `run()` D2D-copies caller-provided inputs into those slots, launches the graph,
///   then D2D-copies the captured output into a fresh tensor and synchronizes.
/// - Shape/dtype/device mismatch at run-time is an error.
pub struct CudaGraphModule {
    cuda: Device,
    in_specs: Vec<(DType, Vec<usize>)>,
    in_slots: Vec<Tensor>,
    out_slot: Tensor,
    graph: CudaGraph,
    // Pinned host buffers created during capture (e.g. via clone_htod) must remain alive
    // for the lifetime of the captured graph.
    _keepalive: Vec<Box<dyn Any + Send + Sync>>,
}

impl CudaGraphModule {
    pub fn device(&self) -> &Device {
        &self.cuda
    }

    pub fn capture<F>(example_inputs: &[Tensor], build: F) -> Result<Self>
    where
        F: Fn(&[Tensor]) -> Result<Tensor>,
    {
        if example_inputs.is_empty() {
            crate::bail!("cuda graph capture: expected at least 1 input")
        }

        let cuda = example_inputs[0].device().clone();
        if !cuda.is_cuda() {
            crate::bail!("cuda graph capture: inputs must be cuda tensors")
        }
        for (i, t) in example_inputs.iter().enumerate() {
            if !t.device().same_device(&cuda) {
                crate::bail!("cuda graph capture: input[{i}] is on a different device")
            }
        }

        // Disable cudarc per-slice CUDA event tracking so captures don't record events.
        // Safety: this module uses a single stream and synchronizes explicitly.
        unsafe { cuda.as_cuda_device()?.disable_event_tracking() };

        let debug = std::env::var("CANDLE_CUDA_GRAPH_DEBUG").is_ok()
            || std::env::var("VOXCPM_CUDA_GRAPH_DEBUG").is_ok();
        if debug {
            eprintln!(
                "event_tracking={} (expected false for cuda graph capture)",
                cuda.as_cuda_device()?.is_event_tracking()
            );
        }

        let pools_supported = cuda
            .as_cuda_device()?
            .cuda_stream()
            .context()
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED)
            .map_err(cudarc_err)?;
        if pools_supported <= 0 {
            crate::bail!(
                "CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED is {pools_supported}; async alloc (has_async_alloc) is disabled"
            )
        }
        if debug {
            eprintln!("memory_pools_supported={pools_supported} (async alloc enabled)");
        }

        let in_specs: Vec<_> = example_inputs
            .iter()
            .map(|t| (t.dtype(), t.dims().to_vec()))
            .collect();
        let in_slots: Vec<_> = example_inputs
            .iter()
            .map(|t| Tensor::zeros(t.shape(), t.dtype(), &cuda))
            .collect::<Result<Vec<_>>>()?;

        // Copy example inputs into the captured input slots (D2D memcpy async).
        for (i, (slot, src)) in in_slots.iter().zip(example_inputs).enumerate() {
            if slot.dtype() != src.dtype() || slot.dims() != src.dims() {
                crate::bail!(
                    "cuda graph capture: internal slot[{i}] spec mismatch: {:?}{:?} vs {:?}{:?}",
                    slot.dtype(),
                    slot.dims(),
                    src.dtype(),
                    src.dims()
                )
            }
            slot.inplace_op2(src, &CopyDtod)?;
        }

        // Warm-up outside capture to force kernel/module loading.
        let _warm = build(&in_slots)?;
        cuda.as_cuda_device()?
            .cuda_stream()
            .synchronize()
            .map_err(cudarc_err)?;

        let stream = cuda.as_cuda_device()?.cuda_stream();
        stream.synchronize().map_err(cudarc_err)?;

        // Install the global capture state so clone_htod can keep pinned host pointers alive.
        let cap = GlobalCaptureGuard::begin(cuda.as_cuda_device()?)?;

        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(cudarc_err)?;

        let out_slot = build(&in_slots)?;

        let graph = stream
            .end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .map_err(cudarc_err)?
            .ok_or_else(|| crate::Error::Msg("cuda graph capture returned null graph".into()))?;

        // Ensure capture completes before returning.
        stream.synchronize().map_err(cudarc_err)?;

        let keepalive = cap.finish();

        Ok(Self {
            cuda,
            in_specs,
            in_slots,
            out_slot,
            graph,
            _keepalive: keepalive,
        })
    }

    pub fn run(&self, inputs: &[Tensor]) -> Result<Tensor> {
        if inputs.len() != self.in_specs.len() {
            crate::bail!(
                "cuda graph run: input count mismatch {} != {}",
                inputs.len(),
                self.in_specs.len()
            )
        }
        for (i, (t, (dt, dims))) in inputs.iter().zip(self.in_specs.iter()).enumerate() {
            if t.dtype() != *dt {
                crate::bail!(
                    "cuda graph run: input[{i}] dtype mismatch {:?} != {:?}",
                    t.dtype(),
                    dt
                )
            }
            if t.dims() != dims.as_slice() {
                crate::bail!(
                    "cuda graph run: input[{i}] shape mismatch {:?} != {:?}",
                    t.dims(),
                    dims
                )
            }
            if !t.device().is_cuda() || !t.device().same_device(&self.cuda) {
                crate::bail!("cuda graph run: input[{i}] must be on the same cuda device/stream")
            }
        }

        // Copy caller inputs into captured slots (D2D memcpy async).
        for (slot, src) in self.in_slots.iter().zip(inputs.iter()) {
            slot.inplace_op2(src, &CopyDtod)?;
        }

        // Ensure input copies complete before replay.
        self.cuda
            .as_cuda_device()?
            .cuda_stream()
            .synchronize()
            .map_err(cudarc_err)?;

        // Replay captured graph.
        self.graph.launch().map_err(cudarc_err)?;

        // Ensure replay completes before consuming outputs.
        self.cuda
            .as_cuda_device()?
            .cuda_stream()
            .synchronize()
            .map_err(cudarc_err)?;

        // Copy output snapshot and return it.
        let out_copy = Tensor::zeros(self.out_slot.shape(), self.out_slot.dtype(), &self.cuda)?;
        out_copy.inplace_op2(&self.out_slot, &CopyDtod)?;

        // Always sync before returning.
        self.cuda
            .as_cuda_device()?
            .cuda_stream()
            .synchronize()
            .map_err(cudarc_err)?;

        Ok(out_copy)
    }
}
