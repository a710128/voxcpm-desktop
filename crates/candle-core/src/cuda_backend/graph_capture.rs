//! Internal CUDA graph capture state.
//!
//! Goals:
//! - Ensure at most one CUDA graph capture is active globally.
//! - Provide a keepalive store for pinned host buffers used by HtoD copies during capture.
//!
//! This is intentionally `pub(crate)` and consumed by:
//! - `crate::cuda_graph` (capture/replay helper)
//! - `CudaDevice::clone_htod` (to keep host pointers alive across graph replays)

use std::any::Any;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use super::{CudaDevice, DeviceId, WrapErr};
use crate::Result;

use cudarc::driver::sys;
use cudarc::driver::{CudaContext, CudaEvent, CudaStream, DeviceRepr, HostSlice, SyncOnDrop};

#[derive(Debug)]
pub(crate) struct CaptureCtx {
    owner: std::thread::ThreadId,
    device_id: DeviceId,
    stream_ptr: usize,
    keepalive: Mutex<Vec<Box<dyn Any + Send + Sync>>>,
}

impl CaptureCtx {
    fn check_owner(&self) -> Result<()> {
        if std::thread::current().id() != self.owner {
            crate::bail!(
                "cuda graph capture is owned by a different thread (only the capture thread may call clone_htod during capture)"
            )
        }
        Ok(())
    }

    fn check_device_stream(&self, dev: &CudaDevice) -> Result<()> {
        if dev.id() != self.device_id {
            crate::bail!(
                "cuda graph capture is active on a different device (expected {:?}, got {:?})",
                self.device_id,
                dev.id()
            )
        }
        let ptr = Arc::as_ptr(&dev.cuda_stream()) as usize;
        if ptr != self.stream_ptr {
            crate::bail!(
                "cuda graph capture is active on a different stream (expected {:#x}, got {:#x})",
                self.stream_ptr,
                ptr
            )
        }
        Ok(())
    }

    pub(crate) fn validate(&self, dev: &CudaDevice) -> Result<()> {
        self.check_owner()?;
        self.check_device_stream(dev)?;
        Ok(())
    }

    pub(crate) fn push_keepalive(&self, v: Box<dyn Any + Send + Sync>) {
        // Keepalive is best-effort; if the mutex is poisoned, we still want to fail
        // deterministically elsewhere (capture will likely fail anyway).
        self.keepalive.lock().unwrap().push(v)
    }

    pub(crate) fn take_keepalive(&self) -> Vec<Box<dyn Any + Send + Sync>> {
        std::mem::take(&mut *self.keepalive.lock().unwrap())
    }
}

static GLOBAL_CAPTURE: OnceLock<Mutex<Option<Arc<CaptureCtx>>>> = OnceLock::new();

pub(crate) fn active_capture() -> Option<Arc<CaptureCtx>> {
    GLOBAL_CAPTURE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap()
        .clone()
}

pub(crate) struct GlobalCaptureGuard {
    ctx: Arc<CaptureCtx>,
    active: bool,
}

impl GlobalCaptureGuard {
    pub(crate) fn begin(dev: &CudaDevice) -> Result<Self> {
        let stream = dev.cuda_stream();
        let ctx = Arc::new(CaptureCtx {
            owner: std::thread::current().id(),
            device_id: dev.id(),
            stream_ptr: Arc::as_ptr(&stream) as usize,
            keepalive: Mutex::new(Vec::new()),
        });

        let slot = GLOBAL_CAPTURE.get_or_init(|| Mutex::new(None));
        let mut slot = slot.lock().unwrap();
        if slot.is_some() {
            crate::bail!("a cuda graph capture is already active")
        }
        *slot = Some(ctx.clone());
        drop(slot);

        Ok(Self { ctx, active: true })
    }

    pub(crate) fn finish(mut self) -> Vec<Box<dyn Any + Send + Sync>> {
        self.active = false;
        let keepalive = self.ctx.take_keepalive();
        let slot = GLOBAL_CAPTURE.get_or_init(|| Mutex::new(None));
        *slot.lock().unwrap() = None;
        keepalive
    }
}

impl Drop for GlobalCaptureGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let slot = GLOBAL_CAPTURE.get_or_init(|| Mutex::new(None));
        *slot.lock().unwrap() = None;
        // Keepalive drops with ctx.
    }
}

/// A pinned host allocation with a stable pointer for CUDA graph replay.
///
/// Stored as a non-generic keepalive so `CudaDevice::clone_htod` does not require
/// additional bounds on `T`.
#[derive(Debug)]
pub(crate) struct GraphPinnedHostAlloc {
    ptr: *mut c_void,
    bytes: usize,
    event: CudaEvent,
}

unsafe impl Send for GraphPinnedHostAlloc {}
unsafe impl Sync for GraphPinnedHostAlloc {}

impl GraphPinnedHostAlloc {
    pub(crate) fn alloc(ctx: &Arc<CudaContext>, bytes: usize) -> Result<Self> {
        ctx.bind_to_thread().w()?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            sys::cuMemHostAlloc(
                &mut ptr as *mut *mut c_void,
                bytes,
                sys::CU_MEMHOSTALLOC_WRITECOMBINED,
            )
            .result()
            .w()?;
        }
        debug_assert!(!ptr.is_null());

        // A blocking-sync event is fine here: we only ever use it for lifetime tracking.
        let event = ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))
            .w()?;

        Ok(Self { ptr, bytes, event })
    }

    pub(crate) fn copy_from_bytes(&mut self, src: *const u8, bytes: usize) {
        assert!(bytes <= self.bytes);
        unsafe {
            std::ptr::copy_nonoverlapping(src, self.ptr as *mut u8, bytes);
        }
    }

    pub(crate) fn view<'a, T: DeviceRepr>(&'a self, len: usize) -> GraphPinnedHostView<'a, T> {
        let bytes = len * std::mem::size_of::<T>();
        assert!(bytes <= self.bytes);
        GraphPinnedHostView {
            alloc: self,
            len,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl Drop for GraphPinnedHostAlloc {
    fn drop(&mut self) {
        let _ = self.event.synchronize();
        unsafe {
            let _ = sys::cuMemFreeHost(self.ptr).result();
        }
    }
}

pub(crate) struct GraphPinnedHostView<'a, T: DeviceRepr> {
    alloc: &'a GraphPinnedHostAlloc,
    len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T: DeviceRepr> HostSlice<T> for GraphPinnedHostView<'a, T> {
    fn len(&self) -> usize {
        self.len
    }

    unsafe fn stream_synced_slice<'b>(
        &'b self,
        stream: &'b CudaStream,
    ) -> (&'b [T], SyncOnDrop<'b>) {
        let _ = stream.wait(&self.alloc.event);
        (
            std::slice::from_raw_parts(self.alloc.ptr as *const T, self.len),
            SyncOnDrop::Record(Some((&self.alloc.event, stream))),
        )
    }

    unsafe fn stream_synced_mut_slice<'b>(
        &'b mut self,
        stream: &'b CudaStream,
    ) -> (&'b mut [T], SyncOnDrop<'b>) {
        let _ = stream.wait(&self.alloc.event);
        (
            std::slice::from_raw_parts_mut(self.alloc.ptr as *mut T, self.len),
            SyncOnDrop::Record(Some((&self.alloc.event, stream))),
        )
    }
}
