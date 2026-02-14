use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::infer_actor::InferenceActorHandle;
use voxcpm_ipc::JobId;

pub(crate) struct CurrentJob {
    pub(crate) job_id: JobId,
    pub(crate) cancel: Arc<AtomicBool>,
}

pub(crate) struct EngineState {
    pub(crate) infer: Option<InferenceActorHandle>,
    pub(crate) current: Option<CurrentJob>,
}
