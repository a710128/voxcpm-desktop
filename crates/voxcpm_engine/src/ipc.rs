use tokio::sync::mpsc;

use voxcpm_ipc::{EngineError, EngineOp, EngineToHost, JobId};

pub(crate) type OutTx = mpsc::Sender<EngineToHost>;

pub(crate) async fn send_error(
    out_tx: &OutTx,
    job_id: JobId,
    op: EngineOp,
    code: &str,
    message: String,
    retryable: bool,
) {
    let _ = out_tx
        .send(EngineToHost::Error(EngineError {
            job_id,
            op,
            code: code.to_string(),
            message,
            retryable,
        }))
        .await;
}
