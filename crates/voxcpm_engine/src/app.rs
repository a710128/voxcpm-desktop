use std::sync::Arc;

use std::collections::HashMap;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{mpsc, Mutex as AsyncMutex};
use tokio::task::JoinSet;

use crate::download::{handle_download_model, DownloadCancel};
use crate::infer_actor::InferenceActorHandle;
use crate::inference::{handle_exit, handle_generate, handle_load_model, handle_stop};
use crate::ipc::OutTx;
use crate::state::EngineState;

use voxcpm_ipc::{
    encode_frame, try_decode_frame, EngineEvent, EngineOp, EngineResponse, EngineToHost,
    HostToEngine, ListDevicesRequest, ListDevicesResponse, LogLevel, DEFAULT_MAX_FRAME_LEN,
};

use voxcpm_ipc::{CancelDownloadRequest, CancelDownloadResponse, JobId};

struct DownloadTask {
    cancel: DownloadCancel,
    handle: tokio::task::JoinHandle<()>,
}

pub(crate) async fn run() -> Result<(), String> {
    let (out_tx, mut out_rx) = mpsc::channel::<EngineToHost>(1024);

    // Track in-flight download tasks so we can cancel them by job_id.
    let download_tasks: Arc<AsyncMutex<HashMap<JobId, DownloadTask>>> =
        Arc::new(AsyncMutex::new(HashMap::new()));

    // Single writer task for stdout (binary-safe framing).
    let stdout_task = tokio::spawn(async move {
        let mut stdout = tokio::io::stdout();
        while let Some(msg) = out_rx.recv().await {
            let frame = encode_frame(&msg).map_err(|e| e.to_string())?;
            stdout.write_all(&frame).await.map_err(|e| e.to_string())?;
            stdout.flush().await.map_err(|e| e.to_string())?;
        }
        Ok::<(), String>(())
    });

    let infer = InferenceActorHandle::start();
    let st = Arc::new(AsyncMutex::new(EngineState {
        infer: Some(infer),
        current: None,
    }));

    let mut stdin = tokio::io::stdin();
    let mut buf = bytes::BytesMut::with_capacity(64 * 1024);
    let mut tmp = [0u8; 8192];

    let mut tasks = JoinSet::<()>::new();

    'stdin: loop {
        let n = stdin.read(&mut tmp).await.map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);

        while let Some(msg) = try_decode_frame::<HostToEngine>(&mut buf, DEFAULT_MAX_FRAME_LEN)
            .map_err(|e| e.to_string())?
        {
            match msg {
                HostToEngine::Exit(req) => {
                    // Exit is handled in-line to ensure the response is flushed before
                    // we abort other work.
                    handle_exit(st.clone(), out_tx.clone(), req).await?;
                    break 'stdin;
                }
                HostToEngine::CancelDownload(req) => {
                    handle_cancel_download(download_tasks.clone(), out_tx.clone(), req).await;
                }
                HostToEngine::DownloadModel(req) => {
                    let job_id = req.job_id;
                    let out_tx = out_tx.clone();
                    let download_tasks2 = download_tasks.clone();
                    let cancel = DownloadCancel::new();
                    let cancel2 = cancel.clone();
                    let h = tokio::spawn(async move {
                        let out_tx2 = out_tx.clone();
                        if let Err(e) = handle_download_model(out_tx2, req, Some(cancel2)).await {
                            let _ = out_tx
                                .send(EngineToHost::Event(EngineEvent::Log {
                                    level: LogLevel::Error,
                                    message: format!("handle_download_model failed: {e}"),
                                }))
                                .await;
                        }
                        let mut g = download_tasks2.lock().await;
                        g.remove(&job_id);
                    });
                    let mut g = download_tasks.lock().await;
                    g.insert(job_id, DownloadTask { cancel, handle: h });
                }
                other => {
                    let out_tx = out_tx.clone();
                    let st = st.clone();
                    tasks.spawn(async move {
                        let out_tx2 = out_tx.clone();
                        if let Err(e) = handle_msg(st, out_tx2, other).await {
                            let _ = out_tx
                                .send(EngineToHost::Event(EngineEvent::Log {
                                    level: LogLevel::Error,
                                    message: format!("handle_msg failed: {e}"),
                                }))
                                .await;
                        }
                    });
                }
            }
        }
    }

    // Best-effort: stop in-flight work promptly.
    tasks.abort_all();
    while tasks.join_next().await.is_some() {}

    // Stop in-flight downloads (they are spawned outside JoinSet).
    let downloads_all = {
        let mut g = download_tasks.lock().await;
        std::mem::take(&mut *g)
    };
    for (_, t) in downloads_all {
        t.handle.abort();
    }

    // Stop the inference actor (after cancel was signaled in handle_exit).
    let infer = {
        let mut stg = st.lock().await;
        stg.infer.take()
    };
    if let Some(infer) = infer {
        let _ = tokio::task::spawn_blocking(move || infer.shutdown_and_join()).await;
    }

    drop(out_tx);
    let _ = stdout_task.await;
    Ok(())
}

async fn handle_cancel_download(
    download_tasks: Arc<AsyncMutex<HashMap<JobId, DownloadTask>>>,
    out_tx: OutTx,
    req: CancelDownloadRequest,
) {
    let job_id = req.job_id;
    let target = req.target_job_id;

    // Signal cancel ASAP; never block on stdout backpressure.
    let cancelled = {
        let g = download_tasks.lock().await;
        if let Some(t) = g.get(&target) {
            t.cancel.cancel();
            Some(target)
        } else {
            None
        }
    };

    // Best-effort: send Ack/Error/Response without blocking the main loop.
    // If stdout is back-pressured, these sends may be delayed, but cancellation is already signaled.
    let out_tx2 = out_tx.clone();
    let download_tasks2 = download_tasks.clone();
    tokio::spawn(async move {
        let _ = out_tx2
            .send(EngineToHost::Ack {
                job_id,
                op: EngineOp::CancelDownload,
            })
            .await;

        if let Some(cancelled_job_id) = cancelled {
            // Unblock the host waiting on the original download_model call.
            let _ = out_tx2
                .send(EngineToHost::Error(voxcpm_ipc::EngineError {
                    job_id: cancelled_job_id,
                    op: EngineOp::DownloadModel,
                    code: "cancelled".to_string(),
                    message: "cancelled".to_string(),
                    retryable: false,
                }))
                .await;

            // Fallback: if the task doesn't exit promptly (e.g. stuck in a preflight request),
            // abort it so we don't keep downloading in the background.
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                let mut g = download_tasks2.lock().await;
                if let Some(t) = g.remove(&cancelled_job_id) {
                    t.handle.abort();
                }
            });
        }

        let _ = out_tx2
            .send(EngineToHost::Response(EngineResponse::CancelDownload(
                CancelDownloadResponse {
                    job_id,
                    cancelled_job_id: cancelled,
                },
            )))
            .await;
    });
}

async fn handle_msg(
    st: Arc<AsyncMutex<EngineState>>,
    out_tx: OutTx,
    msg: HostToEngine,
) -> Result<(), String> {
    match msg {
        HostToEngine::DownloadModel(req) => handle_download_model(out_tx, req, None).await,
        // CancelDownload is handled in-line in the main loop.
        HostToEngine::CancelDownload(_) => Ok(()),
        HostToEngine::LoadModel(req) => handle_load_model(st, out_tx, req).await,
        HostToEngine::Generate(req) => handle_generate(st, out_tx, req).await,
        HostToEngine::StopGenerate(req) => handle_stop(st, out_tx, req).await,
        HostToEngine::Exit(req) => handle_exit(st, out_tx, req).await,
        HostToEngine::ListDevices(req) => handle_list_devices(out_tx, req).await,
    }
}

async fn handle_list_devices(out_tx: OutTx, req: ListDevicesRequest) -> Result<(), String> {
    let job_id = req.job_id;
    let _ = out_tx
        .send(EngineToHost::Ack {
            job_id,
            op: EngineOp::ListDevices,
        })
        .await;

    let devices = list_devices();

    let _ = out_tx
        .send(EngineToHost::Response(EngineResponse::ListDevices(
            ListDevicesResponse { job_id, devices },
        )))
        .await;
    Ok(())
}

fn list_devices() -> Vec<String> {
    let mut devices = vec!["cpu".to_string()];
    devices.extend(list_cuda_devices());
    devices.extend(list_metal_devices());
    devices
}

#[cfg(feature = "cuda")]
fn list_cuda_devices() -> Vec<String> {
    // Use cudarc (driver API) to query device count.
    use cudarc::driver::result;
    if result::init().is_err() {
        return vec![];
    }
    let n = match result::device::get_count() {
        Ok(n) => n,
        Err(_) => return vec![],
    };
    let n = if n < 0 { 0 } else { n as usize };
    (0..n).map(|i| format!("cuda:{i}")).collect()
}

#[cfg(not(feature = "cuda"))]
fn list_cuda_devices() -> Vec<String> {
    vec![]
}

#[cfg(feature = "metal")]
fn list_metal_devices() -> Vec<String> {
    // candle-metal-kernels currently exposes only the system default device (0 or 1).
    let devs = candle_metal_kernels::metal::Device::all();
    (0..devs.len()).map(|i| format!("metal:{i}")).collect()
}

#[cfg(not(feature = "metal"))]
fn list_metal_devices() -> Vec<String> {
    vec![]
}
