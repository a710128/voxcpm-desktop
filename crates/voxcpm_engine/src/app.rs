use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{mpsc, Mutex as AsyncMutex};
use tokio::task::JoinSet;

use crate::download::handle_download_model;
use crate::inference::{handle_exit, handle_generate, handle_load_model, handle_stop};
use crate::infer_actor::InferenceActorHandle;
use crate::ipc::OutTx;
use crate::state::EngineState;

use voxcpm_ipc::{
    encode_frame, try_decode_frame, EngineEvent, EngineToHost, HostToEngine, LogLevel,
    DEFAULT_MAX_FRAME_LEN,
};

pub(crate) async fn run() -> Result<(), String> {
    let (out_tx, mut out_rx) = mpsc::channel::<EngineToHost>(1024);

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

async fn handle_msg(
    st: Arc<AsyncMutex<EngineState>>,
    out_tx: OutTx,
    msg: HostToEngine,
) -> Result<(), String> {
    match msg {
        HostToEngine::DownloadModel(req) => handle_download_model(out_tx, req).await,
        HostToEngine::LoadModel(req) => handle_load_model(st, out_tx, req).await,
        HostToEngine::Generate(req) => handle_generate(st, out_tx, req).await,
        HostToEngine::StopGenerate(req) => handle_stop(st, out_tx, req).await,
        HostToEngine::Exit(req) => handle_exit(st, out_tx, req).await,
    }
}
