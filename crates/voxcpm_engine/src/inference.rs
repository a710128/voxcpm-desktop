use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::sync::Mutex as AsyncMutex;

use crate::ipc::{send_error, OutTx};
use crate::state::{CurrentJob, EngineState};

use voxcpm_ipc::{
    EngineOp, EngineResponse, EngineToHost, ExitRequest, ExitResponse, GenerateRequest,
    LoadModelRequest, StopGenerateRequest, StopGenerateResponse,
};

pub(crate) async fn handle_load_model(
    st: Arc<AsyncMutex<EngineState>>,
    out_tx: OutTx,
    req: LoadModelRequest,
) -> Result<(), String> {
    let job_id = req.job_id;
    let _ = out_tx
        .send(EngineToHost::Ack {
            job_id,
            op: EngineOp::LoadModel,
        })
        .await;

    // Reject load while generating (single-flight generate).
    let infer = {
        let stg = st.lock().await;
        if stg.current.is_some() {
            send_error(
                &out_tx,
                job_id,
                EngineOp::LoadModel,
                "busy",
                "engine is busy".into(),
                true,
            )
            .await;
            return Ok(());
        }
        stg.infer
            .as_ref()
            .ok_or_else(|| "engine is shutting down".to_string())?
            .clone()
    };

    match infer.load_model(req).await {
        Ok(resp) => {
            let _ = out_tx
                .send(EngineToHost::Response(EngineResponse::LoadModel(resp)))
                .await;
        }
        Err(e) => {
            send_error(
                &out_tx,
                job_id,
                EngineOp::LoadModel,
                &e.code,
                e.message,
                e.retryable,
            )
            .await;
        }
    }

    Ok(())
}

pub(crate) async fn handle_generate(
    st: Arc<AsyncMutex<EngineState>>,
    out_tx: OutTx,
    req: GenerateRequest,
) -> Result<(), String> {
    let job_id = req.job_id;
    let _ = out_tx
        .send(EngineToHost::Ack {
            job_id,
            op: EngineOp::Generate,
        })
        .await;

    let (infer, cancel) = {
        let mut stg = st.lock().await;
        if stg.current.is_some() {
            send_error(
                &out_tx,
                job_id,
                EngineOp::Generate,
                "busy",
                "engine is busy".into(),
                true,
            )
            .await;
            return Ok(());
        }

        let cancel = Arc::new(AtomicBool::new(false));
        stg.current = Some(CurrentJob {
            job_id,
            cancel: cancel.clone(),
        });

        let infer = stg
            .infer
            .as_ref()
            .ok_or_else(|| "engine is shutting down".to_string())?
            .clone();
        (infer, cancel)
    };

    let res = infer.generate(req, cancel, out_tx.clone()).await;

    {
        let mut stg = st.lock().await;
        if stg.current.as_ref().is_some_and(|j| j.job_id == job_id) {
            stg.current = None;
        }
    }

    match res {
        Ok(resp) => {
            let _ = out_tx
                .send(EngineToHost::Response(EngineResponse::Generate(resp)))
                .await;
        }
        Err(e) => {
            send_error(
                &out_tx,
                job_id,
                EngineOp::Generate,
                &e.code,
                e.message,
                e.retryable,
            )
            .await;
        }
    }

    Ok(())
}

pub(crate) async fn handle_stop(
    st: Arc<AsyncMutex<EngineState>>,
    out_tx: OutTx,
    req: StopGenerateRequest,
) -> Result<(), String> {
    let job_id = req.job_id;
    let _ = out_tx
        .send(EngineToHost::Ack {
            job_id,
            op: EngineOp::StopGenerate,
        })
        .await;

    let cancelled = {
        let stg = st.lock().await;
        stg.current.as_ref().map(|j| {
            j.cancel.store(true, Ordering::Relaxed);
            j.job_id
        })
    };

    let _ = out_tx
        .send(EngineToHost::Response(EngineResponse::StopGenerate(
            StopGenerateResponse {
                job_id,
                cancelled_job_id: cancelled,
            },
        )))
        .await;
    Ok(())
}

pub(crate) async fn handle_exit(
    st: Arc<AsyncMutex<EngineState>>,
    out_tx: OutTx,
    req: ExitRequest,
) -> Result<(), String> {
    let job_id = req.job_id;
    let _ = out_tx
        .send(EngineToHost::Ack {
            job_id,
            op: EngineOp::Exit,
        })
        .await;

    {
        let stg = st.lock().await;
        if let Some(j) = stg.current.as_ref() {
            j.cancel.store(true, Ordering::Relaxed);
        }
    }

    let _ = out_tx
        .send(EngineToHost::Response(EngineResponse::Exit(ExitResponse {
            job_id,
        })))
        .await;

    Ok(())
}
