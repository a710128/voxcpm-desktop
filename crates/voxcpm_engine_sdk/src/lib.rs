use std::collections::HashMap;
use std::path::Path;
use std::process::Stdio;
use std::sync::{Arc, Mutex};

use bytes::BytesMut;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{broadcast, oneshot};

use voxcpm_ipc::{
    encode_frame, try_decode_frame, EngineError, EngineEvent, EngineOp, EngineResponse,
    EngineToHost, HostToEngine, JobId, DEFAULT_MAX_FRAME_LEN,
};

#[derive(Debug, thiserror::Error)]
pub enum SdkError {
    #[error("io error: {0}")]
    Io(String),
    #[error("ipc error: {0}")]
    Ipc(String),
    #[error("engine error: {code}: {message}")]
    Engine { code: String, message: String },
    #[error("engine terminated")]
    EngineTerminated,
    #[error("engine is busy")]
    Busy,
}

#[derive(Debug, Clone)]
pub struct ModelHandle {
    pub model_id: String,
    pub step_samples: u64,
    pub sample_rate: u32,
}

#[derive(Debug, Clone)]
pub struct GenerateResult {
    pub wav_bytes: Vec<u8>,
    pub sample_rate: u32,
    pub samples: u64,
    pub duration_ms: u64,
}

#[derive(Debug, Clone)]
pub enum Event {
    Engine(EngineEvent),
}

#[derive(Debug)]
struct Pending {
    op: EngineOp,
    tx: oneshot::Sender<Result<EngineResponse, EngineError>>,
}

pub struct EngineSdk {
    write: tokio::sync::Mutex<
        Box<dyn Fn(Vec<u8>) -> tokio::task::JoinHandle<Result<(), SdkError>> + Send + Sync>,
    >,
    pending: Arc<Mutex<HashMap<JobId, Pending>>>,
    events: broadcast::Sender<Event>,
    drop_kill: Mutex<Option<Box<dyn FnOnce() + Send + 'static>>>,
}

impl EngineSdk {
    pub async fn spawn(engine_bin: impl AsRef<Path>) -> Result<Self, SdkError> {
        let mut cmd = Command::new(engine_bin.as_ref());
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| SdkError::Io(e.to_string()))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| SdkError::Io("missing stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| SdkError::Io("missing stdout".into()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| SdkError::Io("missing stderr".into()))?;

        Self::from_stdio(child, stdin, stdout, stderr).await
    }

    async fn from_stdio(
        child: Child,
        stdin: ChildStdin,
        stdout: ChildStdout,
        mut stderr: tokio::process::ChildStderr,
    ) -> Result<Self, SdkError> {
        let (events_tx, _events_rx) = broadcast::channel(1024);
        let pending: Arc<Mutex<HashMap<JobId, Pending>>> = Arc::new(Mutex::new(HashMap::new()));
        let child_arc: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(Some(child)));

        // Writer closure: always non-blocking for caller (does async write on a task).
        let stdin = Arc::new(tokio::sync::Mutex::new(stdin));
        let write_fn: Box<
            dyn Fn(Vec<u8>) -> tokio::task::JoinHandle<Result<(), SdkError>> + Send + Sync,
        > = {
            let stdin = stdin.clone();
            Box::new(move |bytes: Vec<u8>| {
                let stdin = stdin.clone();
                tokio::spawn(async move {
                    let mut w = stdin.lock().await;
                    w.write_all(&bytes)
                        .await
                        .map_err(|e| SdkError::Io(e.to_string()))?;
                    w.flush().await.map_err(|e| SdkError::Io(e.to_string()))?;
                    Ok(())
                })
            })
        };

        let drop_kill: Box<dyn FnOnce() + Send + 'static> = {
            let child_arc = child_arc.clone();
            Box::new(move || {
                if let Some(mut c) = child_arc.lock().unwrap().take() {
                    let _ = c.start_kill();
                }
            })
        };

        let sdk = Self {
            write: tokio::sync::Mutex::new(write_fn),
            pending: pending.clone(),
            events: events_tx.clone(),
            drop_kill: Mutex::new(Some(drop_kill)),
        };

        // Stdout reader/decoder task.
        {
            let pending = pending.clone();
            let events_tx = events_tx.clone();
            let child_arc = child_arc.clone();
            tokio::spawn(async move {
                let mut r = stdout;
                let mut buf = BytesMut::with_capacity(64 * 1024);
                let mut tmp = [0u8; 8192];

                loop {
                    match r.read(&mut tmp).await {
                        Ok(0) => break,
                        Ok(n) => {
                            buf.extend_from_slice(&tmp[..n]);
                            loop {
                                let msg = match try_decode_frame::<EngineToHost>(
                                    &mut buf,
                                    DEFAULT_MAX_FRAME_LEN,
                                ) {
                                    Ok(Some(m)) => m,
                                    Ok(None) => break,
                                    Err(e) => {
                                        let _ = events_tx.send(Event::Engine(EngineEvent::Log {
                                            level: voxcpm_ipc::LogLevel::Error,
                                            message: format!("ipc decode error: {e}"),
                                        }));
                                        break;
                                    }
                                };

                                match msg {
                                    EngineToHost::Event(ev) => {
                                        let _ = events_tx.send(Event::Engine(ev));
                                    }
                                    EngineToHost::Ack { .. } => {
                                        // Optional; ignore.
                                    }
                                    EngineToHost::Response(resp) => {
                                        let job_id = match &resp {
                                            EngineResponse::DownloadModel(r) => r.job_id,
                                            EngineResponse::CancelDownload(r) => r.job_id,
                                            EngineResponse::LoadModel(r) => r.job_id,
                                            EngineResponse::Generate(r) => r.job_id,
                                            EngineResponse::StopGenerate(r) => r.job_id,
                                            EngineResponse::Exit(r) => r.job_id,
                                            EngineResponse::ListDevices(r) => r.job_id,
                                        };
                                        if let Some(p) = pending.lock().unwrap().remove(&job_id) {
                                            let _ = p.tx.send(Ok(resp));
                                        }
                                    }
                                    EngineToHost::Error(err) => {
                                        if let Some(p) = pending.lock().unwrap().remove(&err.job_id)
                                        {
                                            let _ = p.tx.send(Err(err));
                                        }
                                    }
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }

                // Engine terminated: fail all pending.
                let pending_all = std::mem::take(&mut *pending.lock().unwrap());
                for (job_id, p) in pending_all {
                    let _ = p.tx.send(Err(EngineError {
                        job_id,
                        op: p.op,
                        code: "engine_terminated".into(),
                        message: "engine terminated".into(),
                        retryable: false,
                    }));
                }
                *child_arc.lock().unwrap() = None;
            });
        }

        // Stderr -> log events.
        {
            let events_tx = events_tx.clone();
            tokio::spawn(async move {
                let mut tmp = [0u8; 8192];
                loop {
                    match stderr.read(&mut tmp).await {
                        Ok(0) => break,
                        Ok(n) => {
                            let s = String::from_utf8_lossy(&tmp[..n]).to_string();
                            let _ = events_tx.send(Event::Engine(EngineEvent::Log {
                                level: voxcpm_ipc::LogLevel::Info,
                                message: s,
                            }));
                        }
                        Err(_) => break,
                    }
                }
            });
        }

        Ok(sdk)
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.events.subscribe()
    }

    async fn call(
        &self,
        job_id: JobId,
        op: EngineOp,
        msg: HostToEngine,
    ) -> Result<EngineResponse, SdkError> {
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self.pending.lock().unwrap();
            if pending.contains_key(&job_id) {
                return Err(SdkError::Ipc("duplicate job_id".into()));
            }
            pending.insert(job_id, Pending { op, tx });
        }

        let frame = encode_frame(&msg).map_err(|e| SdkError::Ipc(e.to_string()))?;

        // IMPORTANT: only hold the write lock for the actual write.
        // Long-running ops (e.g. DownloadModel) wait for their response while other
        // control ops (e.g. CancelDownload) must still be able to send messages.
        {
            let write = self.write.lock().await;
            let jh = (write)(frame);
            jh.await.map_err(|e| SdkError::Io(e.to_string()))??;
        }

        let resp = rx.await.map_err(|_| SdkError::EngineTerminated)?;
        match resp {
            Ok(r) => Ok(r),
            Err(e) => {
                if e.code == "busy" {
                    Err(SdkError::Busy)
                } else {
                    Err(SdkError::Engine {
                        code: e.code,
                        message: e.message,
                    })
                }
            }
        }
    }

    pub async fn download_model(
        &self,
        job_id: JobId,
        repo_id: String,
        revision: String,
        cache_dir: String,
        endpoint: Option<String>,
    ) -> Result<String, SdkError> {
        let resp = self
            .call(
                job_id,
                EngineOp::DownloadModel,
                HostToEngine::DownloadModel(voxcpm_ipc::DownloadModelRequest {
                    job_id,
                    repo_id,
                    revision,
                    cache_dir,
                    endpoint,
                }),
            )
            .await?;
        match resp {
            EngineResponse::DownloadModel(r) => Ok(r.model_dir),
            _ => Err(SdkError::Ipc("unexpected response".into())),
        }
    }

    pub async fn load_model(
        &self,
        job_id: JobId,
        model_dir: String,
        device_spec: String,
        weights_prefix: Option<String>,
        optimize: bool,
    ) -> Result<ModelHandle, SdkError> {
        let resp = self
            .call(
                job_id,
                EngineOp::LoadModel,
                HostToEngine::LoadModel(voxcpm_ipc::LoadModelRequest {
                    job_id,
                    model_dir,
                    device_spec,
                    weights_prefix,
                    optimize,
                }),
            )
            .await?;
        match resp {
            EngineResponse::LoadModel(r) => Ok(ModelHandle {
                model_id: r.model_id,
                step_samples: r.step_samples,
                sample_rate: r.sample_rate,
            }),
            _ => Err(SdkError::Ipc("unexpected response".into())),
        }
    }

    pub async fn generate(
        &self,
        job_id: JobId,
        model_id: String,
        text: String,
        prompt_wav: Option<Vec<u8>>,
        prompt_text: Option<String>,
        seed: u64,
        max_steps: usize,
        guidance_scale: f64,
        emit_every_ms: u64,
    ) -> Result<GenerateResult, SdkError> {
        let resp = self
            .call(
                job_id,
                EngineOp::Generate,
                HostToEngine::Generate(voxcpm_ipc::GenerateRequest {
                    job_id,
                    model_id,
                    text,
                    prompt_wav,
                    prompt_text,
                    seed,
                    max_steps,
                    guidance_scale,
                    emit_every_ms,
                }),
            )
            .await?;
        match resp {
            EngineResponse::Generate(r) => Ok(GenerateResult {
                wav_bytes: r.wav_bytes,
                sample_rate: r.sample_rate,
                samples: r.samples,
                duration_ms: r.duration_ms,
            }),
            _ => Err(SdkError::Ipc("unexpected response".into())),
        }
    }

    pub async fn stop_generation(&self, job_id: JobId) -> Result<Option<JobId>, SdkError> {
        let resp = self
            .call(
                job_id,
                EngineOp::StopGenerate,
                HostToEngine::StopGenerate(voxcpm_ipc::StopGenerateRequest { job_id }),
            )
            .await?;
        match resp {
            EngineResponse::StopGenerate(r) => Ok(r.cancelled_job_id),
            _ => Err(SdkError::Ipc("unexpected response".into())),
        }
    }

    pub async fn cancel_download(
        &self,
        job_id: JobId,
        target_job_id: JobId,
    ) -> Result<Option<JobId>, SdkError> {
        let resp = self
            .call(
                job_id,
                EngineOp::CancelDownload,
                HostToEngine::CancelDownload(voxcpm_ipc::CancelDownloadRequest {
                    job_id,
                    target_job_id,
                }),
            )
            .await?;
        match resp {
            EngineResponse::CancelDownload(r) => Ok(r.cancelled_job_id),
            _ => Err(SdkError::Ipc("unexpected response".into())),
        }
    }

    pub async fn exit(&self, job_id: JobId) -> Result<(), SdkError> {
        let _ = self
            .call(
                job_id,
                EngineOp::Exit,
                HostToEngine::Exit(voxcpm_ipc::ExitRequest { job_id }),
            )
            .await?;
        Ok(())
    }

    pub async fn list_devices(&self, job_id: JobId) -> Result<Vec<String>, SdkError> {
        let resp = self
            .call(
                job_id,
                EngineOp::ListDevices,
                HostToEngine::ListDevices(voxcpm_ipc::ListDevicesRequest { job_id }),
            )
            .await?;
        match resp {
            EngineResponse::ListDevices(r) => Ok(r.devices),
            _ => Err(SdkError::Ipc("unexpected response".into())),
        }
    }
}

impl Drop for EngineSdk {
    fn drop(&mut self) {
        if let Some(kill) = self.drop_kill.lock().unwrap().take() {
            kill();
        }
    }
}

// -------- Tauri sidecar integration (optional) --------

#[cfg(feature = "tauri-shell")]
mod tauri_sidecar {
    use super::*;
    use tauri_plugin_shell::process::{CommandChild, CommandEvent};
    use tauri_plugin_shell::ShellExt;

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    use std::ffi::OsString;

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn prepend_path_like(orig: Option<OsString>, prefix: &std::path::Path, sep: char) -> String {
        let mut s = prefix.to_string_lossy().to_string();
        if let Some(orig) = orig {
            let orig = orig.to_string_lossy();
            if !orig.is_empty() {
                s.push(sep);
                s.push_str(&orig);
            }
        }
        s
    }

    impl EngineSdk {
        pub async fn spawn_tauri_sidecar(
            app: &tauri::AppHandle,
            sidecar: &str,
        ) -> Result<Self, SdkError> {
            #[allow(unused_mut)]
            let mut cmd = app
                .shell()
                .sidecar(sidecar)
                .map_err(|e| SdkError::Io(e.to_string()))?;

            // Ensure cudarc dynamic-loading can find bundled CUDA runtime libraries.
            // We rely on per-process environment injection (sidecar only).
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            {
                use tauri::Manager;

                let resource_dir = app
                    .path()
                    .resource_dir()
                    .map_err(|e| SdkError::Io(e.to_string()))?;

                #[cfg(target_os = "linux")]
                {
                    let cuda_dir = resource_dir.join("cuda/linux-x64-cuda12.2");
                    let ld =
                        prepend_path_like(std::env::var_os("LD_LIBRARY_PATH"), &cuda_dir, ':');
                    cmd.env("LD_LIBRARY_PATH", ld);
                }

                #[cfg(target_os = "windows")]
                {
                    let cuda_dir = resource_dir.join("cuda/win-x64-cuda12.4");
                    let path = prepend_path_like(std::env::var_os("PATH"), &cuda_dir, ';');
                    cmd.env("PATH", path);
                }
            }

            // IMPORTANT: our IPC uses binary frames on stdout, so we must enable raw output.
            let (mut rx, child) = cmd
                .set_raw_out(true)
                .spawn()
                .map_err(|e| SdkError::Io(e.to_string()))?;

            let (events_tx, _events_rx) = broadcast::channel(1024);
            let pending: Arc<Mutex<HashMap<JobId, Pending>>> = Arc::new(Mutex::new(HashMap::new()));
            let child_arc: Arc<Mutex<Option<CommandChild>>> = Arc::new(Mutex::new(Some(child)));

            let write_fn: Box<
                dyn Fn(Vec<u8>) -> tokio::task::JoinHandle<Result<(), SdkError>> + Send + Sync,
            > = {
                let child_arc = child_arc.clone();
                Box::new(move |bytes: Vec<u8>| {
                    let child_arc = child_arc.clone();
                    tokio::spawn(async move {
                        tokio::task::spawn_blocking(move || {
                            let mut g = child_arc.lock().unwrap();
                            let child = g.as_mut().ok_or(SdkError::EngineTerminated)?;
                            child
                                .write(&bytes)
                                .map_err(|e| SdkError::Io(e.to_string()))?;
                            Ok(())
                        })
                        .await
                        .map_err(|e| SdkError::Io(e.to_string()))?
                    })
                })
            };

            let drop_kill: Box<dyn FnOnce() + Send + 'static> = {
                let child_arc = child_arc.clone();
                Box::new(move || {
                    if let Some(c) = child_arc.lock().unwrap().take() {
                        let _ = c.kill();
                    }
                })
            };

            let sdk = Self {
                write: tokio::sync::Mutex::new(write_fn),
                pending: pending.clone(),
                events: events_tx.clone(),
                drop_kill: Mutex::new(Some(drop_kill)),
            };

            // stdout/stderr from plugin-shell is already chunked events.
            {
                let pending = pending.clone();
                let events_tx = events_tx.clone();
                let child_arc2 = child_arc.clone();
                tokio::spawn(async move {
                    let mut buf = BytesMut::with_capacity(64 * 1024);
                    while let Some(ev) = rx.recv().await {
                        match ev {
                            CommandEvent::Stdout(bytes) => {
                                buf.extend_from_slice(&bytes);
                                loop {
                                    let msg = match try_decode_frame::<EngineToHost>(
                                        &mut buf,
                                        DEFAULT_MAX_FRAME_LEN,
                                    ) {
                                        Ok(Some(m)) => m,
                                        Ok(None) => break,
                                        Err(e) => {
                                            let _ =
                                                events_tx.send(Event::Engine(EngineEvent::Log {
                                                    level: voxcpm_ipc::LogLevel::Error,
                                                    message: format!("ipc decode error: {e}"),
                                                }));
                                            break;
                                        }
                                    };
                                    match msg {
                                        EngineToHost::Event(ev) => {
                                            let _ = events_tx.send(Event::Engine(ev));
                                        }
                                        EngineToHost::Ack { .. } => {}
                                        EngineToHost::Response(resp) => {
                                            let job_id = match &resp {
                                                EngineResponse::DownloadModel(r) => r.job_id,
                                                EngineResponse::CancelDownload(r) => r.job_id,
                                                EngineResponse::LoadModel(r) => r.job_id,
                                                EngineResponse::Generate(r) => r.job_id,
                                                EngineResponse::StopGenerate(r) => r.job_id,
                                                EngineResponse::Exit(r) => r.job_id,
                                                EngineResponse::ListDevices(r) => r.job_id,
                                            };
                                            if let Some(p) = pending.lock().unwrap().remove(&job_id)
                                            {
                                                let _ = p.tx.send(Ok(resp));
                                            }
                                        }
                                        EngineToHost::Error(err) => {
                                            if let Some(p) =
                                                pending.lock().unwrap().remove(&err.job_id)
                                            {
                                                let _ = p.tx.send(Err(err));
                                            }
                                        }
                                    }
                                }
                            }
                            CommandEvent::Stderr(bytes) => {
                                let s = String::from_utf8_lossy(&bytes).to_string();
                                let _ = events_tx.send(Event::Engine(EngineEvent::Log {
                                    level: voxcpm_ipc::LogLevel::Info,
                                    message: s,
                                }));
                            }
                            CommandEvent::Error(e) => {
                                let _ = events_tx.send(Event::Engine(EngineEvent::Log {
                                    level: voxcpm_ipc::LogLevel::Error,
                                    message: format!("engine error: {e}"),
                                }));
                                break;
                            }
                            CommandEvent::Terminated(p) => {
                                let _ = events_tx.send(Event::Engine(EngineEvent::Log {
                                    level: voxcpm_ipc::LogLevel::Warn,
                                    message: format!("engine terminated: {p:?}"),
                                }));
                                break;
                            }
                            _ => {}
                        }
                    }
                    let pending_all = std::mem::take(&mut *pending.lock().unwrap());
                    for (job_id, p) in pending_all {
                        let _ = p.tx.send(Err(EngineError {
                            job_id,
                            op: p.op,
                            code: "engine_terminated".into(),
                            message: "engine terminated".into(),
                            retryable: false,
                        }));
                    }
                    *child_arc2.lock().unwrap() = None;
                });
            }

            Ok(sdk)
        }
    }
}
