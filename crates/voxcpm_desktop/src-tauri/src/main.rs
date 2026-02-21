use std::collections::HashMap;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::ipc::{InvokeResponseBody, Response};
use tauri::{Emitter, Manager};

use voxcpm_engine_sdk::{EngineSdk, Event, ModelHandle, SdkError};
use voxcpm_ipc::EngineEvent;

static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

const DEFAULT_REPO_ID: &str = "openbmb/VoxCPM1.5";
const DEFAULT_REVISION: &str = "main";
const HF_MIRROR_ENDPOINT: &str = "https://hf-mirror.com";

fn next_job_id() -> u64 {
    NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Clone)]
struct ModelKey {
    model_dir: String,
    device_spec: String,
    weights_prefix: Option<String>,
    optimize: bool,
}

impl PartialEq for ModelKey {
    fn eq(&self, other: &Self) -> bool {
        self.model_dir == other.model_dir
            && self.device_spec == other.device_spec
            && self.weights_prefix == other.weights_prefix
            && self.optimize == other.optimize
    }
}

impl Eq for ModelKey {}

impl Hash for ModelKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.model_dir.hash(state);
        self.device_spec.hash(state);
        self.weights_prefix.hash(state);
        self.optimize.hash(state);
    }
}

struct AppState {
    engine: EngineSdk,
    models: Mutex<HashMap<ModelKey, ModelHandle>>,
    prepared_default: Mutex<Option<PreparedDefault>>,
    active_download_job_id: AtomicU64,
    active_prepare_cancel_nonce: AtomicU64,
    active_generate_job_id: AtomicU64,
}

#[derive(Debug, Clone)]
struct PreparedDefault {
    device_spec: String,
    repo_id: String,
    model_dir: String,
    model: ModelHandle,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CapabilitiesResponse {
    devices: Vec<String>,
    mirror_default: bool,
    default_model: DefaultModelInfo,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DefaultModelInfo {
    repo_id: String,
    revision: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PrepareDefaultModelParams {
    device_spec: String,
    #[serde(default)]
    repo_id: Option<String>,
    #[serde(default)]
    mirror: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PrepareDefaultModelResult {
    model_loaded: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerateV1Params {
    device_spec: String,
    target_text: String,
    #[serde(default)]
    reference_audio_path: Option<String>,
    #[serde(default)]
    reference_audio_bytes: Option<Vec<u8>>,
    #[serde(default)]
    reference_text: Option<String>,
    cfg_value: f64,
    inference_steps: usize,
    #[serde(default)]
    seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InferParams {
    model_dir: String,
    text: String,
    #[serde(default)]
    prompt_wav: Option<String>,
    #[serde(default)]
    device_spec: Option<String>,
    #[serde(default)]
    weights_prefix: Option<String>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    max_steps: Option<usize>,
    #[serde(default)]
    guidance_scale: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EnsureModelParams {
    #[serde(default = "default_hf_repo_id")]
    repo_id: String,
    #[serde(default)]
    revision: Option<String>,
}

fn default_hf_repo_id() -> String {
    DEFAULT_REPO_ID.to_string()
}

fn normalize_opt_str(s: Option<String>) -> Option<String> {
    let s = s?.trim().to_string();
    if s.is_empty() || s.eq_ignore_ascii_case("none") {
        None
    } else {
        Some(s)
    }
}

#[tauri::command]
async fn ensure_model(
    window: tauri::WebviewWindow,
    params: EnsureModelParams,
) -> Result<String, String> {
    let app = window.app_handle();
    let st = app.state::<AppState>();

    let app_data = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let cache_root = app_data.join("hf");
    fs::create_dir_all(&cache_root).map_err(|e| e.to_string())?;

    let revision = params.revision.unwrap_or_else(|| "main".to_string());

    let job_id = next_job_id();
    st.active_download_job_id.store(job_id, Ordering::Relaxed);
    let model_dir = st
        .engine
        .download_model(
            job_id,
            params.repo_id,
            revision,
            cache_root.to_string_lossy().to_string(),
            None,
        )
        .await
        .map_err(|e| e.to_string())?;

    // No longer an active download job.
    st.active_download_job_id.store(0, Ordering::Relaxed);

    let _ = window.emit(
        "voxcpm:download",
        serde_json::json!({"stage":"done","model_dir":model_dir}),
    );
    Ok(model_dir)
}

#[tauri::command]
async fn infer(window: tauri::WebviewWindow, params: InferParams) -> Result<Response, String> {
    let app = window.app_handle();
    let st = app.state::<AppState>();

    let device_spec = params.device_spec.unwrap_or_else(|| "cpu".to_string());
    let weights_prefix = normalize_opt_str(params.weights_prefix);
    let optimize = false;

    let key = ModelKey {
        model_dir: params.model_dir.clone(),
        device_spec: device_spec.clone(),
        weights_prefix: weights_prefix.clone(),
        optimize,
    };

    // Don't hold a `MutexGuard` across `.await` (Tauri commands require `Send`).
    let cached = {
        let g = st.models.lock().unwrap();
        g.get(&key).cloned()
    };

    let model = if let Some(m) = cached {
        m
    } else {
        let job_id = next_job_id();
        let m = st
            .engine
            .load_model(
                job_id,
                params.model_dir,
                device_spec,
                weights_prefix,
                optimize,
            )
            .await
            .map_err(|e| e.to_string())?;
        st.models.lock().unwrap().insert(key, m.clone());
        m
    };

    let prompt_bytes: Option<Vec<u8>> = match normalize_opt_str(params.prompt_wav) {
        None => None,
        Some(p) => Some(fs::read(p).map_err(|e| e.to_string())?),
    };

    let job_id = next_job_id();
    st.active_generate_job_id.store(job_id, Ordering::Relaxed);
    let gen = st
        .engine
        .generate(
            job_id,
            model.model_id,
            params.text,
            prompt_bytes,
            None,
            params.seed.unwrap_or(42),
            params.max_steps.unwrap_or(10),
            params.guidance_scale.unwrap_or(2.0),
            200,
        )
        .await
        .map_err(|e| e.to_string())?;

    Ok(Response::new(InvokeResponseBody::Raw(gen.wav_bytes)))
}

#[tauri::command]
async fn stop_generate(app: tauri::AppHandle) -> Result<Option<u64>, String> {
    let st = app.state::<AppState>();
    let job_id = next_job_id();
    st.engine
        .stop_generation(job_id)
        .await
        .map_err(|e| e.to_string())
}

fn default_model_cache_dir(app: &tauri::AppHandle, repo_id: &str) -> Result<PathBuf, String> {
    let app_data = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let repo_key = repo_id.replace('/', "__");
    Ok(app_data
        .join("voxcpm")
        .join("models")
        .join("default")
        .join(repo_key)
        .join(DEFAULT_REVISION))
}

fn emit_stage(app: &tauri::AppHandle, stage: &str, message: Option<String>) {
    let payload = match message {
        Some(message) => serde_json::json!({"stage": stage, "message": message}),
        None => serde_json::json!({"stage": stage}),
    };
    let _ = app.emit("voxcpm:stage", payload);
}

#[tauri::command]
async fn get_capabilities(app: tauri::AppHandle) -> Result<CapabilitiesResponse, String> {
    let st = app.state::<AppState>();
    let job_id = next_job_id();
    eprintln!("get_capabilities: start job_id={job_id}");
    let devices = tokio::time::timeout(Duration::from_secs(5), st.engine.list_devices(job_id))
        .await
        .map_err(|_| "list_devices timed out".to_string())?
        .map_err(|e| e.to_string())?;
    eprintln!("get_capabilities: ok job_id={job_id} devices={devices:?}");
    Ok(CapabilitiesResponse {
        devices,
        mirror_default: false,
        default_model: DefaultModelInfo {
            repo_id: DEFAULT_REPO_ID.to_string(),
            revision: DEFAULT_REVISION.to_string(),
        },
    })
}

#[tauri::command]
async fn prepare_default_model(
    app: tauri::AppHandle,
    params: PrepareDefaultModelParams,
) -> Result<PrepareDefaultModelResult, String> {
    let st = app.state::<AppState>();

    // Capture a nonce so we can ignore cancellation races.
    let cancel_nonce = st.active_prepare_cancel_nonce.load(Ordering::Relaxed);

    let repo_id = normalize_opt_str(params.repo_id).unwrap_or_else(|| DEFAULT_REPO_ID.to_string());

    // Fast-path: already prepared for this device.
    if let Some(p) = st.prepared_default.lock().unwrap().as_ref() {
        if p.device_spec == params.device_spec && p.repo_id == repo_id {
            emit_stage(&app, "ready", None);
            return Ok(PrepareDefaultModelResult { model_loaded: true });
        }
    }

    let cache_root = default_model_cache_dir(&app, &repo_id)?;
    fs::create_dir_all(&cache_root).map_err(|e| e.to_string())?;

    let endpoint = params.mirror.then(|| HF_MIRROR_ENDPOINT.to_string());

    emit_stage(&app, "download", None);
    let dl_job_id = next_job_id();
    st.active_download_job_id
        .store(dl_job_id, Ordering::Relaxed);
    let model_dir = st
        .engine
        .download_model(
            dl_job_id,
            repo_id.clone(),
            DEFAULT_REVISION.to_string(),
            cache_root.to_string_lossy().to_string(),
            endpoint,
        )
        .await
        .map_err(|e| {
            // Back-triggered cancellation should not surface as an error modal.
            let is_cancelled = matches!(
                &e,
                SdkError::Engine { code, .. } if code == "cancelled"
            ) || st.active_prepare_cancel_nonce.load(Ordering::Relaxed) != cancel_nonce;
            if !is_cancelled {
                emit_stage(&app, "error", Some(e.to_string()));
            }
            e.to_string()
        })?;

    // Download completed; stop treating it as the active cancellable job.
    st.active_download_job_id.store(0, Ordering::Relaxed);

    emit_stage(&app, "verify", None);
    emit_stage(&app, "load", None);

    let lm_job_id = next_job_id();
    let model = st
        .engine
        .load_model(
            lm_job_id,
            model_dir.clone(),
            params.device_spec.clone(),
            None,
            false,
        )
        .await
        .map_err(|e| {
            emit_stage(&app, "error", Some(e.to_string()));
            e.to_string()
        })?;

    *st.prepared_default.lock().unwrap() = Some(PreparedDefault {
        device_spec: params.device_spec,
        repo_id,
        model_dir,
        model,
    });
    emit_stage(&app, "ready", None);
    Ok(PrepareDefaultModelResult { model_loaded: true })
}

#[tauri::command]
async fn cancel_prepare_default_model(app: tauri::AppHandle) -> Result<Option<u64>, String> {
    let st = app.state::<AppState>();
    // Bump nonce so in-flight prepare_default_model can treat subsequent errors as cancelled.
    st.active_prepare_cancel_nonce.fetch_add(1, Ordering::Relaxed);

    // Stop forwarding download progress after cancellation.
    let target = st.active_download_job_id.swap(0, Ordering::Relaxed);
    if target == 0 {
        return Ok(None);
    }

    let job_id = next_job_id();
    st.engine
        .cancel_download(job_id, target)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn generate_v1(app: tauri::AppHandle, params: GenerateV1Params) -> Result<Response, String> {
    let st = app.state::<AppState>();

    let prepared = st
        .prepared_default
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| "model not prepared: call prepare_default_model first".to_string())?;
    if prepared.device_spec != params.device_spec {
        return Err("model not prepared for this device_spec".to_string());
    }

    let target_text = params.target_text.trim().to_string();
    if target_text.is_empty() {
        return Err("target_text is empty".to_string());
    }

    let prompt_wav_bytes: Option<Vec<u8>> = match params.reference_audio_bytes {
        Some(b) if !b.is_empty() => Some(b),
        _ => match normalize_opt_str(params.reference_audio_path) {
            None => None,
            Some(p) => Some(fs::read(p).map_err(|e| e.to_string())?),
        },
    };
    let prompt_text = normalize_opt_str(params.reference_text);
    if prompt_wav_bytes.is_some() && prompt_text.is_none() {
        return Err("reference_text is required when reference_audio_path is set".to_string());
    }

    let job_id = next_job_id();
    st.active_generate_job_id.store(job_id, Ordering::Relaxed);
    let gen = st
        .engine
        .generate(
            job_id,
            prepared.model.model_id,
            target_text,
            prompt_wav_bytes,
            prompt_text,
            params.seed.unwrap_or(42),
            params.inference_steps,
            params.cfg_value,
            200,
        )
        .await
        .map_err(|e| e.to_string())?;
    Ok(Response::new(InvokeResponseBody::Raw(gen.wav_bytes)))
}

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let app_handle = app.handle().clone();
            let engine = tauri::async_runtime::block_on(async {
                EngineSdk::spawn_tauri_sidecar(&app_handle, "engine").await
            })
            .map_err(|e| {
                let err: Box<dyn std::error::Error> = Box::new(e);
                tauri::Error::Setup(err.into())
            })?;

            let st = AppState {
                engine,
                models: Mutex::new(HashMap::new()),
                prepared_default: Mutex::new(None),
                active_download_job_id: AtomicU64::new(0),
                active_prepare_cancel_nonce: AtomicU64::new(0),
                active_generate_job_id: AtomicU64::new(0),
            };
            app.manage(st);

            // Forward engine events to the frontend.
            let mut rx = {
                let st = app_handle.state::<AppState>();
                st.engine.subscribe()
            };
            let app_handle2 = app_handle.clone();
            tauri::async_runtime::spawn(async move {
                while let Ok(ev) = rx.recv().await {
                    match ev {
                        Event::Engine(EngineEvent::DownloadProgress(p)) => {
                            let st = app_handle2.state::<AppState>();
                            let active = st.active_download_job_id.load(Ordering::Relaxed);
                            // Only forward the active download job to the frontend.
                            // When there's no active job, drop all download progress.
                            if active == 0 || p.job_id != active {
                                continue;
                            }
                            let _ = app_handle2.emit(
                                "voxcpm:download",
                                serde_json::json!({
                                    "stage": "downloading",
                                    "file": p.file,
                                    "done": p.done,
                                    "total": p.total,
                                    "bytesDownloaded": p.bytes_downloaded,
                                    "bytesTotal": p.bytes_total,
                                    "percent": p.percent
                                }),
                            );
                        }
                        Event::Engine(EngineEvent::GenerateProgress(p)) => {
                            let st = app_handle2.state::<AppState>();
                            let active = st.active_generate_job_id.load(Ordering::Relaxed);
                            if active != 0 && p.job_id != active {
                                continue;
                            }
                            let _ = app_handle2.emit(
                                "voxcpm:progress",
                                serde_json::json!({
                                    "event": "progress",
                                    "stage": "generating",
                                    "seq": p.steps_done,
                                    "progress": {
                                        "steps_done": p.steps_done,
                                        "step_samples": p.step_samples,
                                        "sample_rate": p.sample_rate,
                                        "generated_samples": p.generated_samples,
                                        "generated_ms": p.generated_ms
                                    }
                                }),
                            );
                        }
                        Event::Engine(EngineEvent::Log { level: _, message }) => {
                            let _ = app_handle2.emit("voxcpm:log", message);
                        }
                    }
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            ensure_model,
            infer,
            stop_generate,
            get_capabilities,
            prepare_default_model,
            cancel_prepare_default_model,
            generate_v1
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|_app_handle, _event| {
        // Engine lifetime is tied to AppState (killed on drop).
    });
}
