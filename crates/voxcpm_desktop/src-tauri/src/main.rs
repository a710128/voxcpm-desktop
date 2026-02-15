use std::collections::HashMap;
use std::fs;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use serde::Deserialize;
use tauri::ipc::{InvokeResponseBody, Response};
use tauri::{Emitter, Manager};

use voxcpm_engine_sdk::{EngineSdk, Event, ModelHandle};
use voxcpm_ipc::EngineEvent;

static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

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
    "openbmb/VoxCPM1.5".to_string()
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
async fn ensure_model(window: tauri::WebviewWindow, params: EnsureModelParams) -> Result<String, String> {
    let app = window.app_handle();
    let st = app.state::<AppState>();

    let app_data = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let cache_root = app_data.join("hf");
    fs::create_dir_all(&cache_root).map_err(|e| e.to_string())?;

    let revision = params.revision.unwrap_or_else(|| "main".to_string());

    let job_id = next_job_id();
    let model_dir = st
        .engine
        .download_model(
            job_id,
            params.repo_id,
            revision,
            cache_root.to_string_lossy().to_string(),
        )
        .await
        .map_err(|e| e.to_string())?;

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
    let gen = st
        .engine
        .generate(
            job_id,
            model.model_id,
            params.text,
            prompt_bytes,
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
            };
            app.manage(st);

            // Forward engine events to the frontend.
            let st = app_handle.state::<AppState>();
            let mut rx = st.engine.subscribe();
            tauri::async_runtime::spawn(async move {
                while let Ok(ev) = rx.recv().await {
                    match ev {
                        Event::Engine(EngineEvent::DownloadProgress(p)) => {
                            let _ = app_handle.emit(
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
                            let _ = app_handle.emit(
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
                            let _ = app_handle.emit("voxcpm:log", message);
                        }
                    }
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![ensure_model, infer, stop_generate])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|_app_handle, _event| {
        // Engine lifetime is tied to AppState (killed on drop).
    });
}
