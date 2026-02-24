use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot};

use crate::audio::{decode_audio_mono_f32, encode_wav_f32_mono};
use crate::ipc::OutTx;
use crate::util::prod_usize;

use voxcpm::{GenerateArgs, VoxCpm, WavInput};
use voxcpm_ipc::{EngineEvent, EngineToHost, GenerateProgress, GenerateRequest, GenerateResponse, LoadModelRequest, LoadModelResponse};

pub(crate) struct ActorError {
    pub(crate) code: String,
    pub(crate) message: String,
    pub(crate) retryable: bool,
}

type ActorResult<T> = Result<T, ActorError>;

#[derive(Clone)]
pub(crate) struct InferenceActorHandle {
    tx: mpsc::Sender<InferenceCmd>,
    join: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl InferenceActorHandle {
    pub(crate) fn start() -> Self {
        let (tx, mut rx) = mpsc::channel::<InferenceCmd>(64);

        let join = std::thread::Builder::new()
            .name("voxcpm-infer".to_string())
            .spawn(move || {
                let mut st = ActorState {
                    models: HashMap::new(),
                    next_model_id: 1,
                };

                // Single-threaded loop: all VoxCpm/CUDA graph work lives here.
                while let Some(cmd) = rx.blocking_recv() {
                    match cmd {
                        InferenceCmd::LoadModel { req, reply } => {
                            let _ = reply.send(st.load_model(req));
                        }
                        InferenceCmd::Generate {
                            req,
                            cancel,
                            out_tx,
                            reply,
                        } => {
                            let _ = reply.send(st.generate(req, cancel, out_tx));
                        }
                    }
                }
            })
            .expect("spawn voxcpm inference actor");

        Self {
            tx,
            join: Arc::new(Mutex::new(Some(join))),
        }
    }

    pub(crate) fn shutdown_and_join(self) -> Result<(), String> {
        // Dropping the sender closes the actor channel; the actor thread exits once it
        // finishes any in-flight generate() call and then observes the closed channel.
        drop(self.tx);

        let join = self.join.lock().unwrap().take();
        if let Some(j) = join {
            j.join()
                .map_err(|_| "inference actor thread panicked".to_string())?;
        }
        Ok(())
    }

    pub(crate) async fn load_model(
        &self,
        req: LoadModelRequest,
    ) -> ActorResult<LoadModelResponse> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(InferenceCmd::LoadModel { req, reply: tx })
            .await
            .map_err(|_| ActorError {
                code: "actor".to_string(),
                message: "inference actor stopped".to_string(),
                retryable: true,
            })?;
        rx.await
            .map_err(|_| ActorError {
                code: "actor".to_string(),
                message: "inference actor dropped reply".to_string(),
                retryable: true,
            })?
    }

    pub(crate) async fn generate(
        &self,
        req: GenerateRequest,
        cancel: Arc<AtomicBool>,
        out_tx: OutTx,
    ) -> ActorResult<GenerateResponse> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(InferenceCmd::Generate {
                req,
                cancel,
                out_tx,
                reply: tx,
            })
            .await
            .map_err(|_| ActorError {
                code: "actor".to_string(),
                message: "inference actor stopped".to_string(),
                retryable: true,
            })?;
        rx.await
            .map_err(|_| ActorError {
                code: "actor".to_string(),
                message: "inference actor dropped reply".to_string(),
                retryable: true,
            })?
    }
}

enum InferenceCmd {
    LoadModel {
        req: LoadModelRequest,
        reply: oneshot::Sender<ActorResult<LoadModelResponse>>,
    },
    Generate {
        req: GenerateRequest,
        cancel: Arc<AtomicBool>,
        out_tx: OutTx,
        reply: oneshot::Sender<ActorResult<GenerateResponse>>,
    },
}

struct LoadedModel {
    model: VoxCpm,
    step_samples: u64,
    sample_rate: u32,
}

struct ActorState {
    models: HashMap<String, LoadedModel>,
    next_model_id: u64,
}

impl ActorState {
    fn load_model(&mut self, req: LoadModelRequest) -> ActorResult<LoadModelResponse> {
        let model_dir = PathBuf::from(&req.model_dir);
        let mut builder = VoxCpm::builder().model_dir(model_dir).show_progress(false);
        if let Some(pfx) = req.weights_prefix.as_deref() {
            builder = builder.weights_prefix(pfx);
        }

        let device_spec = req.device_spec.trim().to_string();
        if device_spec.is_empty() || device_spec.eq_ignore_ascii_case("cpu") {
            builder = builder.device_str("cpu");
        } else {
            // Keep device parsing centralized in voxcpm.
            builder = builder.device_str(device_spec.clone());
        }

        #[allow(unused_mut)]
        let mut model = builder.build().map_err(|e| ActorError {
            code: "load_failed".to_string(),
            message: e.to_string(),
            retryable: false,
        })?;

        // Best-effort CUDA graph capture. If it fails, runtime falls back to the
        // non-optimized path automatically.
        #[cfg(feature = "cuda")]
        {
            if device_spec.to_ascii_lowercase().starts_with("cuda:") {
                let _ = model.optimize();
            }
        }

        let patch_size = model
            .config
            .patch_size()
            .ok_or_else(|| ActorError {
                code: "load_failed".to_string(),
                message: "missing patch_size in config.json".to_string(),
                retryable: false,
            })?;
        let audiovae_cfg = model.config.audiovae().map_err(|e| ActorError {
            code: "load_failed".to_string(),
            message: e.to_string(),
            retryable: false,
        })?;
        let hop = prod_usize(&audiovae_cfg.decoder_rates);
        if hop == 0 {
            return Err(ActorError {
                code: "load_failed".to_string(),
                message: "invalid decoder_rates: product is 0".to_string(),
                retryable: false,
            });
        }
        let sample_rate = audiovae_cfg.sample_rate;
        let step_samples: u64 = (patch_size as u64)
            .checked_mul(hop as u64)
            .ok_or_else(|| ActorError {
                code: "load_failed".to_string(),
                message: "step_samples overflow".to_string(),
                retryable: false,
            })?;

        let id = self.next_model_id;
        self.next_model_id += 1;
        let model_id = format!("m{id}");
        self.models.insert(
            model_id.clone(),
            LoadedModel {
                model,
                step_samples,
                sample_rate,
            },
        );

        Ok(LoadModelResponse {
            job_id: req.job_id,
            model_id,
            step_samples,
            sample_rate,
        })
    }

    fn generate(
        &mut self,
        req: GenerateRequest,
        cancel: Arc<AtomicBool>,
        out_tx: OutTx,
    ) -> ActorResult<GenerateResponse> {
        let job_id = req.job_id;
        let m = self
            .models
            .get_mut(&req.model_id)
            .ok_or_else(|| ActorError {
                code: "not_found".to_string(),
                message: format!("unknown model_id: {}", req.model_id),
                retryable: false,
            })?;

        let step_samples = m.step_samples;
        let sample_rate = m.sample_rate;

        let emit_every = Duration::from_millis(req.emit_every_ms.max(1));
        let last_emit = Arc::new(Mutex::new(
            Instant::now().checked_sub(emit_every).unwrap(),
        ));

        let out_tx_cb = out_tx.clone();
        let last_emit2 = last_emit.clone();
        let cb = voxcpm::GenerateProgressCallback::new(move |steps_done: usize| {
            let now = Instant::now();
            {
                let mut last = last_emit2.lock().unwrap();
                if now.duration_since(*last) < emit_every {
                    return;
                }
                // Note: this callback runs on the same actor thread.
                *last = now;
            }

            let generated_samples: u64 = (steps_done as u64).saturating_mul(step_samples);
            let generated_ms: u64 = ((generated_samples as u128) * 1000u128 / (sample_rate as u128))
                .min(u64::MAX as u128) as u64;

            let _ = out_tx_cb.try_send(EngineToHost::Event(EngineEvent::GenerateProgress(
                GenerateProgress {
                    job_id,
                    steps_done,
                    step_samples,
                    sample_rate,
                    generated_samples,
                    generated_ms,
                },
            )));
        });

        m.model.set_progress_callback(Some(cb));
        m.model.set_cancel_flag(Some(cancel));

        let result: ActorResult<GenerateResponse> = (|| {
            let prompt = req
                .prompt_audio_bytes
                .as_deref()
                .map(decode_audio_mono_f32)
                .transpose()
                .map_err(|e| ActorError {
                    code: "infer_failed".to_string(),
                    message: e,
                    retryable: false,
                })?;

            let (prompt_pcm, prompt_sr) = match prompt {
                Some((pcm, sr)) => (Some(pcm), sr),
                None => (None, 0u32),
            };

            let prompt_in = prompt_pcm.as_ref().map(|pcm| WavInput::Samples {
                pcm_f32: pcm.as_slice(),
                sample_rate: prompt_sr,
            });
            let gen_args = GenerateArgs {
                text: &req.text,
                prompt_text: req.prompt_text.as_deref(),
                prompt_wav: prompt_in,
                seed: req.seed,
                max_steps: req.max_steps,
                guidance_scale: req.guidance_scale,
            };

            let audio = m.model.generate(gen_args).map_err(|e| {
                let msg = e.to_string();
                let code = if msg.trim() == "cancelled" {
                    "cancelled"
                } else {
                    "infer_failed"
                };
                ActorError {
                    code: code.to_string(),
                    message: msg,
                    retryable: false,
                }
            })?;

            let wav_bytes = encode_wav_f32_mono(audio.sample_rate, &audio.pcm_f32)
                .map_err(|e| ActorError {
                    code: "infer_failed".to_string(),
                    message: e,
                    retryable: false,
                })?;
            let samples = audio.pcm_f32.len() as u64;
            let dur_ms: u64 = ((samples as u128) * 1000u128 / (audio.sample_rate as u128))
                .min(u64::MAX as u128) as u64;
            Ok(GenerateResponse {
                job_id,
                wav_bytes,
                sample_rate: audio.sample_rate,
                samples,
                duration_ms: dur_ms,
            })
        })();

        m.model.set_progress_callback(None);
        m.model.set_cancel_flag(None);
        result
    }
}

// Ensure the cancel flag is shared across threads.
impl Drop for ActorState {
    fn drop(&mut self) {
        // Best-effort: try to free CUDA resources on this actor thread.
        for (_, m) in self.models.drain() {
            drop(m);
        }
    }
}
