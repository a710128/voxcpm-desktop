//! Candle-based Rust API for VoxCPM.
//!
//! This crate is implemented incrementally: first load config/tokenizer/weights
//! (Milestone 1), then add model execution (next milestones).

mod arange_cache;
mod audio;
mod config;
#[cfg(feature = "cuda")]
pub mod cuda_graph;
mod error;
pub mod model;
mod tokenizer;
mod weights;

use candle_core::{DType, Device, Tensor};
use candle_nn::{linear_no_bias, ops, Module};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub use config::VoxCpmConfig;
pub use error::{Result, VoxCpmError};
pub use tokenizer::VoxCpmTokenizer;
pub use weights::ModelPaths;

/// Optional per-step generation progress callback.
///
/// Called with `steps_done` where one step is one iteration of the patch generation loop.
#[derive(Clone)]
pub struct GenerateProgressCallback(Arc<dyn Fn(usize) + Send + Sync + 'static>);

impl GenerateProgressCallback {
    pub fn new(f: impl Fn(usize) + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }

    fn call(&self, steps_done: usize) {
        (self.0)(steps_done)
    }
}

impl fmt::Debug for GenerateProgressCallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("GenerateProgressCallback(..)")
    }
}

#[derive(Debug)]
pub struct VoxCpm {
    device: Device,
    dtype: DType,
    pub config: VoxCpmConfig,
    pub tokenizer: VoxCpmTokenizer,
    pub paths: ModelPaths,

    weights_prefix: Option<String>,
    runtime: Option<VoxCpmRuntime>,

    show_progress: bool,

    progress_callback: Option<GenerateProgressCallback>,

    cancel_flag: Option<Arc<AtomicBool>>,
}

#[derive(Debug, Default)]
pub struct VoxCpmBuilder {
    pub model_dir: Option<PathBuf>,
    pub paths: Option<ModelPaths>,

    /// Optional device spec string such as "cpu" or "cuda:0".
    ///
    /// Parsed and created inside `build()`.
    pub device_spec: Option<String>,

    /// Escape hatch: directly provide a `Device`.
    ///
    /// If set, this takes precedence over `device`.
    pub device_override: Option<Device>,

    pub dtype: Option<DType>,
    pub weights_prefix: Option<String>,
    pub show_progress: bool,

    pub progress_callback: Option<GenerateProgressCallback>,
}

struct Progress {
    enabled: bool,
    total: usize,
    start: Instant,
    last_draw: Instant,
}

impl Progress {
    fn new(enabled: bool, total: usize) -> Self {
        let now = Instant::now();
        Self {
            enabled,
            total,
            start: now,
            last_draw: now,
        }
    }

    fn draw(&mut self, done: usize) {
        if !self.enabled {
            return;
        }
        // Throttle redraws to reduce overhead.
        let now = Instant::now();
        if done < self.total && now.duration_since(self.last_draw).as_millis() < 100 {
            return;
        }
        self.last_draw = now;

        let done = done.min(self.total);
        let frac = if self.total == 0 {
            1.0
        } else {
            (done as f64) / (self.total as f64)
        };

        let width = 30usize;
        let filled = (frac * width as f64).round() as usize;
        let filled = filled.min(width);
        let mut bar = String::with_capacity(width);
        for i in 0..width {
            bar.push(if i < filled { '=' } else { ' ' });
        }

        let elapsed = now.duration_since(self.start).as_secs_f64();
        let eta = if done == 0 || done >= self.total {
            0.0
        } else {
            elapsed * ((self.total - done) as f64) / (done as f64)
        };
        let (eta_m, eta_s) = ((eta as u64) / 60, (eta as u64) % 60);
        let pct = (frac * 100.0).min(100.0);

        eprint!(
            "\r[{bar}] {done}/{tot} {pct:5.1}% ETA {eta_m:02}:{eta_s:02}",
            tot = self.total
        );
        let _ = std::io::Write::flush(&mut std::io::stderr());
    }

    fn finish(&mut self) {
        if !self.enabled {
            return;
        }
        self.draw(self.total);
        eprintln!();
    }
}

#[derive(Debug)]
struct VoxCpmRuntime {
    base_lm: crate::model::minicpm4::MiniCpmModel,
    residual_lm: crate::model::minicpm4::MiniCpmModel,
    feat_encoder: crate::model::locenc::VoxCpmLocEnc,
    feat_decoder: crate::model::unified_cfm::UnifiedCfm<crate::model::locdit::VoxCpmLocDiT>,
    fsq_layer: crate::model::fsq::ScalarQuantizationLayer,

    enc_to_lm_proj: candle_nn::Linear,
    lm_to_dit_proj: candle_nn::Linear,
    res_to_dit_proj: candle_nn::Linear,

    stop_proj: candle_nn::Linear,
    stop_head: candle_nn::Linear,

    audio_vae: crate::model::audiovae::AudioVae,

    patch_size: usize,
    feat_dim: usize,
    max_length: usize,
    chunk_size: usize,
    sample_rate: u32,

    #[cfg(feature = "cuda")]
    optimized: Option<VoxCpmOptimizedRuntime>,
}

#[cfg(feature = "cuda")]
struct VoxCpmOptimizedRuntime {
    // CUDA graph that advances base+residual cached LMs for one token.
    base_res_step_graph: crate::cuda_graph::CudaGraphModule,
    // CUDA graph for UnifiedCfm's CFG else-branch (dphi_dt computation), reused per Euler step.
    cfm_cfg_graph: crate::cuda_graph::CudaGraphModule,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for VoxCpmOptimizedRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoxCpmOptimizedRuntime").finish()
    }
}

#[derive(Debug, Clone)]
pub struct GenerateArgs<'a> {
    pub text: &'a str,
    /// Reference text for `prompt_wav` (if provided).
    ///
    /// When both `prompt_wav` and `prompt_text` are set, VoxCPM will prepend the
    /// reference text to `text` when building model inputs.
    pub prompt_text: Option<&'a str>,
    pub prompt_wav: Option<WavInput<'a>>,
    pub seed: u64,
    pub max_steps: usize,
    pub guidance_scale: f64,
}

#[derive(Debug, Clone)]
pub enum WavInput<'a> {
    Path(&'a Path),
    Samples {
        pcm_f32: &'a [f32],
        sample_rate: u32,
    },
}

#[derive(Debug, Clone)]
pub struct GeneratedAudio {
    /// PCM samples (typically normalized to [-1, 1]).
    pub pcm_f32: Vec<f32>,
    pub sample_rate: u32,
}

impl VoxCpm {
    pub fn builder() -> VoxCpmBuilder {
        VoxCpmBuilder::default()
    }

    /// Install (or clear) a cancellation flag for the next `generate()` call.
    ///
    /// If set, `generate()` will periodically check this flag and return `VoxCpmError::Cancelled`
    /// when it becomes true.
    pub fn set_cancel_flag(&mut self, flag: Option<Arc<AtomicBool>>) {
        self.cancel_flag = flag;
    }

    /// Load the MiniCPM4 text model from `model.safetensors`.
    ///
    /// This is an internal building block used by later milestones.
    pub fn load_minicpm4(&self) -> Result<crate::model::minicpm4::MiniCpmModel> {
        let cfg = self.config.minicpm4()?;
        let vb = self.paths.model_var_builder(self.dtype, &self.device)?;

        // Heuristic: most converted checkpoints namespace the base LM under `base_lm.`.
        let vb = if vb.contains_tensor("embed_tokens.weight") {
            vb
        } else if vb.contains_tensor("base_lm.embed_tokens.weight") {
            vb.pp("base_lm")
        } else {
            return Err(VoxCpmError::InvalidArg(
                "cannot locate MiniCPM4 weights in model.safetensors (expected embed_tokens.weight or base_lm.embed_tokens.weight)"
                    .into(),
            ));
        };
        Ok(crate::model::minicpm4::MiniCpmModel::new(cfg, vb)?)
    }

    /// Same as [`VoxCpm::load_minicpm4`], but allows specifying an additional
    /// safetensors key prefix if the conversion script emitted one.
    pub fn load_minicpm4_with_prefix(
        &self,
        prefix: Option<&str>,
    ) -> Result<crate::model::minicpm4::MiniCpmModel> {
        let cfg = self.config.minicpm4()?;
        let vb = self.paths.model_var_builder(self.dtype, &self.device)?;
        let vb = match prefix {
            None => vb,
            Some(p) => vb.pp(p),
        };
        Ok(crate::model::minicpm4::MiniCpmModel::new(cfg, vb)?)
    }

    /// Load AudioVAE weights if `audiovae.safetensors` (preferred) or `audiovae.pth` exists in the model directory.
    pub fn load_audiovae(&self) -> Result<Option<crate::model::audiovae::AudioVae>> {
        self.load_audiovae_with_prefix(None)
    }

    pub fn load_audiovae_with_prefix(
        &self,
        prefix: Option<&str>,
    ) -> Result<Option<crate::model::audiovae::AudioVae>> {
        // AudioVAE is executed in FP32 only.
        //
        // Rationale: VoxCPM AudioVAE uses Snake (sin/recip) and some elementwise ops that are
        // backend-dependent for BF16/FP16 on GPU, and we also need stable wave outputs.
        // Keeping the entire AudioVAE in FP32 avoids dtype-mismatch and missing-kernel issues.
        let Some(vb) = self.paths.audiovae_var_builder(DType::F32, &self.device)? else {
            return Ok(None);
        };
        let cfg = self.config.audiovae()?;
        let vb = match prefix {
            None => vb,
            Some(p) => vb.pp(p),
        };
        Ok(Some(crate::model::audiovae::AudioVae::new(cfg, vb)?))
    }

    /// Utility for prompt-audio preprocessing.
    ///
    /// Loads WAV/PCM, mixes down to mono, resamples to `audiovae.cfg.sample_rate`, then
    /// runs `AudioVae::encode` (returns `mu`, matching the reference PyTorch inference).
    pub fn encode_prompt_wav(
        &self,
        audiovae: &crate::model::audiovae::AudioVae,
        prompt_wav: &WavInput<'_>,
    ) -> Result<Tensor> {
        let sr = audiovae.cfg.sample_rate;
        let pcm = crate::audio::load_prompt_mono_f32(prompt_wav, sr)?;
        if pcm.is_empty() {
            return Err(VoxCpmError::InvalidArg("prompt_wav is empty".into()));
        }
        let t = pcm.len();
        let x = Tensor::from_vec(pcm, (1usize, t), &self.device)?;
        // AudioVAE runs in FP32 regardless of main model dtype.
        let x = x.to_dtype(DType::F32)?;
        Ok(audiovae.encode(&x, sr)?)
    }

    /// Generate audio from the provided arguments.
    pub fn generate(&mut self, args: GenerateArgs<'_>) -> Result<GeneratedAudio> {
        if self.runtime.is_none() {
            self.runtime = Some(self.load_runtime()?);
        }

        // CUDA-only fast path: enabled iff optimize() was called successfully.
        let use_optimized = {
            #[cfg(feature = "cuda")]
            {
                if !self.device.is_cuda() {
                    false
                } else if let Some(rt) = self.runtime.as_ref() {
                    match rt.optimized.as_ref() {
                        Some(_opt) => true,
                        None => false,
                    }
                } else {
                    false
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        };

        // Map Rust-facing args to the Python reference defaults.
        // - `max_steps`: diffusion steps per patch.
        // - `max_len`: number of patches, heuristically bounded by text length.
        let n_timesteps = args.max_steps.max(1);
        // Keep max_len heuristic based on target text only.
        let target_text_length = self.tokenizer.encode_ids(args.text)?.len();
        let mut max_len = (target_text_length as f64 * 6.0 + 10.0).ceil() as usize;
        max_len = max_len.min(2000).max(1);
        let min_len = 2usize;
        let cfg_value = args.guidance_scale;

        let mut progress = Progress::new(self.show_progress, max_len);

        // Build aligned (text, mask, prompt-feat) inputs; only needs an immutable runtime view.
        let rt_ref = self.runtime.as_ref().unwrap();

        // If prompt audio is provided, optionally prepend its transcript so the text side
        // matches the audio prefix content.
        let full_text_buf;
        let full_text: &str = match (args.prompt_wav.as_ref(), args.prompt_text) {
            (Some(_), Some(pt)) if !pt.is_empty() => {
                full_text_buf = format!("{pt}{}", args.text);
                &full_text_buf
            }
            _ => args.text,
        };

        let (text_token, text_mask, audio_feat, audio_mask, audio_patch_count) =
            self.build_inputs(full_text, args.prompt_wav.as_ref(), rt_ref)?;

        // After inputs are constructed, we only mutate the runtime state (KV caches).
        let rt = self.runtime.as_mut().unwrap();
        let device = self.device.clone();

        // If optimized, clear existing KV caches in-place without reallocating.
        if use_optimized {
            rt.base_lm.reset_cache_inplace()?;
            rt.residual_lm.reset_cache_inplace()?;
        }

        // Feature embedding (prompt audio patches live at the tail positions where `audio_mask==1`).
        let feat_embed = rt.feat_encoder.forward(&audio_feat)?; // [1, T, H_enc]
        let feat_embed = rt.enc_to_lm_proj.forward(&feat_embed)?; // [1, T, H_lm]

        // Text embedding.
        let scale_emb = if rt.base_lm.cfg().use_mup {
            rt.base_lm.cfg().scale_emb
        } else {
            1.0
        };
        let text_embed = (&rt.base_lm.embed_tokens(&text_token)? * scale_emb)?;

        // Combine embeddings according to masks.
        let text_m = text_mask.unsqueeze(2)?; // [1, T, 1]
        let audio_m = audio_mask.unsqueeze(2)?; // [1, T, 1]
        let combined_embed =
            (text_embed.broadcast_mul(&text_m)? + feat_embed.broadcast_mul(&audio_m)?)?;

        // Reset caches per generation call.
        if !use_optimized {
            rt.base_lm.setup_cache(1, rt.max_length)?;
            rt.residual_lm.setup_cache(1, rt.max_length)?;
        }

        let total_len = text_token.dims2()?.1;
        let pos = arange_cache::arange_u32(total_len, &device)?;

        // Prefill base LM cache.
        let (mut enc_outputs, kv_cache_tuple) =
            rt.base_lm.forward_with_cache(&combined_embed, &pos, true)?;
        rt.base_lm.fill_caches(&kv_cache_tuple)?;

        // FSQ only applies to audio positions during prefill.
        let enc_q = rt.fsq_layer.forward(&enc_outputs)?;
        enc_outputs = (enc_q.broadcast_mul(&audio_m)? + enc_outputs.broadcast_mul(&text_m)?)?;

        let lm_hidden = enc_outputs.narrow(1, total_len - 1, 1)?.squeeze(1)?; // [1, H]

        // Prefill residual LM cache.
        let residual_inp = (enc_outputs + feat_embed.broadcast_mul(&audio_m)?)?;
        let (residual_enc_outputs, residual_kv_cache_tuple) =
            rt.residual_lm
                .forward_with_cache(&residual_inp, &pos, true)?;
        rt.residual_lm.fill_caches(&residual_kv_cache_tuple)?;

        let mut residual_hidden = residual_enc_outputs
            .narrow(1, total_len - 1, 1)?
            .squeeze(1)?; // [1, H]
        let mut lm_hidden = lm_hidden;

        // Conditioning patch: last patch in the input sequence.
        let mut prefix_feat_cond = audio_feat.narrow(1, total_len - 1, 1)?.squeeze(1)?; // [1, P, D]

        // In Python: use the last (streaming_prefix_len-1) prompt patches as initial context.
        let streaming_prefix_len = 3usize;
        let context_len = audio_patch_count.min(streaming_prefix_len.saturating_sub(1));
        let mut pred_feat_seq: Vec<Tensor> = Vec::new();
        if context_len > 0 {
            // Prompt patches are the last `audio_patch_count` items in the input sequence.
            let start = total_len - context_len;
            for i in 0..context_len {
                pred_feat_seq.push(audio_feat.narrow(1, start + i, 1)?);
            }
        }

        // Keep cfg on-device (scalar tensor) for UnifiedCfm.
        let cfg_value_t = Tensor::from_vec(vec![cfg_value as f32], (1,), audio_feat.device())?;

        // Autoregressive patch generation loop.
        for i in 0..max_len {
            if let Some(cancel) = self.cancel_flag.as_ref() {
                if cancel.load(Ordering::Relaxed) {
                    progress.finish();
                    return Err(VoxCpmError::Cancelled);
                }
            }
            let dit_hidden_1 = rt.lm_to_dit_proj.forward(&lm_hidden)?;
            let dit_hidden_2 = rt.res_to_dit_proj.forward(&residual_hidden)?;
            let mu = (dit_hidden_1 + dit_hidden_2)?; // [1, H_dit]

            let cond = prefix_feat_cond.transpose(1, 2)?.contiguous()?; // [1, D, P]
            let seed_i = args.seed.wrapping_add(i as u64);
            let pred_feat = if use_optimized {
                #[cfg(feature = "cuda")]
                {
                    let opt = rt.optimized.as_ref().unwrap();
                    rt.feat_decoder
                        .sample_optimized_cuda(
                            &mu,
                            &cond,
                            rt.patch_size,
                            n_timesteps,
                            seed_i,
                            1.0,
                            &cfg_value_t,
                            1.0,
                            true,
                            &opt.cfm_cfg_graph,
                        )?
                        .transpose(1, 2)?
                        .contiguous()?
                }
                #[cfg(not(feature = "cuda"))]
                {
                    // Not reachable: use_optimized is never set without the cuda feature.
                    rt.feat_decoder
                        .sample(
                            &mu,
                            &cond,
                            rt.patch_size,
                            n_timesteps,
                            seed_i,
                            1.0,
                            &cfg_value_t,
                            1.0,
                            true,
                        )?
                        .transpose(1, 2)?
                        .contiguous()?
                }
            } else {
                rt.feat_decoder
                    .sample(
                        &mu,
                        &cond,
                        rt.patch_size,
                        n_timesteps,
                        seed_i,
                        1.0,
                        &cfg_value_t,
                        1.0,
                        true,
                    )?
                    .transpose(1, 2)?
                    .contiguous()?
            }; // [1, P, D]

            let pred_patch = pred_feat.unsqueeze(1)?; // [1, 1, P, D]
            let curr_embed = rt.feat_encoder.forward(&pred_patch)?;
            let curr_embed = rt.enc_to_lm_proj.forward(&curr_embed)?; // [1, 1, H]

            pred_feat_seq.push(pred_patch);
            prefix_feat_cond = pred_feat;

            let stop_logits = rt.stop_proj.forward(&lm_hidden)?;
            let stop_logits = ops::silu(&stop_logits)?;
            let stop_logits = rt.stop_head.forward(&stop_logits)?; // [1, 2]
                                                                   // `to_vec*` expects CPU f32 storage; on GPU we may be running BF16.
            let stop_v = stop_logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            let stop_flag = if stop_v[0][1] > stop_v[0][0] { 1 } else { 0 };

            // Progress is based on generated patch count (prompt context not included).
            progress.draw(i + 1);
            if let Some(cb) = self.progress_callback.as_ref() {
                cb.call(i + 1);
            }
            if let Some(cancel) = self.cancel_flag.as_ref() {
                if cancel.load(Ordering::Relaxed) {
                    progress.finish();
                    return Err(VoxCpmError::Cancelled);
                }
            }
            if i > min_len && stop_flag == 1 {
                progress.finish();
                break;
            }

            // Advance cached LMs by one step.
            if use_optimized {
                #[cfg(feature = "cuda")]
                {
                    let opt = rt.optimized.as_ref().unwrap();

                    // Keep Rust-side cache position in sync (graph only replays CUDA work).
                    let pos_base = rt.base_lm.step()?;
                    let pos_res = rt.residual_lm.step()?;
                    if pos_base != pos_res {
                        return Err(VoxCpmError::InvalidArg(format!(
                            "optimized step: base/residual cache pos mismatch {pos_base} != {pos_res}"
                        )));
                    }
                    let pos_ids = Tensor::full(pos_base, (1usize,), &device)?;

                    let outs = opt
                        .base_res_step_graph
                        .run(&[curr_embed.clone(), pos_ids])?;
                    if outs.len() != 2 {
                        return Err(VoxCpmError::InvalidArg(format!(
                            "optimized step graph returned {} outputs (expected 2)",
                            outs.len()
                        )));
                    }
                    lm_hidden = outs[0].clone();
                    residual_hidden = outs[1].clone();
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let lm_next = rt.base_lm.forward_step_cached(&curr_embed)?;
                    lm_hidden = lm_next.squeeze(1)?;
                    lm_hidden = rt.fsq_layer.forward(&lm_hidden)?;
                    let res_inp = (&lm_hidden + &curr_embed.squeeze(1)?)?.unsqueeze(1)?;
                    let res_next = rt.residual_lm.forward_step_cached(&res_inp)?;
                    residual_hidden = res_next.squeeze(1)?;
                }
            } else {
                let lm_next = rt.base_lm.forward_step_cached(&curr_embed)?; // [1, 1, H]
                lm_hidden = lm_next.squeeze(1)?;
                lm_hidden = rt.fsq_layer.forward(&lm_hidden)?; // [1, H]

                let res_inp = (&lm_hidden + &curr_embed.squeeze(1)?)?.unsqueeze(1)?;
                let res_next = rt.residual_lm.forward_step_cached(&res_inp)?;
                residual_hidden = res_next.squeeze(1)?;
            }
        }

        progress.finish();

        if pred_feat_seq.is_empty() {
            return Err(VoxCpmError::InvalidArg(
                "generation produced no latent patches".into(),
            ));
        }

        let pred_refs: Vec<&Tensor> = pred_feat_seq.iter().collect();
        let pred_seq = Tensor::cat(&pred_refs, 1)?; // [1, T, P, D]

        // Rearrange: [B, T, P, D] -> [B, D, T*P]
        let z = pred_seq.transpose(2, 3)?.transpose(1, 2)?; // [1, D, T, P]
        let (b, d, t, p) = z.dims4()?;
        let z = z.reshape((b, d, t * p))?;

        let wav = rt.audio_vae.decode(&z.to_dtype(DType::F32)?)?; // [1, 1, samples]
        let wav = wav.squeeze(1)?.squeeze(0)?; // [samples]
        let mut pcm = wav.to_vec1::<f32>()?;

        // Drop the prompt context patches (if any) to avoid repeating prompt tail.
        if context_len > 0 {
            let drop = context_len * rt.patch_size * rt.chunk_size;
            if drop < pcm.len() {
                pcm.drain(0..drop);
            } else {
                pcm.clear();
            }
        }

        Ok(GeneratedAudio {
            pcm_f32: pcm,
            sample_rate: rt.sample_rate,
        })
    }

    /// Update the generation progress callback.
    ///
    /// This is intended for long-lived engine processes that reuse a single `VoxCpm` instance
    /// across multiple generate calls.
    pub fn set_progress_callback(&mut self, cb: Option<GenerateProgressCallback>) {
        self.progress_callback = cb;
    }

    /// Capture CUDA graphs for generation.
    ///
    /// CUDA-only. After this succeeds, `generate()` will use CUDA graphs when possible.
    pub fn optimize(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if !self.device.is_cuda() {
                return Err(VoxCpmError::InvalidArg(
                    "optimize() requires a CUDA device".into(),
                ));
            }
            if self.runtime.is_none() {
                self.runtime = Some(self.load_runtime()?);
            }
            let rt = self.runtime.as_mut().unwrap();

            // Pre-allocate caches once so capture can reference stable device pointers.
            rt.base_lm.setup_cache(1, rt.max_length)?;
            rt.residual_lm.setup_cache(1, rt.max_length)?;
            rt.base_lm.reset_cache_inplace()?;
            rt.residual_lm.reset_cache_inplace()?;

            use crate::cuda_graph::CudaGraphModule;

            // (1) Combined base+residual cached step graph.
            let h = rt.base_lm.cfg().hidden_size;
            let curr_ex = Tensor::zeros((1usize, 1usize, h), self.dtype, &self.device)?;
            let pos_ex = Tensor::zeros((1usize,), DType::U32, &self.device)?;
            let base_lm = &mut rt.base_lm;
            let residual_lm = &mut rt.residual_lm;
            let fsq_layer = &rt.fsq_layer;
            let base_res_step_graph = CudaGraphModule::capture(&[curr_ex, pos_ex], |ins| {
                let curr = &ins[0];
                let pos = &ins[1];
                let lm_next = base_lm.forward_step(curr, pos)?; // [1, 1, H]
                let lm_hidden = lm_next.squeeze(1)?;
                let lm_hidden = fsq_layer.forward(&lm_hidden)?; // [1, H]

                let curr_s = curr.squeeze(1)?;
                let res_inp = (&lm_hidden + &curr_s)?.unsqueeze(1)?;
                let res_next = residual_lm.forward_step(&res_inp, pos)?;
                let res_hidden = res_next.squeeze(1)?;

                Ok(vec![lm_hidden, res_hidden])
            })?;

            // (2) UnifiedCfm CFG else-branch graph (dphi_dt).
            // Shapes are fixed per model config: B=1, C=feat_dim, T=patch_size.
            let c = rt.feat_dim;
            let t = rt.patch_size;
            let dit_h = self.config.dit_config()?.hidden_dim;
            let x_ex = Tensor::zeros((1usize, c, t), self.dtype, &self.device)?;
            let mu_ex = Tensor::zeros((1usize, dit_h), self.dtype, &self.device)?;
            let t_ex = Tensor::zeros((1usize,), DType::F32, &self.device)?;
            let dt_ex = Tensor::zeros((1usize,), DType::F32, &self.device)?;
            let cond_ex = Tensor::zeros((1usize, c, t), self.dtype, &self.device)?;
            // cfg is already normalized to [1,1,1] in model dtype by the caller.
            let cfg_ex = Tensor::zeros((1usize, 1usize, 1usize), self.dtype, &self.device)?;

            let estimator = &rt.feat_decoder.estimator;
            let mean_mode = rt.feat_decoder.mean_mode;
            let cfm_cfg_graph = CudaGraphModule::capture(
                &[x_ex, mu_ex, t_ex, dt_ex, cond_ex, cfg_ex],
                move |ins| {
                    let x = &ins[0];
                    let mu = &ins[1];
                    let t_prev = &ins[2];
                    let dt_f32 = &ins[3];
                    let cond = &ins[4];
                    let cfg = &ins[5];

                    let (b, c, tt) = x.dims3()?;
                    let x_in = Tensor::cat(&[x, x], 0)?; // [2B, C, T]
                    let zeros_mu = Tensor::zeros_like(mu)?;
                    let mu_in = Tensor::cat(&[mu, &zeros_mu], 0)?; // [2B, H]

                    let t_in = t_prev.broadcast_as((2 * b,))?;
                    let dt_in = if mean_mode {
                        dt_f32.broadcast_as((2 * b,))?
                    } else {
                        Tensor::zeros((2 * b,), DType::F32, x.device())?
                    };
                    let cond_in = Tensor::cat(&[cond, cond], 0)?;

                    let pred = estimator.forward(&x_in, &mu_in, &t_in, &cond_in, &dt_in)?;
                    let pos = pred.narrow(0, 0, b)?;
                    let neg = pred.narrow(0, b, b)?;

                    // use_cfg_zero_star = true.
                    let pos_flat = pos.reshape((b, c * tt))?;
                    let neg_flat = neg.reshape((b, c * tt))?;
                    let dot = (pos_flat * &neg_flat)?.sum_keepdim(1)?;
                    let denom = neg_flat.sqr()?.sum_keepdim(1)?;
                    let st = dot.broadcast_div(&(denom + 1e-8)?)?;
                    let st = st.reshape((b, 1, 1))?;
                    let neg_scaled = neg.broadcast_mul(&st)?;

                    let diff = (pos - &neg_scaled)?;
                    let dphi_dt = (neg_scaled + diff.broadcast_mul(cfg)?)?;
                    Ok(vec![dphi_dt])
                },
            )?;

            // Capture will have mutated caches due to the example step; clear them.
            rt.base_lm.reset_cache_inplace()?;
            rt.residual_lm.reset_cache_inplace()?;

            rt.optimized = Some(VoxCpmOptimizedRuntime {
                base_res_step_graph,
                cfm_cfg_graph,
            });

            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(VoxCpmError::InvalidArg(
                "optimize() requires the cuda feature".into(),
            ))
        }
    }

    fn load_runtime(&self) -> Result<VoxCpmRuntime> {
        let vb0 = self
            .paths
            .model_var_builder(self.dtype, &self.device)
            .map_err(|e| VoxCpmError::InvalidArg(format!("model weights not available: {e}")))?;
        let vb0 = match self.weights_prefix.as_deref() {
            None => vb0,
            Some(p) => vb0.pp(p),
        };

        let patch_size = self.config.patch_size().unwrap_or(2);
        let feat_dim = self.config.feat_dim().unwrap_or(64);
        let max_length = self.config.max_length();

        // Warm cached arange tensors once per init-thread.
        // This avoids repeated host->device uploads for common dynamic position-id tensors.
        arange_cache::warm_u32(max_length, &self.device)?;

        // Base LM: either at root or under `base_lm.`.
        let vb_base = if vb0.contains_tensor("embed_tokens.weight") {
            vb0.clone()
        } else if vb0.contains_tensor("base_lm.embed_tokens.weight") {
            vb0.pp("base_lm")
        } else {
            return Err(VoxCpmError::InvalidArg(
                "cannot locate base_lm weights (expected embed_tokens.weight or base_lm.embed_tokens.weight)".into(),
            ));
        };
        let base_cfg = self.config.minicpm4()?;
        let base_lm = crate::model::minicpm4::MiniCpmModel::new(base_cfg.clone(), vb_base)?;

        // Residual LM: same config but fewer layers and no vocab.
        let mut res_cfg = base_cfg;
        res_cfg.num_hidden_layers = self.config.residual_lm_num_layers();
        res_cfg.vocab_size = 0;
        let residual_lm =
            crate::model::minicpm4::MiniCpmModel::new(res_cfg, vb0.pp("residual_lm"))?;

        // Local Encoder.
        let enc_cfg = self.config.locenc_minicpm4()?;
        let feat_encoder =
            crate::model::locenc::VoxCpmLocEnc::new(enc_cfg, feat_dim, vb0.pp("feat_encoder"))?;

        // Local DiT estimator + UnifiedCFM wrapper.
        let dit_cfg = self.config.locdit_minicpm4()?;
        let estimator = crate::model::locdit::VoxCpmLocDiT::new(
            dit_cfg,
            feat_dim,
            vb0.pp("feat_decoder").pp("estimator"),
        )?;
        let mean_mode = self.config.dit_mean_mode().unwrap_or(false);
        let feat_decoder =
            crate::model::unified_cfm::UnifiedCfm::new(feat_dim, estimator, mean_mode);

        // Projection layers.
        let fsq_layer = crate::model::fsq::ScalarQuantizationLayer::new(
            base_lm.cfg().hidden_size,
            base_lm.cfg().hidden_size,
            self.config.scalar_quantization_latent_dim(),
            self.config.scalar_quantization_scale(),
            vb0.pp("fsq_layer"),
        )?;

        let enc_to_lm_proj = candle_nn::linear(
            self.config.encoder_config()?.hidden_dim,
            base_lm.cfg().hidden_size,
            vb0.pp("enc_to_lm_proj"),
        )?;
        let lm_to_dit_proj = candle_nn::linear(
            base_lm.cfg().hidden_size,
            self.config.dit_config()?.hidden_dim,
            vb0.pp("lm_to_dit_proj"),
        )?;
        let res_to_dit_proj = candle_nn::linear(
            base_lm.cfg().hidden_size,
            self.config.dit_config()?.hidden_dim,
            vb0.pp("res_to_dit_proj"),
        )?;

        // Stop predictor.
        let stop_proj = candle_nn::linear(
            base_lm.cfg().hidden_size,
            base_lm.cfg().hidden_size,
            vb0.pp("stop_proj"),
        )?;
        let stop_head = linear_no_bias(base_lm.cfg().hidden_size, 2, vb0.pp("stop_head"))?;

        // Audio VAE.
        let audio_vae = self.load_audiovae()?.ok_or_else(|| {
            VoxCpmError::InvalidArg(
                "missing AudioVAE weights (audiovae.safetensors or audiovae.pth)".into(),
            )
        })?;
        let chunk_size = audio_vae.chunk_size();
        let sample_rate = audio_vae.cfg.sample_rate;

        Ok(VoxCpmRuntime {
            base_lm,
            residual_lm,
            feat_encoder,
            feat_decoder,
            fsq_layer,
            enc_to_lm_proj,
            lm_to_dit_proj,
            res_to_dit_proj,
            stop_proj,
            stop_head,
            audio_vae,
            patch_size,
            feat_dim,
            max_length,
            chunk_size,
            sample_rate,

            #[cfg(feature = "cuda")]
            optimized: None,
        })
    }

    fn build_inputs(
        &self,
        text: &str,
        prompt_wav: Option<&WavInput<'_>>,
        rt: &VoxCpmRuntime,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, usize)> {
        let mut text_ids = self.tokenizer.encode_ids(text)?;
        // Matches the Python reference model.
        const AUDIO_START_TOKEN: u32 = 101;
        text_ids.push(AUDIO_START_TOKEN);
        let text_length = text_ids.len();

        // Prompt audio to patch features.
        let (audio_feat_tail, audio_patch_count) = if let Some(wav) = prompt_wav {
            let mut pcm = crate::audio::load_prompt_mono_f32(wav, rt.sample_rate)?;
            if pcm.is_empty() {
                (None, 0usize)
            } else {
                // Left pad waveform to a multiple of patch_len so the *end* aligns with patch boundaries.
                let patch_len = rt.patch_size * rt.chunk_size;
                let rem = pcm.len() % patch_len;
                if rem != 0 {
                    let pad = patch_len - rem;
                    let mut padded = vec![0f32; pad];
                    padded.extend_from_slice(&pcm);
                    pcm = padded;
                }
                let t = pcm.len();
                // AudioVAE runs in FP32 regardless of main model dtype.
                let x = Tensor::from_vec(pcm, (1usize, t), &self.device)?.to_dtype(DType::F32)?;
                let z = rt.audio_vae.encode(&x, rt.sample_rate)?; // [1, D, T_latent]
                let (_b, d, t_latent) = z.dims3()?;
                if d != rt.feat_dim {
                    return Err(VoxCpmError::InvalidArg(format!(
                        "AudioVAE latent_dim={d} != feat_dim={} (check config.json)",
                        rt.feat_dim
                    )));
                }
                if t_latent % rt.patch_size != 0 {
                    return Err(VoxCpmError::InvalidArg(format!(
                        "prompt latent length {t_latent} is not divisible by patch_size={} (pad logic mismatch)",
                        rt.patch_size
                    )));
                }
                let audio_len = t_latent / rt.patch_size;

                // Rearrange: [1, D, T] -> [audio_len, P, D]
                let z = z.squeeze(0)?; // [D, T]
                let z = z.reshape((d, audio_len, rt.patch_size))?;
                let z = z.transpose(0, 1)?.transpose(1, 2)?; // [audio_len, P, D]
                (Some(z), audio_len)
            }
        } else {
            (None, 0usize)
        };

        // Construct aligned token + feature sequences.
        let total_len = text_length + audio_patch_count;
        let mut token_ids_i64 = Vec::with_capacity(total_len);
        token_ids_i64.extend(text_ids.into_iter().map(|v| v as i64));
        token_ids_i64.extend(std::iter::repeat(0i64).take(audio_patch_count));
        let text_token = Tensor::from_vec(token_ids_i64, (1usize, total_len), &self.device)?;

        let mut text_mask_v = Vec::with_capacity(total_len);
        text_mask_v.extend(std::iter::repeat(1f32).take(text_length));
        text_mask_v.extend(std::iter::repeat(0f32).take(audio_patch_count));
        let mut audio_mask_v = Vec::with_capacity(total_len);
        audio_mask_v.extend(std::iter::repeat(0f32).take(text_length));
        audio_mask_v.extend(std::iter::repeat(1f32).take(audio_patch_count));

        let text_mask = Tensor::from_vec(text_mask_v, (1usize, total_len), &self.device)?
            .to_dtype(self.dtype)?;
        let audio_mask = Tensor::from_vec(audio_mask_v, (1usize, total_len), &self.device)?
            .to_dtype(self.dtype)?;

        // audio_feat: [1, T, P, D]
        let audio_feat = if let Some(tail) = audio_feat_tail {
            let pad = Tensor::zeros(
                (text_length, rt.patch_size, rt.feat_dim),
                self.dtype,
                &self.device,
            )?;
            let feat = Tensor::cat(&[&pad, &tail.to_dtype(self.dtype)?], 0)?;
            feat.unsqueeze(0)?
        } else {
            Tensor::zeros(
                (1usize, total_len, rt.patch_size, rt.feat_dim),
                self.dtype,
                &self.device,
            )?
        };

        Ok((
            text_token,
            text_mask,
            audio_feat,
            audio_mask,
            audio_patch_count,
        ))
    }
}

impl VoxCpmBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Provide a model directory and let VoxCPM discover required files.
    pub fn model_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.model_dir = Some(dir.into());
        self
    }

    /// Override model paths instead of discovering from `model_dir`.
    pub fn paths(mut self, paths: ModelPaths) -> Self {
        self.paths = Some(paths);
        self
    }

    /// Select a device by string spec (e.g. "cpu" or "cuda:0").
    ///
    /// The device is created inside `build()`.
    pub fn device_str(mut self, spec: impl Into<String>) -> Self {
        self.device_spec = Some(spec.into());
        self
    }

    /// Escape hatch: provide a fully constructed device.
    ///
    /// If set, this takes precedence over `device_str`.
    pub fn device_override(mut self, device: Device) -> Self {
        self.device_override = Some(device);
        self
    }

    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Optional root prefix for all model weights in `model.safetensors`.
    ///
    /// Example: if the conversion script produced keys like `model.base_lm.*`, pass `"model"`.
    pub fn weights_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.weights_prefix = Some(prefix.into());
        self
    }

    pub fn show_progress(mut self, show: bool) -> Self {
        self.show_progress = show;
        self
    }

    /// Provide a per-step generation progress callback.
    pub fn progress_callback(mut self, cb: GenerateProgressCallback) -> Self {
        self.progress_callback = Some(cb);
        self
    }

    pub fn build(self) -> Result<VoxCpm> {
        let paths = match self.paths {
            Some(p) => p,
            None => {
                let dir = self.model_dir.ok_or_else(|| {
                    VoxCpmError::InvalidArg(
                        "missing model_dir: set builder.model_dir(...) or builder.paths(...)"
                            .into(),
                    )
                })?;
                ModelPaths::discover(&dir)?
            }
        };

        let config_s = std::fs::read_to_string(&paths.config_json)?;
        let config = VoxCpmConfig::from_json_str(&config_s)?;
        let tokenizer = VoxCpmTokenizer::from_tokenizer_json(&paths.tokenizer_json)?;

        let device = match self.device_override {
            Some(d) => d,
            None => {
                let spec = self.device_spec.as_deref().unwrap_or("cpu");
                Self::create_device_from_spec(spec)?
            }
        };

        // DType selection (model weights + runtime buffers).
        // Priority:
        // 1) Explicit builder override.
        // 2) CUDA: pick based on SM (ignore config.json dtype).
        // 3) Non-CUDA: config.json dtype.
        // 4) Default: BF16.
        let mut dtype = self.dtype.unwrap_or_else(|| {
            #[cfg(feature = "cuda")]
            if device.is_cuda() {
                // sm < 80 -> fp16, sm >= 80 -> bf16. If we cannot query SM, default bf16.
                return match cuda_sm(device.as_cuda_device().ok()) {
                    Some(sm) if sm < 80 => DType::F16,
                    Some(_) => DType::BF16,
                    None => DType::BF16,
                };
            }
            config.dtype().unwrap_or(DType::BF16)
        });
        // CPU fallback: prefer fp32 for broad op coverage and stability.
        if matches!(device, Device::Cpu) {
            dtype = DType::F32;
        }

        // CUDA device init: keep behavior centralized here.
        #[cfg(feature = "cuda")]
        if device.is_cuda() {
            // When using the CUDA async allocator (memory pools), set the default pool's
            // release threshold to UINT64_MAX for stability (e.g. CUDA graph capture).
            crate::cuda_graph::set_mempool_release_threshold_max(&device)?;

            // Disable cudarc per-slice CUDA event tracking globally for this device.
            // This makes CUDA graph capture/replay reliable (no event nodes recorded).
            // Safety: toggles a device-wide setting; callers should not rely on event tracking.
            unsafe { device.as_cuda_device()?.disable_event_tracking() };
        }

        Ok(VoxCpm {
            device,
            dtype,
            config,
            tokenizer,
            paths,
            weights_prefix: self.weights_prefix,
            runtime: None,
            show_progress: self.show_progress,
            progress_callback: self.progress_callback,
            cancel_flag: None,
        })
    }

    fn create_device_from_spec(spec: &str) -> Result<Device> {
        let s = spec.trim();
        if s.is_empty() || s.eq_ignore_ascii_case("cpu") {
            return Ok(Device::Cpu);
        }

        // Apple GPU backend (Candle Metal). Accept common aliases.
        if s.eq_ignore_ascii_case("metal") || s.eq_ignore_ascii_case("mps") {
            #[cfg(feature = "metal")]
            {
                return Ok(Device::new_metal(0)?);
            }
            #[cfg(not(feature = "metal"))]
            {
                return Err(VoxCpmError::InvalidArg(
                    "requested Metal device but crate not built with feature metal".into(),
                ));
            }
        }
        if let Some(rest) = s.strip_prefix("metal:").or_else(|| s.strip_prefix("mps:")) {
            let idx: usize = rest
                .parse()
                .map_err(|_| VoxCpmError::InvalidArg(format!("invalid metal device index: {s}")))?;
            #[cfg(feature = "metal")]
            {
                return Ok(Device::new_metal(idx)?);
            }
            #[cfg(not(feature = "metal"))]
            {
                let _ = idx;
                return Err(VoxCpmError::InvalidArg(
                    "requested Metal device but crate not built with feature metal".into(),
                ));
            }
        }
        if let Some(rest) = s.strip_prefix("cuda:") {
            let idx: usize = rest
                .parse()
                .map_err(|_| VoxCpmError::InvalidArg(format!("invalid cuda device index: {s}")))?;
            #[cfg(feature = "cuda")]
            {
                return Ok(Device::new_cuda_with_stream(idx)?);
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = idx;
                return Err(VoxCpmError::InvalidArg(
                    "requested CUDA device but crate not built with feature cuda".into(),
                ));
            }
        }
        Err(VoxCpmError::InvalidArg(format!(
            "unsupported device spec: {s} (use cpu, cuda:N, metal[:N])"
        )))
    }
}

#[cfg(feature = "cuda")]
fn cuda_sm(cuda: Option<&candle_core::cuda_backend::CudaDevice>) -> Option<u32> {
    use candle_core::cuda_backend::cudarc::driver::sys;

    let cuda = cuda?;
    let stream = cuda.cuda_stream();
    let ctx = stream.context();
    let major = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .ok()?;
    let minor = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .ok()?;
    if major < 0 || minor < 0 {
        return None;
    }
    Some((major as u32) * 10 + (minor as u32))
}
