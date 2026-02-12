//! Candle-based Rust API for VoxCPM.
//!
//! This crate is implemented incrementally: first load config/tokenizer/weights
//! (Milestone 1), then add model execution (next milestones).

mod audio;
mod config;
mod error;
pub mod model;
mod tokenizer;
mod weights;

use candle_core::{DType, Device, Tensor};
use candle_nn::{linear_no_bias, ops, Module};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub use config::VoxCpmConfig;
pub use error::{Result, VoxCpmError};
pub use tokenizer::VoxCpmTokenizer;
pub use weights::ModelPaths;

#[derive(Debug)]
pub struct VoxCpm {
    device: Device,
    dtype: DType,
    model_dir: PathBuf,
    pub config: VoxCpmConfig,
    pub tokenizer: VoxCpmTokenizer,
    pub paths: ModelPaths,

    weights_prefix: Option<String>,
    runtime: Option<VoxCpmRuntime>,

    show_progress: bool,
}

#[derive(Debug, Default)]
pub struct VoxCpmBuilder {
    pub device: Option<Device>,
    pub dtype: Option<DType>,
    pub weights_prefix: Option<String>,
    pub show_progress: bool,
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
}

#[derive(Debug, Clone)]
pub struct GenerateArgs<'a> {
    pub text: &'a str,
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
    /// Create a model instance from a directory on disk.
    pub fn from_dir<P: AsRef<Path>>(dir: P, builder: VoxCpmBuilder) -> Result<Self> {
        let model_dir = dir.as_ref().to_path_buf();
        let paths = ModelPaths::discover(&model_dir)?;

        let config_s = std::fs::read_to_string(&paths.config_json)?;
        let config = VoxCpmConfig::from_json_str(&config_s)?;

        let tokenizer = VoxCpmTokenizer::from_tokenizer_json(&paths.tokenizer_json)?;

        let device = builder.device.unwrap_or(Device::Cpu);
        let dtype = builder.dtype.or(config.dtype()).unwrap_or(DType::BF16);
        // CPU fallback: prefer fp32 for broad op coverage and stability.
        let dtype = if matches!(device, Device::Cpu) {
            DType::F32
        } else {
            dtype
        };

        Ok(Self {
            device,
            dtype,
            model_dir,
            config,
            tokenizer,
            paths,

            weights_prefix: builder.weights_prefix,
            runtime: None,

            show_progress: builder.show_progress,
        })
    }

    pub fn builder() -> VoxCpmBuilder {
        VoxCpmBuilder::default()
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

        // Map Rust-facing args to the Python reference defaults.
        // - `max_steps`: diffusion steps per patch.
        // - `max_len`: number of patches, heuristically bounded by text length.
        let n_timesteps = args.max_steps.max(1);
        let target_text_length = self.tokenizer.encode_ids(args.text)?.len();
        let mut max_len = (target_text_length as f64 * 6.0 + 10.0).ceil() as usize;
        max_len = max_len.min(2000).max(1);
        let min_len = 2usize;
        let cfg_value = args.guidance_scale;

        let mut progress = Progress::new(self.show_progress, max_len);

        // Build aligned (text, mask, prompt-feat) inputs; only needs an immutable runtime view.
        let rt_ref = self.runtime.as_ref().unwrap();
        let (text_token, text_mask, audio_feat, audio_mask, audio_patch_count) =
            self.build_inputs(args.text, args.prompt_wav.as_ref(), rt_ref)?;

        // After inputs are constructed, we only mutate the runtime state (KV caches).
        let rt = self.runtime.as_mut().unwrap();
        let device = self.device.clone();

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
        rt.base_lm.setup_cache(1, rt.max_length)?;
        rt.residual_lm.setup_cache(1, rt.max_length)?;

        let total_len = text_token.dims2()?.1;
        let pos: Vec<u32> = (0..total_len as u32).collect();
        let pos = Tensor::from_vec(pos, (total_len,), &device)?;

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

        // Autoregressive patch generation loop.
        for i in 0..max_len {
            let dit_hidden_1 = rt.lm_to_dit_proj.forward(&lm_hidden)?;
            let dit_hidden_2 = rt.res_to_dit_proj.forward(&residual_hidden)?;
            let mu = (dit_hidden_1 + dit_hidden_2)?; // [1, H_dit]

            let cond = prefix_feat_cond.transpose(1, 2)?.contiguous()?; // [1, D, P]
            let seed_i = args.seed.wrapping_add(i as u64);
            let pred_feat = rt
                .feat_decoder
                .sample(
                    &mu,
                    &cond,
                    rt.patch_size,
                    n_timesteps,
                    seed_i,
                    1.0,
                    cfg_value,
                    1.0,
                    true,
                )?
                .transpose(1, 2)?
                .contiguous()?; // [1, P, D]

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
            if i > min_len && stop_flag == 1 {
                progress.finish();
                break;
            }

            // Advance cached LMs by one step.
            let lm_next = rt.base_lm.forward_step_cached(&curr_embed)?; // [1, 1, H]
            lm_hidden = lm_next.squeeze(1)?;
            lm_hidden = rt.fsq_layer.forward(&lm_hidden)?; // [1, H]

            let res_inp = (&lm_hidden + &curr_embed.squeeze(1)?)?.unsqueeze(1)?;
            let res_next = rt.residual_lm.forward_step_cached(&res_inp)?;
            residual_hidden = res_next.squeeze(1)?;
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

    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
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
}
