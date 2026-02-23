use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::Module;
use rubato::Resampler;
use voxcpm::model;
use voxcpm::{ModelPaths, Result, VoxCpmConfig, VoxCpmError, VoxCpmTokenizer};

fn arange_u32(len: usize, dev: &Device) -> Result<Tensor> {
    let mut v = Vec::with_capacity(len);
    for i in 0..len {
        v.push(i as u32);
    }
    Ok(Tensor::from_vec(v, (len,), dev)?)
}

fn usage() -> ! {
    eprintln!(
        "usage: parity_e2e_prefill_decode1 <model_dir> --devices cpu,metal:0 --text <text> [--prompt-wav none|path] [--prompt-text <text>] [--weights-prefix none|PREFIX] [--atol 3e-4] [--rtol 3e-4]\n\nNotes:\n- Candle->Candle parity across devices (CPU baseline vs targets).\n- Focus: prefill last hidden + prefill KV cache + decode1 last hidden + decode1 new KV slice.\n- decode1 uses teacher-forced patch = last prefill input patch (no sampling).\n\nExample:\n  CANDLE_METAL_ENABLE_FAST_MATH=0 cargo run -p voxcpm --features metal --example parity_e2e_prefill_decode1 -- \\n    \"/path/to/model_dir\" --devices cpu,metal:0 --text \"hello\" --prompt-wav none"
    );
    std::process::exit(2);
}

#[derive(Debug)]
struct Args {
    model_dir: PathBuf,
    devices: Vec<String>,
    text: String,
    prompt_wav: Option<PathBuf>,
    prompt_text: Option<String>,
    weights_prefix: Option<String>,
    atol: f64,
    rtol: f64,
}

fn parse_args() -> Args {
    let mut it = std::env::args().skip(1);
    let model_dir: PathBuf = it.next().map(PathBuf::from).unwrap_or_else(|| usage());

    let mut devices: Option<String> = None;
    let mut text: Option<String> = None;
    let mut prompt_wav: Option<PathBuf> = None;
    let mut prompt_text: Option<String> = None;
    let mut weights_prefix: Option<String> = None;
    let mut atol = 3e-4;
    let mut rtol = 3e-4;

    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--devices" => devices = Some(it.next().unwrap_or_else(|| usage())),
            "--text" => text = Some(it.next().unwrap_or_else(|| usage())),
            "--prompt-wav" => {
                let v = it.next().unwrap_or_else(|| usage());
                prompt_wav = if v == "none" {
                    None
                } else {
                    Some(PathBuf::from(v))
                };
            }
            "--prompt-text" => prompt_text = Some(it.next().unwrap_or_else(|| usage())),
            "--weights-prefix" => {
                let v = it.next().unwrap_or_else(|| usage());
                weights_prefix = if v == "none" { None } else { Some(v) };
            }
            "--atol" => {
                atol = it
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--rtol" => {
                rtol = it
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            _ => usage(),
        }
    }

    let devices = devices.unwrap_or_else(|| "cpu,metal:0".to_owned());
    let devices: Vec<String> = devices
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_owned())
        .collect();
    if devices.is_empty() || !devices[0].eq_ignore_ascii_case("cpu") {
        eprintln!("--devices must start with cpu");
        usage();
    }

    let text = text.unwrap_or_else(|| usage());

    Args {
        model_dir,
        devices,
        text,
        prompt_wav,
        prompt_text,
        weights_prefix,
        atol,
        rtol,
    }
}

fn device_from_spec(spec: &str) -> Result<Device> {
    if spec.is_empty() || spec.eq_ignore_ascii_case("cpu") {
        return Ok(Device::Cpu);
    }
    if let Some(idx) = spec.strip_prefix("metal") {
        let idx = idx.strip_prefix(':').unwrap_or("0");
        #[cfg(feature = "metal")]
        {
            let idx = idx.parse::<usize>().map_err(|e| {
                VoxCpmError::InvalidArg(format!("invalid metal index {idx:?}: {e}"))
            })?;
            return Ok(Device::new_metal(idx)?);
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = idx;
            return Err(VoxCpmError::InvalidArg(
                "device spec metal[:N] requires the metal feature".into(),
            ));
        }
    }
    if let Some(idx) = spec.strip_prefix("mps") {
        let idx = idx.strip_prefix(':').unwrap_or("0");
        #[cfg(feature = "metal")]
        {
            let idx = idx
                .parse::<usize>()
                .map_err(|e| VoxCpmError::InvalidArg(format!("invalid mps index {idx:?}: {e}")))?;
            return Ok(Device::new_metal(idx)?);
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = idx;
            return Err(VoxCpmError::InvalidArg(
                "device spec mps[:N] requires the metal feature".into(),
            ));
        }
    }
    if let Some(idx) = spec.strip_prefix("cuda:") {
        #[cfg(feature = "cuda")]
        {
            let idx = idx
                .parse::<usize>()
                .map_err(|e| VoxCpmError::InvalidArg(format!("invalid cuda index {idx:?}: {e}")))?;
            return Ok(Device::new_cuda(idx)?);
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = idx;
            return Err(VoxCpmError::InvalidArg(
                "device spec cuda:N requires the cuda feature".into(),
            ));
        }
    }
    Err(VoxCpmError::InvalidArg(format!(
        "unsupported device spec {spec:?} (expected cpu, metal[:N], mps[:N], cuda:N)"
    )))
}

#[derive(Debug, Clone, Copy)]
struct DiffStats {
    n: usize,
    bad: usize,
    max_abs: f64,
    max_rel: f64,
    mean_abs: f64,
    rmse: f64,
}

fn diff_stats(a: &Tensor, b: &Tensor, atol: f64, rtol: f64) -> Result<DiffStats> {
    if a.dims() != b.dims() {
        return Err(VoxCpmError::InvalidArg(format!(
            "shape mismatch: {:?} vs {:?}",
            a.dims(),
            b.dims()
        )));
    }

    let a = a
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?;
    let b = b
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?;
    let av = a.to_vec1::<f32>()?;
    let bv = b.to_vec1::<f32>()?;

    let mut max_abs = 0f64;
    let mut max_rel = 0f64;
    let mut bad = 0usize;
    let mut sum_abs = 0f64;
    let mut sum_sq = 0f64;
    for (x, y) in av.iter().zip(bv.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        if !x.is_finite() || !y.is_finite() {
            return Err(VoxCpmError::InvalidArg(
                "non-finite value encountered".into(),
            ));
        }
        let d = x - y;
        let abs = d.abs();
        let rel = abs / (y.abs().max(1e-12));
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        sum_abs += abs;
        sum_sq += d * d;
        if abs > atol + rtol * y.abs() {
            bad += 1;
        }
    }
    let n = av.len();
    let mean_abs = if n == 0 { 0.0 } else { sum_abs / (n as f64) };
    let rmse = if n == 0 {
        0.0
    } else {
        (sum_sq / (n as f64)).sqrt()
    };
    Ok(DiffStats {
        n,
        bad,
        max_abs,
        max_rel,
        mean_abs,
        rmse,
    })
}

fn report(label: &str, spec: &str, a: &Tensor, b: &Tensor, atol: f64, rtol: f64) -> Result<()> {
    let s = diff_stats(a, b, atol, rtol)?;
    eprintln!(
        "  {label} {spec} n={} bad={} max_abs={:.6e} max_rel={:.6e} mean_abs={:.6e} rmse={:.6e}",
        s.n, s.bad, s.max_abs, s.max_rel, s.mean_abs, s.rmse
    );
    if s.bad != 0 {
        return Err(VoxCpmError::InvalidArg(format!(
            "allclose failed for {label} {spec} (bad={}/{}, atol={atol}, rtol={rtol})",
            s.bad, s.n
        )));
    }
    Ok(())
}

fn wav_to_mono_f32(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut r = hound::WavReader::open(path)
        .map_err(|e| VoxCpmError::InvalidArg(format!("open wav {path:?}: {e}")))?;
    let spec = r.spec();
    let sr = spec.sample_rate;
    let ch = spec.channels as usize;
    if ch == 0 {
        return Err(VoxCpmError::InvalidArg("wav has 0 channels".into()));
    }

    let mut samples = Vec::<f32>::new();
    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => {
            let iter = r
                .samples::<f32>()
                .map(|s| s.map_err(|e| VoxCpmError::InvalidArg(format!("read wav: {e}"))));
            for s in iter {
                samples.push(s?);
            }
        }
        (hound::SampleFormat::Int, 16) => {
            let iter = r
                .samples::<i16>()
                .map(|s| s.map_err(|e| VoxCpmError::InvalidArg(format!("read wav: {e}"))));
            for s in iter {
                samples.push((s? as f32) / 32768.0);
            }
        }
        (hound::SampleFormat::Int, 32) => {
            let iter = r
                .samples::<i32>()
                .map(|s| s.map_err(|e| VoxCpmError::InvalidArg(format!("read wav: {e}"))));
            for s in iter {
                samples.push((s? as f32) / 2147483648.0);
            }
        }
        other => {
            return Err(VoxCpmError::InvalidArg(format!(
                "unsupported wav format {:?} (expected f32/i16/i32)",
                other
            )))
        }
    }

    if samples.is_empty() {
        return Ok((samples, sr));
    }

    if ch == 1 {
        return Ok((samples, sr));
    }

    // Interleaved -> average channels.
    if samples.len() % ch != 0 {
        return Err(VoxCpmError::InvalidArg(
            "wav samples not divisible by channels".into(),
        ));
    }
    let frames = samples.len() / ch;
    let mut mono = Vec::with_capacity(frames);
    for i in 0..frames {
        let mut acc = 0f32;
        for c in 0..ch {
            acc += samples[i * ch + c];
        }
        mono.push(acc / (ch as f32));
    }
    Ok((mono, sr))
}

fn resample_if_needed(pcm: &[f32], sr_in: u32, sr_out: u32) -> Result<Vec<f32>> {
    if sr_in == sr_out {
        return Ok(pcm.to_vec());
    }

    // Sinc resampler (mono).
    let ratio = (sr_out as f64) / (sr_in as f64);
    let chunk = 1024usize;
    let sinc_len = 256usize;
    let f_cutoff = 0.95;
    let oversampling = 256usize;
    let window = rubato::WindowFunction::BlackmanHarris2;
    let params = rubato::SincInterpolationParameters {
        sinc_len,
        f_cutoff,
        interpolation: rubato::SincInterpolationType::Linear,
        oversampling_factor: oversampling,
        window,
    };
    let mut resampler = rubato::SincFixedIn::<f32>::new(ratio, 2.0, params, chunk, 1)
        .map_err(|e| VoxCpmError::InvalidArg(format!("resampler init: {e}")))?;

    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < pcm.len() {
        let end = (pos + chunk).min(pcm.len());
        let mut block = vec![0f32; chunk];
        block[..(end - pos)].copy_from_slice(&pcm[pos..end]);
        pos = end;

        let y = resampler
            .process(&[block], None)
            .map_err(|e| VoxCpmError::InvalidArg(format!("resampler process: {e}")))?;
        out.extend_from_slice(&y[0]);
    }

    Ok(out)
}

fn build_inputs(
    tokenizer: &VoxCpmTokenizer,
    text: &str,
    prompt_wav: Option<&Path>,
    prompt_text: Option<&str>,
    patch_size: usize,
    feat_dim: usize,
    audio_vae: Option<&model::audiovae::AudioVae>,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor, Tensor, Tensor, usize)> {
    let full_text_buf;
    let full_text = match (prompt_wav, prompt_text) {
        (Some(_), Some(pt)) if !pt.is_empty() => {
            full_text_buf = format!("{pt}{text}");
            full_text_buf.as_str()
        }
        _ => text,
    };

    let mut text_ids = tokenizer.encode_ids(full_text)?;
    const AUDIO_START_TOKEN: u32 = 101;
    text_ids.push(AUDIO_START_TOKEN);
    let text_length = text_ids.len();

    let (audio_feat_tail, audio_patch_count) = if let Some(wav_path) = prompt_wav {
        let Some(audio_vae) = audio_vae else {
            return Err(VoxCpmError::InvalidArg(
                "prompt wav provided but audio_vae is not loaded".into(),
            ));
        };
        let (pcm, sr_in) = wav_to_mono_f32(wav_path)?;
        if pcm.is_empty() {
            (None, 0usize)
        } else {
            let pcm = resample_if_needed(&pcm, sr_in, audio_vae.cfg.sample_rate)?;

            // Left pad to patch boundary (end-aligned).
            let patch_len = patch_size * audio_vae.chunk_size();
            let rem = pcm.len() % patch_len;
            let pcm = if rem == 0 {
                pcm
            } else {
                let pad = patch_len - rem;
                let mut padded = vec![0f32; pad];
                padded.extend_from_slice(&pcm);
                padded
            };

            let t = pcm.len();
            let x = Tensor::from_vec(pcm, (1usize, t), device)?.to_dtype(DType::F32)?;
            let z = audio_vae.encode(&x, audio_vae.cfg.sample_rate)?; // [1, D, T_latent]
            let (_b, d, t_latent) = z.dims3()?;
            if d != feat_dim {
                return Err(VoxCpmError::InvalidArg(format!(
                    "AudioVAE latent_dim={d} != feat_dim={feat_dim}"
                )));
            }
            if t_latent % patch_size != 0 {
                return Err(VoxCpmError::InvalidArg(format!(
                    "prompt latent length {t_latent} is not divisible by patch_size={patch_size}"
                )));
            }
            let audio_len = t_latent / patch_size;
            let z = z.squeeze(0)?; // [D, T]
            let z = z.reshape((d, audio_len, patch_size))?;
            let z = z.transpose(0, 1)?.transpose(1, 2)?; // [audio_len, P, D]
            (Some(z), audio_len)
        }
    } else {
        (None, 0usize)
    };

    let total_len = text_length + audio_patch_count;
    let mut token_ids_i64 = Vec::with_capacity(total_len);
    token_ids_i64.extend(text_ids.into_iter().map(|v| v as i64));
    token_ids_i64.extend(std::iter::repeat(0i64).take(audio_patch_count));
    let text_token = Tensor::from_vec(token_ids_i64, (1usize, total_len), device)?;

    let mut text_mask_v = Vec::with_capacity(total_len);
    text_mask_v.extend(std::iter::repeat(1f32).take(text_length));
    text_mask_v.extend(std::iter::repeat(0f32).take(audio_patch_count));
    let mut audio_mask_v = Vec::with_capacity(total_len);
    audio_mask_v.extend(std::iter::repeat(0f32).take(text_length));
    audio_mask_v.extend(std::iter::repeat(1f32).take(audio_patch_count));

    let text_mask = Tensor::from_vec(text_mask_v, (1usize, total_len), device)?.to_dtype(dtype)?;
    let audio_mask =
        Tensor::from_vec(audio_mask_v, (1usize, total_len), device)?.to_dtype(dtype)?;

    // audio_feat: [1, T, P, D]
    let audio_feat = if let Some(tail) = audio_feat_tail {
        let pad = Tensor::zeros((text_length, patch_size, feat_dim), dtype, device)?;
        let feat = Tensor::cat(&[&pad, &tail.to_dtype(dtype)?], 0)?;
        feat.unsqueeze(0)?
    } else {
        Tensor::zeros((1usize, total_len, patch_size, feat_dim), dtype, device)?
    };

    Ok((
        text_token,
        text_mask,
        audio_feat,
        audio_mask,
        audio_patch_count,
    ))
}

struct Run {
    total_len: usize,
    // Prefill
    base_prefill_last_hidden: Tensor, // [1, H]
    res_prefill_last_hidden: Tensor,  // [1, H]
    base_prefill_kv: Vec<(Tensor, Tensor)>,
    res_prefill_kv: Vec<(Tensor, Tensor)>,
    // Decode1 (full forward on total_len+1)
    base_decode1_hidden: Tensor,                // [1, H]
    res_decode1_hidden: Tensor,                 // [1, H]
    base_decode1_kv_new: Vec<(Tensor, Tensor)>, // per-layer [1, kvh, hd]
    res_decode1_kv_new: Vec<(Tensor, Tensor)>,
}

fn model_vb(
    paths: &ModelPaths,
    dev: &Device,
    dtype: DType,
    weights_prefix: Option<&str>,
) -> Result<candle_nn::VarBuilder<'static>> {
    let mut vb = paths.model_var_builder(dtype, dev)?;
    if let Some(p) = weights_prefix {
        vb = vb.pp(p);
    }
    Ok(vb)
}

fn audiovae_vb(
    paths: &ModelPaths,
    dev: &Device,
    weights_prefix: Option<&str>,
) -> Result<candle_nn::VarBuilder<'static>> {
    let mut vb = paths
        .audiovae_var_builder(DType::F32, dev)?
        .ok_or_else(|| VoxCpmError::InvalidArg("missing audiovae weights".into()))?;
    if let Some(p) = weights_prefix {
        vb = vb.pp(p);
    }
    Ok(vb)
}

fn slice_kv_new(kv: &[(Tensor, Tensor)], pos: usize) -> Result<Vec<(Tensor, Tensor)>> {
    let mut out = Vec::with_capacity(kv.len());
    for (k, v) in kv.iter() {
        // k/v: [bs, kvh, seq, hd] where seq = pos+1
        let k_new = k.narrow(2, pos, 1)?.squeeze(2)?; // [bs, kvh, hd]
        let v_new = v.narrow(2, pos, 1)?.squeeze(2)?;
        out.push((k_new, v_new));
    }
    Ok(out)
}

fn run_one(
    spec: &str,
    paths: &ModelPaths,
    cfg: &VoxCpmConfig,
    tokenizer: &VoxCpmTokenizer,
    text: &str,
    prompt_wav: Option<&Path>,
    prompt_text: Option<&str>,
    weights_prefix: Option<&str>,
) -> Result<Run> {
    let dev = device_from_spec(spec)?;
    let dtype = DType::F32;

    let vb0 = model_vb(paths, &dev, dtype, weights_prefix)?;
    let vb_base = if vb0.contains_tensor("embed_tokens.weight") {
        vb0.clone()
    } else if vb0.contains_tensor("base_lm.embed_tokens.weight") {
        vb0.pp("base_lm")
    } else {
        return Err(VoxCpmError::InvalidArg(
            "cannot locate base_lm weights".into(),
        ));
    };

    let patch_size = cfg.patch_size().unwrap_or(2);
    let feat_dim = cfg.feat_dim().unwrap_or(64);

    // Optional: prompt wav -> requires AudioVAE.
    let audio_vae = if prompt_wav.is_some() {
        let vae_cfg = cfg.audiovae()?;
        let vb = audiovae_vb(paths, &dev, weights_prefix)?;
        Some(model::audiovae::AudioVae::new(vae_cfg, vb)?)
    } else {
        None
    };

    let (text_token, text_mask, audio_feat, audio_mask, _audio_patch_count) = build_inputs(
        tokenizer,
        text,
        prompt_wav,
        prompt_text,
        patch_size,
        feat_dim,
        audio_vae.as_ref(),
        &dev,
        dtype,
    )?;

    let total_len = text_token.dims2()?.1;
    let pos = arange_u32(total_len, &dev)?;

    // Base + residual models.
    let base_cfg = cfg.minicpm4()?;
    let base_lm = model::minicpm4::MiniCpmModel::new(base_cfg.clone(), vb_base)?;

    let mut res_cfg = base_cfg;
    res_cfg.num_hidden_layers = cfg.residual_lm_num_layers();
    res_cfg.vocab_size = 0;
    let residual_lm = model::minicpm4::MiniCpmModel::new(res_cfg, vb0.pp("residual_lm"))?;

    // Local encoder + projections + fsq.
    let enc_cfg = cfg.locenc_minicpm4()?;
    let feat_encoder = model::locenc::VoxCpmLocEnc::new(enc_cfg, feat_dim, vb0.pp("feat_encoder"))?;
    let enc_to_lm_proj = candle_nn::linear(
        cfg.encoder_config()?.hidden_dim,
        base_lm.cfg().hidden_size,
        vb0.pp("enc_to_lm_proj"),
    )?;
    let fsq_layer = model::fsq::ScalarQuantizationLayer::new(
        base_lm.cfg().hidden_size,
        base_lm.cfg().hidden_size,
        cfg.scalar_quantization_latent_dim(),
        cfg.scalar_quantization_scale(),
        vb0.pp("fsq_layer"),
    )?;

    // Embeddings.
    let scale_emb = if base_lm.cfg().use_mup {
        base_lm.cfg().scale_emb
    } else {
        1.0
    };
    let text_embed = (&base_lm.embed_tokens(&text_token)? * scale_emb)?;
    let feat_embed = feat_encoder.forward(&audio_feat)?; // [1, T, H_enc]
    let feat_embed = enc_to_lm_proj.forward(&feat_embed)?; // [1, T, H_lm]

    let text_m = text_mask.unsqueeze(2)?;
    let audio_m = audio_mask.unsqueeze(2)?;
    let combined_embed =
        (text_embed.broadcast_mul(&text_m)? + feat_embed.broadcast_mul(&audio_m)?)?;

    // Prefill base.
    let (mut enc_outputs, base_prefill_kv) =
        base_lm.forward_with_cache(&combined_embed, &pos, true)?;
    let enc_q = fsq_layer.forward(&enc_outputs)?;
    enc_outputs = (enc_q.broadcast_mul(&audio_m)? + enc_outputs.broadcast_mul(&text_m)?)?;
    let base_prefill_last_hidden = enc_outputs.narrow(1, total_len - 1, 1)?.squeeze(1)?;

    // Prefill residual.
    let residual_inp = (enc_outputs + feat_embed.broadcast_mul(&audio_m)?)?;
    let (residual_enc_outputs, res_prefill_kv) =
        residual_lm.forward_with_cache(&residual_inp, &pos, true)?;
    let res_prefill_last_hidden = residual_enc_outputs
        .narrow(1, total_len - 1, 1)?
        .squeeze(1)?;

    // Teacher-forced decode1: use last prefill patch as the next patch.
    let next_feat = audio_feat.narrow(1, total_len - 1, 1)?.squeeze(1)?; // [1, P, D]
    let pred_patch = next_feat.unsqueeze(1)?; // [1, 1, P, D]
    let curr_embed = feat_encoder.forward(&pred_patch)?; // [1, 1, H_enc]
    let curr_embed = enc_to_lm_proj.forward(&curr_embed)?; // [1, 1, H_lm]

    // Decode1 base: full forward on concatenated sequence.
    let full_base_inp = Tensor::cat(&[&combined_embed, &curr_embed], 1)?; // [1, total_len+1, H]
    let pos_full = arange_u32(total_len + 1, &dev)?;
    let (base_full_out, base_full_kv) =
        base_lm.forward_with_cache(&full_base_inp, &pos_full, true)?;
    let base_decode1_hidden = base_full_out.narrow(1, total_len, 1)?.squeeze(1)?;
    let base_decode1_hidden = fsq_layer.forward(&base_decode1_hidden)?;
    let base_decode1_kv_new = slice_kv_new(&base_full_kv, total_len)?;

    // Decode1 residual: append step input.
    let res_step_inp = (&base_decode1_hidden + &curr_embed.squeeze(1)?)?.unsqueeze(1)?; // [1,1,H]
    let full_res_inp = Tensor::cat(&[&residual_inp, &res_step_inp], 1)?;
    let (res_full_out, res_full_kv) =
        residual_lm.forward_with_cache(&full_res_inp, &pos_full, true)?;
    let res_decode1_hidden = res_full_out.narrow(1, total_len, 1)?.squeeze(1)?;
    let res_decode1_kv_new = slice_kv_new(&res_full_kv, total_len)?;

    Ok(Run {
        total_len,
        base_prefill_last_hidden,
        res_prefill_last_hidden,
        base_prefill_kv,
        res_prefill_kv,
        base_decode1_hidden,
        res_decode1_hidden,
        base_decode1_kv_new,
        res_decode1_kv_new,
    })
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = parse_args();
    let paths = ModelPaths::discover(&args.model_dir)?;
    let cfg = VoxCpmConfig::from_json_str(&std::fs::read_to_string(&paths.config_json)?)?;
    let tokenizer = VoxCpmTokenizer::from_tokenizer_json(&paths.tokenizer_json)?;

    let weights_prefix = args.weights_prefix.as_deref();
    let prompt_wav = args.prompt_wav.as_deref();
    let prompt_text = args.prompt_text.as_deref();

    let base = run_one(
        "cpu",
        &paths,
        &cfg,
        &tokenizer,
        &args.text,
        prompt_wav,
        prompt_text,
        weights_prefix,
    )?;
    eprintln!(
        "baseline=cpu total_len={} atol={} rtol={}",
        base.total_len, args.atol, args.rtol
    );

    for spec in args.devices.iter().skip(1) {
        eprintln!("compare cpu vs {spec}");
        let other = run_one(
            spec,
            &paths,
            &cfg,
            &tokenizer,
            &args.text,
            prompt_wav,
            prompt_text,
            weights_prefix,
        )?;
        if other.total_len != base.total_len {
            return Err(VoxCpmError::InvalidArg(
                "total_len mismatch across devices".into(),
            ));
        }

        report(
            "prefill base last_hidden",
            spec,
            &other.base_prefill_last_hidden,
            &base.base_prefill_last_hidden,
            args.atol,
            args.rtol,
        )?;
        report(
            "prefill residual last_hidden",
            spec,
            &other.res_prefill_last_hidden,
            &base.res_prefill_last_hidden,
            args.atol,
            args.rtol,
        )?;

        for (i, ((k, v), (k_ref, v_ref))) in other
            .base_prefill_kv
            .iter()
            .zip(base.base_prefill_kv.iter())
            .enumerate()
        {
            report(
                &format!("prefill base k layer={i}"),
                spec,
                k,
                k_ref,
                args.atol,
                args.rtol,
            )?;
            report(
                &format!("prefill base v layer={i}"),
                spec,
                v,
                v_ref,
                args.atol,
                args.rtol,
            )?;
        }
        for (i, ((k, v), (k_ref, v_ref))) in other
            .res_prefill_kv
            .iter()
            .zip(base.res_prefill_kv.iter())
            .enumerate()
        {
            report(
                &format!("prefill residual k layer={i}"),
                spec,
                k,
                k_ref,
                args.atol,
                args.rtol,
            )?;
            report(
                &format!("prefill residual v layer={i}"),
                spec,
                v,
                v_ref,
                args.atol,
                args.rtol,
            )?;
        }

        report(
            "decode1 base hidden",
            spec,
            &other.base_decode1_hidden,
            &base.base_decode1_hidden,
            args.atol,
            args.rtol,
        )?;
        report(
            "decode1 residual hidden",
            spec,
            &other.res_decode1_hidden,
            &base.res_decode1_hidden,
            args.atol,
            args.rtol,
        )?;

        for (i, ((k, v), (k_ref, v_ref))) in other
            .base_decode1_kv_new
            .iter()
            .zip(base.base_decode1_kv_new.iter())
            .enumerate()
        {
            report(
                &format!("decode1 base k_new layer={i}"),
                spec,
                k,
                k_ref,
                args.atol,
                args.rtol,
            )?;
            report(
                &format!("decode1 base v_new layer={i}"),
                spec,
                v,
                v_ref,
                args.atol,
                args.rtol,
            )?;
        }
        for (i, ((k, v), (k_ref, v_ref))) in other
            .res_decode1_kv_new
            .iter()
            .zip(base.res_decode1_kv_new.iter())
            .enumerate()
        {
            report(
                &format!("decode1 residual k_new layer={i}"),
                spec,
                k,
                k_ref,
                args.atol,
                args.rtol,
            )?;
            report(
                &format!("decode1 residual v_new layer={i}"),
                spec,
                v,
                v_ref,
                args.atol,
                args.rtol,
            )?;
        }
    }

    eprintln!("ok");
    Ok(())
}
