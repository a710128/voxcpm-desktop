//! Local-only numeric parity tests (real weights, fixtures not committed).
//!
//! Generate fixtures with:
//!   python3 tools/export_parity.py --model_dir models/VoxCPM1.5 --out_dir ./parity_out
//!
//! Run with:
//!   VOXCPM_MODEL_DIR=$PWD/models/VoxCPM1.5 \
//!   VOXCPM_PARITY_DIR=$PWD/parity_out \
//!   cargo test -p voxcpm -- --ignored

use candle_core::{DType, Device, Tensor};
use safetensors::{tensor::Dtype as SdDtype, SafeTensors};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use voxcpm::model;
use voxcpm::{ModelPaths, Result, VoxCpmConfig, VoxCpmError};

fn env_path(key: &str) -> PathBuf {
    std::env::var_os(key)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("missing env var {key}"))
}

fn opt_env_str(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.is_empty())
}

fn read_json(path: &Path) -> serde_json::Value {
    let s =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&s).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn load_case_tensors(case_dir: &Path, device: &Device) -> HashMap<String, Tensor> {
    let io = case_dir.join("io.safetensors");
    let bytes = std::fs::read(&io).unwrap_or_else(|e| panic!("read {}: {e}", io.display()));
    let st = SafeTensors::deserialize(&bytes)
        .unwrap_or_else(|e| panic!("deserialize {}: {e}", io.display()));

    let mut out = HashMap::new();
    for name in st.names() {
        let view = st
            .tensor(name)
            .unwrap_or_else(|e| panic!("tensor {name} in {}: {e}", io.display()));
        let shape: Vec<usize> = view.shape().iter().map(|v| *v as usize).collect();
        let t = match view.dtype() {
            SdDtype::F32 => {
                let raw = view.data();
                if raw.len() % 4 != 0 {
                    panic!("invalid f32 byte length for {name}")
                }
                let mut v = Vec::<f32>::with_capacity(raw.len() / 4);
                for chunk in raw.chunks_exact(4) {
                    v.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Tensor::from_vec(v, shape.as_slice(), device)
                    .unwrap_or_else(|e| panic!("to tensor {name}: {e}"))
            }
            SdDtype::I64 => {
                let raw = view.data();
                if raw.len() % 8 != 0 {
                    panic!("invalid i64 byte length for {name}")
                }
                let mut v = Vec::<i64>::with_capacity(raw.len() / 8);
                for chunk in raw.chunks_exact(8) {
                    v.push(i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]));
                }
                Tensor::from_vec(v, shape.as_slice(), device)
                    .unwrap_or_else(|e| panic!("to tensor {name}: {e}"))
            }
            other => panic!("unsupported fixture dtype {other:?} for {name}"),
        };
        out.insert(name.to_owned(), t);
    }
    out
}

fn allclose(a: &Tensor, b: &Tensor, atol: f64, rtol: f64) -> Result<()> {
    // Make both contiguous so flatten order is deterministic.
    let a = a.to_dtype(DType::F32)?.contiguous()?;
    let b = b.to_dtype(DType::F32)?.contiguous()?;
    if a.dims() != b.dims() {
        return Err(VoxCpmError::InvalidArg(format!(
            "shape mismatch: {:?} vs {:?}",
            a.dims(),
            b.dims()
        )));
    }
    let af = a.flatten_all()?;
    let bf = b.flatten_all()?;
    let av = af.to_vec1::<f32>()?;
    let bv = bf.to_vec1::<f32>()?;

    let mut max_abs = 0f64;
    let mut max_rel = 0f64;
    let mut bad = 0usize;
    for (x, y) in av.iter().zip(bv.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        if !x.is_finite() || !y.is_finite() {
            return Err(VoxCpmError::InvalidArg(
                "non-finite value encountered".into(),
            ));
        }
        let abs = (x - y).abs();
        let rel = abs / (y.abs().max(1e-12));
        if abs > max_abs {
            max_abs = abs;
        }
        if rel > max_rel {
            max_rel = rel;
        }
        if abs > atol + rtol * y.abs() {
            bad += 1;
        }
    }
    if bad != 0 {
        return Err(VoxCpmError::InvalidArg(format!(
            "allclose failed (bad={bad}/{}, atol={atol}, rtol={rtol}, max_abs={max_abs:.6e}, max_rel={max_rel:.6e})",
            av.len()
        )));
    }
    Ok(())
}

fn case_meta(case_dir: &Path) -> serde_json::Value {
    read_json(&case_dir.join("meta.json"))
}

fn meta_atol_rtol(meta: &serde_json::Value) -> (f64, f64) {
    let atol = meta.get("atol").and_then(|v| v.as_f64()).unwrap_or(1e-4);
    let rtol = meta.get("rtol").and_then(|v| v.as_f64()).unwrap_or(1e-4);
    (atol, rtol)
}

#[test]
#[ignore]
fn parity_minicpm_cache_consistency() -> Result<()> {
    let dev = Device::Cpu;
    let model_dir = env_path("VOXCPM_MODEL_DIR");
    let parity_dir = env_path("VOXCPM_PARITY_DIR");

    let paths = ModelPaths::discover(&model_dir)?;
    let cfg = VoxCpmConfig::from_json_str(&std::fs::read_to_string(&paths.config_json)?)?;
    let lm = cfg.minicpm4()?;

    let case_dir = parity_dir.join("minicpm.cache_consistency");
    let meta = case_meta(&case_dir);
    let (atol, rtol) = meta_atol_rtol(&meta);
    let ts = load_case_tensors(&case_dir, &dev);
    let x_seq = ts.get("input/x_seq").unwrap();
    let x_next = ts.get("input/x_next").unwrap();
    let y_step_ref = ts.get("expected/y_step").unwrap();
    let y_full_last_ref = ts.get("expected/y_full_last").unwrap();

    // Load base_lm weights.
    let vb = model_vb(&paths, DType::F32, &dev)?;
    let mut model = model::minicpm4::MiniCpmModel::new(lm, vb.pp("base_lm"))?;

    let (_bs, seq, _h) = x_seq.dims3()?;
    let pos: Vec<u32> = (0..seq as u32).collect();
    let pos = Tensor::from_vec(pos, (seq,), &dev)?;

    // Prefill caches from a full forward.
    let (_y_prefill, caches) = model.forward_with_cache(x_seq, &pos, true)?;
    model.setup_cache(1, seq + 2)?;
    model.fill_caches(&caches)?;

    // Incremental step should match the Python reference's cached step output.
    let y_step = model.forward_step_cached(x_next)?.squeeze(1)?;
    allclose(&y_step, y_step_ref, atol, rtol)?;

    // Also sanity-check that the cached-step equals a full forward on concatenated sequence.
    allclose(&y_step, y_full_last_ref, atol, rtol)
}

fn model_vb(
    paths: &ModelPaths,
    dtype: DType,
    dev: &Device,
) -> Result<candle_nn::VarBuilder<'static>> {
    let mut vb = paths.model_var_builder(dtype, dev)?;
    if let Some(p) = opt_env_str("VOXCPM_WEIGHTS_PREFIX") {
        vb = vb.pp(p);
    }
    Ok(vb)
}

fn audiovae_vb(
    paths: &ModelPaths,
    dtype: DType,
    dev: &Device,
) -> Result<candle_nn::VarBuilder<'static>> {
    let mut vb = paths
        .audiovae_var_builder(dtype, dev)?
        .ok_or_else(|| VoxCpmError::InvalidArg("missing audiovae.safetensors".into()))?;
    if let Some(p) = opt_env_str("VOXCPM_WEIGHTS_PREFIX") {
        vb = vb.pp(p);
    }
    Ok(vb)
}

#[test]
#[ignore]
fn parity_minicpm_rmsnorm_l0() -> Result<()> {
    let dev = Device::Cpu;
    let model_dir = env_path("VOXCPM_MODEL_DIR");
    let parity_dir = env_path("VOXCPM_PARITY_DIR");

    let paths = ModelPaths::discover(&model_dir)?;
    let cfg = VoxCpmConfig::from_json_str(&std::fs::read_to_string(&paths.config_json)?)?;
    let lm = cfg.minicpm4()?;

    let case_dir = parity_dir.join("minicpm.rmsnorm.l0");
    let meta = case_meta(&case_dir);
    let (atol, rtol) = meta_atol_rtol(&meta);
    let ts = load_case_tensors(&case_dir, &dev);
    let x = ts.get("input/x").unwrap();
    let y_ref = ts.get("expected/y").unwrap();

    let vb = model_vb(&paths, DType::F32, &dev)?;
    let rms = model::minicpm4::MiniCpmRmsNorm::new(
        lm.hidden_size,
        lm.rms_norm_eps,
        vb.pp("base_lm").pp("layers").pp("0").pp("input_layernorm"),
    )?;
    let y = rms.forward(x)?;
    allclose(&y, y_ref, atol, rtol)
}

#[test]
#[ignore]
fn parity_minicpm_mlp_l0() -> Result<()> {
    let dev = Device::Cpu;
    let model_dir = env_path("VOXCPM_MODEL_DIR");
    let parity_dir = env_path("VOXCPM_PARITY_DIR");

    let paths = ModelPaths::discover(&model_dir)?;
    let cfg = VoxCpmConfig::from_json_str(&std::fs::read_to_string(&paths.config_json)?)?;
    let lm = cfg.minicpm4()?;

    let case_dir = parity_dir.join("minicpm.mlp.l0");
    let meta = case_meta(&case_dir);
    let (atol, rtol) = meta_atol_rtol(&meta);
    let ts = load_case_tensors(&case_dir, &dev);
    let x = ts.get("input/x").unwrap();
    let y_ref = ts.get("expected/y").unwrap();

    let vb = model_vb(&paths, DType::F32, &dev)?;
    let mlp =
        model::minicpm4::MiniCpmMlp::new(&lm, vb.pp("base_lm").pp("layers").pp("0").pp("mlp"))?;
    let y = mlp.forward(x)?;
    allclose(&y, y_ref, atol, rtol)
}

#[test]
#[ignore]
fn parity_minicpm_attn_l0() -> Result<()> {
    let dev = Device::Cpu;
    let model_dir = env_path("VOXCPM_MODEL_DIR");
    let parity_dir = env_path("VOXCPM_PARITY_DIR");

    let paths = ModelPaths::discover(&model_dir)?;
    let cfg = VoxCpmConfig::from_json_str(&std::fs::read_to_string(&paths.config_json)?)?;
    let lm = cfg.minicpm4()?;

    let case_dir = parity_dir.join("minicpm.attn.l0");
    let meta = case_meta(&case_dir);
    let (atol, rtol) = meta_atol_rtol(&meta);
    let ts = load_case_tensors(&case_dir, &dev);

    let x = ts.get("input/x").unwrap();
    let pos = ts.get("input/position_ids").unwrap();
    let y_ref = ts.get("expected/y").unwrap();
    let k_ref = ts.get("expected/k_cache").unwrap();
    let v_ref = ts.get("expected/v_cache").unwrap();

    let vb = model_vb(&paths, DType::F32, &dev)?;
    let rope = std::sync::Arc::new(model::minicpm4::RotaryLongRope::new(&lm, vb.device())?);
    let attn = model::minicpm4::MiniCpmAttention::new(
        &lm,
        rope,
        vb.pp("base_lm").pp("layers").pp("0").pp("self_attn"),
    )?;
    let (y, (k, v)) = attn.forward_with_cache(x, pos, true)?;
    // Check caches first to localize issues (RoPE/proj vs softmax/out).
    allclose(&k, k_ref, atol, rtol)?;
    allclose(&v, v_ref, atol, rtol)?;
    allclose(&y, y_ref, atol, rtol)
}

#[test]
#[ignore]
fn parity_minicpm_kproj_l0() -> Result<()> {
    let dev = Device::Cpu;
    let model_dir = env_path("VOXCPM_MODEL_DIR");
    let parity_dir = env_path("VOXCPM_PARITY_DIR");

    let paths = ModelPaths::discover(&model_dir)?;
    let cfg = VoxCpmConfig::from_json_str(&std::fs::read_to_string(&paths.config_json)?)?;
    let lm = cfg.minicpm4()?;

    let case_dir = parity_dir.join("minicpm.kproj.l0");
    let meta = case_meta(&case_dir);
    let (atol, rtol) = meta_atol_rtol(&meta);
    let ts = load_case_tensors(&case_dir, &dev);

    let x = ts.get("input/x").unwrap();
    let pos = ts.get("input/position_ids").unwrap();
    let k_lin_ref = ts.get("expected/k_lin").unwrap();
    let k_pre_ref = ts.get("expected/k_pre").unwrap();
    let k_post_ref = ts.get("expected/k_post").unwrap();

    // Use the production attention module's projection path.
    let vb = model_vb(&paths, DType::F32, &dev)?;
    let rope = std::sync::Arc::new(model::minicpm4::RotaryLongRope::new(&lm, vb.device())?);
    let attn = model::minicpm4::MiniCpmAttention::new(
        &lm,
        rope,
        vb.pp("base_lm").pp("layers").pp("0").pp("self_attn"),
    )?;
    let (k_lin, k_pre, k_post) = attn.debug_kproj_tensors(x, pos)?;
    allclose(&k_lin, k_lin_ref, atol, rtol)?;
    allclose(&k_pre, k_pre_ref, atol, rtol)?;
    allclose(&k_post, k_post_ref, atol, rtol)
}

#[test]
#[ignore]
fn parity_minicpm_rope() -> Result<()> {
    let dev = Device::Cpu;
    let model_dir = env_path("VOXCPM_MODEL_DIR");
    let parity_dir = env_path("VOXCPM_PARITY_DIR");

    let paths = ModelPaths::discover(&model_dir)?;
    let cfg = VoxCpmConfig::from_json_str(&std::fs::read_to_string(&paths.config_json)?)?;
    let lm = cfg.minicpm4()?;

    let case_dir = parity_dir.join("minicpm.rope");
    let meta = case_meta(&case_dir);
    let (atol, rtol) = meta_atol_rtol(&meta);
    let ts = load_case_tensors(&case_dir, &dev);
    let pos = ts.get("input/position_ids").unwrap();
    let cos_ref = ts.get("expected/cos").unwrap();
    let sin_ref = ts.get("expected/sin").unwrap();

    let rope = model::minicpm4::RotaryLongRope::new(&lm, &dev)?;
    let (cos, sin) = rope.get_cos_sin(pos)?;
    allclose(&cos, cos_ref, atol, rtol)?;
    allclose(&sin, sin_ref, atol, rtol)
}

#[test]
#[ignore]
fn parity_minicpm_model_forward_step() -> Result<()> {
    let dev = Device::Cpu;
    let model_dir = env_path("VOXCPM_MODEL_DIR");
    let parity_dir = env_path("VOXCPM_PARITY_DIR");

    let paths = ModelPaths::discover(&model_dir)?;
    let cfg = VoxCpmConfig::from_json_str(&std::fs::read_to_string(&paths.config_json)?)?;
    let lm = cfg.minicpm4()?;

    let case_dir = parity_dir.join("minicpm.model.forward_step");
    let meta = case_meta(&case_dir);
    let (atol, rtol) = meta_atol_rtol(&meta);
    let ts = load_case_tensors(&case_dir, &dev);

    let x_seq = ts.get("input/x_seq").unwrap();
    let y_ref = ts.get("expected/y_seq").unwrap();
    let (b, seq, _) = x_seq.dims3()?;

    let vb = model_vb(&paths, DType::F32, &dev)?;
    let mut m = model::minicpm4::MiniCpmModel::new(lm.clone(), vb.pp("base_lm"))?;
    m.setup_cache(b, seq + 2)?;
    let mut outs = Vec::with_capacity(seq);
    for i in 0..seq {
        let xi = x_seq.narrow(1, i, 1)?; // [B,1,H]
        let pos = Tensor::from_vec(vec![i as u32; b], (b,), &dev)?;
        let yi = m.forward_step(&xi, &pos)?; // [B,1,H]
        outs.push(yi);
    }
    let out_refs: Vec<&Tensor> = outs.iter().collect();
    let y = Tensor::cat(&out_refs, 1)?;
    allclose(&y, y_ref, atol, rtol)
}

#[test]
#[ignore]
fn parity_fsq() -> Result<()> {
    let dev = Device::Cpu;
    let model_dir = env_path("VOXCPM_MODEL_DIR");
    let parity_dir = env_path("VOXCPM_PARITY_DIR");

    let paths = ModelPaths::discover(&model_dir)?;
    let cfg = VoxCpmConfig::from_json_str(&std::fs::read_to_string(&paths.config_json)?)?;
    let lm = cfg.minicpm4()?;

    let case_dir = parity_dir.join("fsq");
    let meta = case_meta(&case_dir);
    let (atol, rtol) = meta_atol_rtol(&meta);
    let ts = load_case_tensors(&case_dir, &dev);
    let x = ts.get("input/x").unwrap();
    let y_ref = ts.get("expected/y").unwrap();

    let vb = model_vb(&paths, DType::F32, &dev)?;
    let fsq = model::fsq::ScalarQuantizationLayer::new(
        lm.hidden_size,
        lm.hidden_size,
        cfg.scalar_quantization_latent_dim(),
        cfg.scalar_quantization_scale(),
        vb.pp("fsq_layer"),
    )?;
    let y = fsq.forward(x)?;
    allclose(&y, y_ref, atol, rtol)
}

#[test]
#[ignore]
fn parity_locenc() -> Result<()> {
    let dev = Device::Cpu;
    let model_dir = env_path("VOXCPM_MODEL_DIR");
    let parity_dir = env_path("VOXCPM_PARITY_DIR");

    let paths = ModelPaths::discover(&model_dir)?;
    let cfg = VoxCpmConfig::from_json_str(&std::fs::read_to_string(&paths.config_json)?)?;
    let enc_cfg = cfg.locenc_minicpm4()?;
    let feat_dim = cfg.feat_dim().unwrap_or(64);

    let case_dir = parity_dir.join("locenc");
    let meta = case_meta(&case_dir);
    let (atol, rtol) = meta_atol_rtol(&meta);
    let ts = load_case_tensors(&case_dir, &dev);
    let x = ts.get("input/x").unwrap();
    let y_ref = ts.get("expected/y").unwrap();

    let vb = model_vb(&paths, DType::F32, &dev)?;
    let locenc = model::locenc::VoxCpmLocEnc::new(enc_cfg, feat_dim, vb.pp("feat_encoder"))?;
    let y = locenc.forward(x)?;
    allclose(&y, y_ref, atol, rtol)
}

#[test]
#[ignore]
fn parity_locdit() -> Result<()> {
    let dev = Device::Cpu;
    let model_dir = env_path("VOXCPM_MODEL_DIR");
    let parity_dir = env_path("VOXCPM_PARITY_DIR");

    let paths = ModelPaths::discover(&model_dir)?;
    let cfg = VoxCpmConfig::from_json_str(&std::fs::read_to_string(&paths.config_json)?)?;
    let dit_cfg = cfg.locdit_minicpm4()?;
    let feat_dim = cfg.feat_dim().unwrap_or(64);

    let case_dir = parity_dir.join("locdit");
    let meta = case_meta(&case_dir);
    let (atol, rtol) = meta_atol_rtol(&meta);
    let ts = load_case_tensors(&case_dir, &dev);

    let x = ts.get("input/x").unwrap();
    let mu = ts.get("input/mu").unwrap();
    let t = ts.get("input/t").unwrap();
    let cond = ts.get("input/cond").unwrap();
    let dt = ts.get("input/dt").unwrap();
    let y_ref = ts.get("expected/y").unwrap();

    let vb = model_vb(&paths, DType::F32, &dev)?;
    let dit =
        model::locdit::VoxCpmLocDiT::new(dit_cfg, feat_dim, vb.pp("feat_decoder").pp("estimator"))?;
    let y = dit.forward(x, mu, t, cond, dt)?;
    allclose(&y, y_ref, atol, rtol)
}

#[test]
#[ignore]
fn parity_cfm_solve_euler() -> Result<()> {
    let dev = Device::Cpu;
    let model_dir = env_path("VOXCPM_MODEL_DIR");
    let parity_dir = env_path("VOXCPM_PARITY_DIR");

    let paths = ModelPaths::discover(&model_dir)?;
    let cfg = VoxCpmConfig::from_json_str(&std::fs::read_to_string(&paths.config_json)?)?;
    let dit_cfg = cfg.locdit_minicpm4()?;
    let feat_dim = cfg.feat_dim().unwrap_or(64);

    let case_dir = parity_dir.join("cfm.solve_euler");
    let meta = case_meta(&case_dir);
    let (atol, rtol) = meta_atol_rtol(&meta);
    let cfg_value = meta
        .get("cfg_value")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let use_zero_star = meta
        .get("use_cfg_zero_star")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let ts = load_case_tensors(&case_dir, &dev);
    let x0 = ts.get("input/x0").unwrap();
    let t_span = ts.get("input/t_span").unwrap();
    let mu = ts.get("input/mu").unwrap();
    let cond = ts.get("input/cond").unwrap();
    let y_ref = ts.get("expected/y").unwrap();

    let t_span = t_span.to_dtype(DType::F32)?;

    let vb = model_vb(&paths, DType::F32, &dev)?;
    let estimator =
        model::locdit::VoxCpmLocDiT::new(dit_cfg, feat_dim, vb.pp("feat_decoder").pp("estimator"))?;
    let cfm = model::unified_cfm::UnifiedCfm::new(feat_dim, estimator, false);
    let cfg_value_t = Tensor::from_vec(vec![cfg_value as f32], (1usize,), &dev)?;
    let y = cfm.solve_euler(x0, &t_span, mu, cond, &cfg_value_t, use_zero_star)?;
    allclose(&y, y_ref, atol, rtol)
}

#[test]
#[ignore]
fn parity_audiovae_decode() -> Result<()> {
    let dev = Device::Cpu;
    let model_dir = env_path("VOXCPM_MODEL_DIR");
    let parity_dir = env_path("VOXCPM_PARITY_DIR");

    let paths = ModelPaths::discover(&model_dir)?;
    let cfg = VoxCpmConfig::from_json_str(&std::fs::read_to_string(&paths.config_json)?)?;
    let vae_cfg = cfg.audiovae()?;

    let case_dir = parity_dir.join("audiovae.decode");
    let meta = case_meta(&case_dir);
    let (atol, rtol) = meta_atol_rtol(&meta);
    let ts = load_case_tensors(&case_dir, &dev);
    let z = ts.get("input/z").unwrap();
    let y_ref = ts.get("expected/audio").unwrap();

    let vb = audiovae_vb(&paths, DType::F32, &dev)?;
    let vae = model::audiovae::AudioVae::new(vae_cfg, vb)?;
    let y = vae.decode(z)?;
    allclose(&y, y_ref, atol, rtol)
}
