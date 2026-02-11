//! Internal KV-cache self-consistency checks (no Python fixtures).
//!
//! Run with:
//!   VOXCPM_MODEL_DIR=$PWD/models/VoxCPM1.5 cargo test -p voxcpm -- --ignored

use candle_core::{DType, Device, Tensor};
use voxcpm::{ModelPaths, Result, VoxCpmConfig, VoxCpmError};

fn env_path(key: &str) -> std::path::PathBuf {
    std::env::var_os(key)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| panic!("missing env var {key}"))
}

fn allclose(a: &Tensor, b: &Tensor, atol: f64, rtol: f64) -> Result<()> {
    let a = a.to_dtype(DType::F32)?.contiguous()?;
    let b = b.to_dtype(DType::F32)?.contiguous()?;
    if a.dims() != b.dims() {
        return Err(VoxCpmError::InvalidArg(format!(
            "shape mismatch: {:?} vs {:?}",
            a.dims(),
            b.dims()
        )));
    }
    let av = a.flatten_all()?.to_vec1::<f32>()?;
    let bv = b.flatten_all()?.to_vec1::<f32>()?;
    let mut max_abs = 0f64;
    for (x, y) in av.iter().zip(bv.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        let abs = (x - y).abs();
        if abs > max_abs {
            max_abs = abs;
        }
        if abs > atol + rtol * y.abs() {
            return Err(VoxCpmError::InvalidArg(format!(
                "allclose failed (atol={atol} rtol={rtol} max_abs={max_abs:.6e})"
            )));
        }
    }
    Ok(())
}

#[test]
#[ignore]
fn minicpm_base_lm_cache_self_consistency() -> Result<()> {
    let dev = Device::Cpu;
    let model_dir = env_path("VOXCPM_MODEL_DIR");
    let paths = ModelPaths::discover(&model_dir)?;
    let cfg = VoxCpmConfig::from_json_str(&std::fs::read_to_string(&paths.config_json)?)?;
    let lm_cfg = cfg.minicpm4()?;

    // Random embedding inputs (post-embed space).
    let seq = 8usize;
    let x_seq = Tensor::randn(0f32, 1f32, (1usize, seq, lm_cfg.hidden_size), &dev)?;
    let x_next = Tensor::randn(0f32, 1f32, (1usize, 1usize, lm_cfg.hidden_size), &dev)?;

    let vb = paths.model_var_builder(DType::F32, &dev)?.pp("base_lm");
    let mut model = voxcpm::model::minicpm4::MiniCpmModel::new(lm_cfg, vb)?;

    // Full forward on concatenated sequence.
    let pos_full: Vec<u32> = (0..(seq as u32 + 1)).collect();
    let pos_full = Tensor::from_vec(pos_full, (seq + 1,), &dev)?;
    let x_full = Tensor::cat(&[&x_seq, &x_next], 1)?;
    let (y_full, _kv_full) = model.forward_with_cache(&x_full, &pos_full, true)?;
    let y_full_last = y_full.narrow(1, seq, 1)?.squeeze(1)?; // [1, H]

    // Prefill caches + one cached step.
    let pos: Vec<u32> = (0..seq as u32).collect();
    let pos = Tensor::from_vec(pos, (seq,), &dev)?;
    let (_y_prefill, kv) = model.forward_with_cache(&x_seq, &pos, true)?;
    model.setup_cache(1, seq + 2)?;
    model.fill_caches(&kv)?;
    let y_step = model.forward_step_cached(&x_next)?.squeeze(1)?; // [1, H]

    // This should be extremely tight (same backend, same dtype).
    allclose(&y_step, &y_full_last, 1e-5, 1e-5)
}
