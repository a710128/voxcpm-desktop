use std::path::PathBuf;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use voxcpm::model;
use voxcpm::{ModelPaths, Result, VoxCpmConfig, VoxCpmError};

fn usage() -> ! {
    eprintln!(
        "usage: parity_test_candle <model_dir> [--devices cpu,metal:0] [--cases all|rope,rmsnorm,mlp,attn,kproj,cache,step,fsq,locenc,locdit,cfm,audiovae] [--atol 3e-4] [--rtol 3e-4] [--bs 1] [--seq 8] [--prefix 4] [--patch 8] [--latent-t 8] [--weights-prefix none|PREFIX]\n\nExamples:\n  cargo run -p voxcpm --features metal --example parity_test_candle -- models/VoxCPM1.5 --devices cpu,metal:0\n  CANDLE_METAL_ENABLE_FAST_MATH=0 cargo run -p voxcpm --features metal --example parity_test_candle -- models/VoxCPM1.5 --devices cpu,metal:0 --cases attn,kproj\n\nNotes:\n- This is Candle->Candle parity only (no Python export fixtures).\n- Baseline is always CPU F32; targets are the non-cpu devices in --devices.\n- All comparisons are done on CPU in F32 for portability across backends." );
    std::process::exit(2);
}

#[derive(Debug)]
struct Args {
    model_dir: PathBuf,
    devices: Vec<String>,
    cases: Vec<String>,
    atol: f64,
    rtol: f64,
    bs: usize,
    seq: usize,
    prefix: usize,
    patch: usize,
    latent_t: usize,
    weights_prefix: Option<String>,
}

fn parse_args() -> Args {
    let mut it = std::env::args().skip(1);
    let model_dir: PathBuf = it.next().map(PathBuf::from).unwrap_or_else(|| usage());

    let mut devices = "cpu,metal:0".to_owned();
    let mut cases = "all".to_owned();
    let mut atol = 3e-4;
    let mut rtol = 3e-4;
    let mut bs = 1usize;
    let mut seq = 8usize;
    let mut prefix = 4usize;
    let mut patch = 8usize;
    let mut latent_t = 8usize;
    let mut weights_prefix: Option<String> = None;

    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--devices" => devices = it.next().unwrap_or_else(|| usage()),
            "--cases" => cases = it.next().unwrap_or_else(|| usage()),
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
            "--bs" => {
                bs = it
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--seq" => {
                seq = it
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--prefix" => {
                prefix = it
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--patch" => {
                patch = it
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--latent-t" => {
                latent_t = it
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--weights-prefix" => {
                let v = it.next().unwrap_or_else(|| usage());
                weights_prefix = if v == "none" { None } else { Some(v) };
            }
            _ => usage(),
        }
    }

    let devices: Vec<String> = devices
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_owned())
        .collect();
    if devices.is_empty() {
        usage();
    }

    let cases: Vec<String> = if cases == "all" {
        vec![
            "rope", "rmsnorm", "mlp", "attn", "kproj", "cache", "step", "fsq", "locenc", "locdit",
            "cfm", "audiovae",
        ]
        .into_iter()
        .map(|s| s.to_owned())
        .collect()
    } else {
        cases
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_owned())
            .collect()
    };

    Args {
        model_dir,
        devices,
        cases,
        atol,
        rtol,
        bs,
        seq,
        prefix,
        patch,
        latent_t,
        weights_prefix,
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
        // Alias for metal.
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
    // Match the parity test harness behavior: check shape before flattening.
    if a.dims() != b.dims() {
        return Err(VoxCpmError::InvalidArg(format!(
            "shape mismatch: {:?} vs {:?}",
            a.dims(),
            b.dims()
        )));
    }

    // Compare on CPU in f32 for portability across backends.
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

fn report_allclose(
    label: &str,
    spec: &str,
    a: &Tensor,
    b: &Tensor,
    atol: f64,
    rtol: f64,
) -> Result<()> {
    let s = diff_stats(a, b, atol, rtol)?;
    eprintln!(
        "  {label} {spec} n={} bad={} max_abs={:.6e} max_rel={:.6e} mean_abs={:.6e} rmse={:.6e}",
        s.n, s.bad, s.max_abs, s.max_rel, s.mean_abs, s.rmse
    );
    if s.bad != 0 {
        return Err(VoxCpmError::InvalidArg(format!(
            "allclose failed for {label} {spec} (bad={}/{}, atol={atol}, rtol={rtol}, max_abs={:.6e}, max_rel={:.6e}, rmse={:.6e})",
            s.bad, s.n, s.max_abs, s.max_rel, s.rmse
        )));
    }
    Ok(())
}

fn model_vb(
    paths: &ModelPaths,
    dtype: DType,
    dev: &Device,
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
    dtype: DType,
    dev: &Device,
    weights_prefix: Option<&str>,
) -> Result<candle_nn::VarBuilder<'static>> {
    let mut vb = paths
        .audiovae_var_builder(dtype, dev)?
        .ok_or_else(|| VoxCpmError::InvalidArg("missing audiovae.safetensors".into()))?;
    if let Some(p) = weights_prefix {
        vb = vb.pp(p);
    }
    Ok(vb)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = parse_args();
    let weights_prefix = args.weights_prefix.as_deref();

    let paths = ModelPaths::discover(&args.model_dir)?;
    let cfg = VoxCpmConfig::from_json_str(&std::fs::read_to_string(&paths.config_json)?)?;

    // Parse target devices (baseline is always CPU).
    let mut targets = Vec::new();
    for spec in args.devices.iter() {
        if spec.eq_ignore_ascii_case("cpu") {
            continue;
        }
        targets.push((spec.clone(), device_from_spec(spec)?));
    }
    if targets.is_empty() {
        eprintln!("no target devices specified; nothing to compare (try: --devices cpu,metal:0)");
        return Ok(());
    }

    eprintln!(
        "baseline=cpu dtype=f32 targets={:?} atol={} rtol={}",
        targets.iter().map(|(s, _)| s).collect::<Vec<_>>(),
        args.atol,
        args.rtol
    );

    for case in args.cases.iter() {
        eprintln!("case={case}");
        match case.as_str() {
            "rope" => case_rope(
                &paths,
                &cfg,
                &targets,
                weights_prefix,
                args.bs,
                args.seq,
                args.atol,
                args.rtol,
            )?,
            "rmsnorm" => case_rmsnorm_l0(
                &paths,
                &cfg,
                &targets,
                weights_prefix,
                args.bs,
                args.seq,
                args.atol,
                args.rtol,
            )?,
            "mlp" => case_mlp_l0(
                &paths,
                &cfg,
                &targets,
                weights_prefix,
                args.bs,
                args.seq,
                args.atol,
                args.rtol,
            )?,
            "attn" => case_attn_l0(
                &paths,
                &cfg,
                &targets,
                weights_prefix,
                args.bs,
                args.seq,
                args.atol,
                args.rtol,
            )?,
            "kproj" => case_kproj_l0(
                &paths,
                &cfg,
                &targets,
                weights_prefix,
                args.bs,
                args.seq,
                args.atol,
                args.rtol,
            )?,
            "cache" => case_cache_consistency(
                &paths,
                &cfg,
                &targets,
                weights_prefix,
                args.bs,
                args.seq,
                args.atol,
                args.rtol,
            )?,
            "step" => case_model_forward_step(
                &paths,
                &cfg,
                &targets,
                weights_prefix,
                args.bs,
                args.seq,
                args.atol,
                args.rtol,
            )?,
            "fsq" => case_fsq(
                &paths,
                &cfg,
                &targets,
                weights_prefix,
                args.bs,
                args.seq,
                args.atol,
                args.rtol,
            )?,
            "locenc" => case_locenc(
                &paths,
                &cfg,
                &targets,
                weights_prefix,
                args.bs,
                args.seq,
                args.prefix,
                args.atol,
                args.rtol,
            )?,
            "locdit" => case_locdit(
                &paths,
                &cfg,
                &targets,
                weights_prefix,
                args.bs,
                args.seq,
                args.prefix,
                args.atol,
                args.rtol,
            )?,
            "cfm" => case_cfm_solve_euler(
                &paths,
                &cfg,
                &targets,
                weights_prefix,
                args.bs,
                args.patch,
                args.prefix,
                args.atol,
                args.rtol,
            )?,
            "audiovae" => case_audiovae_decode(
                &paths,
                &cfg,
                &targets,
                weights_prefix,
                args.bs,
                args.latent_t,
                args.atol,
                args.rtol,
            )?,
            other => {
                return Err(VoxCpmError::InvalidArg(format!(
                    "unknown case {other:?} (use --cases all or a comma-separated list)"
                )))
            }
        }
    }

    eprintln!("ok");
    Ok(())
}

fn mk_pos(bs: usize, seq: usize, dev: &Device) -> Result<Tensor> {
    let mut pos = Vec::with_capacity(bs * seq);
    for _ in 0..bs {
        for i in 0..seq {
            pos.push(i as u32);
        }
    }
    Ok(Tensor::from_vec(pos, (bs, seq), dev)?)
}

fn case_rope(
    _paths: &ModelPaths,
    cfg: &VoxCpmConfig,
    targets: &[(String, Device)],
    _weights_prefix: Option<&str>,
    bs: usize,
    seq: usize,
    atol: f64,
    rtol: f64,
) -> Result<()> {
    let lm = cfg.minicpm4()?;
    let dev = Device::Cpu;
    let pos_cpu = mk_pos(bs, seq, &dev)?;
    let rope_cpu = model::minicpm4::RotaryLongRope::new(&lm, &dev)?;
    let (cos_cpu, sin_cpu) = rope_cpu.get_cos_sin(&pos_cpu)?;

    for (spec, dev2) in targets.iter() {
        let pos = pos_cpu.to_device(dev2)?;
        let rope = model::minicpm4::RotaryLongRope::new(&lm, dev2)?;
        let (cos, sin) = rope.get_cos_sin(&pos)?;
        report_allclose("rope cos", spec, &cos, &cos_cpu, atol, rtol)?;
        report_allclose("rope sin", spec, &sin, &sin_cpu, atol, rtol)?;
    }
    Ok(())
}

fn case_rmsnorm_l0(
    paths: &ModelPaths,
    cfg: &VoxCpmConfig,
    targets: &[(String, Device)],
    weights_prefix: Option<&str>,
    bs: usize,
    seq: usize,
    atol: f64,
    rtol: f64,
) -> Result<()> {
    let lm = cfg.minicpm4()?;
    let dev = Device::Cpu;
    let x_cpu = Tensor::randn(0f32, 1f32, (bs, seq, lm.hidden_size), &dev)?;
    let vb = model_vb(paths, DType::F32, &dev, weights_prefix)?;
    let rms = model::minicpm4::MiniCpmRmsNorm::new(
        lm.hidden_size,
        lm.rms_norm_eps,
        vb.pp("base_lm").pp("layers").pp("0").pp("input_layernorm"),
    )?;
    let y_cpu = rms.forward(&x_cpu)?;

    for (spec, dev2) in targets.iter() {
        let x = x_cpu.to_device(dev2)?;
        let vb = model_vb(paths, DType::F32, dev2, weights_prefix)?;
        let rms = model::minicpm4::MiniCpmRmsNorm::new(
            lm.hidden_size,
            lm.rms_norm_eps,
            vb.pp("base_lm").pp("layers").pp("0").pp("input_layernorm"),
        )?;
        let y = rms.forward(&x)?;
        report_allclose("rmsnorm", spec, &y, &y_cpu, atol, rtol)?;
    }
    Ok(())
}

fn case_mlp_l0(
    paths: &ModelPaths,
    cfg: &VoxCpmConfig,
    targets: &[(String, Device)],
    weights_prefix: Option<&str>,
    bs: usize,
    seq: usize,
    atol: f64,
    rtol: f64,
) -> Result<()> {
    let lm = cfg.minicpm4()?;
    let dev = Device::Cpu;
    let x_cpu = Tensor::randn(0f32, 1f32, (bs, seq, lm.hidden_size), &dev)?;
    let vb = model_vb(paths, DType::F32, &dev, weights_prefix)?;
    let mlp =
        model::minicpm4::MiniCpmMlp::new(&lm, vb.pp("base_lm").pp("layers").pp("0").pp("mlp"))?;
    let y_cpu = mlp.forward(&x_cpu)?;

    for (spec, dev2) in targets.iter() {
        let x = x_cpu.to_device(dev2)?;
        let vb = model_vb(paths, DType::F32, dev2, weights_prefix)?;
        let mlp =
            model::minicpm4::MiniCpmMlp::new(&lm, vb.pp("base_lm").pp("layers").pp("0").pp("mlp"))?;
        let y = mlp.forward(&x)?;
        report_allclose("mlp", spec, &y, &y_cpu, atol, rtol)?;
    }
    Ok(())
}

fn case_attn_l0(
    paths: &ModelPaths,
    cfg: &VoxCpmConfig,
    targets: &[(String, Device)],
    weights_prefix: Option<&str>,
    bs: usize,
    seq: usize,
    atol: f64,
    rtol: f64,
) -> Result<()> {
    let lm = cfg.minicpm4()?;
    let dev = Device::Cpu;
    let x_cpu = Tensor::randn(0f32, 1f32, (bs, seq, lm.hidden_size), &dev)?;
    let pos_cpu = mk_pos(bs, seq, &dev)?;
    let vb = model_vb(paths, DType::F32, &dev, weights_prefix)?;
    let rope = Arc::new(model::minicpm4::RotaryLongRope::new(&lm, vb.device())?);
    let attn = model::minicpm4::MiniCpmAttention::new(
        &lm,
        rope,
        vb.pp("base_lm").pp("layers").pp("0").pp("self_attn"),
    )?;
    let (y_cpu, (k_cpu, v_cpu)) = attn.forward_with_cache(&x_cpu, &pos_cpu, true)?;

    for (spec, dev2) in targets.iter() {
        let x = x_cpu.to_device(dev2)?;
        let pos = pos_cpu.to_device(dev2)?;
        let vb = model_vb(paths, DType::F32, dev2, weights_prefix)?;
        let rope = Arc::new(model::minicpm4::RotaryLongRope::new(&lm, vb.device())?);
        let attn = model::minicpm4::MiniCpmAttention::new(
            &lm,
            rope,
            vb.pp("base_lm").pp("layers").pp("0").pp("self_attn"),
        )?;
        let (y, (k, v)) = attn.forward_with_cache(&x, &pos, true)?;
        report_allclose("attn k_cache", spec, &k, &k_cpu, atol, rtol)?;
        report_allclose("attn v_cache", spec, &v, &v_cpu, atol, rtol)?;
        report_allclose("attn y", spec, &y, &y_cpu, atol, rtol)?;
    }
    Ok(())
}

fn case_kproj_l0(
    paths: &ModelPaths,
    cfg: &VoxCpmConfig,
    targets: &[(String, Device)],
    weights_prefix: Option<&str>,
    bs: usize,
    seq: usize,
    atol: f64,
    rtol: f64,
) -> Result<()> {
    let lm = cfg.minicpm4()?;
    let dev = Device::Cpu;
    let x_cpu = Tensor::randn(0f32, 1f32, (bs, seq, lm.hidden_size), &dev)?;
    let pos_cpu = mk_pos(bs, seq, &dev)?;
    let vb = model_vb(paths, DType::F32, &dev, weights_prefix)?;
    let rope = Arc::new(model::minicpm4::RotaryLongRope::new(&lm, vb.device())?);
    let attn = model::minicpm4::MiniCpmAttention::new(
        &lm,
        rope,
        vb.pp("base_lm").pp("layers").pp("0").pp("self_attn"),
    )?;
    let (k_lin_cpu, k_pre_cpu, k_post_cpu) = attn.debug_kproj_tensors(&x_cpu, &pos_cpu)?;

    for (spec, dev2) in targets.iter() {
        let x = x_cpu.to_device(dev2)?;
        let pos = pos_cpu.to_device(dev2)?;
        let vb = model_vb(paths, DType::F32, dev2, weights_prefix)?;
        let rope = Arc::new(model::minicpm4::RotaryLongRope::new(&lm, vb.device())?);
        let attn = model::minicpm4::MiniCpmAttention::new(
            &lm,
            rope,
            vb.pp("base_lm").pp("layers").pp("0").pp("self_attn"),
        )?;
        let (k_lin, k_pre, k_post) = attn.debug_kproj_tensors(&x, &pos)?;
        report_allclose("kproj k_lin", spec, &k_lin, &k_lin_cpu, atol, rtol)?;
        report_allclose("kproj k_pre", spec, &k_pre, &k_pre_cpu, atol, rtol)?;
        report_allclose("kproj k_post", spec, &k_post, &k_post_cpu, atol, rtol)?;
    }
    Ok(())
}

fn case_cache_consistency(
    paths: &ModelPaths,
    cfg: &VoxCpmConfig,
    targets: &[(String, Device)],
    weights_prefix: Option<&str>,
    bs: usize,
    seq: usize,
    atol: f64,
    rtol: f64,
) -> Result<()> {
    let lm = cfg.minicpm4()?;
    let dev = Device::Cpu;
    let x_seq_cpu = Tensor::randn(0f32, 1f32, (bs, seq, lm.hidden_size), &dev)?;
    let x_next_cpu = Tensor::randn(0f32, 1f32, (bs, 1usize, lm.hidden_size), &dev)?;

    let vb = model_vb(paths, DType::F32, &dev, weights_prefix)?;
    let mut m = model::minicpm4::MiniCpmModel::new(lm.clone(), vb.pp("base_lm"))?;
    let pos: Vec<u32> = (0..seq as u32).collect();
    let pos = Tensor::from_vec(pos, (seq,), &dev)?;
    let (_y_prefill, caches) = m.forward_with_cache(&x_seq_cpu, &pos, true)?;
    m.setup_cache(bs, seq + 2)?;
    m.fill_caches(&caches)?;
    let y_step_cpu = m.forward_step_cached(&x_next_cpu)?.squeeze(1)?;

    for (spec, dev2) in targets.iter() {
        let x_seq = x_seq_cpu.to_device(dev2)?;
        let x_next = x_next_cpu.to_device(dev2)?;

        let vb = model_vb(paths, DType::F32, dev2, weights_prefix)?;
        let mut m = model::minicpm4::MiniCpmModel::new(lm.clone(), vb.pp("base_lm"))?;
        let pos: Vec<u32> = (0..seq as u32).collect();
        let pos = Tensor::from_vec(pos, (seq,), dev2)?;
        let (_y_prefill, caches) = m.forward_with_cache(&x_seq, &pos, true)?;
        m.setup_cache(bs, seq + 2)?;
        m.fill_caches(&caches)?;
        let y_step = m.forward_step_cached(&x_next)?.squeeze(1)?;
        report_allclose("cache y_step", spec, &y_step, &y_step_cpu, atol, rtol)?;
    }
    Ok(())
}

fn case_model_forward_step(
    paths: &ModelPaths,
    cfg: &VoxCpmConfig,
    targets: &[(String, Device)],
    weights_prefix: Option<&str>,
    bs: usize,
    seq: usize,
    atol: f64,
    rtol: f64,
) -> Result<()> {
    let lm = cfg.minicpm4()?;
    let dev = Device::Cpu;
    let x_seq_cpu = Tensor::randn(0f32, 1f32, (bs, seq, lm.hidden_size), &dev)?;

    let vb = model_vb(paths, DType::F32, &dev, weights_prefix)?;
    let mut m = model::minicpm4::MiniCpmModel::new(lm.clone(), vb.pp("base_lm"))?;
    m.setup_cache(bs, seq + 2)?;
    let mut outs = Vec::with_capacity(seq);
    for i in 0..seq {
        let xi = x_seq_cpu.narrow(1, i, 1)?;
        let pos = Tensor::from_vec(vec![i as u32; bs], (bs,), &dev)?;
        outs.push(m.forward_step(&xi, &pos)?);
    }
    let out_refs: Vec<&Tensor> = outs.iter().collect();
    let y_cpu = Tensor::cat(&out_refs, 1)?;

    for (spec, dev2) in targets.iter() {
        let x_seq = x_seq_cpu.to_device(dev2)?;
        let vb = model_vb(paths, DType::F32, dev2, weights_prefix)?;
        let mut m = model::minicpm4::MiniCpmModel::new(lm.clone(), vb.pp("base_lm"))?;
        m.setup_cache(bs, seq + 2)?;
        let mut outs = Vec::with_capacity(seq);
        for i in 0..seq {
            let xi = x_seq.narrow(1, i, 1)?;
            let pos = Tensor::from_vec(vec![i as u32; bs], (bs,), dev2)?;
            outs.push(m.forward_step(&xi, &pos)?);
        }
        let out_refs: Vec<&Tensor> = outs.iter().collect();
        let y = Tensor::cat(&out_refs, 1)?;
        report_allclose("step y_seq", spec, &y, &y_cpu, atol, rtol)?;
    }
    Ok(())
}

fn case_fsq(
    paths: &ModelPaths,
    cfg: &VoxCpmConfig,
    targets: &[(String, Device)],
    weights_prefix: Option<&str>,
    bs: usize,
    seq: usize,
    atol: f64,
    rtol: f64,
) -> Result<()> {
    let lm = cfg.minicpm4()?;
    let dev = Device::Cpu;
    let x_cpu = Tensor::randn(0f32, 1f32, (bs, seq, lm.hidden_size), &dev)?;

    let vb = model_vb(paths, DType::F32, &dev, weights_prefix)?;
    let fsq = model::fsq::ScalarQuantizationLayer::new(
        lm.hidden_size,
        lm.hidden_size,
        cfg.scalar_quantization_latent_dim(),
        cfg.scalar_quantization_scale(),
        vb.pp("fsq_layer"),
    )?;
    let y_cpu = fsq.forward(&x_cpu)?;

    for (spec, dev2) in targets.iter() {
        let x = x_cpu.to_device(dev2)?;
        let vb = model_vb(paths, DType::F32, dev2, weights_prefix)?;
        let fsq = model::fsq::ScalarQuantizationLayer::new(
            lm.hidden_size,
            lm.hidden_size,
            cfg.scalar_quantization_latent_dim(),
            cfg.scalar_quantization_scale(),
            vb.pp("fsq_layer"),
        )?;
        let y = fsq.forward(&x)?;
        report_allclose("fsq", spec, &y, &y_cpu, atol, rtol)?;
    }
    Ok(())
}

fn case_locenc(
    paths: &ModelPaths,
    cfg: &VoxCpmConfig,
    targets: &[(String, Device)],
    weights_prefix: Option<&str>,
    bs: usize,
    t: usize,
    p: usize,
    atol: f64,
    rtol: f64,
) -> Result<()> {
    let enc_cfg = cfg.locenc_minicpm4()?;
    let feat_dim = cfg.feat_dim().unwrap_or(64);
    let dev = Device::Cpu;
    let x_cpu = Tensor::randn(0f32, 1f32, (bs, t, p, feat_dim), &dev)?;

    let vb = model_vb(paths, DType::F32, &dev, weights_prefix)?;
    let locenc =
        model::locenc::VoxCpmLocEnc::new(enc_cfg.clone(), feat_dim, vb.pp("feat_encoder"))?;
    let y_cpu = locenc.forward(&x_cpu)?;

    for (spec, dev2) in targets.iter() {
        let x = x_cpu.to_device(dev2)?;
        let vb = model_vb(paths, DType::F32, dev2, weights_prefix)?;
        let locenc =
            model::locenc::VoxCpmLocEnc::new(enc_cfg.clone(), feat_dim, vb.pp("feat_encoder"))?;
        let y = locenc.forward(&x)?;
        report_allclose("locenc", spec, &y, &y_cpu, atol, rtol)?;
    }
    Ok(())
}

fn case_locdit(
    paths: &ModelPaths,
    cfg: &VoxCpmConfig,
    targets: &[(String, Device)],
    weights_prefix: Option<&str>,
    n: usize,
    t_len: usize,
    prefix: usize,
    atol: f64,
    rtol: f64,
) -> Result<()> {
    let dit_cfg = cfg.locdit_minicpm4()?;
    let feat_dim = cfg.feat_dim().unwrap_or(64);
    let dev = Device::Cpu;

    let x_cpu = Tensor::randn(0f32, 1f32, (n, feat_dim, t_len), &dev)?;
    let mu_cpu = Tensor::randn(0f32, 1f32, (n, dit_cfg.hidden_size), &dev)?;
    let cond_cpu = Tensor::randn(0f32, 1f32, (n, feat_dim, prefix), &dev)?;
    let t_cpu = Tensor::from_vec(vec![0.5f32; n], (n,), &dev)?;
    let dt_cpu = Tensor::from_vec(vec![0.2f32; n], (n,), &dev)?;

    let vb = model_vb(paths, DType::F32, &dev, weights_prefix)?;
    let dit = model::locdit::VoxCpmLocDiT::new(
        dit_cfg.clone(),
        feat_dim,
        vb.pp("feat_decoder").pp("estimator"),
    )?;
    let y_cpu = dit.forward(&x_cpu, &mu_cpu, &t_cpu, &cond_cpu, &dt_cpu)?;

    for (spec, dev2) in targets.iter() {
        let x = x_cpu.to_device(dev2)?;
        let mu = mu_cpu.to_device(dev2)?;
        let cond = cond_cpu.to_device(dev2)?;
        let t = t_cpu.to_device(dev2)?;
        let dt = dt_cpu.to_device(dev2)?;

        let vb = model_vb(paths, DType::F32, dev2, weights_prefix)?;
        let dit = model::locdit::VoxCpmLocDiT::new(
            dit_cfg.clone(),
            feat_dim,
            vb.pp("feat_decoder").pp("estimator"),
        )?;
        let y = dit.forward(&x, &mu, &t, &cond, &dt)?;
        report_allclose("locdit", spec, &y, &y_cpu, atol, rtol)?;
    }
    Ok(())
}

fn case_cfm_solve_euler(
    paths: &ModelPaths,
    cfg: &VoxCpmConfig,
    targets: &[(String, Device)],
    weights_prefix: Option<&str>,
    bs: usize,
    patch: usize,
    prefix: usize,
    atol: f64,
    rtol: f64,
) -> Result<()> {
    let dit_cfg = cfg.locdit_minicpm4()?;
    let feat_dim = cfg.feat_dim().unwrap_or(64);
    let dev = Device::Cpu;

    let x0_cpu = Tensor::randn(0f32, 1f32, (bs, feat_dim, patch), &dev)?;
    let mu_cpu = Tensor::randn(0f32, 1f32, (bs, dit_cfg.hidden_size), &dev)?;
    let cond_cpu = Tensor::randn(0f32, 1f32, (bs, feat_dim, prefix), &dev)?;
    let cfg_value_cpu = Tensor::from_vec(vec![1.0f32], (1usize,), &dev)?;

    let n_steps = 4usize;
    let mut ts = Vec::with_capacity(n_steps + 1);
    for i in 0..=n_steps {
        ts.push(1.0f32 - (i as f32) / (n_steps as f32));
    }
    let t_span_cpu = Tensor::from_vec(ts, (n_steps + 1,), &dev)?;

    let vb = model_vb(paths, DType::F32, &dev, weights_prefix)?;
    let estimator = model::locdit::VoxCpmLocDiT::new(
        dit_cfg.clone(),
        feat_dim,
        vb.pp("feat_decoder").pp("estimator"),
    )?;
    let cfm = model::unified_cfm::UnifiedCfm::new(feat_dim, estimator, false);
    let y_cpu = cfm.solve_euler(
        &x0_cpu,
        &t_span_cpu,
        &mu_cpu,
        &cond_cpu,
        &cfg_value_cpu,
        true,
    )?;

    for (spec, dev2) in targets.iter() {
        let x0 = x0_cpu.to_device(dev2)?;
        let t_span = t_span_cpu.to_device(dev2)?;
        let mu = mu_cpu.to_device(dev2)?;
        let cond = cond_cpu.to_device(dev2)?;
        let cfg_value = cfg_value_cpu.to_device(dev2)?;

        let vb = model_vb(paths, DType::F32, dev2, weights_prefix)?;
        let estimator = model::locdit::VoxCpmLocDiT::new(
            dit_cfg.clone(),
            feat_dim,
            vb.pp("feat_decoder").pp("estimator"),
        )?;
        let cfm = model::unified_cfm::UnifiedCfm::new(feat_dim, estimator, false);
        let y = cfm.solve_euler(&x0, &t_span, &mu, &cond, &cfg_value, true)?;
        report_allclose("cfm solve_euler", spec, &y, &y_cpu, atol, rtol)?;
    }

    Ok(())
}

fn case_audiovae_decode(
    paths: &ModelPaths,
    cfg: &VoxCpmConfig,
    targets: &[(String, Device)],
    weights_prefix: Option<&str>,
    bs: usize,
    latent_t: usize,
    atol: f64,
    rtol: f64,
) -> Result<()> {
    let vae_cfg = cfg.audiovae()?;
    let dev = Device::Cpu;

    let z_cpu = Tensor::randn(0f32, 1f32, (bs, vae_cfg.latent_dim, latent_t), &dev)?;
    let vb = audiovae_vb(paths, DType::F32, &dev, weights_prefix)?;
    let vae = model::audiovae::AudioVae::new(vae_cfg.clone(), vb)?;
    let y_cpu = vae.decode(&z_cpu)?;

    for (spec, dev2) in targets.iter() {
        let z = z_cpu.to_device(dev2)?;
        let vb = audiovae_vb(paths, DType::F32, dev2, weights_prefix)?;
        let vae = model::audiovae::AudioVae::new(vae_cfg.clone(), vb)?;
        let y = vae.decode(&z)?;
        report_allclose("audiovae decode", spec, &y, &y_cpu, atol, rtol)?;
    }
    Ok(())
}
