use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");
    println!("cargo:rerun-if-env-changed=CANDLE_CUDA_ARCH_LIST");
    println!("cargo:rerun-if-env-changed=CANDLE_CUDA_PTX_ARCH");
    // Used by bindgen_cuda; set this in CI to avoid calling nvidia-smi.
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

    // Keep Rust-side and CUDA-side consistent: this fork intentionally disables MoE kernels.
    // We use this cfg to provide stub symbols for candle-nn so Windows/MSVC linking works.
    println!("cargo:rustc-cfg=candle_disable_moe");

    // Build for PTX
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-DCANDLE_DISABLE_MOE=1")
        .arg("-O3")
        // Only build PTX for the modules we actually embed.
        .kernel_paths(vec![
            "src/affine.cu",
            "src/binary.cu",
            "src/cast.cu",
            "src/conv.cu",
            "src/fill.cu",
            "src/indexing.cu",
            "src/quantized.cu",
            "src/reduce.cu",
            "src/sort.cu",
            "src/ternary.cu",
            "src/unary.cu",
        ]);
    let bindings = builder.build_ptx().unwrap();
    bindings.write(&ptx_path).unwrap();

    // MOE kernels are disabled in this fork, keep PTX generation simple.

    // Build multi-arch fatbins for the main kernel modules.
    // Runtime behavior: load fatbin first (SASS if matching), fallback to PTX.
    //
    // Notes:
    // - We intentionally do not probe GPUs here. Users/CI provide arch list via env.
    // - Default arch set targets cc >= 6.1.
    let arch_list =
        env::var("CANDLE_CUDA_ARCH_LIST").unwrap_or_else(|_| "61,70,75,80,86,89,90".to_string());
    let ptx_arch = env::var("CANDLE_CUDA_PTX_ARCH").unwrap_or_else(|_| "61".to_string());
    let arch_list = parse_arch_list(&arch_list);
    let ptx_arch = parse_arch_token(&ptx_arch);

    build_fatbin(&out_dir, "affine", "src/affine.cu", &arch_list, &ptx_arch);
    build_fatbin(&out_dir, "binary", "src/binary.cu", &arch_list, &ptx_arch);
    build_fatbin(&out_dir, "cast", "src/cast.cu", &arch_list, &ptx_arch);
    build_fatbin(&out_dir, "conv", "src/conv.cu", &arch_list, &ptx_arch);
    build_fatbin(&out_dir, "fill", "src/fill.cu", &arch_list, &ptx_arch);
    build_fatbin(
        &out_dir,
        "indexing",
        "src/indexing.cu",
        &arch_list,
        &ptx_arch,
    );
    build_fatbin(
        &out_dir,
        "quantized",
        "src/quantized.cu",
        &arch_list,
        &ptx_arch,
    );
    build_fatbin(&out_dir, "reduce", "src/reduce.cu", &arch_list, &ptx_arch);
    build_fatbin(&out_dir, "sort", "src/sort.cu", &arch_list, &ptx_arch);
    build_fatbin(&out_dir, "ternary", "src/ternary.cu", &arch_list, &ptx_arch);
    build_fatbin(&out_dir, "unary", "src/unary.cu", &arch_list, &ptx_arch);

    // NOTE: libmoe kernels intentionally not built/linked.
}

fn parse_arch_list(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    for token in s.split(',') {
        let t = token.trim();
        if t.is_empty() {
            continue;
        }
        out.push(parse_arch_token(t));
    }
    if out.is_empty() {
        panic!("CANDLE_CUDA_ARCH_LIST is empty");
    }
    out
}

fn parse_arch_token(token: &str) -> String {
    // Accept forms: "61", "75", "90", "90a".
    let t = token.trim().to_ascii_lowercase();
    if t.ends_with('a') {
        let n = &t[..t.len() - 1];
        if n.parse::<u32>().is_err() {
            panic!("invalid arch token: {token}");
        }
        return format!("{n}a");
    }
    if t.parse::<u32>().is_err() {
        panic!("invalid arch token: {token}");
    }
    t
}

fn build_fatbin(out_dir: &PathBuf, name: &str, cu: &str, sm_list: &[String], ptx_arch: &str) {
    let out = out_dir.join(format!("{name}.fatbin"));

    // nvcc --fatbin produces an image loadable via cuModuleLoadData.
    // Include multiple SASS images + one PTX image for forward compatibility.
    let mut cmd = Command::new("nvcc");
    cmd.arg("--fatbin")
        .arg(cu)
        .arg("-o")
        .arg(&out)
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-DCANDLE_DISABLE_MOE=1")
        .arg("-O3")
        .arg("-Isrc");

    // SASS targets.
    for sm in sm_list {
        let compute = sm.trim_end_matches('a');
        cmd.arg("-gencode")
            .arg(format!("arch=compute_{compute},code=sm_{sm}"));
    }
    // PTX fallback target.
    cmd.arg("-gencode")
        .arg(format!("arch=compute_{ptx_arch},code=compute_{ptx_arch}"));

    // Keep Linux builds usable for linking into shared libs.
    // This mirrors the existing build.rs behavior for the MOE static lib.
    let target = env::var("TARGET").unwrap_or_default();
    if !target.contains("msvc") {
        cmd.arg("-Xcompiler").arg("-fPIC");
    } else {
        cmd.arg("-D_USE_MATH_DEFINES");
    }

    // Ensure stable error output when nvcc isn't available.
    let status = cmd.status().unwrap_or_else(|e| {
        panic!("failed to spawn nvcc for {name}: {e}");
    });
    if !status.success() {
        panic!("nvcc fatbin build failed for {name}");
    }
}
