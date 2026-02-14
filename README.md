# Candle VoxCPM (Rust)

This repo re-implements VoxCPM inference in Rust on top of Candle (end-to-end TTS / voice cloning).

## Repository Layout

- `crates/voxcpm/`: Rust inference library (Candle). Public API: `VoxCpm::from_dir` / `VoxCpm::generate`.
- `crates/voxcpm_engine/`: Sidecar inference process (async IPC over stdin/stdout; in-memory audio bytes).
- `crates/voxcpm_engine_sdk/`: Rust SDK to spawn/manage the engine process (used by desktop).
- `crates/voxcpm_desktop/`: Desktop app (React + Vite frontend, Tauri v2 backend).
- `tools/`: helper scripts (e.g. convert PyTorch checkpoints to `.safetensors`).

## Rust Inference (CLI example)

Convert weights (produce `model.safetensors` / `audiovae.safetensors` for the Rust side):

```bash
python tools/convert_weights.py /path/to/model_dir
```

Run end-to-end inference (writes `./out.wav`):

```bash
cargo run -p voxcpm --example infer -- <model_dir> "<text>" [prompt_wav|none] [weights_prefix|none] [cpu|cuda:N]
```

CUDA requires building with feature `cuda`:

```bash
cargo run -p voxcpm --features cuda --example infer -- <model_dir> "<text>" none none cuda:0
```

## VoxCPM Desktop (Tauri)

The desktop app runs VoxCPM inference in an external sidecar binary named `engine`.
The Tauri backend spawns the sidecar once and reuses it across multiple `infer` calls.

### Prerequisites

- Rust toolchain
- Node.js + npm
- Tauri v2 system dependencies per OS (see official docs): https://v2.tauri.app/start/prerequisites/
- Tauri CLI (`cargo tauri`) installed (see official docs): https://v2.tauri.app/reference/cli/

Platform notes (practical defaults):

- Linux: you will need WebKitGTK development packages for Wry (see Tauri prereqs).
- macOS: install Xcode Command Line Tools (`xcode-select --install`). Metal backend is macOS-only.
- Windows: install Visual Studio Build Tools (Desktop development with C++). CUDA backend is commonly Windows/Linux only.

Note: the Tauri Rust crate lives in `crates/voxcpm_desktop/src-tauri/` and is excluded from the root Cargo workspace.

### Build The Engine Sidecar

Tauri v2 bundles sidecars using a per-target-triple filename:
`crates/voxcpm_desktop/src-tauri/binaries/engine-$TARGET_TRIPLE[.exe]`.

1) Find your host target triple:

```bash
rustc --print host-tuple
```

2) Build `engine` (pick one backend):

```bash
# CPU
cargo build -p voxcpm_engine --bin engine --release

# CUDA (Linux/Windows, requires NVIDIA driver)
cargo build -p voxcpm_engine --bin engine --release --features cuda

# Metal (macOS)
cargo build -p voxcpm_engine --bin engine --release --features metal
```

3) Copy/rename the binary into the Tauri sidecar folder:

```bash
# Example (Linux):
cp target/release/engine crates/voxcpm_desktop/src-tauri/binaries/engine-x86_64-unknown-linux-gnu
```

On Windows, use `engine.exe` and keep the `.exe` suffix in the final filename.

### Run In Dev Mode

From `crates/voxcpm_desktop/`:

```bash
npm install
cargo tauri dev
```

### Build Release Bundles

From `crates/voxcpm_desktop/` (make sure the sidecar exists first):

```bash
cargo tauri build
```

Cross-platform builds are typically done on each target OS in CI (Windows/macOS/Linux).
For official CI templates, see: https://v2.tauri.app/distribute/pipelines/github/

### Device Support (CPU/CUDA/Metal)

The UI/backend passes a string `deviceSpec` to the engine:

- CPU: `cpu`
- CUDA: `cuda:0`, `cuda:1`, ...
  - The engine must be built with `--features cuda`.
  - On CUDA loads, the engine will best-effort call `model.optimize()` (CUDA graph capture). If it fails, it automatically falls back to the normal (non-graph) path.
- Metal: `metal:0`
  - macOS only.
  - The engine must be built with `--features metal`.

### Model Download Notes

The engine downloads model files via Hugging Face HTTP APIs and supports:

- `HF_ENDPOINT` override
- `HF_TOKEN` for gated/private repos
