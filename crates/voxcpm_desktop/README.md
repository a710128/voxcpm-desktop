# VoxCPM Desktop (Tauri)

Local desktop app for VoxCPM inference.

- Frontend: React + Vite
- Backend: Tauri v2 Rust commands
- Inference: external sidecar binary (engine) to isolate GPU stacks

## Development Notes

The Tauri app spawns a sidecar named `engine` and talks to it over framed binary IPC (stdin/stdout).

The `engine` process is spawned once at desktop startup and reused across multiple `infer` calls.
It shuts down when the desktop app exits.

### Build the engine sidecar

1) Build the engine:

   - CPU:
     - `cargo build -p voxcpm_engine --bin engine --release`
   - CUDA:
     - `cargo build -p voxcpm_engine --bin engine --release --features cuda`
   - Metal (macOS):
     - `cargo build -p voxcpm_engine --bin engine --release --features metal`

2) Copy/rename the binary to `src-tauri/binaries/engine-$TARGET_TRIPLE[.exe]`.
   - Find your triple: `rustc --print host-tuple`

Tauri v2 requires the `-$TARGET_TRIPLE` suffix for bundled sidecars.

For a more complete cross-platform build guide (Windows/macOS/Linux + CPU/CUDA/Metal), see the repo root `README.md`.
