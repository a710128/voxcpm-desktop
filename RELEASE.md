# Release Guide (VoxCPM_Desktop)

This document describes the release/distribution plan for the Tauri desktop app.

Scope:
- OS: Windows / macOS / Linux
- Hardware variants:
  - CPU build (always available)
  - CUDA build (Windows/Linux only)
    - Windows: CUDA 12.4 runtime bundle
    - Linux: CUDA 12.2 runtime bundle
  - Metal support on macOS (via a Metal-enabled engine build)
- Update model: manual downloads via GitHub Releases (no in-app auto-updater requirement)
- Model/weights: not bundled in installers; downloaded on-demand at runtime

Repo notes (current architecture):
- The Tauri app bundles and launches an inference sidecar binary named `engine`.
- Sidecar bundling is configured via `crates/voxcpm_desktop/src-tauri/tauri.conf.json`.

## Goals

- One GUI installer per OS/arch per variant (CPU vs CUDA where applicable).
- CPU path must always work on a clean machine without CUDA.
- GPU acceleration is opt-in (via selecting a CUDA-capable deviceSpec when available).
- Clear, unambiguous artifact naming.
- Reproducible build/release steps and a minimal QA checklist.

## Release Matrix

### macOS

- Target: universal2, minimum macOS 13+
- GUI: 1 dmg asset
- Engine bundled inside GUI:
  - Single `engine` sidecar built with Metal support (CPU is always available as a runtime device)

Runtime behavior (current implementation):
- The app queries `engine.list_devices()` and picks a default device spec (prefer CUDA, else Metal, else CPU).
- The user can select `deviceSpec` (persisted in localStorage) and the app will reuse it if it is still present in `devices`.
- There is no automatic “try Metal then fallback to CPU on load failure” retry today; failures surface to UI.

### Windows

- Target: x86_64
- GUI: 2 NSIS installers (CPU and CUDA)
  - CPU build: CPU-only engine
  - CUDA build: CUDA-enabled engine + bundled CUDA runtime dependencies (CUDA 12.4)

Runtime behavior (current implementation):
- The app queries `engine.list_devices()` and picks a default device spec (prefer CUDA, else Metal, else CPU).
- CUDA build is GPU-only by policy: if NVIDIA driver is missing/too old, or required CUDA runtime libraries are missing, the engine will exit non-zero and the GUI will surface an error.
- CUDA build is allowed to be installed on non-NVIDIA machines, but it is expected to fail to start.
- There is no automatic retry/fallback (e.g. try CUDA then CPU) today.

### Linux

- Target: x86_64
- GUI: 2 AppImage assets (CPU and CUDA)
  - CPU build: CPU-only engine
  - CUDA build: CUDA-enabled engine + bundled CUDA runtime dependencies (CUDA 12.2)

Runtime behavior (current implementation):
- Same as Windows.

## Artifact Naming

App name (fixed): `VoxCPM_Desktop`

Version:
- Tag: `vX.Y.Z`
- Release assets use `X.Y.Z` (without the leading `v`).

### GUI assets

- Windows (CPU): `VoxCPM_Desktop_X.Y.Z_windows-x64_setup_cpu.exe`
- Windows (CUDA 12.4): `VoxCPM_Desktop_X.Y.Z_windows-x64_setup_cuda12_4.exe`
- macOS: `VoxCPM_Desktop_X.Y.Z_macos-universal.dmg`
- Linux (CPU): `VoxCPM_Desktop_X.Y.Z_linux-x64_cpu.AppImage`
- Linux (CUDA 12.2): `VoxCPM_Desktop_X.Y.Z_linux-x64_cuda12_2.AppImage`

### Integrity files

For every downloadable asset (GUI), publish:
- `*.sha256` (required)
- `*.sig` (recommended)

## Engine Sidecar

- The GUI bundles a single sidecar binary named `engine`.
- Device selection is done via `deviceSpec` strings returned by `engine.list_devices()`:
  - CPU: `cpu`
  - CUDA: `cuda:0`, `cuda:1`, ... (only present when the engine is built with CUDA support and CUDA driver initialization succeeds)
  - Metal: `metal:0` (only present when the engine is built with Metal support)
- The frontend default selection prefers CUDA, then Metal, then CPU, and persists the last selected `deviceSpec` if still available.
- There is no automatic backend retry/fallback on runtime model-load failure today; failures are surfaced to UI.

## CUDA Driver Requirement Policy

Minimum (as a product policy):
- Windows CUDA 12.4 build requires NVIDIA driver >= 525.
- Linux CUDA 12.2 build requires NVIDIA driver >= 525.

Clarification:
- CUDA builds are GPU-only by policy.
- If no NVIDIA driver is installed, or the driver is too old, the engine is expected to fail-fast (exit non-zero) during startup/probe.

Recommended (for stability):
- Newer drivers are strongly recommended, especially if the engine uses PTX or NVRTC at runtime.
- If CUDA fails due to driver/library incompatibility, CUDA devices may be unavailable and/or model load may fail; users can switch deviceSpec to `cpu`.

## CUDA Availability Notes (Windows/Linux)

- CUDA devices are only reported by the engine when:
  - the engine is built with CUDA support, and
  - CUDA driver initialization succeeds at runtime.
- If CUDA cannot be initialized, the engine is expected to fail-fast (exit non-zero).

## CUDA Runtime Bundling (Windows/Linux)

The CUDA installers/AppImages are self-contained and must include the CUDA runtime/toolkit user-mode libraries needed by the engine.

Important constraints (current architecture):
- The engine uses `cudarc` dynamic-loading. This means CUDA libraries are loaded at runtime via OS loader search paths (not via static link dependencies).
- Because libraries may only be loaded when certain code paths execute, we maintain an explicit per-OS runtime library list (the source of truth) and do a startup probe that attempts to load them all.

Policy:
- Missing/incorrect NVIDIA driver OR missing any required CUDA runtime library is a fatal error for CUDA builds (engine exits non-zero; GUI surfaces an error).

Bundling layout and loader behavior:
- Windows (CUDA 12.4): ship required `*.dll` next to `engine.exe` (or otherwise ensure the engine process resolves DLLs from the bundled directory first).
- Linux (CUDA 12.2): ship required `lib*.so*` inside the AppImage and inject `LD_LIBRARY_PATH` when spawning the sidecar so `dlopen()` resolves the bundled libraries.

Release requirements:
- Maintain a versioned CUDA runtime manifest per OS (file list + version + sha256) and use it to fetch NVIDIA official redistributables during CI builds.
- The engine startup probe must print a clear error message describing which driver/library check failed.

## Model/Weights Distribution

- Installers do not include model weights.
- On first run (or when needed), download models into a user cache directory.
- Current implementation uses ETag-based caching and temp-file downloads; hash validation and resumable downloads are not implemented yet.
- Keep model cache separate from application binaries.

## GitHub Releases Process

For every release `vX.Y.Z`:
- Create tag `vX.Y.Z`.
- Create a GitHub Release.
- Upload GUI assets for Windows/macOS/Linux.
- Upload corresponding `sha256` (and optionally `sig`) files for all assets.

Retention:
- Keep at least the last N stable releases (e.g. 5) for rollback and debugging.

## Minimal QA Checklist

Windows:
- No NVIDIA GPU: install and run CPU; model download works.
- NVIDIA GPU + driver meets policy: CPU build works; CUDA build lists `cuda:0` and CUDA inference works.
- CUDA build on non-NVIDIA machine: engine is expected to fail-fast and GUI should surface an error.
- CUDA build with missing CUDA runtime DLL: engine is expected to fail-fast and GUI should surface an error.

Linux:
- No NVIDIA GPU: AppImage runs; CPU inference + model download works.
- NVIDIA GPU + driver meets policy: CUDA build lists `cuda:0` and CUDA inference works.
- CUDA build with missing NVIDIA driver OR missing CUDA runtime .so: engine is expected to fail-fast and GUI should surface an error.

macOS:
- Metal path works on supported macOS.
- CPU mode still works via `deviceSpec=cpu`.
- App passes codesign/notarization for end-user install.
