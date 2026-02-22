# Release Guide (VoxCPM_Desktop)

This document describes the release/distribution plan for the Tauri desktop app.

Scope:
- OS: Windows / macOS / Linux
- Hardware variants:
  - CPU build (always available)
  - CUDA build (Windows/Linux only, CUDA 12.2)
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
  - CUDA build: CUDA-enabled engine + bundled CUDA runtime dependencies

Runtime behavior (current implementation):
- The app queries `engine.list_devices()` and picks a default device spec (prefer CUDA, else Metal, else CPU).
- If the CUDA build cannot initialize CUDA driver APIs at runtime, it simply won't report `cuda:N` devices and the app will run on CPU.
- CUDA build must still start and run in CPU mode on machines without an NVIDIA GPU/driver.
- There is no automatic “try CUDA then fallback to CPU on load failure” retry today; failures surface to UI.

### Linux

- Target: x86_64
- GUI: 2 AppImage assets (CPU and CUDA)
  - CPU build: CPU-only engine
  - CUDA build: CUDA-enabled engine + bundled CUDA runtime dependencies

Runtime behavior (current implementation):
- Same as Windows.

## Artifact Naming

App name (fixed): `VoxCPM_Desktop`

Version:
- Tag: `vX.Y.Z`
- Release assets use `X.Y.Z` (without the leading `v`).

### GUI assets

- Windows (CPU): `VoxCPM_Desktop_X.Y.Z_windows-x64_setup_cpu.exe`
- Windows (CUDA 12.2): `VoxCPM_Desktop_X.Y.Z_windows-x64_setup_cuda12_2.exe`
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
- Windows/Linux CUDA builds require NVIDIA driver >= 525.

Clarification:
- The driver requirement only applies to using `cuda:N` devices.
- If no NVIDIA driver is installed, the CUDA build should still run in CPU mode (and `list_devices()` should not include `cuda:N`).

Recommended (for stability):
- Newer drivers are strongly recommended, especially if the engine uses PTX or NVRTC at runtime.
- If CUDA fails due to driver/library incompatibility, CUDA devices may be unavailable and/or model load may fail; users can switch deviceSpec to `cpu`.

## CUDA Availability Notes (Windows/Linux)

- CUDA devices are only reported by the engine when:
  - the engine is built with CUDA support, and
  - CUDA driver initialization succeeds at runtime.
- If CUDA cannot be initialized, `list_devices()` will not include `cuda:N`, and the app will run on CPU.

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
- CUDA build on non-NVIDIA machine: should still start and run CPU (and not list `cuda:N`).

Linux:
- No NVIDIA GPU: AppImage runs; CPU inference + model download works.
- NVIDIA GPU + driver meets policy: CUDA build lists `cuda:0` and CUDA inference works.

macOS:
- Metal path works on supported macOS.
- CPU mode still works via `deviceSpec=cpu`.
- App passes codesign/notarization for end-user install.
