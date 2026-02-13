# Project Overview

This repo re-implements VoxCPM inference in Rust on top of Candle (end-to-end TTS / voice cloning). 

## Repository Layout

- `crates/voxcpm/`: Rust inference library (Candle). Public API: `crates/voxcpm/src/lib.rs`
- `tools/`: helper scripts (e.g. convert PyTorch checkpoints to `.safetensors`)
- `models/`, `parity_out/`, `*.wav`: local-only weights/fixtures/audio outputs, ignored by default in `.gitignore`

## Common Commands

- Convert weights (produce `model.safetensors` / `audiovae.safetensors` for the Rust side)
  - `python tools/convert_weights.py /path/to/model_dir`
- Rust end-to-end inference example (writes `./out.wav`)
  - `cargo run -p voxcpm --example infer -- <model_dir> "<text>" [prompt_wav|none] [weights_prefix|none] [cpu|cuda:N]`
  - CUDA requires a feature flag: `cargo run -p voxcpm --features cuda --example infer -- ... cuda:0`
- Local numeric parity tests (fixtures not committed; tests are ignored by default)
  - `python3 tools/export_parity.py --model_dir <model_dir> --out_dir ./parity_out`
  - `VOXCPM_MODEL_DIR=<model_dir> VOXCPM_PARITY_DIR=./parity_out cargo test -p voxcpm -- --ignored`

## Key Entry Points

- Rust: `crates/voxcpm/src/lib.rs` (`VoxCpm::from_dir` / `VoxCpm::generate`)
- Python (reference): `VoxCPM/src/voxcpm/core.py`, `VoxCPM/src/voxcpm/model/voxcpm.py`

# Agent Rules

## MOST IMPORTANT RULES

- DO NOT modify AGENTS.md unless the user permits it by explicitly asking you to do so.
- DO NOT consider about the compatibility. This project is under development now, not being deployed yet.
- When you adding new feature or fixing a bug, please consider about how to minimize the impact on the existing code, which means modify codes as less as possible.
- When you want to ask user for making decisions, if there is a `question` tool or `ask_user` tool, prefer to use it instead of asking the user directly. If there is no suitable tool, ask the user directly.
- Before making any changes, write a TODO list to remind yourself to implement the features later, if there is a `todo` tool, prefer to use it instead of writing the TODO list manually.
- Before you start writing code, you need to inform the user of the solution you intend to adopt. Only after discussing and confirming with the user should you begin the work.


## Git Commit Message Style

The git commit message style should be composed of two parts: title and body. The title should be `<type>: <subject>` (such as `feat: add new feature` or `fix: fix bug`). The body should be the detailed description of the changes in list format.

It's important that DO NOT include any symbol like $ or ` in the title or body. This would cause the bash shell to interpret the title or body as a command or a code block.
