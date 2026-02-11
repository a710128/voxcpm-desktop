#!/usr/bin/env python3
"""Convert VoxCPM PyTorch checkpoints to safetensors.

This repo's Rust side expects (eventually):
  - model.safetensors
  - audiovae.safetensors
  - (optional) lora_weights.safetensors

This script converts common PyTorch checkpoint formats into safetensors and
strips known wrapper prefixes (e.g. "module.", "_orig_mod.").

Usage:
  python tools/convert_weights.py /path/to/model_dir
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def _load_torch_state_dict(path: Path) -> Dict[str, Any]:
    import torch  # type: ignore

    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict):
        # Common wrappers.
        for k in ("state_dict", "model", "net", "module"):
            if k in obj and isinstance(obj[k], dict):
                maybe = obj[k]
                if all(isinstance(v, torch.Tensor) for v in maybe.values()):
                    obj = maybe
                    break
    if not isinstance(obj, dict):
        raise TypeError(f"unsupported checkpoint format: {path} (type={type(obj)})")
    if not obj:
        raise ValueError(f"empty checkpoint: {path}")
    if not all(isinstance(k, str) for k in obj.keys()):
        raise TypeError(f"checkpoint keys must be strings: {path}")

    sd: Dict[str, Any] = {}
    for k, v in obj.items():
        if not isinstance(v, torch.Tensor):
            continue
        sd[k] = v.detach().cpu().contiguous()
    if not sd:
        raise ValueError(f"no tensors found in checkpoint: {path}")
    return sd


def _strip_prefixes(sd: Dict[str, Any], prefixes: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        out[nk] = v
    return out


def _save_safetensors(sd: Dict[str, Any], out_path: Path) -> None:
    from safetensors.torch import save_file  # type: ignore

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    save_file(sd, str(tmp), metadata={"format": "pt"})
    os.replace(str(tmp), str(out_path))


def _convert_one(
    in_path: Path, out_path: Path, prefixes: Tuple[str, ...], overwrite: bool
) -> None:
    if out_path.exists() and not overwrite:
        print(f"skip: {out_path} already exists")
        return
    print(f"load: {in_path}")
    sd = _load_torch_state_dict(in_path)
    sd = _strip_prefixes(sd, prefixes)
    print(f"save: {out_path} (tensors={len(sd)})")
    _save_safetensors(sd, out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "model_dir", type=Path, help="Directory containing VoxCPM checkpoints"
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output .safetensors files",
    )
    ap.add_argument(
        "--strip-prefix",
        action="append",
        default=["_orig_mod.", "module."],
        help="Prefix to strip from checkpoint keys (repeatable)",
    )
    args = ap.parse_args()

    model_dir: Path = args.model_dir
    if not model_dir.is_dir():
        raise SystemExit(f"not a directory: {model_dir}")

    prefixes = tuple(args.strip_prefix)

    # Main model.
    out_model = model_dir / "model.safetensors"
    in_model_candidates = [
        model_dir / "pytorch_model.bin",
        model_dir / "model.pth",
        model_dir / "model.pt",
    ]
    in_model = next((p for p in in_model_candidates if p.is_file()), None)
    if in_model is None:
        if out_model.exists():
            print(f"ok: {out_model} already exists")
        else:
            print(
                "warn: no main model checkpoint found (pytorch_model.bin/model.pth/model.pt)"
            )
    else:
        _convert_one(in_model, out_model, prefixes, args.overwrite)

    # AudioVAE.
    out_vae = model_dir / "audiovae.safetensors"
    in_vae_candidates = [
        model_dir / "audiovae.pth",
        model_dir / "audiovae.pt",
    ]
    in_vae = next((p for p in in_vae_candidates if p.is_file()), None)
    if in_vae is None:
        if out_vae.exists():
            print(f"ok: {out_vae} already exists")
        else:
            print("note: no audiovae checkpoint found (audiovae.pth/audiovae.pt)")
    else:
        _convert_one(in_vae, out_vae, prefixes, args.overwrite)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
