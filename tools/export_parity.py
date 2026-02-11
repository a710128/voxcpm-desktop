#!/usr/bin/env python3
"""Export local numeric parity fixtures (PyTorch -> safetensors).

This script is intentionally local-only: it writes fixtures under an output
directory (e.g. ./parity_out/) that should not be committed.

Primary goal: use the *original* VoxCPM PyTorch modules as reference (import and
run them), so the fixtures reflect upstream behavior as closely as possible.

Notes:
- Fixtures are exported on CPU + FP32 for determinism.
- MiniCPM attention uses PyTorch SDPA in upstream. By default we keep the
  upstream behavior. For debugging, `--attn_impl explicit` can monkeypatch
  attention to explicit matmul+mask+softmax (legacy).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch

try:
    from safetensors.torch import load_file, safe_open, save_file
except Exception as e:
    raise SystemExit(
        "Missing python package 'safetensors'. Install with: python3 -m pip install --user safetensors\n"
        f"Original error: {e}"
    )


def _add_voxcpm_src_to_syspath(voxcpm_src_dir: Path) -> None:
    pkg_dir = voxcpm_src_dir / "voxcpm"
    if not pkg_dir.is_dir():
        raise SystemExit(f"Invalid VoxCPM src dir: {voxcpm_src_dir} (missing voxcpm/)")
    sys.path.insert(0, str(voxcpm_src_dir))


def _set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Some builds may not support full determinism; continue.
        pass


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_case(
    out_dir: Path, case: str, tensors: Dict[str, torch.Tensor], meta: Dict[str, Any]
) -> None:
    case_dir = out_dir / case
    _mkdir(case_dir)
    io_path = case_dir / "io.safetensors"
    meta_path = case_dir / "meta.json"

    # Always save CPU tensors for portability.
    tensors_cpu = {
        k: v.detach().to(device="cpu").contiguous() for k, v in tensors.items()
    }
    save_file(tensors_cpu, str(io_path))
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def _load_prefix(st_path: Path, prefix: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(prefix):
                out[k[len(prefix) :]] = f.get_tensor(k)
    return out


def _load_key(st_path: Path, key: str) -> torch.Tensor:
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _patch_minicpm_attention_explicit() -> None:
    """Monkeypatch MiniCPM attention to avoid SDPA.

    This is a debugging aid only.
    """

    from voxcpm.modules.minicpm4 import model as minicpm4_model  # type: ignore

    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        # x: [B, kvh, T, Hd] -> [B, h, T, Hd]
        if n_rep == 1:
            return x
        return x.repeat_interleave(n_rep, dim=1)

    def _causal_mask(
        q_len: int, k_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        # Use the same trick as Rust: log(tril(ones)) -> {0, -inf}.
        allow = torch.tril(
            torch.ones((q_len, k_len), device=device, dtype=torch.float32)
        )
        m = torch.log(allow)
        return m.to(dtype)

    def _explicit_attn(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool
    ) -> torch.Tensor:
        # q,k,v: [B, H, Tq/Tk, Hd]
        hd = q.size(-1)
        scale = 1.0 / math.sqrt(float(hd))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if is_causal:
            q_len, k_len = scores.size(-2), scores.size(-1)
            m = _causal_mask(q_len, k_len, scores.device, scores.dtype)
            scores = scores + m.view(1, 1, q_len, k_len)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    def forward(self, hidden_states, position_emb, is_causal):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_emb
        query_states, key_states = minicpm4_model.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        past_key_value = (key_states, value_states)

        repeat = self.num_heads // self.num_key_value_heads
        k = _repeat_kv(key_states, repeat)
        v = _repeat_kv(value_states, repeat)
        attn_output = _explicit_attn(query_states, k, v, is_causal=is_causal)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output, past_key_value

    def forward_step(self, hidden_states, position_emb, position_id, kv_cache):
        # hidden_states: [B, hidden]
        bsz, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, 1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, 1, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, 1, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_emb
        query_states, key_states = minicpm4_model.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        key_cache, value_cache = kv_cache
        # kv_cache: [B, kvh, max_len, Hd]
        pos = int(position_id.view(-1)[0].item())
        key_cache[:, :, pos : pos + 1, :] = key_states
        value_cache[:, :, pos : pos + 1, :] = value_states

        # Slice filled prefix [0..pos].
        k_all = key_cache[:, :, : pos + 1, :]
        v_all = value_cache[:, :, : pos + 1, :]

        repeat = self.num_heads // self.num_key_value_heads
        k_all = _repeat_kv(k_all, repeat)
        v_all = _repeat_kv(v_all, repeat)

        # No explicit mask: k_all only contains past+current.
        out = _explicit_attn(query_states, k_all, v_all, is_causal=False)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .reshape(bsz, 1, self.num_heads * self.head_dim)
        )
        out = self.o_proj(out)
        return out[:, 0, :]

    minicpm4_model.MiniCPMAttention.forward = forward
    minicpm4_model.MiniCPMAttention.forward_step = forward_step


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./parity_out")
    ap.add_argument(
        "--cases",
        type=str,
        default="all",
        help="Comma-separated case ids or 'all'",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--attn_impl",
        type=str,
        default="original",
        choices=["original", "explicit"],
        help="MiniCPM attention implementation used for reference outputs",
    )
    ap.add_argument(
        "--voxcpm_src",
        type=str,
        default=None,
        help="Path to VoxCPM/src (defaults to ./VoxCPM/src in this repo)",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    _set_determinism(args.seed)

    repo_root = Path(__file__).resolve().parent.parent
    voxcpm_src = (
        Path(args.voxcpm_src) if args.voxcpm_src else (repo_root / "VoxCPM" / "src")
    )
    _add_voxcpm_src_to_syspath(voxcpm_src)

    # Now we can import voxcpm.modules.* safely.
    from voxcpm.modules.minicpm4 import MiniCPM4Config, MiniCPMModel  # type: ignore
    from voxcpm.modules.locenc import VoxCPMLocEnc  # type: ignore
    from voxcpm.modules.locdit import CfmConfig, UnifiedCFM, VoxCPMLocDiT  # type: ignore
    from voxcpm.modules.layers import ScalarQuantizationLayer  # type: ignore
    from voxcpm.modules.audiovae import AudioVAE, AudioVAEConfig  # type: ignore

    if args.attn_impl == "explicit":
        _patch_minicpm_attention_explicit()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    _mkdir(out_dir)

    cfg = json.loads((model_dir / "config.json").read_text())
    lm_cfg = MiniCPM4Config(**cfg["lm_config"])
    residual_cfg = MiniCPM4Config(
        **{**cfg["lm_config"], "num_hidden_layers": int(cfg["residual_lm_num_layers"])}
    )

    enc_overrides = cfg["encoder_config"]
    enc_cfg_dict = {**cfg["lm_config"]}
    enc_cfg_dict.update(
        {
            "hidden_size": int(enc_overrides["hidden_dim"]),
            "intermediate_size": int(enc_overrides["ffn_dim"]),
            "num_attention_heads": int(enc_overrides["num_heads"]),
            "num_hidden_layers": int(enc_overrides["num_layers"]),
            "vocab_size": 0,
        }
    )
    enc_kv = enc_overrides.get("kv_channels", None)
    if enc_kv is not None:
        enc_cfg_dict["kv_channels"] = int(enc_kv)
    else:
        enc_cfg_dict.pop("kv_channels", None)
    enc_cfg = MiniCPM4Config(**enc_cfg_dict)

    dit_overrides = cfg["dit_config"]
    dit_cfg_dict = {**cfg["lm_config"]}
    dit_cfg_dict.update(
        {
            "hidden_size": int(dit_overrides["hidden_dim"]),
            "intermediate_size": int(dit_overrides["ffn_dim"]),
            "num_attention_heads": int(dit_overrides["num_heads"]),
            "num_hidden_layers": int(dit_overrides["num_layers"]),
            "vocab_size": 0,
        }
    )
    dit_kv = dit_overrides.get("kv_channels", None)
    if dit_kv is not None:
        dit_cfg_dict["kv_channels"] = int(dit_kv)
    else:
        dit_cfg_dict.pop("kv_channels", None)
    dit_cfg = MiniCPM4Config(**dit_cfg_dict)

    patch_size = int(cfg["patch_size"])
    feat_dim = int(cfg["feat_dim"])
    sq_latent_dim = int(cfg["scalar_quantization_latent_dim"])
    sq_scale = int(cfg["scalar_quantization_scale"])

    model_st = model_dir / "model.safetensors"
    vae_st = model_dir / "audiovae.safetensors"

    cases_wanted = (
        [c.strip() for c in args.cases.split(",")] if args.cases != "all" else ["all"]
    )

    def want(case_id: str) -> bool:
        return "all" in cases_wanted or case_id in cases_wanted

    # ------------------------------------------------------------------
    # MiniCPM primitives
    # ------------------------------------------------------------------
    if want("minicpm.rmsnorm.l0"):
        from voxcpm.modules.minicpm4 import model as minicpm4_model  # type: ignore

        x = torch.randn((2, 3, lm_cfg.hidden_size), device="cpu", dtype=torch.float32)
        rms = minicpm4_model.MiniCPMRMSNorm(
            lm_cfg.hidden_size, eps=float(lm_cfg.rms_norm_eps)
        ).to(torch.float32)
        rms.load_state_dict(
            {"weight": _load_key(model_st, "base_lm.layers.0.input_layernorm.weight")},
            strict=True,
        )
        rms.eval()
        y = rms(x)
        _write_case(
            out_dir,
            "minicpm.rmsnorm.l0",
            {"input/x": x, "expected/y": y},
            {"atol": 1e-5, "rtol": 1e-5, "note": "base_lm.layers.0.input_layernorm"},
        )

    if want("minicpm.mlp.l0"):
        from voxcpm.modules.minicpm4 import model as minicpm4_model  # type: ignore

        x = torch.randn((2, 3, lm_cfg.hidden_size), device="cpu", dtype=torch.float32)
        mlp = minicpm4_model.MiniCPMMLP(lm_cfg).to(torch.float32)
        mlp.load_state_dict(
            _load_prefix(model_st, "base_lm.layers.0.mlp."), strict=True
        )
        mlp.eval()
        y = mlp(x)
        _write_case(
            out_dir,
            "minicpm.mlp.l0",
            {"input/x": x, "expected/y": y},
            {"atol": 2e-4, "rtol": 2e-4, "note": "base_lm.layers.0.mlp"},
        )

    if want("minicpm.attn.l0"):
        from voxcpm.modules.minicpm4 import model as minicpm4_model  # type: ignore

        seq = 16
        x = torch.randn((1, seq, lm_cfg.hidden_size), device="cpu", dtype=torch.float32)
        pos_ids = torch.arange(0, seq, dtype=torch.long, device="cpu")
        rope = minicpm4_model.MiniCPMLongRoPE(lm_cfg)
        cos, sin = rope(pos_ids)

        attn = minicpm4_model.MiniCPMAttention(lm_cfg, layer_idx=0).to(torch.float32)
        sd = _load_prefix(model_st, "base_lm.layers.0.self_attn.")
        attn.load_state_dict(sd, strict=True)
        attn.eval()
        y, (k_cache, v_cache) = attn(x, (cos, sin), is_causal=True)

        _write_case(
            out_dir,
            "minicpm.attn.l0",
            {
                "input/x": x,
                "input/position_ids": pos_ids,
                "expected/y": y,
                "expected/k_cache": k_cache,
                "expected/v_cache": v_cache,
            },
            {
                "atol": 3e-4,
                "rtol": 3e-4,
                "note": f"base_lm.layers.0.self_attn (attn_impl={args.attn_impl})",
            },
        )

    if want("minicpm.kproj.l0"):
        # Export k_proj intermediate tensors to localize parity issues.
        from voxcpm.modules.minicpm4 import model as minicpm4_model  # type: ignore

        seq = 16
        x = torch.randn((1, seq, lm_cfg.hidden_size), device="cpu", dtype=torch.float32)
        pos_ids = torch.arange(0, seq, dtype=torch.long, device="cpu")
        rope = minicpm4_model.MiniCPMLongRoPE(lm_cfg)
        cos, sin = rope(pos_ids)

        w_k = _load_key(model_st, "base_lm.layers.0.self_attn.k_proj.weight").to(
            torch.float32
        )
        k_lin = torch.matmul(x, w_k.t())

        head_dim = (
            (lm_cfg.hidden_size // lm_cfg.num_attention_heads)
            if lm_cfg.kv_channels is None
            else int(lm_cfg.kv_channels)
        )
        kvh = int(lm_cfg.num_key_value_heads)
        k_pre = k_lin.view(1, seq, kvh, head_dim).transpose(1, 2).contiguous()

        # Apply RoPE (matches apply_rotary_pos_emb behavior for k).
        q_dummy = torch.zeros(
            (1, lm_cfg.num_attention_heads, seq, head_dim), dtype=torch.float32
        )
        _, k_post = minicpm4_model.apply_rotary_pos_emb(q_dummy, k_pre, cos, sin)
        k_post = k_post.contiguous()

        _write_case(
            out_dir,
            "minicpm.kproj.l0",
            {
                "input/x": x,
                "input/position_ids": pos_ids,
                "expected/w_k": w_k,
                "expected/k_lin": k_lin,
                "expected/k_pre": k_pre,
                "expected/k_post": k_post,
            },
            {"atol": 3e-4, "rtol": 3e-4, "note": "base_lm.layers.0.self_attn.k_proj"},
        )

    if want("minicpm.rope"):
        from voxcpm.modules.minicpm4 import model as minicpm4_model  # type: ignore

        seq = 16
        pos_ids = torch.arange(0, seq, dtype=torch.long, device="cpu")
        rope = minicpm4_model.MiniCPMLongRoPE(lm_cfg)
        cos, sin = rope(pos_ids)
        # Match Rust's get_cos_sin: [bs=1, 1, seq, hd]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        _write_case(
            out_dir,
            "minicpm.rope",
            {"input/position_ids": pos_ids, "expected/cos": cos, "expected/sin": sin},
            {"atol": 1e-6, "rtol": 1e-6, "note": "MiniCPMLongRoPE cache slice"},
        )

    if want("minicpm.model.forward_step"):
        # Full MiniCPMModel forward_step with KV cache (patched attention).
        seq = 8
        x_seq = torch.randn(
            (1, seq, lm_cfg.hidden_size), device="cpu", dtype=torch.float32
        )
        model = MiniCPMModel(lm_cfg).to(torch.float32)
        model.load_state_dict(_load_prefix(model_st, "base_lm."), strict=True)
        model.eval()
        model.setup_cache(
            batch_size=1,
            max_length=seq + 2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        ys = []
        for i in range(seq):
            y_i = model.forward_step(
                x_seq[:, i, :], torch.tensor([i], dtype=torch.long)
            )
            ys.append(y_i)
        y = torch.stack(ys, dim=1)
        _write_case(
            out_dir,
            "minicpm.model.forward_step",
            {"input/x_seq": x_seq, "expected/y_seq": y},
            {"atol": 5e-4, "rtol": 5e-4, "note": "base_lm forward_step seq"},
        )

    if want("minicpm.cache_consistency"):
        # End-to-end cache check: prefill (full forward) -> fill static caches -> forward_step
        # should match the last token of a full forward on the concatenated sequence.
        seq = 8
        x_seq = torch.randn(
            (1, seq, lm_cfg.hidden_size), device="cpu", dtype=torch.float32
        )
        x_next = torch.randn(
            (1, 1, lm_cfg.hidden_size), device="cpu", dtype=torch.float32
        )

        model = MiniCPMModel(lm_cfg).to(torch.float32)
        model.load_state_dict(_load_prefix(model_st, "base_lm."), strict=True)
        model.eval()

        # Prefill without using the static cache.
        y_prefill, kv_tuple = model(inputs_embeds=x_seq, is_causal=True)

        # Fill static cache and run one incremental step.
        model.setup_cache(
            batch_size=1,
            max_length=seq + 2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        model.kv_cache.fill_caches(kv_tuple)
        pos = torch.tensor([model.kv_cache.step()], device="cpu", dtype=torch.long)
        y_step = model.forward_step(x_next[:, 0, :], pos)

        # Full forward on the concatenated sequence (reference for the incremental step).
        y_full, _ = model(
            inputs_embeds=torch.cat([x_seq, x_next], dim=1), is_causal=True
        )
        y_full_last = y_full[:, -1, :]

        _write_case(
            out_dir,
            "minicpm.cache_consistency",
            {
                "input/x_seq": x_seq,
                "input/x_next": x_next,
                "expected/y_prefill": y_prefill,
                "expected/y_step": y_step,
                "expected/y_full_last": y_full_last,
            },
            {
                "atol": 8e-4,
                "rtol": 8e-4,
                "note": f"MiniCPM cache consistency (attn_impl={args.attn_impl})",
                "seq": seq,
            },
        )

    # ------------------------------------------------------------------
    # FSQ layer
    # ------------------------------------------------------------------
    if want("fsq"):
        x = torch.randn((2, 5, lm_cfg.hidden_size), device="cpu", dtype=torch.float32)
        fsq = ScalarQuantizationLayer(
            in_dim=lm_cfg.hidden_size,
            out_dim=lm_cfg.hidden_size,
            latent_dim=sq_latent_dim,
            scale=sq_scale,
        ).to(torch.float32)
        fsq.load_state_dict(_load_prefix(model_st, "fsq_layer."), strict=True)
        fsq.eval()
        y = fsq(x)
        _write_case(
            out_dir,
            "fsq",
            {"input/x": x, "expected/y": y},
            {"atol": 1e-5, "rtol": 1e-5},
        )

    # ------------------------------------------------------------------
    # Local encoder / DiT / CFM
    # ------------------------------------------------------------------
    if want("locenc"):
        x = torch.randn((1, 2, patch_size, feat_dim), device="cpu", dtype=torch.float32)
        locenc = VoxCPMLocEnc(enc_cfg, input_dim=feat_dim).to(torch.float32)
        locenc.load_state_dict(_load_prefix(model_st, "feat_encoder."), strict=True)
        locenc.eval()
        y = locenc(x)
        _write_case(
            out_dir,
            "locenc",
            {"input/x": x, "expected/y": y},
            {"atol": 5e-4, "rtol": 5e-4},
        )

    if want("locdit"):
        b = 1
        prefix = 4
        x = torch.randn((b, feat_dim, patch_size), device="cpu", dtype=torch.float32)
        mu = torch.randn((b, dit_cfg.hidden_size), device="cpu", dtype=torch.float32)
        cond = torch.randn((b, feat_dim, prefix), device="cpu", dtype=torch.float32)
        t = torch.tensor([0.5], device="cpu", dtype=torch.float32)
        dt = torch.tensor([0.0], device="cpu", dtype=torch.float32)

        estimator = VoxCPMLocDiT(dit_cfg, in_channels=feat_dim).to(torch.float32)
        estimator.load_state_dict(
            _load_prefix(model_st, "feat_decoder.estimator."), strict=True
        )
        estimator.eval()
        y = estimator(x, mu, t, cond, dt)
        _write_case(
            out_dir,
            "locdit",
            {
                "input/x": x,
                "input/mu": mu,
                "input/t": t,
                "input/cond": cond,
                "input/dt": dt,
                "expected/y": y,
            },
            {"atol": 5e-4, "rtol": 5e-4},
        )

    if want("cfm.solve_euler"):
        b = 1
        prefix = 4
        n_steps = 10
        x0 = torch.randn((b, feat_dim, patch_size), device="cpu", dtype=torch.float32)
        mu = torch.randn((b, dit_cfg.hidden_size), device="cpu", dtype=torch.float32)
        cond = torch.randn((b, feat_dim, prefix), device="cpu", dtype=torch.float32)
        t_span = torch.linspace(
            1.0, 0.0, n_steps + 1, device="cpu", dtype=torch.float32
        )

        cfm_params = CfmConfig(**dit_overrides.get("cfm_config", {}))
        estimator = VoxCPMLocDiT(dit_cfg, in_channels=feat_dim).to(torch.float32)
        estimator.load_state_dict(
            _load_prefix(model_st, "feat_decoder.estimator."), strict=True
        )
        estimator.eval()

        cfm = UnifiedCFM(
            in_channels=feat_dim,
            cfm_params=cfm_params,
            estimator=estimator,
            mean_mode=False,
        ).to(torch.float32)
        cfm.eval()
        y = cfm.solve_euler(
            x=x0,
            t_span=t_span,
            mu=mu,
            cond=cond,
            cfg_value=float(cfm_params.inference_cfg_rate),
            use_cfg_zero_star=True,
        )
        _write_case(
            out_dir,
            "cfm.solve_euler",
            {
                "input/x0": x0,
                "input/t_span": t_span,
                "input/mu": mu,
                "input/cond": cond,
                "expected/y": y,
            },
            {
                "atol": 8e-4,
                "rtol": 8e-4,
                "cfg_value": float(cfm_params.inference_cfg_rate),
                "use_cfg_zero_star": True,
            },
        )

    # ------------------------------------------------------------------
    # AudioVAE decode
    # ------------------------------------------------------------------
    if want("audiovae.decode"):
        vae_cfg = AudioVAEConfig(**cfg["audio_vae_config"])
        vae = AudioVAE(config=vae_cfg).to(torch.float32)
        vae.load_state_dict(load_file(str(vae_st), device="cpu"), strict=True)
        vae.eval()

        # Keep small latent length to keep fixtures compact.
        z = torch.randn((1, vae_cfg.latent_dim, 8), device="cpu", dtype=torch.float32)
        audio = vae.decode(z)
        _write_case(
            out_dir,
            "audiovae.decode",
            {"input/z": z, "expected/audio": audio},
            {"atol": 2e-4, "rtol": 2e-4},
        )

    print(f"Wrote parity fixtures to: {out_dir}")


if __name__ == "__main__":
    main()
