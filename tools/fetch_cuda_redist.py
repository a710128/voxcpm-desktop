#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile


CUDA_REDIST_BASE_URL = "https://developer.download.nvidia.com/compute/cuda/redist/"


def _fetch_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _version_tuple(label: str):
    # label like 12.4.1
    return tuple(int(x) for x in label.split("."))


def _latest_redistrib_label(cuda_minor: str) -> str:
    # Parse the directory index for redistrib_*.json and pick the latest patch.
    # This keeps the policy "latest patch" while still using the official manifest.
    html = _fetch_bytes(CUDA_REDIST_BASE_URL).decode("utf-8", errors="replace")
    labels = re.findall(r"redistrib_(\d+\.\d+\.\d+)\.json", html)
    labels = [l for l in labels if l.startswith(cuda_minor + ".")]
    if not labels:
        raise RuntimeError(f"no redistrib json found for cuda minor {cuda_minor}")
    return max(labels, key=_version_tuple)


def _download_to(url: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(dst_path, "wb") as f:
        shutil.copyfileobj(resp, f)


def _copy_matching_from_dir(extracted_root: str, out_dir: str, *, exts, must_contain) -> int:
    copied = 0
    for root, _dirs, files in os.walk(extracted_root):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), extracted_root)
            rel_norm = rel.replace("\\", "/")
            if must_contain and must_contain not in rel_norm:
                continue
            if not any(name.lower().endswith(ext) for ext in exts):
                continue
            src = os.path.join(root, name)
            dst = os.path.join(out_dir, name)
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    return copied


def _copy_linux_so_from_dir(extracted_root: str, out_dir: str) -> int:
    copied = 0
    for root, _dirs, files in os.walk(extracted_root):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), extracted_root)
            rel_norm = rel.replace("\\", "/")
            if "/lib/" not in rel_norm:
                continue
            # Accept versioned SONAMEs like libcublas.so.12.4.5.8
            if ".so" not in name:
                continue
            src = os.path.join(root, name)
            dst = os.path.join(out_dir, name)
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    return copied


def _extract_archive(archive_path: str, tmp_dir: str) -> str:
    extracted_root = os.path.join(tmp_dir, "extracted")
    os.makedirs(extracted_root, exist_ok=True)
    if archive_path.lower().endswith(".zip"):
        with zipfile.ZipFile(archive_path) as z:
            z.extractall(extracted_root)
    elif archive_path.lower().endswith(".tar.xz") or archive_path.lower().endswith(".tar.gz"):
        with tarfile.open(archive_path) as t:
            t.extractall(extracted_root)
    else:
        raise RuntimeError(f"unsupported archive format: {archive_path}")
    return extracted_root


def _copy_component_license(extracted_root: str, out_dir: str, component: str) -> None:
    # Each archive contains exactly one LICENSE.txt at a stable location.
    license_src = None
    for root, _dirs, files in os.walk(extracted_root):
        for name in files:
            if name.lower() != "license.txt":
                continue
            license_src = os.path.join(root, name)
            break
        if license_src:
            break
    if not license_src:
        return
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"LICENSE.{component}.txt")
    shutil.copy2(license_src, dst)


def _ensure_linux_soname_links(out_dir: str, lib_base: str) -> None:
    # Ensure libcublas.so exists if libcublas.so.<major> exists, etc.
    prefix = f"lib{lib_base}.so."
    candidates = []
    for name in os.listdir(out_dir):
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            try:
                major = int(suffix.split(".")[0])
            except ValueError:
                continue
            candidates.append((major, name))
    if not candidates:
        return
    major, target = max(candidates, key=lambda x: x[0])
    link_name = os.path.join(out_dir, f"lib{lib_base}.so")
    if os.path.exists(link_name):
        return
    os.symlink(target, link_name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda-minor", required=True, help="e.g. 12.4 or 12.2")
    ap.add_argument("--platform", required=True, help="e.g. windows-x86_64 or linux-x86_64")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--components",
        required=True,
        nargs="+",
        help="e.g. cuda_cudart libcublas libcurand",
    )
    args = ap.parse_args()

    label = _latest_redistrib_label(args.cuda_minor)
    manifest_url = f"{CUDA_REDIST_BASE_URL}redistrib_{label}.json"
    manifest = json.loads(_fetch_bytes(manifest_url).decode("utf-8"))

    os.makedirs(args.out_dir, exist_ok=True)

    for component in args.components:
        if component not in manifest:
            raise RuntimeError(f"component not in manifest: {component}")
        comp = manifest[component]
        if args.platform not in comp:
            raise RuntimeError(f"platform {args.platform} not in component {component}")
        pkg = comp[args.platform]
        rel = pkg["relative_path"]
        expected_sha = pkg["sha256"]
        url = f"{CUDA_REDIST_BASE_URL}{rel}"

        with tempfile.TemporaryDirectory() as td:
            archive_path = os.path.join(td, os.path.basename(rel))
            _download_to(url, archive_path)
            actual_sha = _sha256_file(archive_path)
            if actual_sha.lower() != expected_sha.lower():
                raise RuntimeError(
                    f"sha256 mismatch for {component} {args.platform}: got {actual_sha}, expected {expected_sha}"
                )
            extracted_root = _extract_archive(archive_path, td)
            _copy_component_license(extracted_root, args.out_dir, component)

            if args.platform.startswith("windows-"):
                # Prefer bin/*.dll from the archive.
                copied = _copy_matching_from_dir(
                    extracted_root, args.out_dir, exts=[".dll"], must_contain="/bin/"
                )
                if copied == 0:
                    raise RuntimeError(
                        f"no dlls copied for {component} {args.platform} (unexpected archive layout)"
                    )
            elif args.platform.startswith("linux-"):
                copied = _copy_linux_so_from_dir(extracted_root, args.out_dir)
                if copied == 0:
                    raise RuntimeError(
                        f"no so files copied for {component} {args.platform} (unexpected archive layout)"
                    )
            else:
                raise RuntimeError(f"unknown platform: {args.platform}")

    # Ensure common soname links exist for dynamic-loading on Linux.
    if args.platform.startswith("linux-"):
        for base in ["cudart", "cublas", "cublasLt", "curand"]:
            _ensure_linux_soname_links(args.out_dir, base)

    print(
        f"ok: fetched CUDA redist {label} for {args.platform} into {args.out_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
