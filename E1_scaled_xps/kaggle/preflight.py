"""
Verify a Kaggle session can run E1 -- without changing it.

E1 imports only torch, transformers, datasets, huggingface_hub, numpy, scipy,
scikit-learn and pandas, all of which the Kaggle base image already provides at
versions that work. The repo's requirements.txt is a snapshot of the *local*
development environment: it pins numpy==1.26.4, and installing it on Kaggle
downgrades numpy underneath a pandas that was compiled against numpy 2.x
headers, which fails at import with

    ValueError: numpy.dtype size changed, may indicate binary incompatibility.
                Expected 96 from C header, got 88 from PyObject

It also drags in torch==2.5.1, opacus and a git checkout of peft, none of which
E1 touches. So the Kaggle setup cell installs nothing and runs this instead.

    python E1_scaled_xps/kaggle/preflight.py

Exits 0 if the session is usable, 1 with a specific remedy if not.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

_E1_DIR = Path(__file__).parent.parent.resolve()
_ROOT = _E1_DIR.parent

# (module, attribute holding the version, minimum, why)
REQUIRED = [
    ("numpy",           "__version__", (1, 24), "arrays everywhere"),
    ("scipy",           "__version__", (1, 10), "stats.ks_2samp, rankdata, brentq"),
    ("sklearn",         "__version__", (1, 3),  "roc_curve in metrics/compute_metrics.py"),
    ("pandas",          "__version__", (2, 0),  "results/e1_metrics.csv"),
    ("torch",           "__version__", (2, 1),  "training + forward passes"),
    ("transformers",    "__version__", (4, 44), "GPTNeoX / GPT-Neo loading"),
    ("datasets",        "__version__", (2, 19), "corpus download"),
    ("huggingface_hub", "__version__", (0, 24), "artifact persistence"),
]

MIN_FREE_GB = 12.0


def _ver(mod) -> tuple[int, ...]:
    raw = getattr(mod, "__version__", "0")
    out = []
    for part in raw.split(".")[:3]:
        digits = "".join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def main() -> int:
    problems: list[str] = []
    print(f"python           {sys.version.split()[0]}")

    versions: dict[str, str] = {}
    for name, _attr, minimum, why in REQUIRED:
        try:
            mod = importlib.import_module(name)
        except Exception as exc:
            # A broken ABI pairing surfaces here, not as a missing package.
            hint = ""
            if "numpy.dtype size changed" in str(exc):
                hint = ("\n      -> numpy was downgraded under a package built against "
                        "numpy 2.x.\n         Almost always caused by `pip install -r "
                        "requirements.txt`, which pins\n         numpy==1.26.4 for the local "
                        "dev box. Factory-reset the session\n         (Run > Factory reset) "
                        "and do not run that install.")
            problems.append(f"{name}: import failed -- {type(exc).__name__}: {exc}{hint}")
            continue
        versions[name] = getattr(mod, "__version__", "?")
        got = _ver(mod)
        flag = " " if got >= minimum else "!"
        print(f"{flag}{name:16s} {versions[name]:12s} ({why})")
        if got < minimum:
            problems.append(
                f"{name} {versions[name]} < required {'.'.join(map(str, minimum))} -- "
                f"needed for {why}. Install just this one: pip install -q -U {name}")

    # datasets 3.0 removed loading-script support and 5.0 removed
    # trust_remote_code, so pile-of-law cannot be opened via load_dataset on a
    # current image. corpora.py reads its raw jsonl.xz shards off the Hub
    # instead, so this is informational only -- but say so, because the raw
    # path is easy to mistake for a bug.
    if "datasets" in versions and _ver(importlib.import_module("datasets"))[0] >= 3:
        print(f"  [note] datasets {versions['datasets']} has no loading scripts; the "
              f"E1b legal corpus loads via raw shards (corpora.read_hub_jsonl).")

    if "torch" in versions:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            dtype = "bfloat16" if major >= 8 else "float16"
            print(f" gpu              {name} sm_{major}{minor}  {mem:.0f}GB  "
                  f"x{torch.cuda.device_count()}  -> autocast {dtype}")

            # The card being present does not mean this torch build has kernels
            # for it. Kaggle's torch 2.10+cu128 is compiled for sm_70 and up,
            # so the P100 (sm_60) is unusable: every CUDA launch fails with "no
            # kernel image is available for execution on the device", but only
            # once training starts. Catch it here, in seconds.
            arches = torch.cuda.get_arch_list()
            print(f" torch arch list  {' '.join(arches)}")
            if not any(a.startswith(f"sm_{major}{minor}") for a in arches):
                problems.append(
                    f"{name} is sm_{major}{minor}, but this torch build only has kernels "
                    f"for [{' '.join(arches)}]. Switch the accelerator to GPU T4 x2 "
                    f"(sm_75) -- do not use the P100 on this image.")
            else:
                # Cheap proof that a kernel actually launches.
                try:
                    x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
                    torch.mm(x, x).sum().item()
                    torch.cuda.synchronize()
                    print(" cuda smoke test  ok (fp16 matmul)")
                except Exception as exc:
                    problems.append(
                        f"CUDA is reported available but a trivial fp16 matmul failed: "
                        f"{type(exc).__name__}: {str(exc)[:160]}")
        else:
            print(" gpu              NONE (accelerator is off -- training will crawl)")
            problems.append("No CUDA device. Set Settings > Accelerator before running e1a/e1b.")

    free = shutil.disk_usage(str(_ROOT)).free / 1e9
    print(f" disk free        {free:.1f}GB")
    if free < MIN_FREE_GB:
        problems.append(
            f"Only {free:.1f}GB free; a 410M run peaks near 6GB and wants "
            f"{MIN_FREE_GB:.0f}GB of headroom. Delete /kaggle/working leftovers.")

    repo, tok = os.environ.get("E1_HF_REPO"), os.environ.get("HF_TOKEN")
    print(f" E1_HF_REPO       {repo or 'UNSET'}")
    print(f" HF_TOKEN         {'set' if tok else 'UNSET'}")
    if not (repo and tok):
        problems.append(
            "E1_HF_REPO/HF_TOKEN unset: every artifact stays on the session disk and "
            "dies with it. This is how the first 410M attempt lost a finished run.")

    print()
    if problems:
        print(f"NOT READY -- {len(problems)} problem(s):")
        for p in problems:
            print(f"  * {p}")
        return 1
    print("READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
