"""
Kaggle session entrypoint for E1.

Paste ONE of the cells below into a Kaggle notebook. Every stage is idempotent
and resumable, so if the session dies you rerun the same cell and it picks up
where it left off.

    # ---- cell 1: setup (all sessions) ----
    # Do NOT `pip install -r requirements.txt` here. That file pins the local
    # dev box (numpy==1.26.4, torch==2.5.1, opacus, peft-from-git); on Kaggle it
    # downgrades numpy under a pandas compiled against numpy 2.x and every
    # `import pandas` afterwards dies with "numpy.dtype size changed". The base
    # image already has everything E1 imports -- preflight.py proves it.
    !git clone https://github.com/<you>/MIA_Finetuned_LLMs.git /kaggle/working/mia
    %cd /kaggle/working/mia

    import os
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["E1_HF_REPO"] = "<user>/mia-e1-artifacts"
    os.environ["WANDB_MODE"] = "offline"

    !python E1_scaled_xps/kaggle/preflight.py     # must print READY

    # ---- cell 2: one-off data prep ----
    !python data/prepare_enron.py                                  # only if raw_data/pool.jsonl is absent
    !python E1_scaled_xps/corpora.py --prepare enron --splits --corpus enron
    !python -m pytest E1_scaled_xps/tests/test_e1.py -q             # Week 1 gate

    # ---- cell 3: E1a, one model per session ----
    # Accelerator MUST be "GPU T4 x2", not P100. Kaggle's torch is built for
    # sm_70+ and the P100 is sm_60, so every CUDA launch on it fails once
    # training starts. preflight.py refuses to run on an unsupported card.
    !python E1_scaled_xps/kaggle/run_kaggle.py e1a --model pythia-70m
    !python E1_scaled_xps/kaggle/run_kaggle.py e1a --model pythia-160m
    !python E1_scaled_xps/kaggle/run_kaggle.py e1a --model pythia-410m   # ~2.5h/N, split across sessions

    # ---- cell 4: E1b cross-domain (needs the news/legal pools first) ----
    # Probe before preparing: a dead config or renamed column costs seconds
    # here and an hour mid-session otherwise.
    !python E1_scaled_xps/corpora.py --probe news
    !python E1_scaled_xps/corpora.py --probe legal
    !python E1_scaled_xps/corpora.py --prepare news  --splits --corpus news
    # pile-of-law/echr has ~7.1k train rows, so 10k cannot be filled; 6000 still
    # leaves 2000 members + 2000 non-members + a 2000-sequence eval split.
    !python E1_scaled_xps/corpora.py --prepare legal --splits --corpus legal --pool-size 6000
    !python E1_scaled_xps/kaggle/run_kaggle.py e1b

    # ---- cell 5: evaluation (2xT4 accelerator; CPU-bound after caching) ----
    !python E1_scaled_xps/kaggle/run_kaggle.py eval

Training already writes the log-prob caches inline, so `eval` is normally pure
CPU. `cache` only exists for checkpoints trained before inline caching, or
after a --no-cache run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_E1_DIR = Path(__file__).parent.parent.resolve()
_ROOT = _E1_DIR.parent

# Kaggle's default 12h wall clock. Leaving an hour of headroom means the final
# state flush and Hub upload always complete.
DEFAULT_MAX_HOURS = 11.0


def _run(args: list[str]) -> int:
    print(f"\n$ {' '.join(args)}\n", flush=True)
    return subprocess.call(args, cwd=str(_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="Kaggle session driver for E1.")
    ap.add_argument("stage", choices=["e1a", "e1b", "cache", "eval", "neighbors"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS)
    ap.add_argument("--bootstrap", type=int, default=None)
    ap.add_argument("--keep-epochs", type=int, nargs="*", default=None,
                    help="Epochs whose checkpoints survive pruning. Needed only for configs "
                         "later fed to E1d / E5d / the neighbourhood attack.")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="Run even if the environment check fails.")
    args = ap.parse_args()

    py = sys.executable

    # A broken env (typically numpy downgraded by requirements.txt) otherwise
    # surfaces hours in, as an ImportError inside a worker.
    if args.stage in ("e1a", "e1b", "cache", "eval"):
        if _run([py, str(_E1_DIR / "kaggle" / "preflight.py")]) != 0:
            if args.skip_preflight:
                print("[preflight] failed; continuing because --skip-preflight was given.",
                      file=sys.stderr)
            else:
                print("\n[preflight] Environment is not ready -- see above. "
                      "Refusing to start; pass --skip-preflight to override.",
                      file=sys.stderr)
                raise SystemExit(2)

    if not os.environ.get("E1_HF_REPO"):
        print("[warn] E1_HF_REPO is unset -- artifacts stay local and will be lost "
              "when this session ends. Set it in cell 1.")

    if args.stage in ("e1a", "e1b"):
        cmd = [py, str(_E1_DIR / "train_e1.py"),
               "--experiment", args.stage,
               "--max-hours", str(args.max_hours)]
        for flag, val in (("--model", args.model), ("--corpus", args.corpus)):
            if val:
                cmd += [flag, val]
        if args.n:
            cmd += ["--n", str(args.n)]
        if args.keep_epochs:
            cmd += ["--keep-epochs"] + [str(e) for e in args.keep_epochs]
        raise SystemExit(_run(cmd))

    if args.stage == "cache":
        cmd = [py, str(_E1_DIR / "cache_logprobs.py")]
        cmd += ["--run-id", args.run_id] if args.run_id else ["--all"]
        raise SystemExit(_run(cmd))

    if args.stage == "neighbors":
        if not (args.run_id and args.epoch):
            ap.error("neighbors needs --run-id and --epoch")
        raise SystemExit(_run([py, str(_E1_DIR / "neighbors.py"),
                               "--run-id", args.run_id, "--epoch", str(args.epoch)]))

    cmd = [py, str(_E1_DIR / "eval_e1.py")]
    if args.run_id:
        cmd += ["--run-id", args.run_id]
    if args.bootstrap is not None:
        cmd += ["--bootstrap", str(args.bootstrap)]
    raise SystemExit(_run(cmd))


if __name__ == "__main__":
    main()
