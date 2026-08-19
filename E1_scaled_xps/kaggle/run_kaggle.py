"""
Kaggle session entrypoint for E1.

Paste ONE of the cells below into a Kaggle notebook. Every stage is idempotent
and resumable, so if the session dies you rerun the same cell and it picks up
where it left off.

    # ---- cell 1: setup (all sessions) ----
    !git clone https://github.com/<you>/MIA_Finetuned_LLMs.git /kaggle/working/mia
    %cd /kaggle/working/mia
    !pip install -q -r requirements.txt

    import os
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["E1_HF_REPO"] = "<user>/mia-e1-artifacts"
    os.environ["WANDB_MODE"] = "offline"

    # ---- cell 2: one-off data prep ----
    !python data/prepare_enron.py                                  # only if raw_data/pool.jsonl is absent
    !python E1_scaled_xps/corpora.py --prepare enron --splits --corpus enron
    !python -m pytest E1_scaled_xps/tests/test_e1.py -q             # Week 1 gate

    # ---- cell 3: E1a, one model per session (P100) ----
    !python E1_scaled_xps/kaggle/run_kaggle.py e1a --model pythia-70m
    !python E1_scaled_xps/kaggle/run_kaggle.py e1a --model pythia-160m
    !python E1_scaled_xps/kaggle/run_kaggle.py e1a --model pythia-410m   # ~2.5h/N, split across sessions

    # ---- cell 4: E1b cross-domain (needs the news/legal pools first) ----
    !python E1_scaled_xps/corpora.py --prepare news --splits --corpus news
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
    args = ap.parse_args()

    py = sys.executable
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
