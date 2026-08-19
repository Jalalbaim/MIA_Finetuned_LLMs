"""
HF Hub sync — survives Kaggle's 12h session kill.

extension.md: "Assume every session dies at 11h59." Everything that must
outlive a session goes to a private Hub repo: per-epoch training state (so a
run resumes mid-grid instead of restarting) and the log-prob caches (which are
the actual scientific output).

All functions are no-ops when HF_REPO_ID is unset, so the whole pipeline runs
locally without a token.

Kaggle setup:
    Add-ons -> Secrets -> new secret named HF_TOKEN
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_E1_DIR = Path(__file__).parent.resolve()
if str(_E1_DIR) not in sys.path:
    sys.path.insert(0, str(_E1_DIR))

from config_e1 import HF_REPO_ID


def repo_id() -> str | None:
    """Env var wins over the config constant, so a Kaggle notebook can point at
    a different repo without editing the file."""
    return os.environ.get("E1_HF_REPO") or HF_REPO_ID


def token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def enabled() -> bool:
    return bool(repo_id() and token())


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=token())


def ensure_repo() -> bool:
    """Create the private repo if missing. Returns False when Hub I/O is off."""
    if not enabled():
        rid = repo_id()
        if rid and not token():
            print(f"  [hub] repo {rid} configured but HF_TOKEN is unset -- local-only mode.")
        return False
    _api().create_repo(repo_id(), repo_type="dataset", private=True, exist_ok=True)
    return True


def upload_file(local_path: Path, remote_path: str) -> bool:
    if not enabled() or not Path(local_path).exists():
        return False
    try:
        _api().upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=repo_id(),
            repo_type="dataset",
        )
        print(f"  [hub] uploaded {remote_path}")
        return True
    except Exception as exc:
        # A failed upload must never kill a training run that is otherwise
        # fine -- the local copy still exists and can be pushed later.
        print(f"  [hub] upload FAILED for {remote_path}: {exc}")
        return False


def upload_dir(local_dir: Path, remote_prefix: str) -> bool:
    if not enabled() or not Path(local_dir).exists():
        return False
    try:
        _api().upload_folder(
            folder_path=str(local_dir),
            path_in_repo=remote_prefix,
            repo_id=repo_id(),
            repo_type="dataset",
        )
        print(f"  [hub] uploaded dir {remote_prefix}")
        return True
    except Exception as exc:
        print(f"  [hub] upload FAILED for {remote_prefix}: {exc}")
        return False


def download_file(remote_path: str, local_path: Path) -> bool:
    """Pull one file back, e.g. training_state.pt when resuming in a fresh
    session. Returns False if Hub is off or the file does not exist."""
    if not enabled():
        return False
    from huggingface_hub import hf_hub_download
    try:
        cached = hf_hub_download(
            repo_id=repo_id(),
            filename=remote_path,
            repo_type="dataset",
            token=token(),
        )
    except Exception:
        return False
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(Path(cached).read_bytes())
    print(f"  [hub] restored {remote_path} -> {local_path}")
    return True


def remote_state_path(run_id: str) -> str:
    return f"runs/{run_id}/training_state.pt"


def remote_cache_path(run_id: str, epoch: int, pool: str) -> str:
    return f"cache/{run_id}/epoch{epoch}_{pool}.npz"


def remote_ckpt_prefix(run_id: str, epoch: int) -> str:
    return f"runs/{run_id}/epoch{epoch}"
