"""
RunConfig — the identity of a single fine-tuning run.

Replaces the workshop pattern of encoding run identity in filenames and
recovering it with regexes (four parallel patterns in
metrics/compute_metrics.py, plus matching if/elif chains in
signals/compute_signals.py and kl_estimators/compute_kl.py). That pattern
needs a code change in three files for every new sweep axis; E1 alone adds
model and corpus, and E2/E5 add lr, dup_factor, rank and epsilon.

Here a run is a dataclass. It serialises itself to runs/<run_id>/config.json,
and every downstream artifact is addressed by run_id. Adding an axis means
adding a field.

Run id format follows extension.md:
    pythia410m_enron_N2000_lr5e-5_seed42_a3f2
The trailing 4 hex chars hash *every* field, including ones not shown in the
readable prefix, so two runs differing only in (say) dup_factor cannot collide.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from config_e1 import (
    E1_RUNS_DIR,
    E1_CACHE_DIR,
    EPOCH_GRID,
    LEARNING_RATE,
    MAX_SEQ_LEN,
    EFFECTIVE_BATCH,
    SEED,
)


def _fmt_lr(lr: float) -> str:
    """5e-05 -> '5e-5', 0.0002 -> '2e-4'. Stable across platforms."""
    s = f"{lr:g}"
    return s.replace("e-0", "e-").replace("e+0", "e+")


def _slug(model: str) -> str:
    """'pythia-410m' -> 'pythia410m'."""
    return model.replace("-", "").replace("_", "").replace(".", "p")


@dataclass(frozen=True)
class RunConfig:
    """One fine-tuning run. Immutable so run_id can be cached safely."""

    model: str                       # key into models.MODELS, e.g. "pythia-410m"
    corpus: str                      # key into corpora.CORPORA, e.g. "enron"
    n_members: int
    seed: int = SEED
    lr: float = LEARNING_RATE

    # Adaptation method. E1 only ever uses "full"; the remaining values exist
    # so E5 can reuse this dataclass without a schema change.
    method: str = "full"             # full | lora | mica | dp
    rank: int | None = None          # lora/mica only
    dp_epsilon: float | None = None  # dp only

    # E2 axes, fixed for E1.
    dup_factor: int = 1              # 4 = each member duplicated 4x in the training set

    max_seq_len: int = MAX_SEQ_LEN
    effective_batch: int = EFFECTIVE_BATCH
    epoch_grid: tuple[int, ...] = field(default=tuple(EPOCH_GRID))

    # Derived identity

    @property
    def _payload(self) -> dict:
        """Canonical dict used for hashing. Sorted keys make the hash stable
        across Python versions and dict insertion order."""
        d = asdict(self)
        d["epoch_grid"] = list(self.epoch_grid)
        return d

    @property
    def hash(self) -> str:
        blob = json.dumps(self._payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:4]

    @property
    def run_id(self) -> str:
        parts = [
            _slug(self.model),
            self.corpus,
            f"N{self.n_members}",
            f"lr{_fmt_lr(self.lr)}",
        ]
        if self.method != "full":
            parts.append(self.method)
        if self.rank is not None:
            parts.append(f"r{self.rank}")
        if self.dp_epsilon is not None:
            parts.append(f"eps{self.dp_epsilon:g}")
        if self.dup_factor != 1:
            parts.append(f"dup{self.dup_factor}")
        parts.append(f"seed{self.seed}")
        parts.append(self.hash)
        return "_".join(parts)

    @property
    def max_epochs(self) -> int:
        return max(self.epoch_grid)

    # Filesystem layout

    @property
    def run_dir(self) -> Path:
        return E1_RUNS_DIR / self.run_id

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def state_path(self) -> Path:
        """Optimizer/scheduler/epoch state for resume-after-session-kill."""
        return self.run_dir / "training_state.pt"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "train_log.csv"

    def ckpt_dir(self, epoch: int) -> Path:
        return self.run_dir / f"epoch{epoch}"

    @property
    def reference_cache_path(self) -> Path:
        """P_pre log-probs over this run's pools.

        The reference model does not change across epochs, so recomputing it at
        every grid checkpoint doubles the cost of the eval pass for nothing --
        measured at 410M, caching took 507s per checkpoint against 59s of
        training. Computed once, reused seven times."""
        return E1_CACHE_DIR / self.run_id / "reference.npz"

    def cache_path(self, epoch: int, pool: str = "attack") -> Path:
        """Per-token log-prob cache for one checkpoint. `pool` distinguishes
        the member+nonmember attack pool from the held-out utility pool."""
        return E1_CACHE_DIR / self.run_id / f"epoch{epoch}_{pool}.npz"

    # Persistence

    def save(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = {**self._payload, "run_id": self.run_id}
        self.config_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        return self.config_path

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d.pop("run_id", None)
        d["epoch_grid"] = tuple(d["epoch_grid"])
        return cls(**d)

    @classmethod
    def discover(cls, runs_dir: Path = E1_RUNS_DIR) -> list["RunConfig"]:
        """Every run with a config.json on disk. Replaces filename-regex
        discovery -- the sweep always matches what actually exists."""
        if not runs_dir.exists():
            return []
        found = []
        for cfg_path in sorted(runs_dir.glob("*/config.json")):
            try:
                found.append(cls.load(cfg_path))
            except Exception as exc:
                print(f"  [warn] Unreadable run config {cfg_path}: {exc}")
        return found

    @property
    def done_marker(self) -> Path:
        """Written when the epoch grid finishes.

        Completion cannot be inferred from checkpoints on disk: they are pruned
        once their log-prob cache exists (7 checkpoints + resume state is
        ~10.6GB per 410M run, against ~20GB of Kaggle disk). Without this
        marker, rerunning a finished command would retrain from scratch."""
        return self.run_dir / "COMPLETED"

    def is_complete(self) -> bool:
        return self.done_marker.exists()

    def mark_complete(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.done_marker.write_text(
            json.dumps({"run_id": self.run_id,
                        "cached_epochs": self.cached_epochs()}, indent=2),
            encoding="utf-8",
        )

    def completed_epochs(self) -> list[int]:
        """Epochs in the grid whose checkpoint is fully written on disk.

        Note this shrinks as checkpoints are pruned -- use cached_epochs() to
        ask what is evaluable, and this only to ask what still has a model."""
        return [e for e in sorted(self.epoch_grid)
                if (self.ckpt_dir(e) / "config.json").exists()]

    def cached_epochs(self) -> list[int]:
        """Epochs with a log-prob cache. This is what E1c can evaluate, and it
        survives checkpoint pruning."""
        return [e for e in sorted(self.epoch_grid) if self.cache_path(e).exists()]

    def __str__(self) -> str:
        return self.run_id


# Grid expansion

def expand_grid(
    models: list[str],
    corpora: list[str],
    corpus_sizes: list[int],
    **fixed,
) -> list[RunConfig]:
    """Cartesian product -> RunConfigs. This is the 'for-loop, not a refactor'
    that extension.md asks for: adding a seed axis means adding a loop here,
    with no changes anywhere downstream."""
    return [
        RunConfig(model=m, corpus=c, n_members=n, **fixed)
        for m in models
        for c in corpora
        for n in corpus_sizes
    ]
