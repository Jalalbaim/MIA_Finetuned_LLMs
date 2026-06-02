#!/usr/bin/env python3
"""
Membership assignment for MIA fine-tuning experiments (Step 2).
Usage: python data/membership_assignment.py
"""

import sys
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SEEDS, CORPUS_SIZES, N_NONMEMBERS, POOL_SIZE,
    POOL_FILE, DATA_DIR,
)



def _split_path(split_name: str, n: int, seed: int, out_dir: Path) -> Path:
    return out_dir / f"{split_name}_N{n}_seed{seed}.jsonl"



def make_split(
    pool_path: Path,
    seed: int,
    n_members: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
      pool[:N_NONMEMBERS]                         → nonmembers
      pool[N_NONMEMBERS : N_NONMEMBERS+n_members] → members
      pool[N_NONMEMBERS+n_members :]              → eval set E

    Returns (members, nonmembers, eval_set).
    """
    with pool_path.open(encoding="utf-8") as fh:
        pool = [json.loads(line) for line in fh]

    if len(pool) != POOL_SIZE:
        raise ValueError(
            f"Pool has {len(pool):,} records; expected {POOL_SIZE:,}. "
            "Re-run data/prepare_enron.py."
        )

    need = N_NONMEMBERS + n_members
    if need > POOL_SIZE:
        raise ValueError(
            f"N_NONMEMBERS ({N_NONMEMBERS:,}) + n_members ({n_members:,}) = "
            f"{need:,} exceeds POOL_SIZE ({POOL_SIZE:,})."
        )

    random.seed(seed)
    random.shuffle(pool)

    nonmembers = pool[:N_NONMEMBERS]
    members    = pool[N_NONMEMBERS : N_NONMEMBERS + n_members]
    eval_set   = pool[N_NONMEMBERS + n_members :]

    return members, nonmembers, eval_set


def save_split(
    members:    list[dict],
    nonmembers: list[dict],
    eval_set:   list[dict],
    seed:       int,
    n_members:  int,
    out_dir:    Path = DATA_DIR,
) -> None:
    """Write three split files to out_dir, tagging each record with a 'split' field."""
    out_dir.mkdir(parents=True, exist_ok=True)

    _LABEL = {
        "members":    "member",
        "nonmembers": "nonmember",
        "eval":       "eval",
    }
    for records, key in (
        (members,    "members"),
        (nonmembers, "nonmembers"),
        (eval_set,   "eval"),
    ):
        path = _split_path(key, n_members, seed, out_dir)
        with path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps({**rec, "split": _LABEL[key]}) + "\n")


def load_split(
    seed:      int,
    n_members: int,
    data_dir:  Path = DATA_DIR,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Load and return (members, nonmembers, eval_set) from saved split files.
    Intended for import by downstream modules (fine-tuning, attack signals, KL estimators).
    """
    result = []
    for key in ("members", "nonmembers", "eval"):
        path = _split_path(key, n_members, seed, data_dir)
        with path.open(encoding="utf-8") as fh:
            result.append([json.loads(line) for line in fh])
    members, nonmembers, eval_set = result
    return members, nonmembers, eval_set



def _ids(records: list[dict]) -> set[int]:
    return {r["id"] for r in records}


def _disjoint(*id_sets: set[int]) -> bool:
    total = sum(len(s) for s in id_sets)
    return total == len(set().union(*id_sets))


def main() -> None:
    print(f"Pool  : {POOL_FILE}")
    print(f"Output: {DATA_DIR}\n")

    sep = "+" + "-" * 6 + "+" + "-" * 11 + "+" + "-" * 14 + "+" + "-" * 8 + "+" + "-" * 15 + "+"
    hdr = f"| {'seed':>4} | {'N_members':>9} | {'N_nonmembers':>12} | {'N_eval':>6} | {'overlap_check':>13} |"
    print(sep)
    print(hdr)
    print(sep)

    ref_nonmember_ids: dict[int, set[int]] = {}

    for seed in SEEDS:
        for n in CORPUS_SIZES:
            members, nonmembers, eval_set = make_split(POOL_FILE, seed, n)

            m_ids  = _ids(members)
            nm_ids = _ids(nonmembers)
            e_ids  = _ids(eval_set)
            ok     = "PASS" if _disjoint(m_ids, nm_ids, e_ids) else "FAIL"

            save_split(members, nonmembers, eval_set, seed, n)

            print(
                f"| {seed:>4} | {n:>9,} | {len(nonmembers):>12,} "
                f"| {len(eval_set):>6,} | {ok:>13} |"
            )

            # Consistency invariant: non-members are identical for all N, same seed.
            if n == CORPUS_SIZES[0]:
                ref_nonmember_ids[seed] = nm_ids
            else:
                diff = nm_ids ^ ref_nonmember_ids[seed]
                assert not diff, (
                    f"Non-member consistency violated: seed={seed}, N={n} "
                    f"vs N={CORPUS_SIZES[0]}. "
                    f"Symmetric diff ({len(diff)} ids): {sorted(diff)[:10]} ..."
                )

    print(sep)
    print("\nNon-member consistency across all N values: PASSED for all seeds.")
    print(f"\nWrote {len(SEEDS) * len(CORPUS_SIZES) * 3} files to {DATA_DIR}")


if __name__ == "__main__":
    main()
