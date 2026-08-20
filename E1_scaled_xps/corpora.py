"""
Corpus registry and membership splits for E1.

Three corpora:
  enron  -- reuses the existing raw_data/pool.jsonl (workshop preprocessing
            kept byte-for-byte, which is the point: continuity).
  news   -- post-cutoff articles. Clean-room bound validity, zero pretraining
            contamination. MUST-priority.
  legal  -- Pile-of-Law ECHR subset. Audit/compliance framing. RECOMMENDED,
            and the designated cut if Week 1 slips.

Splits are regenerated at seed 42 into raw_data/e1/<corpus>/, leaving the
workshop's raw_data/*_seed{0,1,2}.jsonl files untouched so existing GPT-Neo
results stay reproducible.

Reuses data/membership_assignment.py::{save_split, load_split} and
data/prepare_enron.py::clean unchanged.

Usage:
    python E1_scaled_xps/corpora.py --prepare enron
    python E1_scaled_xps/corpora.py --prepare news --pool-size 10000
    python E1_scaled_xps/corpora.py --prepare legal
    python E1_scaled_xps/corpora.py --splits            # all prepared corpora
    python E1_scaled_xps/corpora.py --contamination news --reference enron
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

_E1_DIR = Path(__file__).parent.resolve()
_ROOT = _E1_DIR.parent
for _p in (str(_E1_DIR), str(_ROOT), str(_ROOT / "data")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config_e1 import (
    CORPUS_SIZES,
    E1_DATA_DIR,
    MAX_SEQ_LEN,
    N_NONMEMBERS,
    POOL_SIZE,
    SEED,
)
from membership_assignment import save_split, load_split as _load_split_raw


# Corpus registry

@dataclass(frozen=True)
class HFSource:
    """One downloadable source. `configs` holds more than one entry when a
    single config is too small to fill a pool -- they are concatenated."""
    dataset_id: str
    configs: tuple[str | None, ...]
    split: str
    text_column: str


def _months(start: str, end: str) -> tuple[str, ...]:
    """('2024-01', '2025-06') -> ('2024-01', ..., '2025-06')."""
    sy, sm = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    out = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append(f"{y}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return tuple(out)


# The clean-room window. Pythia-deduped is trained on the Pile, assembled in
# 2020, so anything published from 2024 on postdates every target model's
# pretraining data by years -- that margin is the contamination argument.
#
# RealTimeData/bbc_news_alltime is partitioned into YYYY-MM configs running
# 2017-01 .. 2025-06 (verified against the dataset card), with ~400-3400
# articles per month. One month cannot fill a 10k pool, so eighteen are
# concatenated. Do not widen this window backwards past 2021 without
# rechecking the contamination claim.
NEWS_START, NEWS_END = "2024-01", "2025-06"


@dataclass(frozen=True)
class CorpusSpec:
    key: str
    description: str
    # None for `enron`, which reads the already-prepared workshop pool.
    hf_candidates: tuple[HFSource, ...] | None
    min_tokens: int = 50
    max_tokens: int = 1024   # pool-level truncation; E1 truncates again to 256 at load


CORPORA: dict[str, CorpusSpec] = {
    "enron": CorpusSpec(
        key="enron",
        description="Enron email bodies, workshop preprocessing (headers/HTML/quotes stripped, deduped)",
        hf_candidates=None,
    ),
    "news": CorpusSpec(
        key="news",
        description=f"BBC news articles {NEWS_START}..{NEWS_END}. Clean-room: published years after the Pile was assembled.",
        hf_candidates=(
            HFSource("RealTimeData/bbc_news_alltime", _months(NEWS_START, NEWS_END), "train", "content"),
        ),
    ),
    "legal": CorpusSpec(
        key="legal",
        description="Pile-of-Law, ECHR (European Court of Human Rights) opinions subset",
        hf_candidates=(
            HFSource("pile-of-law/pile-of-law", ("echr",), "train", "text"),
        ),
    ),
}


def get_spec(corpus: str) -> CorpusSpec:
    if corpus not in CORPORA:
        raise KeyError(f"Unknown corpus {corpus!r}. Known: {sorted(CORPORA)}")
    return CORPORA[corpus]


def corpus_dir(corpus: str) -> Path:
    return E1_DATA_DIR / corpus


def pool_path(corpus: str) -> Path:
    return corpus_dir(corpus) / "pool.jsonl"


# Pool preparation

def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _write_pool(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for idx, rec in enumerate(records):
            fh.write(json.dumps({"id": idx, **rec}) + "\n")


def prepare_enron(pool_size: int = POOL_SIZE) -> Path:
    """Copy the workshop pool into the E1 corpus layout. Deliberately a copy,
    not a re-derivation: 'keep exact workshop preprocessing' is the whole
    reason Enron is in the grid."""
    src = _ROOT / "raw_data" / "pool.jsonl"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Run `python data/prepare_enron.py` first."
        )
    records = _read_jsonl(src)
    if len(records) < pool_size:
        raise ValueError(f"Workshop pool has {len(records):,} records, need {pool_size:,}.")
    out = pool_path("enron")
    _write_pool([{"text": r["text"], "n_tokens": r["n_tokens"]} for r in records[:pool_size]], out)
    print(f"  enron: copied {pool_size:,} records from workshop pool -> {out}")
    return out


def prepare_hf_corpus(corpus: str, pool_size: int, tokenizer_id: str) -> Path:
    """Download, clean, length-filter and freeze a pool for `news` or `legal`.

    Cleaning reuses data/prepare_enron.py::clean, which strips HTML and
    quoted-reply lines and normalises whitespace -- all appropriate for news
    and court opinions too. Header stripping only triggers on RFC-822 leads,
    so it is a no-op outside Enron."""
    from datasets import concatenate_datasets, load_dataset
    from transformers import AutoTokenizer
    from tqdm import tqdm
    from prepare_enron import clean

    spec = get_spec(corpus)
    if spec.hf_candidates is None:
        raise ValueError(f"Corpus {corpus!r} has no HF source; prepare it directly.")

    ds, text_col, provenance = None, None, {}
    for source in spec.hf_candidates:
        parts, used = [], []
        for cfg in source.configs:
            try:
                part = load_dataset(
                    source.dataset_id, cfg, split=source.split, trust_remote_code=True
                )
                parts.append(part)
                used.append(cfg)
                print(f"    {source.dataset_id} [{cfg}]: {len(part):,} rows")
            except Exception as exc:
                # One missing month should not sink the whole corpus; a source
                # is only a failure if *every* config it lists is unavailable.
                print(f"    {source.dataset_id} [{cfg}]: unavailable ({exc})")
        if parts:
            ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
            text_col = source.text_column
            provenance = {
                "dataset_id": source.dataset_id,
                "configs_used": used,
                "configs_requested": list(source.configs),
                "split": source.split,
                "text_column": source.text_column,
            }
            print(f"  Loaded {len(ds):,} rows from {len(used)} config(s); text column={text_col!r}")
            break

    if ds is None:
        raise RuntimeError(
            f"Every source for corpus {corpus!r} failed. Either add a working "
            f"HFSource to CORPORA[{corpus!r}].hf_candidates, or write your own "
            f"pool to {pool_path(corpus)} (one JSON object per line with a "
            f"'text' field) and skip --prepare."
        )

    print("  Cleaning and deduplicating ...")
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in tqdm(ds, desc="clean"):
        body = clean(item[text_col] or "")
        if not body:
            continue
        key = " ".join(body.lower().split())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(body)
    print(f"  After cleaning / dedup: {len(cleaned):,}")

    print(f"  Tokenizing with {tokenizer_id} (keep {spec.min_tokens}-{spec.max_tokens} tokens) ...")
    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    filtered: list[dict] = []
    for body in tqdm(cleaned, desc="tokenize"):
        ids = tok.encode(body, add_special_tokens=False)
        if len(ids) < spec.min_tokens:
            continue
        ids = ids[: spec.max_tokens]
        filtered.append({"text": tok.decode(ids, skip_special_tokens=True), "n_tokens": len(ids)})
    print(f"  After length filter: {len(filtered):,}")

    if len(filtered) < pool_size:
        raise ValueError(
            f"Only {len(filtered):,} sequences survived for {corpus!r}; need {pool_size:,}.\n"
            f"Either lower --pool-size (N=2000 members + 2000 non-members needs only "
            f"~5,000 to leave a usable eval split), or widen the source -- for 'news', "
            f"move NEWS_START earlier in corpora.py, but not past 2021 without "
            f"rechecking the contamination claim."
        )

    rng = random.Random(SEED)
    rng.shuffle(filtered)
    pool = filtered[:pool_size]

    out = pool_path(corpus)
    _write_pool(pool, out)

    # Provenance sits next to the pool: which configs actually loaded is part
    # of the clean-room claim, and "we used BBC 2024-01..2025-06" has to be
    # checkable later rather than reconstructed from memory.
    provenance.update({
        "corpus": corpus,
        "pool_size": len(pool),
        "n_after_clean_dedup": len(cleaned),
        "n_after_length_filter": len(filtered),
        "tokenizer": tokenizer_id,
        "min_tokens": spec.min_tokens,
        "max_tokens": spec.max_tokens,
        "shuffle_seed": SEED,
    })
    (out.parent / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"  provenance -> {out.parent / 'provenance.json'}")

    lengths = [r["n_tokens"] for r in pool]
    print(
        f"  {corpus}: wrote {len(pool):,} records -> {out}\n"
        f"    token length  mean={statistics.mean(lengths):.1f}  "
        f"median={statistics.median(lengths):.1f}  "
        f"min={min(lengths)}  max={max(lengths)}"
    )
    return out


# Membership splits

def make_split(
    pool: list[dict],
    seed: int,
    n_members: int,
    n_nonmembers: int = N_NONMEMBERS,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Shuffle under `seed`, then slice disjoint nonmember / member / eval sets.

    Same construction as data/membership_assignment.py::make_split, but the
    pool size is read from the pool rather than asserted equal to the root
    config's POOL_SIZE -- the news and legal pools are smaller than Enron's.

    Non-members are taken from the front of the shuffled pool so that, for a
    fixed seed, the non-member set is byte-identical across every N. That is
    what makes the corpus-size sweep a clean comparison: only the member set
    changes."""
    if n_nonmembers + n_members > len(pool):
        raise ValueError(
            f"n_nonmembers ({n_nonmembers:,}) + n_members ({n_members:,}) "
            f"exceeds pool size ({len(pool):,})."
        )
    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    nonmembers = shuffled[:n_nonmembers]
    members = shuffled[n_nonmembers : n_nonmembers + n_members]
    eval_set = shuffled[n_nonmembers + n_members :]
    return members, nonmembers, eval_set


def build_splits(
    corpus: str,
    seed: int = SEED,
    corpus_sizes: list[int] | None = None,
) -> None:
    """Freeze splits for every N. Idempotent: rerunning with the same seed
    reproduces byte-identical files."""
    sizes = corpus_sizes or CORPUS_SIZES
    path = pool_path(corpus)
    if not path.exists():
        raise FileNotFoundError(f"No pool for {corpus!r} at {path}. Run --prepare {corpus} first.")

    pool = _read_jsonl(path)
    out_dir = corpus_dir(corpus)
    print(f"\n{corpus}: pool={len(pool):,}  seed={seed}  -> {out_dir}")

    ref_nonmember_ids: set[int] | None = None
    for n in sizes:
        if N_NONMEMBERS + n > len(pool):
            print(f"  [skip] N={n:,}: pool too small ({len(pool):,} records)")
            continue
        members, nonmembers, eval_set = make_split(pool, seed, n)
        save_split(members, nonmembers, eval_set, seed, n, out_dir=out_dir)

        m_ids = {r["id"] for r in members}
        nm_ids = {r["id"] for r in nonmembers}
        e_ids = {r["id"] for r in eval_set}
        disjoint = len(m_ids) + len(nm_ids) + len(e_ids) == len(m_ids | nm_ids | e_ids)

        if ref_nonmember_ids is None:
            ref_nonmember_ids = nm_ids
        elif nm_ids != ref_nonmember_ids:
            raise AssertionError(
                f"Non-member set changed between N values for {corpus!r} seed={seed} -- "
                f"the corpus-size sweep would not be a controlled comparison."
            )

        print(
            f"  N={n:<6,} members={len(members):<6,} nonmembers={len(nonmembers):<6,} "
            f"eval={len(eval_set):<6,} disjoint={'PASS' if disjoint else 'FAIL'}"
        )


def load_split(corpus: str, n_members: int, seed: int = SEED):
    """(members, nonmembers, eval_set) for one corpus/N/seed."""
    return _load_split_raw(seed, n_members, data_dir=corpus_dir(corpus))


def load_pool(corpus: str) -> list[dict]:
    return _read_jsonl(pool_path(corpus))


# Contamination check (13-gram overlap)

def _ngrams(text: str, n: int) -> set[int]:
    """Hashed word-level n-grams. Hashing rather than storing strings keeps the
    reference index small enough to hold a large corpus in memory."""
    words = text.lower().split()
    if len(words) < n:
        return set()
    return {hash(" ".join(words[i : i + n])) for i in range(len(words) - n + 1)}


def build_ngram_index(texts, n: int = 13) -> set[int]:
    index: set[int] = set()
    for t in texts:
        index |= _ngrams(t, n)
    return index


def contamination_rate(
    corpus_texts: list[str],
    reference_index: set[int],
    n: int = 13,
) -> dict:
    """Fraction of corpus documents sharing at least one n-gram with the
    reference, plus the mean per-document overlap fraction.

    extension.md asks for a 13-gram decontamination check against the Pile and
    for the overlap rate to be reported in the paper. This function computes it
    against whatever reference index you build -- to make the claim about the
    Pile specifically, `reference_index` must be built from Pile text, which
    means supplying a Pile shard. Running it against the Enron pool instead is
    a useful cross-corpus sanity check but is NOT the Pile claim, and the paper
    must not describe it as one."""
    n_hit = 0
    frac_sum = 0.0
    for text in corpus_texts:
        grams = _ngrams(text, n)
        if not grams:
            continue
        hits = len(grams & reference_index)
        if hits:
            n_hit += 1
        frac_sum += hits / len(grams)
    total = len(corpus_texts)
    return {
        "n_documents": total,
        "n_documents_with_overlap": n_hit,
        "document_overlap_rate": n_hit / total if total else 0.0,
        "mean_ngram_overlap_fraction": frac_sum / total if total else 0.0,
        "ngram_n": n,
    }


# Entry point

def probe(corpus: str, n_configs: int = 2) -> bool:
    """Cheaply check that a corpus's HF sources still resolve.

    Downloads a handful of rows per config rather than the whole split, so a
    dead dataset id or a renamed column costs seconds instead of an hour into
    an E1b session. Returns True if any source produced rows.

    This exists because the obvious hand-written probe -- load_dataset on a
    guessed config name -- is where both known E1b failures happened: a config
    ("2025-12") that the dataset does not have, and an environment broken by
    installing requirements.txt on Kaggle.
    """
    from datasets import load_dataset

    spec = get_spec(corpus)
    if spec.hf_candidates is None:
        ok = pool_path(corpus).exists()
        print(f"{corpus}: no HF source; local pool {'present' if ok else 'ABSENT'} "
              f"at {pool_path(corpus)}")
        return ok

    any_ok = False
    for source in spec.hf_candidates:
        cfgs = source.configs[:n_configs]
        print(f"\n{corpus}: {source.dataset_id}  "
              f"({len(source.configs)} config(s), probing {len(cfgs)})")
        for cfg in cfgs:
            try:
                part = load_dataset(source.dataset_id, cfg,
                                    split=f"{source.split}[:5]", trust_remote_code=True)
            except Exception as exc:
                print(f"  [{cfg}] FAIL {type(exc).__name__}: {str(exc)[:200]}")
                continue
            cols = part.column_names
            if source.text_column not in cols:
                print(f"  [{cfg}] rows ok but text column {source.text_column!r} "
                      f"missing; columns are {cols}")
                continue
            chars = len(part[0][source.text_column] or "")
            print(f"  [{cfg}] ok -- columns={cols}, first row {chars} chars")
            any_ok = True
    if not any_ok:
        print(f"\n{corpus}: every probed source failed. Fix "
              f"CORPORA[{corpus!r}].hf_candidates before spending a session.")
    return any_ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare E1 corpora and freeze membership splits.")
    ap.add_argument("--prepare", choices=sorted(CORPORA), default=None,
                    help="Download/clean/freeze the pool for one corpus")
    ap.add_argument("--splits", action="store_true",
                    help="Build membership splits for every corpus with a pool on disk")
    ap.add_argument("--corpus", default=None, help="Restrict --splits to one corpus")
    ap.add_argument("--pool-size", type=int, default=POOL_SIZE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tokenizer", default="EleutherAI/pythia-410m-deduped",
                    help="Tokenizer used for length filtering (all Pythia sizes share one)")
    ap.add_argument("--probe", default=None, choices=sorted(CORPORA),
                    help="Check this corpus's HF sources resolve, without downloading them")
    ap.add_argument("--contamination", default=None, choices=sorted(CORPORA),
                    help="Corpus to check for n-gram overlap")
    ap.add_argument("--reference", default=None, choices=sorted(CORPORA),
                    help="Reference corpus for --contamination")
    args = ap.parse_args()

    if args.probe:
        raise SystemExit(0 if probe(args.probe) else 1)

    if args.prepare == "enron":
        prepare_enron(args.pool_size)
    elif args.prepare:
        prepare_hf_corpus(args.prepare, args.pool_size, args.tokenizer)

    if args.splits:
        targets = [args.corpus] if args.corpus else sorted(CORPORA)
        for c in targets:
            if pool_path(c).exists():
                build_splits(c, seed=args.seed)
            else:
                print(f"\n[skip] {c}: no pool at {pool_path(c)}")

    if args.contamination:
        if not args.reference:
            ap.error("--contamination requires --reference")
        print(f"\nBuilding 13-gram index from {args.reference!r} ...")
        ref_index = build_ngram_index(r["text"] for r in load_pool(args.reference))
        print(f"  {len(ref_index):,} distinct 13-grams")
        texts = [r["text"] for r in load_pool(args.contamination)]
        stats = contamination_rate(texts, ref_index)
        print(f"\n13-gram overlap: {args.contamination!r} vs {args.reference!r}")
        for k, v in stats.items():
            print(f"  {k:<32} {v}")

    if not (args.prepare or args.splits or args.contamination or args.probe):
        ap.print_help()


if __name__ == "__main__":
    main()
