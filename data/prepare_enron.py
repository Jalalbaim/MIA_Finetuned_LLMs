
"""
Prepare the Enron email pool for fine-tuning experiments.
"""

import sys
import re
import json
import random
import statistics
import email as _email
from pathlib import Path
from html.parser import HTMLParser

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_NAME, MAX_SEQ_LEN, MIN_SEQ_LEN, POOL_SIZE, POOL_FILE, DATA_DIR

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from tqdm import tqdm


# Dataset loading 

_CANDIDATES = [
    # (dataset_id,           split,   text_column)
    ("enron_emails",         None,    None),
    ("SetFit/enron_spam",    "train", "text"),
    ("snats/enron",          None,    None),
]


def _find_text_col(features: dict) -> str | None:
    for name in ("text", "message", "body", "email", "content"):
        if name in features:
            return name
    for name, feat in features.items():
        try:
            if feat.dtype == "string":
                return name
        except AttributeError:
            pass
    return None


def load_enron():
    last_exc: Exception | None = None
    for dataset_id, split, col in _CANDIDATES:
        try:
            print(f"Trying dataset {dataset_id!r} ...")
            ds = load_dataset(dataset_id, split=split, trust_remote_code=True)
            if hasattr(ds, "keys"):          # DatasetDict → merge all splits
                ds = concatenate_datasets(list(ds.values()))
            col = col or _find_text_col(ds.features)
            if col is None:
                raise ValueError(
                    f"Cannot identify a text column. "
                    f"Available features: {list(ds.features)}"
                )
            print(f"  Loaded {len(ds):,} rows; text column={col!r}")
            return ds, col
        except Exception as exc:
            print(f"  Skipping: {exc}")
            last_exc = exc

    raise RuntimeError(
        f"All dataset candidates failed. Last error: {last_exc}\n"
        "Add a working dataset id to _CANDIDATES in data/prepare_enron.py."
    )


# HTML stripping 

class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, d: str) -> None:
        self._parts.append(d)

    @property
    def text(self) -> str:
        return " ".join(self._parts)


def _strip_html(text: str) -> str:
    if not re.search(r"<[a-zA-Z/!]", text):
        return text
    s = _HTMLStripper()
    try:
        s.feed(text)
        return s.text
    except Exception:
        return re.sub(r"<[^>]+>", " ", text)  # dumb-regex fallback


# Email body extraction

def _extract_rfc822_body(raw: str) -> str:
    """Return the payload text of an RFC 822 email, stripping all headers."""
    msg = _email.message_from_string(raw)
    parts: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() != "text/plain":
                continue
            try:
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                parts.append(payload.decode(charset, errors="replace"))
            except Exception:
                fb = part.get_payload(decode=False)
                if fb:
                    parts.append(str(fb))
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload is not None:
                charset = msg.get_content_charset() or "utf-8"
                parts.append(payload.decode(charset, errors="replace"))
            else:
                parts.append(str(msg.get_payload() or ""))
        except Exception:
            parts.append(str(msg.get_payload() or ""))

    body = "\n".join(parts)

    # Manual fallback: scan for the blank line separating headers from body
    if not body.strip():
        lines = raw.splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                body = "\n".join(lines[i + 1:])
                break

    return body


_HEADER_LEAD = re.compile(
    r"^(Message-ID:|From |Date:|Subject:|To:|Received:|MIME-Version:|X-[\w-]+:)",
    re.IGNORECASE | re.MULTILINE,
)


def clean(raw: str) -> str:
    """
    Return cleaned email body:
      - parse + strip RFC 822 headers (if present)
      - strip HTML
      - drop quoted-reply lines (lines whose first non-space char is '>')
      - normalise whitespace
    """
    text = _extract_rfc822_body(raw) if _HEADER_LEAD.search(raw[:400]) else raw
    text = _strip_html(text)
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith(">")]
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Main pipeline 

def main() -> None:
    dataset, text_col = load_enron()
    n_raw = len(dataset)
    print(f"\nRaw emails loaded: {n_raw:,}")

    # Clean + deduplicate
    print("Cleaning and deduplicating ...")
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in tqdm(dataset, desc="clean"):
        body = clean(item[text_col] or "")
        if not body:
            continue
        key = " ".join(body.lower().split())   # collapse whitespace for comparison
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(body)

    n_cleaned = len(cleaned)
    print(f"After cleaning / dedup: {n_cleaned:,}")

    # Tokenize, length-filter, truncate, decode back to text
    print(f"Loading tokenizer ({MODEL_NAME}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-Neo has no dedicated pad token

    print(f"Tokenizing (keep {MIN_SEQ_LEN}–{MAX_SEQ_LEN} tokens) ...")
    filtered: list[tuple[str, int]] = []
    for body in tqdm(cleaned, desc="tokenize"):
        ids = tokenizer.encode(body, add_special_tokens=False)
        if len(ids) < MIN_SEQ_LEN:
            continue
        ids = ids[:MAX_SEQ_LEN]
        text_trunc = tokenizer.decode(ids, skip_special_tokens=True)
        filtered.append((text_trunc, len(ids)))

    n_filtered = len(filtered)
    print(f"After length filter (≥{MIN_SEQ_LEN} tokens): {n_filtered:,}")

    if n_filtered < POOL_SIZE:
        raise ValueError(
            f"Only {n_filtered:,} sequences survived cleaning — need {POOL_SIZE:,}.\n"
            "Lower MIN_SEQ_LEN in config.py or source a larger dataset."
        )

    # Shuffle (fixed seed) and subsample to exactly POOL_SIZE
    rng = random.Random(42)
    rng.shuffle(filtered)
    pool = filtered[:POOL_SIZE]

    # Write output
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing {POOL_SIZE:,} records → {POOL_FILE} ...")
    with POOL_FILE.open("w", encoding="utf-8") as fh:
        for idx, (text, n_tok) in enumerate(pool):
            fh.write(json.dumps({"id": idx, "text": text, "n_tokens": n_tok}) + "\n")

    # Summary statistics
    lengths = [n for _, n in pool]
    print(
        "\n── Summary ──────────────────────────────────────────────────\n"
        f"  Raw emails loaded       : {n_raw:>10,}\n"
        f"  After cleaning / dedup  : {n_cleaned:>10,}\n"
        f"  After length filter     : {n_filtered:>10,}\n"
        f"  Final pool size         : {len(pool):>10,}\n"
        f"  Token length  mean      : {statistics.mean(lengths):>10.1f}\n"
        f"  Token length  std       : {statistics.stdev(lengths):>10.1f}\n"
        f"  Token length  min       : {min(lengths):>10,}\n"
        f"  Token length  max       : {max(lengths):>10,}\n"
        f"  Token length  median    : {statistics.median(lengths):>10.1f}\n"
        "─────────────────────────────────────────────────────────────"
    )


if __name__ == "__main__":
    main()
