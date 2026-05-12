"""Tokenize a stricter-filtered subset of FineWeb-Edu into uint16 npy shards.

E20 — data quality filtering. FineWeb-Edu sample-10BT is already filtered to
int_score >= 3. This script adds three additional filters on top:

  1. int_score >= 4              (very high quality only, ~50-70% retention)
  2. token_count >= 100          (drop very short docs, ~95%+ retention)
  3. exact text hash dedup       (SHA-256, drop full-duplicate docs)

Output: build-nanogpt/shards/fineweb_filtered/fwf_*.npy (100M tokens/shard).

Run:
    python build-nanogpt/data_sources/fineweb_filtered.py --target_tokens 6_000_000_000

Default 6.0B tokens covers short (1B) + mid (5B) E20 needs.
sample-10BT has ~10B raw tokens; after int_score >= 4 we expect ~5-7B kept.
If filter is too aggressive, lower --min_int_score to 3 (= sample-10BT default,
effectively just dedup) or stream from sample-100BT for more headroom.

Restart-safe (resumes at next shard index). Delete output dir to start fresh.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    SHARD_SIZE,
    SHARDS_ROOT,
    ShardWriter,
    encode_doc,
    existing_token_count,
    setup_env,
)

DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
TEXT_FIELD = "text"
SOURCE_TAG = "fwf"


def run(
    target_tokens: int,
    shard_size: int,
    min_int_score: int,
    min_token_count: int,
    dedup: bool,
) -> None:
    setup_env()
    from datasets import load_dataset
    from tqdm import tqdm

    out_dir = SHARDS_ROOT / "fineweb_filtered"
    already = existing_token_count(out_dir, SOURCE_TAG)
    if already >= target_tokens:
        print(f"[fwf] already have {already:,} tokens on disk (>= {target_tokens:,}); nothing to do.")
        return

    print(f"[fwf] streaming {DATASET_ID} ({DATASET_CONFIG})")
    print(f"[fwf] filters: int_score >= {min_int_score} | token_count >= {min_token_count} | dedup={dedup}")
    ds = load_dataset(DATASET_ID, name=DATASET_CONFIG, split="train", streaming=True)

    writer = ShardWriter(out_dir, source=SOURCE_TAG, shard_size=shard_size)
    remaining = target_tokens - already
    pbar = tqdm(total=remaining, unit="tok", unit_scale=True, desc="fwf")

    seen_hashes: set[str] = set()  # for exact dedup
    written = 0
    stats = {
        "total": 0,
        "kept": 0,
        "drop_score": 0,
        "drop_length": 0,
        "drop_dup": 0,
    }
    t_last = time.time()

    for doc in ds:
        stats["total"] += 1

        # Filter 1: classifier score
        score = doc.get("int_score", 0)
        if score is None or score < min_int_score:
            stats["drop_score"] += 1
            continue

        # Filter 2: token count (FineWeb-Edu provides this field)
        token_count = doc.get("token_count", 0) or 0
        if token_count < min_token_count:
            stats["drop_length"] += 1
            continue

        # Filter 3: exact text dedup
        if dedup:
            h = hashlib.sha256(doc[TEXT_FIELD].encode("utf-8")).hexdigest()
            if h in seen_hashes:
                stats["drop_dup"] += 1
                continue
            seen_hashes.add(h)

        # Passed all filters — tokenize and write
        toks = encode_doc(doc[TEXT_FIELD])
        writer.add(toks)
        written += len(toks)
        stats["kept"] += 1
        pbar.update(len(toks))

        # Periodic stats print every 60s
        now = time.time()
        if now - t_last > 60:
            ret_rate = stats["kept"] / max(stats["total"], 1)
            print(f"\n[fwf] {stats} | kept rate {ret_rate:.1%} | hash set size {len(seen_hashes):,}", flush=True)
            t_last = now

        if written >= remaining:
            break

    writer.close()
    pbar.close()
    ret_rate = stats["kept"] / max(stats["total"], 1)
    print(f"\n[fwf] FINAL stats: {stats}")
    print(f"[fwf] kept rate: {ret_rate:.2%}")
    print(f"[fwf] hash set size: {len(seen_hashes):,}")
    print(f"[fwf] wrote {written:,} new tokens into {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target_tokens", type=int, default=6_000_000_000,
                   help="stop after this many tokens (default 6B = short 1B + mid 5B)")
    p.add_argument("--shard_size", type=int, default=SHARD_SIZE)
    p.add_argument("--min_int_score", type=int, default=4,
                   help="FineWeb-Edu int_score threshold (0-5); sample-10BT default is 3, "
                        "we set 4 for stricter quality")
    p.add_argument("--min_token_count", type=int, default=100,
                   help="minimum GPT-2 token count per doc")
    p.add_argument("--no_dedup", action="store_true",
                   help="disable SHA-256 exact dedup (default: dedup on)")
    args = p.parse_args()
    run(
        target_tokens=args.target_tokens,
        shard_size=args.shard_size,
        min_int_score=args.min_int_score,
        min_token_count=args.min_token_count,
        dedup=not args.no_dedup,
    )


if __name__ == "__main__":
    main()
