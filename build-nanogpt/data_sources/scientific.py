"""Tokenize scientific papers into uint16 npy shards.

Source: armanc/scientific_papers, config 'arxiv' (HF, public, no auth).
Each row has 'article' (full paper body) and 'abstract'. We use 'article'.
Output: build-nanogpt/shards/scientific/sci_*.npy.

Run:
    python build-nanogpt/data_sources/scientific.py --target_tokens 1_500_000_000

Default 1.5B tokens covers the worst-case E16 full-tier need (15% × 10B = 1.5B).
Restart-safe (resumes at next shard index). Delete output dir to start fresh.
"""

from __future__ import annotations

import argparse
import sys
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

DATASET_ID = "armanc/scientific_papers"
DATASET_CONFIG = "arxiv"
TEXT_FIELD = "article"
SOURCE_TAG = "sci"


def run(target_tokens: int, shard_size: int) -> None:
    setup_env()
    from datasets import load_dataset
    from tqdm import tqdm

    out_dir = SHARDS_ROOT / "scientific"
    already = existing_token_count(out_dir, SOURCE_TAG)
    if already >= target_tokens:
        print(f"[sci] already have {already:,} tokens on disk (>= {target_tokens:,}); nothing to do.")
        return

    print(f"[sci] streaming {DATASET_ID} ({DATASET_CONFIG}), field='{TEXT_FIELD}'")
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", streaming=True,
                      trust_remote_code=True)

    writer = ShardWriter(out_dir, source=SOURCE_TAG, shard_size=shard_size)
    remaining = target_tokens - already
    pbar = tqdm(total=remaining, unit="tok", unit_scale=True, desc="sci")
    written = 0
    for doc in ds:
        text = doc.get(TEXT_FIELD)
        if not text:
            continue
        toks = encode_doc(text)
        writer.add(toks)
        written += len(toks)
        pbar.update(len(toks))
        if written >= remaining:
            break
    writer.close()
    pbar.close()
    print(f"[sci] done: wrote {written:,} new tokens into {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target_tokens", type=int, default=1_500_000_000)
    p.add_argument("--shard_size", type=int, default=SHARD_SIZE)
    args = p.parse_args()
    run(args.target_tokens, args.shard_size)


if __name__ == "__main__":
    main()
