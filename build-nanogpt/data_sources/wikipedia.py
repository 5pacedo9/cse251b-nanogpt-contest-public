"""Tokenize English Wikipedia into uint16 npy shards.

Source: wikimedia/wikipedia, config 20231101.en (HF, public, no auth).
Output: build-nanogpt/shards/wikipedia/wiki_*.npy (100M tokens each).

Run:
    python build-nanogpt/data_sources/wikipedia.py --target_tokens 2_000_000_000

Default 2.0B tokens covers the worst-case E16 full-tier need (20% × 10B = 2B).
The script is restart-safe: existing shards on disk are kept and the writer
resumes at the next index. To start fresh, delete the output directory.
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

DATASET_ID = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.en"
SOURCE_TAG = "wiki"


def run(target_tokens: int, shard_size: int) -> None:
    setup_env()
    from datasets import load_dataset  # imported after env is set
    from tqdm import tqdm

    out_dir = SHARDS_ROOT / "wikipedia"
    already = existing_token_count(out_dir, SOURCE_TAG)
    if already >= target_tokens:
        print(f"[wiki] already have {already:,} tokens on disk (>= {target_tokens:,}); nothing to do.")
        return

    print(f"[wiki] streaming {DATASET_ID} ({DATASET_CONFIG})")
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", streaming=True)

    writer = ShardWriter(out_dir, source=SOURCE_TAG, shard_size=shard_size)
    remaining = target_tokens - already
    pbar = tqdm(total=remaining, unit="tok", unit_scale=True, desc="wiki")
    written = 0
    for doc in ds:
        toks = encode_doc(doc["text"])
        writer.add(toks)
        written += len(toks)
        pbar.update(len(toks))
        if written >= remaining:
            break
    writer.close()
    pbar.close()
    print(f"[wiki] done: wrote {written:,} new tokens into {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target_tokens", type=int, default=2_000_000_000)
    p.add_argument("--shard_size", type=int, default=SHARD_SIZE)
    args = p.parse_args()
    run(args.target_tokens, args.shard_size)


if __name__ == "__main__":
    main()
