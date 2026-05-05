"""Stage 3: assemble data/<mix>/{train,val}.bin from per-source npy shards.

Reads a recipe from mix_recipes.py, plans how many tokens to take from each
source (= ratio × tier_total), then concatenates the first N tokens of each
source's shard pool into a single uint16 train.bin.

train.py samples random offsets from train.bin at training time, so order
within the bin does not matter — concat is fine and deterministic.

val.bin is ALWAYS the FineWeb shard 0 (the existing in-training internal val).
This keeps cross-mix internal val PPL comparable. The contest public val.bin
at the repo root is never touched.

Run examples:
    # build one specific mix
    python build-nanogpt/build_mix_bins.py --mix mix10_fw100

    # build every short-tier E16 mix at once
    python build-nanogpt/build_mix_bins.py --tier short

    # build everything (short + mid + full for every recipe)
    python build-nanogpt/build_mix_bins.py --all

    # dry run — print plan without writing bytes
    python build-nanogpt/build_mix_bins.py --mix mix10_fw80_wiki10_sci5_books5 --dry_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mix_recipes import (  # noqa: E402
    FINEWEB_VAL_SHARD,
    REPO_ROOT,
    RECIPES,
    SOURCE_DIRS,
    SOURCE_TRAIN_GLOB,
    TIER_TOKENS,
    all_recipe_names,
    get_recipe,
)

DATA_ROOT = REPO_ROOT / "data"
COPY_CHUNK = 64 * 1024 * 1024  # 64M tokens / 128 MiB per write — bounded RAM


def _shard_paths(source: str, *, required: bool = True) -> list[Path]:
    src_dir = SOURCE_DIRS[source]
    pattern = SOURCE_TRAIN_GLOB[source]
    paths = sorted(src_dir.glob(pattern)) if src_dir.exists() else []
    if not paths and required:
        raise FileNotFoundError(
            f"no shards for source '{source}' under {src_dir} (pattern {pattern}). "
            f"Run the corresponding tokenizer in data_sources/ first."
        )
    return paths


def _shard_token_count(path: Path) -> int:
    return int(np.load(path, mmap_mode="r").shape[0])


def _available_tokens(source: str) -> int:
    """Tokens currently on disk for a source. Returns 0 if not tokenized yet."""
    return sum(_shard_token_count(p) for p in _shard_paths(source, required=False))


def _stream_tokens(paths: Iterable[Path], n_target: int):
    """Yield (np.uint16 array) chunks summing to n_target tokens, walking shards in order."""
    remaining = n_target
    for path in paths:
        if remaining <= 0:
            return
        arr = np.load(path, mmap_mode="r")  # uint16
        if arr.dtype != np.uint16:
            raise ValueError(f"{path} dtype {arr.dtype}, expected uint16")
        if len(arr) <= remaining:
            yield np.asarray(arr)
            remaining -= len(arr)
        else:
            yield np.asarray(arr[:remaining])
            remaining = 0
    if remaining > 0:
        raise RuntimeError(f"shards exhausted with {remaining:,} tokens still needed")


def plan(recipe: dict) -> dict:
    """Compute per-source token budget and check availability. No writes."""
    tier = recipe["tier"]
    total = TIER_TOKENS[tier]
    plan_per_source = {}
    for src, ratio in recipe["ratios"].items():
        want = int(round(total * ratio))
        have = _available_tokens(src)
        plan_per_source[src] = {"want": want, "have": have, "ratio": ratio}
    return {"name": recipe["name"], "tier": tier, "total": total, "sources": plan_per_source}


def _print_plan(p: dict) -> None:
    print(f"\n=== {p['name']}  (tier={p['tier']}, total={p['total']:,} tokens) ===")
    for src, info in p["sources"].items():
        ok = "OK" if info["have"] >= info["want"] else "SHORT"
        print(f"  {src:8s}  ratio={info['ratio']:.2f}  want={info['want']:>13,}  have={info['have']:>13,}  [{ok}]")


def build(mix_name: str, dry_run: bool = False) -> Path:
    recipe = get_recipe(mix_name)
    p = plan(recipe)
    _print_plan(p)

    short = [src for src, info in p["sources"].items() if info["have"] < info["want"]]

    if dry_run:
        if short:
            print(f"[dry_run] {mix_name} would FAIL -- missing tokens: {short}")
        else:
            print(f"[dry_run] would write data/{mix_name}/train.bin and val.bin")
        return DATA_ROOT / mix_name

    if short:
        raise RuntimeError(
            f"insufficient tokens for {mix_name}: {short}. "
            f"Re-run the corresponding data_sources/*.py with a larger --target_tokens."
        )

    out_dir = DATA_ROOT / mix_name
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    # train.bin: concat the first N tokens of each source in recipe order.
    sources_in_order = list(recipe["ratios"].keys())
    written = 0
    with train_path.open("wb") as fout:
        for src in sources_in_order:
            need = p["sources"][src]["want"]
            paths = _shard_paths(src)
            for chunk in _stream_tokens(paths, need):
                # write in bounded chunks so memmap'd arrays don't materialize all at once
                for i in range(0, len(chunk), COPY_CHUNK):
                    sub = chunk[i : i + COPY_CHUNK]
                    np.asarray(sub, dtype=np.uint16).tofile(fout)
                    written += len(sub)
            print(f"  [{src}] {need:,} tokens appended (running total {written:,})", flush=True)

    # val.bin: always FineWeb shard 0, copied as uint16.
    if not FINEWEB_VAL_SHARD.exists():
        raise FileNotFoundError(
            f"FineWeb val shard {FINEWEB_VAL_SHARD} missing — run build-nanogpt/fineweb.py first."
        )
    val_arr = np.asarray(np.load(FINEWEB_VAL_SHARD, mmap_mode="r"), dtype=np.uint16)
    val_arr.tofile(val_path)

    print(f"[{mix_name}] wrote {train_path} ({written:,} tokens) + {val_path} ({len(val_arr):,} tokens)")
    return out_dir


def _names_for_filter(args) -> list[str]:
    if args.mix:
        return [args.mix]
    if args.all:
        return all_recipe_names()
    if args.tier:
        return [n for n, r in RECIPES.items() if r["tier"] == args.tier]
    raise SystemExit("specify one of --mix / --tier / --all")


def main() -> None:
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--mix", help="single recipe name, e.g. mix10_fw100")
    g.add_argument("--tier", choices=["short", "mid", "full"], help="all recipes in this tier")
    g.add_argument("--all", action="store_true", help="every recipe")
    parser.add_argument("--dry_run", action="store_true",
                        help="print plan and availability check, do not write")
    parser.add_argument("--list", action="store_true", help="list known recipes and exit")
    args = parser.parse_args()

    if args.list:
        for n in all_recipe_names():
            r = RECIPES[n]
            print(f"  {n:42s}  tier={r['tier']:5s}  ratios={r['ratios']}")
        return

    names = _names_for_filter(args)
    print(f"building {len(names)} mix(es): {names}")
    for name in names:
        build(name, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
