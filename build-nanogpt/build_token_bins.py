"""
Build train/val token binaries from token shard files.

This script is intended to work with tokenized shard datasets such as the
`.npy` files produced by `fineweb.py`, but it is also configurable enough to
work with other shard layouts.

Typical usage:
    python build_token_bins.py

By default it reads `.npy` shards from `edu_fineweb10B/` and writes:
    edu_fineweb10B/train.bin
    edu_fineweb10B/val.bin

Configuration is intentionally kept at the top of the file so it is easy to
adapt to a different dataset without touching the rest of the logic.
"""

from pathlib import Path
import random
import sys

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config: update these values for a different dataset.

# Directory containing token shards.
INPUT_DIR = Path(__file__).resolve().parent / "edu_fineweb10B"

# Which files in INPUT_DIR should be considered shard files.
INPUT_GLOB = "*.npy"

# Output location for the merged binaries.
OUTPUT_DIR = INPUT_DIR
TRAIN_OUTPUT_NAME = "train.bin"
VAL_OUTPUT_NAME = "val.bin"

# If filenames contain explicit split tags, these are used first.
TRAIN_PATTERNS = ("train",)
VAL_PATTERNS = ("val",)

# Fallback split if explicit split tags are not present.
# When auto-splitting, shards are assigned at the file level, not token level.
VAL_FRACTION = 0.01
SHUFFLE_BEFORE_SPLIT = False
RANDOM_SEED = 1337

# Expected array format.
EXPECTED_SUFFIX = ".npy"
EXPECTED_DTYPE = np.uint16


def list_input_files() -> list[Path]:
    files = sorted(INPUT_DIR.glob(INPUT_GLOB))
    files = [path for path in files if path.is_file() and path.suffix == EXPECTED_SUFFIX]
    if not files:
        raise FileNotFoundError(f"No input shard files found in {INPUT_DIR} matching {INPUT_GLOB}")
    return files


def matches_any_pattern(path: Path, patterns: tuple[str, ...]) -> bool:
    name = path.name.lower()
    return any(pattern.lower() in name for pattern in patterns)


def split_files(files: list[Path]) -> tuple[list[Path], list[Path]]:
    explicit_train = [path for path in files if matches_any_pattern(path, TRAIN_PATTERNS)]
    explicit_val = [path for path in files if matches_any_pattern(path, VAL_PATTERNS)]

    if explicit_train or explicit_val:
        if not explicit_train:
            raise ValueError("Found explicit val shards but no explicit train shards.")
        if not explicit_val:
            raise ValueError("Found explicit train shards but no explicit val shards.")
        return explicit_train, explicit_val

    files_for_split = list(files)
    if SHUFFLE_BEFORE_SPLIT:
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(files_for_split)

    val_count = max(1, int(round(len(files_for_split) * VAL_FRACTION)))
    if val_count >= len(files_for_split):
        raise ValueError("VAL_FRACTION leaves no train shards. Reduce VAL_FRACTION.")

    val_files = files_for_split[:val_count]
    train_files = files_for_split[val_count:]
    return sorted(train_files), sorted(val_files)


def load_shard(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.dtype != EXPECTED_DTYPE:
        raise TypeError(f"{path} has dtype {arr.dtype}, expected {EXPECTED_DTYPE}.")
    if arr.ndim != 1:
        raise ValueError(f"{path} has ndim={arr.ndim}, expected 1.")
    return arr


def count_tokens(files: list[Path]) -> int:
    total = 0
    for path in files:
        total += len(load_shard(path))
    return total


def write_bin(files: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fout:
        for path in tqdm(files, desc=f"Writing {output_path.name}", unit="shard"):
            arr = np.asarray(load_shard(path), dtype=EXPECTED_DTYPE)
            arr.tofile(fout)


def describe_split(name: str, files: list[Path]) -> None:
    token_count = count_tokens(files)
    print(f"{name}: {len(files)} shard(s), {token_count:,} tokens")


def main() -> int:
    files = list_input_files()
    train_files, val_files = split_files(files)

    print(f"Input dir: {INPUT_DIR}")
    print(f"Found {len(files)} shard(s)")
    describe_split("train", train_files)
    describe_split("val", val_files)

    train_out = OUTPUT_DIR / TRAIN_OUTPUT_NAME
    val_out = OUTPUT_DIR / VAL_OUTPUT_NAME

    write_bin(train_files, train_out)
    write_bin(val_files, val_out)

    print(f"Wrote {train_out}")
    print(f"Wrote {val_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
