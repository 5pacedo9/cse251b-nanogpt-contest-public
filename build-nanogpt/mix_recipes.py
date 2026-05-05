"""Declarative E16 data-mix recipes.

Single source of truth for what each `data/<mix_name>/` directory contains.
Both build_mix_bins.py and downstream training scripts consume this.

Token budgets per tier match recommended_exp.md §2:
    short ~1B  (10 FineWeb shards)
    mid   ~5B  (50 FineWeb shards)
    full  ~10B (all 99 FineWeb shards)

Each source's shard pool is a directory of fixed-size .npy files produced by
the corresponding tokenizer in data_sources/. Recipe assembly takes the first
N tokens from the concatenated shard order — `train.py` samples at random
offsets from the merged train.bin so within-file ordering is irrelevant.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TIER_TOKENS = {
    "short": 1_000_000_000,
    "mid":   5_000_000_000,
    "full":  10_000_000_000,
}

# Where each source's tokenized .npy shards live.
SOURCE_DIRS = {
    "fineweb": REPO_ROOT / "build-nanogpt" / "edu_fineweb10B",
    "wiki":    REPO_ROOT / "build-nanogpt" / "shards" / "wikipedia",
    "sci":     REPO_ROOT / "build-nanogpt" / "shards" / "scientific",
    "books":   REPO_ROOT / "build-nanogpt" / "shards" / "books",
}

# Glob pattern for each source's TRAIN shards (excludes any val shard).
SOURCE_TRAIN_GLOB = {
    "fineweb": "edufineweb_train_*.npy",
    "wiki":    "wiki_*.npy",
    "sci":     "sci_*.npy",
    "books":   "books_*.npy",
}

# FineWeb shard 0 doubles as the training-time internal val (estimate_loss).
# We reuse it across all mixes so cross-mix internal val PPL stays comparable.
# This is NOT the contest public val.bin (which lives at repo root and is
# never touched).
FINEWEB_VAL_SHARD = SOURCE_DIRS["fineweb"] / "edufineweb_val_000000.npy"


def _short(name: str, ratios: dict) -> dict:
    return {"tier": "short", "ratios": ratios, "name": name}


def _mid(name: str, ratios: dict) -> dict:
    return {"tier": "mid", "ratios": ratios, "name": name}


def _full(name: str, ratios: dict) -> dict:
    return {"tier": "full", "ratios": ratios, "name": name}


# ---- E16 mixes (recommended_exp.md §8.3) -----------------------------------

# Short-tier candidates: control + 3 mixes, all at ~1B tokens.
SHORT_MIXES = {
    "mix10_fw100":                     _short("mix10_fw100", {"fineweb": 1.00}),
    "mix10_fw90_wiki10":               _short("mix10_fw90_wiki10", {"fineweb": 0.90, "wiki": 0.10}),
    "mix10_fw80_wiki10_sci5_books5":   _short("mix10_fw80_wiki10_sci5_books5",
                                              {"fineweb": 0.80, "wiki": 0.10, "sci": 0.05, "books": 0.05}),
    "mix10_fw50_wiki20_sci15_books15": _short("mix10_fw50_wiki20_sci15_books15",
                                              {"fineweb": 0.50, "wiki": 0.20, "sci": 0.15, "books": 0.15}),
}


def promote_to_mid(short_recipe: dict) -> dict:
    """Build the mid-tier (5B) version of a short-tier recipe."""
    base = short_recipe["name"].replace("mix10_", "mix50_", 1)
    return _mid(base, dict(short_recipe["ratios"]))


def promote_to_full(short_recipe: dict) -> dict:
    """Build the full-tier (~10B) version of a short-tier recipe."""
    base = short_recipe["name"].replace("mix10_", "mixfull_", 1)
    return _full(base, dict(short_recipe["ratios"]))


# All recipes, including auto-generated mid + full versions of every short mix.
RECIPES: dict[str, dict] = {}
RECIPES.update(SHORT_MIXES)
for _r in SHORT_MIXES.values():
    _mid_r = promote_to_mid(_r)
    _full_r = promote_to_full(_r)
    RECIPES[_mid_r["name"]] = _mid_r
    RECIPES[_full_r["name"]] = _full_r


def get_recipe(name: str) -> dict:
    if name not in RECIPES:
        raise KeyError(f"unknown mix '{name}'. Known: {sorted(RECIPES)}")
    r = RECIPES[name]
    total = sum(r["ratios"].values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"recipe '{name}' ratios sum to {total}, must sum to 1.0")
    return r


def all_recipe_names() -> list[str]:
    return sorted(RECIPES)
