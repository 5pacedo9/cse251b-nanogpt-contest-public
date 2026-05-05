"""Shared helpers for E16 data-source tokenizers.

Every source script (wikipedia.py / scientific.py / books.py) imports from here
so the tokenization, shard format, and HF cache location stay byte-compatible
with build-nanogpt/fineweb.py:

  - GPT-2 BPE via tiktoken
  - one EOT (50256) prepended per document
  - uint16 .npy shards, fixed shard_size (default 100M tokens)
  - HF cache pinned to <repo>/.hf_cache so nothing lands on C:

Sibling-import idiom (since "build-nanogpt" has a hyphen and is not a package):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _common import setup_env, encode_doc, ShardWriter
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import tiktoken

REPO_ROOT = Path(__file__).resolve().parents[2]
HF_CACHE = REPO_ROOT / ".hf_cache"
SHARDS_ROOT = REPO_ROOT / "build-nanogpt" / "shards"
SHARD_SIZE = 100_000_000  # 100M tokens, matches fineweb.py


def setup_env() -> None:
    """Pin HF cache to F: drive (<repo>/.hf_cache). Idempotent — only sets
    vars that the user has not explicitly overridden."""
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE / "datasets"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    hf_home = os.environ["HF_HOME"]
    drive = Path(hf_home).drive.upper()
    if drive == "C:":
        raise RuntimeError(
            f"HF_HOME={hf_home} is on C:. Refusing to run — C: has too little "
            f"free space for HF datasets. Unset HF_HOME or point it under {REPO_ROOT}."
        )


_ENC = None
_EOT = None


def _enc():
    global _ENC, _EOT
    if _ENC is None:
        _ENC = tiktoken.get_encoding("gpt2")
        _EOT = _ENC._special_tokens["<|endoftext|>"]  # 50256
    return _ENC


def encode_doc(text: str) -> np.ndarray:
    """Tokenize one document. Returns uint16 array, EOT-prefixed.

    Matches fineweb.py: the EOT goes BEFORE the document body so a document
    boundary always lines up on a 50256 token at training time.
    """
    enc = _enc()
    ids = [_EOT] + enc.encode_ordinary(text)
    arr = np.asarray(ids, dtype=np.int64)  # int64 to safely range-check
    if not ((arr >= 0).all() and (arr < 2**16).all()):
        raise ValueError("token id outside uint16 range")
    return arr.astype(np.uint16)


class ShardWriter:
    """Buffered writer that emits fixed-size .npy shards.

    Each shard is exactly ``shard_size`` tokens (last shard may be shorter).
    File names: ``<source>_<6-digit idx>.npy`` (e.g. wiki_000000.npy).
    """

    def __init__(self, out_dir: Path, source: str, shard_size: int = SHARD_SIZE) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.shard_size = shard_size
        self.buf = np.empty(shard_size, dtype=np.uint16)
        self.pos = 0
        self.idx = self._resume_idx()
        self.total_written = 0

    def _resume_idx(self) -> int:
        existing = sorted(self.out_dir.glob(f"{self.source}_*.npy"))
        return len(existing)  # next free index

    def add(self, tokens: np.ndarray) -> None:
        """Append a token array. Flushes whenever a shard fills."""
        n = len(tokens)
        if n == 0:
            return
        # fast path: fits in current buffer with room to spare
        if self.pos + n < self.shard_size:
            self.buf[self.pos : self.pos + n] = tokens
            self.pos += n
            return
        # spill across one or more shard boundaries
        offset = 0
        while offset < n:
            room = self.shard_size - self.pos
            take = min(room, n - offset)
            self.buf[self.pos : self.pos + take] = tokens[offset : offset + take]
            self.pos += take
            offset += take
            if self.pos >= self.shard_size:
                self._flush()

    def _flush(self) -> None:
        path = self.out_dir / f"{self.source}_{self.idx:06d}.npy"
        np.save(path, self.buf[: self.pos])
        print(f"\nwrote {path} ({self.pos:,} tokens)", flush=True)
        self.idx += 1
        self.total_written += self.pos
        self.pos = 0

    def close(self) -> None:
        if self.pos > 0:
            self._flush()


def existing_token_count(out_dir: Path, source: str) -> int:
    """Sum tokens already on disk for a source. Used to skip re-tokenization."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return 0
    total = 0
    for p in sorted(out_dir.glob(f"{source}_*.npy")):
        # mmap to avoid loading; only the header is read for shape
        total += int(np.load(p, mmap_mode="r").shape[0])
    return total
