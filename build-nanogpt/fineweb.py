"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import argparse
import multiprocessing as mp
from itertools import islice
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

enc = None
eot = None

def init_tokenizer():
    global enc, eot
    if enc is None:
        enc = tiktoken.get_encoding("gpt2")
        eot = enc._special_tokens['<|endoftext|>'] # end of text token

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    init_tokenizer()
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def parse_args():
    parser = argparse.ArgumentParser(description="Download and tokenize FineWeb-Edu into token shards.")
    parser.add_argument(
        "--nprocs",
        type=int,
        default=1 if os.name == "nt" else max(1, os.cpu_count() // 2),
        help="Number of tokenizer worker processes. Defaults to 1 on Windows for stability.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=16,
        help="Multiprocessing imap chunksize.",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="If set, only process the first N documents. Useful for smoke tests.",
    )
    return parser.parse_args()

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
if __name__ == "__main__":
    args = parse_args()
    init_tokenizer()
    print(f"Loading FineWeb-Edu split '{remote_name}' in streaming mode...")
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)
    print(f"Tokenizer workers: {args.nprocs}, chunksize: {args.chunksize}, shard_size: {shard_size:,}")
    if args.max_docs is not None:
        print(f"Processing only the first {args.max_docs} documents for this run.")
        fw = islice(fw, args.max_docs)

    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    token_iter = None
    if args.nprocs <= 1:
        token_iter = map(tokenize, fw)
    else:
        with mp.Pool(args.nprocs) as pool:
            token_iter = pool.imap(tokenize, fw, chunksize=args.chunksize)
            for tokens in token_iter:
                # is there enough space in the current shard for the new tokens?
                if token_count + len(tokens) < shard_size:
                    # simply append tokens to current shard
                    all_tokens_np[token_count:token_count+len(tokens)] = tokens
                    token_count += len(tokens)
                    # update progress bar
                    if progress_bar is None:
                        progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                    progress_bar.update(len(tokens))
                else:
                    # write the current shard and start a new one
                    split = "val" if shard_index == 0 else "train"
                    filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                    # split the document into whatever fits in this shard; the remainder goes to next one
                    remainder = shard_size - token_count
                    progress_bar.update(remainder)
                    all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                    write_datafile(filename, all_tokens_np)
                    print(f"\nWrote {filename}.npy")
                    shard_index += 1
                    progress_bar = None
                    # populate the next shard with the leftovers of the current doc
                    all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                    token_count = len(tokens)-remainder

    if args.nprocs <= 1:
        for tokens in token_iter:
            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                print(f"\nWrote {filename}.npy")
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
        print(f"\nWrote {filename}.npy")
