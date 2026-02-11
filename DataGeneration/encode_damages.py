'''This script will basically get all the damage texts from pickles saved in a folder,
check how they line up if they are changing or not and pass them through embedding gemma so that we wont have to 
do that later during the training phase. this will make training faster'''

import pdb
import pickle, copy
import os, sys
import glob
import torch
from typing import List, Any
import argparse
import hashlib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DamageEmbeeding import all_text_from_vehicle_sensors, mid_text_from_vehicle_sensors, batch_encode_gemma, load_embeedinggemma
import json, tempfile
import shutil
import numpy as np
from tqdm import tqdm
import gc

#Generate a deterministic SHA-1 hash of any JSON-like object(ignores key order)
def stable_hash(obj):
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

#Checks the deep equality between two nested dicts using stable hashes
def deep_equal(a, b) -> bool:
    return stable_hash(a) == stable_hash(b)

#Binary Search to find the last lindx in seq where elements equal key_obj
def bsearch_last_equal(seq: List[Any], start: int, key_obj: Any) -> int:
    """
    Given a list 'seq' and a 'key_obj', return the last index >= start
    where seq[i] == key_obj (deep equality). Assumes a contiguous 'run'
    of equal items starting at 'start'. Uses binary search to find the end
    of that run in O(log n).
    """
    lo = start
    hi = len(seq) - 1
    ans = start
    key_hash = stable_hash(key_obj)
    def is_equal(i: int) -> bool:
        return stable_hash(seq[i]) == key_hash

    # standard upper-bound style bsearch on the equality predicate
    while lo <= hi:
        mid = (lo + hi) // 2
        if is_equal(mid):
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ans

# Partition list into contiguous runs of identical JSONs using binary search.
def compute_runs(payloads):
    """
    Partition the list into maximal runs of identical JSONs (deep equality).
    Returns list of (start_idx, end_idx, sha1_hash).
    Uses a binary search boundary finder for each run start.
    """
    runs = []
    i = 0
    n = len(payloads)
    while i < n:
        end = bsearch_last_equal(payloads, i, payloads[i])
        h = stable_hash(payloads[i])
        runs.append((i, end, h))
        i = end + 1
    return runs

# Remove duplicate strings and return unique texts plus index mappings.
def dedupe_texts(texts):
    """
    Deduplicate identical text strings to avoid redundant embedding work.
    Returns:
      unique_texts: ndarray of shape (U,) dtype=object
      index_map   : list of length N mapping each original index -> unique index
      inv_map     : dict text->unique_index
    """
    inv = {}
    uniq = []
    index_map = []
    for t in texts:
        if t in inv:
            index_map.append(inv[t])
        else:
            u = len(uniq)
            inv[t] = u
            uniq.append(t)
            index_map.append(u)
    return np.array(uniq, dtype=object), index_map, inv

# Reconstruct full embedding array from unique embeddings using index mapping.
def expand_embeddings(uniq_embs, idx_map):
    D = uniq_embs.shape[1]
    out = np.zeros((len(idx_map), D), dtype=uniq_embs.dtype)
    for i, u in enumerate(idx_map):
        out[i] = uniq_embs[u]
    return out

# Safely overwrite pickle file via temp file to prevent corruption
def atomic_pickle_dump(obj: Any, path: str) -> None:
    """
    Safer in-place write: write to temp file, then replace.
    """
    d = os.path.dirname(path)
    os.makedirs(d or ".", exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=d or ".", delete=False) as tf:
        tmp = tf.name
        pickle.dump(obj, tf, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.move(tmp, path)

# Process a single pickle: detect equal JSONs, parse texts, embed, and update file.
def proces_pickle(path, device, overwrite, topk, keep_cosmetic):
    with open(path, "rb") as f:
        data = pickle.load(f)

    payloads = data["damage_text"]
    
    runs = compute_runs(payloads)
    all_equal = (len(runs) == 1)

    texts_all = []
    texts_mid = []
    texts_none = []
    for pl in tqdm(payloads, desc=f"[{os.path.basename(path)}] Parsing "):
        # functions from your DamageEmbeeding.py
        t_all = all_text_from_vehicle_sensors(pl, K=topk, keep_cosmetic=keep_cosmetic)
        t_mid = mid_text_from_vehicle_sensors(pl, K=topk, keep_cosmetic=keep_cosmetic)
        t_none = json.dumps(pl, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        texts_all.append(t_all)
        texts_mid.append(t_mid)
        texts_none.append(t_none)
    
    # Deduplicate per view to minimize encoding
    uniq_all, map_all, _ = dedupe_texts(texts_all)
    uniq_mid, map_mid, _ = dedupe_texts(texts_mid)
    uniq_none, map_none, _ = dedupe_texts(texts_none)

    torch.cuda.empty_cache()
    model, device = load_embeedinggemma(device)

    # Encode each unique set
    embs_all_u  = batch_encode_gemma(list(uniq_all),  model, device)
    embs_mid_u  = batch_encode_gemma(list(uniq_mid),  model, device)
    embs_none_u = batch_encode_gemma(list(uniq_none), model, device)

    # Expand back to per-item embeddings
    embs_all  = expand_embeddings(embs_all_u,  map_all)
    embs_mid  = expand_embeddings(embs_mid_u,  map_mid)
    embs_none = expand_embeddings(embs_none_u, map_none)

    # Decide write or skip
    if (not overwrite and
        all(k in data for k in ("embedding_all", "embedding_mid", "embedding_none"))):
        print(f"[SKIP] {os.path.basename(path)}: embeddings exist (use --overwrite to refresh)")
        return

    # Attach outputs
    data["embedding_all"]  = embs_all
    data["embedding_mid"]  = embs_mid
    data["embedding_none"] = embs_none
    data["text_uniq"] = uniq_all
    data["damage_text_runs"] = runs
    data["damage_text_all_equal"] = all_equal

    # Save atomically
    atomic_pickle_dump(data, path)
    print(f"[OK]   {os.path.basename(path)}: "
          f"N={len(payloads)}, D={embs_all.shape[1]}, runs={len(runs)}, all_equal={all_equal}")


def main(args):
    folder_path = args.folder
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")
    
    pickle_files = glob.glob(os.path.join(folder_path, args.pattern))
    if not pickle_files:
        print(f"No .pickle files found")

    loaded_data = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for file_path in sorted(pickle_files):
        proces_pickle(
            path=file_path,
            device=device,
            overwrite=args.overwrite,
            topk=args.topk,
            keep_cosmetic=args.keep_cosmetic,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ =="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=str, default="folder holding all the pickles")
    ap.add_argument("--pattern", type=str, default="*.pkl")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite embeddings if already presnet")
    ap.add_argument("--topk", type=int, default=48, help="Top-K records to keep in parsed text (all/mid)")
    ap.add_argument("--keep-cosmetic", action="store_true",
                    help="Keep cosmetic parts in parsing (defaults False)")
    args = ap.parse_args()

    main(args)