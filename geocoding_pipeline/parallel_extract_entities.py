"""
Parallel NER extraction over HuggingFaceFW/fineweb-edu sample-10BT
===================================================================
Splits n_docs across N worker processes using datasets.shard().
Each worker runs the full NER pipeline on its slice and saves a
partial .pkl file. The parent then merges all partials into one.

Usage
-----
# 3 workers, 60 000 docs total (20 000 each)
python run_parallel_ner.py --n_docs 60000 --n_workers 3

# 3 GPUs (one per worker)
python run_parallel_ner.py --n_docs 60000 --n_workers 3 --multi_gpu
"""

import argparse
import multiprocessing as mp
import pickle
import sys
import time
from collections import Counter
from pathlib import Path


# Worker function
def _worker(
    worker_id: int,
    n_workers: int,
    n_docs_per_worker: int,
    dataset_name: str,
    model_name: str,
    device: str,
    batch_size: int,
    seed: int,
    output_dir: str,
):
    """
    Runs inside a subprocess. Loads shard `worker_id / n_workers`,
    takes the first n_docs_per_worker English docs, runs NER, saves partial pkl.
    """
    import re
    import sys
    from datasets import load_dataset
    from tqdm import tqdm

    print(f"[worker {worker_id}] starting  device={device}  "
          f"docs={n_docs_per_worker:,}")

    # Load the right shard
    ds = load_dataset(
        dataset_name,
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=100_000)
    # shard() on a streaming dataset interleaves: shard i takes every n_workers-th example
    ds = ds.shard(num_shards=n_workers, index=worker_id)

    rows = []
    for doc in ds:
        if doc.get("language", "en") == "en":
            rows.append({
                "text":        doc["text"],
                "token_count": doc.get("token_count", 0),
                "score":       doc.get("score", 0),
                "int_score":   doc.get("int_score", 0),
                "url":         doc.get("url", ""),
                "id":           doc["id"]
            })
        if len(rows) >= n_docs_per_worker:
            break

    print(f"[worker {worker_id}] loaded {len(rows):,} docs")

    # Run NER
    from create_geodb import clean_entity

    def _clean(text: str) -> str:
        return clean_entity(text)

    results = {}

    import spacy
    if "cuda" in device:
        activated = spacy.require_gpu()
        print(f"[worker {worker_id}] spaCy GPU: {activated}")
    nlp = spacy.load(model_name, disable=["parser", "lemmatizer"])
    texts = [r["text"] for r in rows]
    for doc, row in tqdm(
        zip(nlp.pipe(texts, batch_size=batch_size), rows),
        total=len(rows),
        desc=f"NER w{worker_id}",
        position=worker_id,
        leave=False,
    ):
        for ent in doc.ents:
            if ent.label_ not in ("GPE", "LOC"):
                continue
            key = _clean(ent.text)
            if len(key) < 2:
                continue
            if key not in results:
                results[key] = {"count": 0, "label": ent.label_, "docs_id": set()}
            results[key]["count"] += 1
            results[key]["docs_id"].add(row["id"])


    # Save partial result
    partial_path = Path(output_dir) / f"_partial_{worker_id}.pkl"
    with open(partial_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_mentions = sum(v["count"] for v in results.values())
    print(f"[worker {worker_id}] done — {n_mentions:,} mentions, "
          f"{len(results):,} entities → {partial_path}")
    return str(partial_path)


# Merge partials
def merge_partials(partial_paths: list[str]) -> dict:
    """
    Merge { entity: {count, label} } dicts from all workers.
    Counts are summed; label is taken from whichever worker saw it first.
    """
    merged: dict = {}
    for path in partial_paths:
        with open(path, "rb") as f:
            partial = pickle.load(f)
        for entity, data in partial.items():
            if entity not in merged:
                merged[entity] = {"count": 0, "label": data["label"], "docs_id": set()}
            merged[entity]["count"] += data["count"]
            merged[entity]["docs_id"].update(data.get("docs_id", set()))
        #Path(path).unlink()   # clean up partial file
    return merged


def parse_args():
    p = argparse.ArgumentParser(
        description="Parallel NER extraction over fineweb-edu",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n_docs",    type=int, default=60_000,
                   help="Total docs to process (split equally across workers)")
    p.add_argument("--n_workers", type=int, default=3,
                   help="Number of parallel worker processes")
    p.add_argument("--dataset",   default="HuggingFaceFW/fineweb-edu",
                   choices=["HuggingFaceFW/fineweb", "HuggingFaceFW/fineweb-edu"],
                   help="Dataset from HuggingFace")
    p.add_argument("--model",     default="en_core_web_trf",
                   help="spaCy model or HF model name")
    p.add_argument("--device",    default="auto",
                   help="'auto', 'cpu', 'cuda', 'cuda:N'")
    p.add_argument("--multi_gpu", action="store_true",
                   help="Give worker i device cuda:i (requires N GPUs)")
    p.add_argument("--batch_size",type=int, default=128,
                   help="Batch size for the NER")
    p.add_argument("--seed",      type=int, default=42,
                   help="Seed of the dataset shuffle")
    p.add_argument("--output_dir",default="results",
                   help="Directory for partial and final pkl files")
    p.add_argument("--output_path", default=None,
                   help="Override final output path (default: auto-named in output_dir)")
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_device = "cuda"
    n_docs_per_worker = args.n_docs // args.n_workers
    # Give any remainder to the last worker
    doc_counts = [n_docs_per_worker] * args.n_workers
    doc_counts[-1] += args.n_docs - n_docs_per_worker * args.n_workers

    print(f"Launching {args.n_workers} workers")
    print(f"  total docs : {args.n_docs:,}")
    print(f"  per worker : {doc_counts}")
    print(f"  model      : {args.model}")
    print(f"  multi_gpu  : {args.multi_gpu}")

    # Build per-worker kwargs
    worker_kwargs = []
    for i in range(args.n_workers):
        if args.multi_gpu and base_device.startswith("cuda"):
            device = f"cuda:{i}"
        else:
            device = base_device
        worker_kwargs.append(dict(
            worker_id=i,
            n_workers=args.n_workers,
            n_docs_per_worker=doc_counts[i],
            dataset_name=args.dataset,
            model_name=args.model,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
            output_dir=str(output_dir),
        ))

    # Launch workers
    # spawn is required for CUDA (fork doesn't work with GPU contexts)
    ctx = mp.get_context("spawn")
    t0 = time.time()

    with ctx.Pool(processes=args.n_workers) as pool:
        partial_paths = pool.starmap(
            _worker,
            [list(kw.values()) for kw in worker_kwargs],
        )

    elapsed = time.time() - t0
    print(f"\nAll workers done in {int(elapsed // 60)} m {int(elapsed % 60)} s")

    # Merge
    print("Merging partial results...")
    merged = merge_partials(partial_paths)

    total_mentions = sum(v["count"] for v in merged.values())
    print(f"Merged: {total_mentions:,} mentions across {len(merged):,} entities")

    # Save
    dataset_str = args.dataset.replace("/", "_").lower()
    if args.output_path:
        final_path = args.output_path
    else:
        final_path = (
            output_dir
            / f"geo_entities_spacy_nworker_{args.n_workers}_{dataset_str}_{args.n_docs}.pkl"
        )
    with open(final_path, "wb") as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved merged result -> {final_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()