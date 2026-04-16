"""
Extract named entities from corpus
=====================================================
Loads a shuffled subset of HuggingFaceFW/fineweb and detects geographical entities using NER
Two NER backends :
    transformers - HuggingFace pipeline on GPU/CPU
                              model: dslim/bert-large-NER  (recommended)
    spacy (default) - spaCy GPU

python 
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
import pickle

from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import pipeline
import time

N_STEPS = 2

# Load dataset subet (shuffled)
def load_corpus(dataset_name: str, n_docs: int, seed: int = 42) -> list[str]:
    print(f"[1/{N_STEPS}] Streaming {n_docs:,} shuffled docs from {dataset_name}...")
    ds = load_dataset(
        dataset_name,
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=n_docs*10)
    texts = []
    pbar = tqdm(total=n_docs, unit="doc", desc="English docs")
    for doc in ds:
        if doc["language"] == "en":
            texts.append(doc["text"])
            pbar.update(1)

        if len(texts) >= n_docs:
            break
    print(f"    {len(texts):,} documents loaded")
    return texts

# Device resolution
def resolve_device(device_arg: str) -> tuple[str, int]:
    """
    Returns (device_str, torch_device_index).
    device_str  : 'cuda', 'mps', or 'cpu'
    torch_index : integer device id for transformers pipeline (-1 = cpu)
    """

    if device_arg == "cpu":
        return "cpu", -1

    if device_arg.startswith("cuda:"):
        idx = int(device_arg.split(":")[1])
        if not torch.cuda.is_available():
            print("     CUDA not available — falling back to CPU")
            return "cpu", -1
        return "cuda", idx

    # "auto" — pick best available
    if torch.cuda.is_available():
        print(f"    CUDA detected: {torch.cuda.get_device_name(0)}"
              f"  ({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")
        return "cuda", 0
    try:
        if torch.backends.mps.is_available():
            print("     Apple MPS detected")
            return "mps", 0
    except AttributeError:
        pass
    print("     No GPU found — using CPU")
    return "cpu", -1



def _clean_entity(text: str) -> str:
    text = text.lower()                     # lower case
    text = re.sub(r"(?:'s|’s)$", "", text)  # possessive removal
    text = re.sub(r"\bthe\b", "", text)     # remove "the" as a word only
    text = text.strip(" .,;:\"'()[]")       # remove any leading and tailing caracters present in the ("")
    text = re.sub(r"\s+", " ", text)        # normalize spaces
    return text


_HF_LOC_LABELS = {"LOC", "GPE", "FAC", "B-LOC", "I-LOC", "B-GPE", "I-GPE"}

def _ner_transformers(
    texts: list[str],
    model_name: str,
    device_index: int,
    batch_size: int,
) -> Counter:

    device_label = f"cuda:{device_index}" if device_index >= 0 else "cpu"
    print(f"  Loading '{model_name}' on {device_label}...")

    ner = pipeline(
        "ner",
        model=model_name,
        aggregation_strategy="simple",   # merge sub-tokens → full spans
        device=device_index,
        batch_size=batch_size,
    )

    entity_counter: Counter = Counter()

    # Truncate texts to avoid OOM on very long documents
    MAX_CHARS = 4096
    clipped = [t[:MAX_CHARS] for t in texts]

    for result in tqdm(
        ner(clipped, batch_size=batch_size),
        total=len(clipped),
        unit="doc",
        desc="NER (transformers)",
    ):
        for ent in result:
            label = ent.get("entity_group") or ent.get("entity", "")
            if label not in _HF_LOC_LABELS:
                continue
            token = _clean_entity(ent["word"])
            if len(token) < 2:
                continue

            entity_counter[token] += 1

    return entity_counter



def _ner_spacy(
    texts: list[str],
    model_name: str,
    device_str: str,
    batch_size: int,
) -> Counter:
    try:
        import spacy
    except ImportError:
        print("spaCy not installed. Run: pip install spacy")
        sys.exit(1)

    if device_str == "cuda":
        activated = spacy.require_gpu()
        status = "GPU" if activated else "GPU requested but not available - using CPU"
        print(f"  spaCy device: {status}")
    else:
        print("  spaCy device: CPU")

    try:
        nlp = spacy.load(model_name, disable=["parser", "lemmatizer"])
    except OSError:
        print(f"  Model '{model_name}' not found. Run:")
        print(f"    python -m spacy download {model_name}")
        sys.exit(1)

    entity_counter: Counter = Counter()

    for doc in tqdm(
        nlp.pipe(texts, batch_size=batch_size),
        total=len(texts),
        unit="doc",
        desc="NER (spaCy)",
    ):
        for ent in doc.ents:
            if ent.label_ not in ("GPE", "LOC"):
                continue
            token = _clean_entity(ent.text)

            if len(token) < 2:
                continue

            entity_counter[token] += 1

    return entity_counter


# ---------------------------------------------------------------------------
# 3e. Public entry point
# ---------------------------------------------------------------------------
def extract_city_counts(
    texts: list[str],
    backend: str,
    model_name: str,
    device_str: str,
    device_index: int,
    batch_size: int,
):
    print(f"[2/{N_STEPS}] Running NER  backend={backend}  model={model_name}  "
          f"device={'cpu' if device_index < 0 else f'cuda:{device_index}'}...")

    if backend == "transformers":
        entity_counter = _ner_transformers(
            texts, model_name, device_index, batch_size
        )
    elif backend == "spacy":
        entity_counter = _ner_spacy(
            texts, model_name, device_str, batch_size
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose: transformers | spacy")

    print(f"    -> {sum(entity_counter.values()):,} entity mentions across "
          f"{len(entity_counter):,} distinct entities")

    return entity_counter



def parse_args():
    p = argparse.ArgumentParser(
        description="City & country mentions in fineweb-edu by continent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--n_docs",    type=int, default=5_000,
                   help="Number of documents to sample")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb", 
                   choices=["HuggingFaceFW/fineweb", "HuggingFaceFW/fineweb-edu"],
                   help="Dataset from huggingface name")
    p.add_argument("--ner", default="spacy", choices=["transformers", "spacy"],
        help=("NER backend. 'transformers' runs a HuggingFace model "
            "'spacy' uses spaCy pipeline "))
    p.add_argument("--model", default="en_core_web_trf",
        help=("Model name. transformers : dslim/bert-large-NER, "
            "Defaults : spacy : en_core_web_trf"))
    p.add_argument(
        "--device", default="auto",
        help=("Device to use. 'auto' picks best available GPU, "
            "'cpu' forces CPU, 'cuda:N' picks a specific GPU index"))
    p.add_argument("--batch_size", type=int, default=128,
                   help="Batch size fed to the NER model")
    p.add_argument("--seed",       type=int, default=42,
                   help="Shuffle seed")

    p.add_argument("--output_dir",     default="results_tmp/",
                   help="Dir to output HTML file")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    if not Path(args.output_dir).exists():
        print(f"Error: The output dir path '{args.output_dir}' does not exist.")
        sys.exit(1)

    dataset_str = args.dataset.replace("/","_").lower()

    # Resolve device
    print(f"[0/{N_STEPS}] Resolving compute device...")
    device_str, device_index = resolve_device(args.device)

    
    # Default models per backend
    model_name = args.model or (
        "dslim/bert-large-NER" if args.ner == "transformers"
        else "en_core_web_trf"
    )

    texts = load_corpus(args.dataset, args.n_docs, seed=args.seed)

    start_time = time.time()
    entity_counter = extract_city_counts(
                                 texts,
                                 backend=args.ner,
                                 model_name=model_name,
                                 device_str=device_str,
                                 device_index=device_index,
                                 batch_size=args.batch_size,
                             )
    exec_time = time.time() - start_time
    print(f"Total execution time on [{args.n_docs}] docs : {int(exec_time // 60)} m, {int(exec_time % 60)} s")

    output_path = f"{args.output_dir}/geo_entity_counter_{dataset_str}_ndocs_{args.n_docs}_{args.ner}.plk"

    # Save
    with open(output_path, "wb") as f:
        pickle.dump(entity_counter, f, protocol=pickle.HIGHEST_PROTOCOL)

    # # Load
    # with open("geo_entity_counter.pkl", "rb") as f:
    #     my_dict = pickle.load(f)

if __name__ == "__main__":
    main()