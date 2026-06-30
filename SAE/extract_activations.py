import argparse
import ast
import pickle

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os


def parse_layers(layer_str):
    """
    Example:
        "[10,15,28]" -> [10, 15, 28]
    """
    if isinstance(layer_str, list):
        return layer_str

    return list(ast.literal_eval(layer_str))


def extract_mlp_activations(
    model,
    tokenizer,
    prompt,
    layers,
    device,
):
    """
    Extract AFTER-MLP activations for selected layers.

    Returns:
        dict[layer_idx] -> tensor(seq_len, hidden_size)
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    activations = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, module_input, module_output):
            # module_output:
            # [batch, seq_len, hidden_size]

            acts = module_output.detach()[0].float().cpu()

            # remove batch dimension
            # -> [seq_len, hidden_size]

            activations[layer_idx] = acts

        return hook

    for layer_idx in layers:
        h = model.model.layers[layer_idx].mlp.register_forward_hook(
            make_hook(layer_idx)
        )
        hooks.append(h)

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return activations


def main():
    parser = argparse.ArgumentParser(
        description="Extract MLP activations from a Hugging Face causal LM."
    )

    parser.add_argument(
        "--input_pkl",
        type=str,
        required=True,
        help="Path to input pickle file."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="HuggingFaceTB/SmolLM3-3B",
        help="Hugging Face model name."
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output pickle path."
    )

    parser.add_argument(
        "--layers",
        type=str,
        required=True,
        help='Layers list, e.g. "[10,15,28]"'
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1
    )

    args = parser.parse_args()

    layers = parse_layers(args.layers)

    print(f"Loading dataframe: {args.input_pkl}")
    df = pd.read_pickle(args.input_pkl)

    required_cols = ["geonameid", "lat", "lon", "prompt"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    print(f"Loading model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16
        #torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to("cuda")

    model.eval()

    device = next(model.parameters()).device

    rows = {layer: [] for layer in layers}
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        geonameid = row["geonameid"]
        prompt = row["prompt"]
        try:
            acts_dict = extract_mlp_activations(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                layers=layers,
                device=device,
            )
            for layer_idx, acts in acts_dict.items():
                # acts: [seq_len, hidden_size]
                mean_pool = acts.mean(dim=0).numpy()
                max_pool = acts.max(dim=0).values.numpy()
                rows[layer_idx].append({
                    "geonameid": geonameid,
                    "prompt": prompt,
                    "layer": layer_idx,
                    "activations": acts.numpy(),
                    "mean_pooling": mean_pool,
                    "max_pooling": max_pool,
                })
        except Exception as e:
            print(f"Error processing geonameid={geonameid}: {e}")

    output_dir = os.path.dirname(args.output_path)
    for layer_idx, layer_rows in rows.items():
        out_df = pd.DataFrame(layer_rows)
        out_path = os.path.join(output_dir, f"mlp_act_world_10k_l{layer_idx}.pkl")
        print(f"Saving layer {layer_idx} to: {out_path}")
        out_df.to_pickle(out_path)

    print("Done.")


if __name__ == "__main__":
    main()