#!/usr/bin/env python3

import argparse
import ast

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("  CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("  CUDA not available. Using CPU.")


def parse_layers(layer_str):
    layers = []
    for layer in layer_str.split(","):
        layers.append(int(layer))
    return layers

def load_model(model_name="HuggingFaceTB/SmolLM3-3B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
        dtype=torch.float16).to(device)

    return model, tokenizer

def make_hook(layer_name):
    def hook(module, inputs, output):
        activations[layer_name] = output.detach()
    return hook

def extract_activations(geoid, text, layers, model, tokenizer):
    global activations
    activations = {}

    hooks = []
    for layer_number in layers:
        # mlp / post_attention_layernorm
        hook = model.model.layers[layer_number].mlp.register_forward_hook(make_hook(f'layer_{layer_number}_output'))
        hooks.append(hook)

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)

    for hook in hooks:
        hook.remove()
    
    results = []
    for layer_number in layers:
        activation_output = activations.get(f'layer_{layer_number}_output')

        # Compute mean and max pooling
        if activation_output is not None:
            activation_output = activation_output.to('cpu')
            mean_pooling = torch.mean(activation_output, dim=1).numpy()
            max_pooling = torch.max(activation_output, dim=1).values.numpy()
        else:
            mean_pooling = None
            max_pooling = None

        # Append results for this layer
        results.append({
            'geonameid': geoid,
            'prompt': text,
            'layer': layer_number,
            'activations': activation_output.numpy() if activation_output is not None else None,
            'mean_pooling': mean_pooling,
            'max_pooling': max_pooling
        })

    return results

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_pkl", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM3-3B")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--layers", type=str, required=True)

    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name)

    layers = parse_layers(args.layers)

    df = pd.read_pickle(args.input_pkl)
    texts = df["prompt"].tolist()
    identifiers = df['geonameid'].tolist()

    df = pd.DataFrame(columns=['geonameid','prompt', 'layer', 'activations', 'mean_pooling', 'max_pooling'])

    for ids,text in tqdm(zip(identifiers,texts), desc="Texts", total=len(texts)):
        results = extract_activations(ids,text, layers, model, tokenizer)
        sub = pd.DataFrame(results)
        df = pd.concat([df, sub])
        
    df.to_pickle(args.output_path)

    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()