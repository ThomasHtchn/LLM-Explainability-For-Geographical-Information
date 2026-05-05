import torch
import pandas as pd
import re
from collections import defaultdict
from geollm_utils import *
import argparse
import os
from tqdm import tqdm

def is_digit_token(token_str):
    token_str = token_str.strip()
    return re.fullmatch(r"[0-9]", token_str) is not None


def get_top_digit(probs, tokenizer, top_k=50):
    top_probs, top_ids = torch.topk(probs, top_k, dim=-1)

    for prob, tid in zip(top_probs[0], top_ids[0]):
        tok = tokenizer.decode([tid.item()]).strip()
        if is_digit_token(tok):
            return tok, prob.item()

    return None, None


def tokenize_prompt(prompt_user, tokenizer, model, is_chat=False, prompt_system="", add_gen_prompt=True, enable_thinking=False):
    if(is_chat):
        messages = [
            {"role": "system", "content":prompt_system},
            {"role": "user", "content": prompt_user}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_gen_prompt,
            enable_thinking=enable_thinking,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
    else:
        inputs = tokenizer(prompt_user, return_tensors="pt").to(model.device)

    return inputs


def analyze_geollm_first_token(
    prompts_path,
    model,
    tokenizer,
    task,
    output_dir,
    start_layer,
    end_layer,
    top_k,
    output_prefix,
):
    prompts = load_geollm_prompts(prompts_path, task)

    os.makedirs(output_dir, exist_ok=True)

    results_per_layer = {
        layer: {"lat": [], "lon": [], "digit": []}
        for layer in range(start_layer, end_layer + 1)
    }

    for prompt in tqdm(prompts, total=len(prompts)):
        lat, lon = get_coordinates(prompt)

        inputs = tokenize_prompt(prompt, tokenizer, model)
        input_ids = inputs["input_ids"]

        gen_outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=3,
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=True
        )

        step_hidden_states = gen_outputs.hidden_states[0]

        for layer_idx in range(start_layer, end_layer + 1):
            h = step_hidden_states[layer_idx]
            last_token_hidden = h[:, -1, :]

            logits = model.lm_head(last_token_hidden)
            probs = torch.softmax(logits, dim=-1)

            digit, prob = get_top_digit(probs, tokenizer, top_k=top_k)

            results_per_layer[layer_idx]["lat"].append(lat)
            results_per_layer[layer_idx]["lon"].append(lon)
            results_per_layer[layer_idx]["digit"].append(digit)

    # Save once per layer
    for layer_idx, data in results_per_layer.items():
        df = pd.DataFrame({
            "latitude": data["lat"],
            "longitude": data["lon"],
            "predicted_digit": data["digit"],
        })

        filepath = os.path.join(output_dir, f"{output_prefix}_{layer_idx}.csv")
        df.to_csv(filepath, index=False)
        print(f"Saved {filepath}")



def parse_args():
    p = argparse.ArgumentParser(
        description="LLM hidden layers Probing using GeoLLM benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--prompts_path",
                help="Path to geollm prompts (jsonl file)")
    p.add_argument("--task",    default="Population Density",
                help="GeoLLM task")
    p.add_argument("--model",    default="HuggingFaceTB/SmolLM3-3B",
                help="LLM model from huggingface used for probing")
    p.add_argument("--start_layer", type=int, default=1,
                help="starting layer")
    p.add_argument("--end_layer", type=int, default=36,
                help="end layer")
    p.add_argument("--top_k", default=10,
                help="Top k to look at")
    p.add_argument("--output_prefix", default="layer",
                help="Prefix of the per layer result files")
    p.add_argument("--output_dir", default="layers_output",
                   help="Path to the dir to save per layer predictions")
    p.add_argument("--end_layer", default="28",
                   help="number of layers")
    return p.parse_args()


def main():
    args = parse_args()

    model, tokenizer = load_local_model(args.model)

    OUTPUT_PREFIX = "layer"

    analyze_geollm_first_token(
        args.prompts_path,
        model,
        tokenizer,
        args.task,
        args.output_dir,
        start_layer=args.start_layer,
        end_layer=int(args.end_layer),
        top_k=args.top_k,
        output_prefix=args.output_prefix,
    )

if __name__ == "__main__":
    main()