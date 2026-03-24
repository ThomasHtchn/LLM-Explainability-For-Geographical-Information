import argparse
#from geollm_scripts.geollm_utils import *
from geollm_utils import *
import os
import pandas as pd
import re
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams


def write_to_csv(latitudes, longitudes, predictions, file_path):
    df = pd.DataFrame({
        'Latitude': latitudes,
        'Longitude': longitudes,
        'Predictions': predictions
    })
    df.to_csv(file_path, index=False)

def write_to_csv_failed_preds(latitudes, longitudes, prompt, completion, file_path):
    df = pd.DataFrame({
        'Latitude': latitudes,
        'Longitude': longitudes,
        'Prompt' : prompt,
        'Completion': completion
    })
    df.to_csv(file_path, index=False)

def get_rating(completion):
    match = re.search(r"(\d+\.\d+)", completion)
    if not match:
        return None
    rating = float(match.group(0))
    return rating

def get_local_prediction(model, tokenizer, prompt, use_chat_template):
    if(use_chat_template):
        system_content, user_content = prompt.split("\n\n", 1)
        messages = [
            {"role": "system", "content" : system_content},
            {"role": "user", "content": user_content}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=20)

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
    completion = tokenizer.decode(output_ids, skip_special_tokens=True)
    rating = get_rating(completion)

    return completion, rating

def load_local_model(model_name, dtype=torch.bfloat16, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype 
    ).to(device)
    return model, tokenizer

def load_local_vllm(model_name, dtype="bfloat16", gpu_percent=0.2, max_len=4096):
    llm = LLM(
        model=model_name,
        dtype=dtype,
        gpu_memory_utilization=gpu_percent,
        max_model_len=max_len
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return llm, tokenizer


def get_local_vllm_pred(llm, tokenizer, prompt, use_chat_template=False, temp=0.6, top_p=0.95, max_tokens=20):
    if(use_chat_template):
        system_content, user_content = prompt.split("\n\n", 1)
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

    params = SamplingParams(temperature=temp, top_p=top_p, max_tokens=max_tokens)

    outputs = llm.generate(prompt, params)
    completion = outputs[0].outputs[0].text
    rating = get_rating(completion)

    return completion, rating


def run_task_for_data(model_name, prompt_file_path, task, use_vllm, use_chat_template):

    prompts = load_geollm_prompts(prompt_file_path, task)

    directory = "results"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    model_str = re.sub(r'[^a-zA-Z0-9_]', '_', model_name)
    task_name = re.sub(r'[^a-zA-Z0-9_]', '_', task)
    prompts_name = re.sub(r'[^a-zA-Z0-9_]', '_', prompt_file_path.split("/")[-1].split(".")[0])
    if(int(use_vllm) == 1):
        base_file_path = f"{directory}/vllm_{model_str}_{task_name}_{prompts_name}.csv"
        failed_path = f"{directory}/failed_preds_vllm_{model_str}_{task_name}_{prompts_name}.csv"
    else:
        base_file_path = f"{directory}/{model_str}_{task_name}_{prompts_name}.csv"
        failed_path = f"{directory}/failed_preds_{model_str}_{task_name}_{prompts_name}.csv"

    i = 0
    latitudes = []
    longitudes = []
    predicted = []

    failed_lats = []
    failed_lons = []
    failed_prompts = []
    failed_completions = []
    failed_count = 0

    start_load_model_time = time.time()
    if(int(use_vllm) == 1):
        print(" >> Loading model with vllm << ")
        local_model, tokenizer = load_local_vllm(model_name)
    else:
        local_model, tokenizer = load_local_model(model_name)
    load_model_time = time.time() - start_load_model_time

    start_time = time.time()

    for prompt in tqdm(prompts, total=len(prompts)):
        try:
            i += 1

            lat, lon = get_coordinates(prompt)

            if(int(use_vllm) == 1):
                completion, most_probable = get_local_vllm_pred(local_model, tokenizer, prompt, use_chat_template)
            else:
                completion, most_probable = get_local_prediction(local_model, tokenizer, prompt, use_chat_template)

            if most_probable is None:
                failed_lats.append(lat)
                failed_lons.append(lon)
                failed_prompts.append(prompt)
                failed_completions.append(completion)
                failed_count += 1
                continue

            latitudes.append(lat)
            longitudes.append(lon)
            predicted.append(most_probable)

        except Exception as e:
            print(f"Error encountered: {e}. Skipping this iteration.")
            continue
    
    end_time = time.time()
    total_time = end_time - start_time
    minutes = int(total_time // 60)
    remaining_seconds = int(total_time % 60)
    print(f"Load model time : {load_model_time:.2f} s -> {int(load_model_time // 60)} min {int(load_model_time % 60)} s")
    print(f"Total time for {len(prompts)} prompts : {total_time:.2f} s -> {minutes} min {remaining_seconds} s")
    print(f"Total fails : {failed_count}, sucess rate : {100 - ((failed_count*100) / len(prompts)):.3f} % \n")

    print(f"Saving predictions at {base_file_path}")
    
    write_to_csv(latitudes, longitudes, predicted, base_file_path)
    if(len(failed_completions) >= 0):
        print(f"Saving failed predictions at {failed_path}")
        write_to_csv_failed_preds(failed_lats, failed_lons, failed_prompts, failed_completions, failed_path)

def main():
    parser = argparse.ArgumentParser(description='Run zero-shot predictions.')
    parser.add_argument('model', type=str, help='The model to use for predictions (model name from hugging face)')
    parser.add_argument('prompts_file', type=str, help='The file containing prompts')
    parser.add_argument('task', type=str, help='The task for predictions')
    parser.add_argument('use_vllm', type=bool, help='Use vllm for inference : True or False')
    parser.add_argument('use_chat_template', type=bool, help='Use chat template for tokenization (Prefix in system prompt)')

    args = parser.parse_args()

    model = args.model
    prompt_file = args.prompts_file
    task = args.task
    use_vllm = args.use_vllm
    use_chat_template= args.use_chat_template
    

    run_task_for_data(model, prompt_file, task, use_vllm, use_chat_template)

if __name__ == "__main__":
    main()
