from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import streamlit as st

def remove_skiplines(stri):
    return stri.replace("\n","\\n")

def remove_spaces(stri):
    return stri.replace(" ", "").lower()

# Creat Pandas dataframe to Columns Prompt and Label such as {Prompt : "The capital of France is", Label : "Paris"}
def generate_country_capital_prompts(file_name):
    df = pd.read_csv(file_name)
    prompt_df = df[["Country Name", "Capital"]].copy()
    prompt_df["Country Name"] = "The capital of " + prompt_df["Country Name"] + " is"
    prompt_df.rename(columns={'Country Name': 'Prompt', 'Capital': 'Label'}, inplace=True)
    return prompt_df

# Load hugging face model
@st.cache_resource
def load_model(model_name="HuggingFaceTB/SmolLM3-3B", device="cuda", type=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "mistralai" in model_name:
        from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend, FineGrainedFP8Config
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device,
            dtype=type,
        ) 
    else :
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=type,
        ).to(device)
    return model, tokenizer

# Tokenize prompt to infer on
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

def get_hidden_states(model, tokenizer, inputs):
    model_outputs = model.generate(**inputs, 
                                   return_dict_in_generate=True, 
                                   output_hidden_states=True,
                                   do_sample=False,
                                   max_new_tokens=5) 
    
    generated_ids = model_outputs.sequences[0][len(inputs.input_ids[0]) :]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"Model output : {prediction}")
    return model_outputs.hidden_states[0]

def print_top_tokens_per_layer(prompt, expected_next_token, model, tokenizer, is_already_tokenized=False, apply_last_norm=False, start_layer=30, end_layer=36, top_k=20):
    if(not is_already_tokenized):
        inputs = tokenize_prompt(prompt, tokenizer, model)#, is_chat=True, prompt_system="The first token you generate should be the answer to the question.")
    else:
        inputs = prompt
    hidden_states = get_hidden_states(model, tokenizer, inputs)
    lm_head = model.lm_head
    #print("Norme moyenne des embeddings du residual stream")
    for i, h in enumerate(hidden_states[start_layer:end_layer+1], start=start_layer):
        last_token_hidden = h[:, -1, :]  

        if(apply_last_norm):
            before_norm = last_token_hidden.norm().mean()
            last_token_hidden = model.model.norm(last_token_hidden)
            after_norm = last_token_hidden.norm().mean()
            #print(f"layer {i}: [avant normalisation : {before_norm:.2f}, après normalisation: {after_norm:.2f}]")
        
        # if(i == end_layer and not apply_last_norm):
        #     print(f"i == {i}")
        #     last_token_hidden = model.model.norm(last_token_hidden)

        logits = lm_head(last_token_hidden)
        probs = torch.softmax(logits, dim=-1)

        top_probs, top_token_ids = torch.topk(probs, top_k, dim=-1)
        decoded_tokens = []
        for j, tid in enumerate(top_token_ids[0]):
            decoded_token = tokenizer.decode([tid.item()])
            if(tid.item() == tokenizer.encode(expected_next_token)[0] or remove_spaces(decoded_token) == remove_spaces(expected_next_token)):
                print(f"    -> {expected_next_token} detected -> top {j+1}")
            decoded_tokens.append(decoded_token)

        probs_list = [p.item() for p in top_probs[0]]
        topk_str = ", ".join([f'"{remove_skiplines(tok)}": {prob:.4f}' 
                            for tok, prob in zip(decoded_tokens, probs_list)])
        print(f"Layer {i}: [{topk_str}]")