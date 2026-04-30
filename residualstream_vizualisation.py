from residualstream_utils import *
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def verify_next_token(model, inputs, first_token_id, second_token_id):
    new_input_ids = torch.cat(
        [inputs["input_ids"], torch.tensor([first_token_id], device=inputs["input_ids"].device)],
        dim=1
    )

    with torch.no_grad():
        outputs = model(input_ids=new_input_ids)

    logits = outputs.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)

    pred_token_id = torch.argmax(probs, dim=-1).item()

    return pred_token_id == second_token_id


# Returns the index of the first layer in which the label's token is detected or -1 if not found among top_k
def first_layer_detection(model, tokenizer, hidden_states, label, country_name, top_k, min_proba_threshold, 
                          inputs, comp_text, comp_token, verif_sec_token):
    for i, h in enumerate(hidden_states, start=0):
        last_token_hidden = h[:, -1, :]  
        #last_token_hidden = model.model.norm(last_token_hidden)

        logits = model.lm_head(last_token_hidden)
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_token_ids = torch.topk(probs, top_k, dim=-1)

        for j, tid in enumerate(top_token_ids[0], start=0):
            decoded_token = tokenizer.decode([tid.item()])
            label_ids = tokenizer.encode(label, add_special_tokens=False)
            encoded_label = label_ids[0]

            # Comparaison strings
            if(comp_text):
                if(remove_spaces(decoded_token) == remove_spaces(label)):
                    prob = top_probs[0][j].item()
                    if prob < min_proba_threshold:
                        continue # filtre les proba trop faible
                    info = f"<br>{country_name} [{decoded_token}] (text : {len(label_ids)}): {prob:.3f} | top {j+1}"
                    return i, info
                
            # Comparaison token ids
            if(comp_token):
                if(tid.item() == encoded_label):
                    prob = top_probs[0][j].item()

                    if prob < min_proba_threshold:
                        continue # filtre les proba trop faible
                    
                    if(len(label_ids) == 1 or not verif_sec_token):
                        info = f"<br>{country_name} [{decoded_token}] (tk) : {prob:.3f} | top {j+1}"
                        return i, info
                    # Multi token labels
                    elif(len(label_ids) > 1 and verif_sec_token):
                        for k in range(1, len(label_ids)):
                            if not verify_next_token(model, inputs, label_ids[0:k], label_ids[k]):
                                continue
                        tks = ",".join(tokenizer.decode(tk) for tk in label_ids)
                        info = f"<br>{country_name} [{tks}] (multi tks : {len(label_ids)}) : {prob:.3f} | top {j+1}"
                        return i, info

    return -1, country_name


# Returns a list with the quantity of expected token found for each layer (not cumulative) and the quantity of not found
def plot_layer_analysis(model, tokenizer, df, nb_layers=36, top_k=10, min_proba_threshold=0.01, 
                        comp_text=True, comp_token=True, verif_sec_token=True):
    layers_count = [0 for _ in range(nb_layers+1)]
    not_found_count = 0
    not_found_capitales_txt = ""
    layers_details = ["" for _ in range(nb_layers + 1)]

    for row in tqdm(df.itertuples(), total=len(df)):
        inputs = tokenize_prompt(row.Prompt, tokenizer, model, is_chat=True, prompt_system="Answer with one word.")
        hidden_states = get_hidden_states(model, inputs)
        #hidden_states = get_hidden_states_from_raw_prompt(model, tokenizer, row.Prompt)
        layer_idx, infos = first_layer_detection(model, tokenizer, hidden_states, row.Label, 
                                                 top_k, min_proba_threshold, inputs,
                                                 comp_text, comp_token, verif_sec_token)
        if(layer_idx >= 0):
            layers_count[layer_idx] += 1
            if(layers_count[layer_idx] <= 20):
                layers_details[layer_idx] += infos
        elif (layer_idx == -1):
            not_found_count += 1
            not_found_capitales_txt += infos
        else:
            print("Error : layer_idx out of bounds")
    #return layers_count, not_found_count, nb_layers, top_k, layers_details, not_found_capitales_txt, min_proba_threshold

    layers = list(range(0, nb_layers + 1))
    title_text = f"A partir de quelle couche de smolLM3 le bon token est détecté parmis le top {top_k} (non cumulatif)<br>avec une probabilité minimum de {min_proba_threshold}, pour 196 pairs de pays / capitale<br>Méthodes de comparaisons : Text = {comp_text}, Token ids = {comp_token}, verif second token = {verif_sec_token}"
    annotation_text = f"Proportion de capitales trouvées : {len(df) - not_found_count} / {len(df)} -> Manquantes : {not_found_capitales_txt}"
    data = pd.DataFrame({
        "Couches": layers,
        "Quantité de bon tokens détectés": layers_count,
        "Details": layers_details,
        "Title" : None,
        "Annotation" : None
    })
    data.iloc[0, data.columns.get_loc('Title')] = title_text
    data.iloc[0, data.columns.get_loc('Annotation')] = annotation_text

    fig = px.bar(
        data,
        x="Couches",
        y="Quantité de bon tokens détectés",
        title=title_text,
        hover_data=["Details"]
    )
    fig.update_traces(marker_color="blue")

    fig.update_layout(
        template="plotly",
        margin=dict(b=140)
    )

    fig.add_annotation(
        text=annotation_text,
        xref="paper",
        yref="paper",
        x=0,
        y=-0.3,
        showarrow=False,
    )
    fig.show()
    return fig

def only_plot_layer_analysis(df):
    title_text = df.iloc[0, df.columns.get_loc('Title')]
    annotation_text = df.iloc[0, df.columns.get_loc('Annotation')]

    fig = px.bar(
        df,
        x="Couches",
        y="Quantité de bon tokens détectés",
        title=title_text,
        hover_data=["Details"]
    )
    fig.update_traces(marker_color="blue")

    fig.update_layout(
        template="plotly",
        margin=dict(b=100)
    )

    fig.add_annotation(
        text=annotation_text,
        xref="paper",
        yref="paper",
        x=0,
        y=-0.3,
        showarrow=False,
    )
    #fig.write_html("test.html")
    fig.show()
    return fig
    

def generate_hidden_states(model, tokenizer, country_name):
    prompt = f"What is the capital city of {country_name}?"
    messages = [
        {"role": "system",
            "content": (
                f"You are an expert geographer. "
                f"You have to give name of the capital city. "
                f"Answer only with the capital city name "
                f"without any other words or repetition of the question. Don't repeat the prompt neither. "
                f"Example of answer: 'Paris'."
            )
        },
        {"role": "user", "content": prompt}
    ]
    # Tokenize input prompt

    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([inputs], return_tensors="pt").to("cuda")
    model_outputs = model.generate(**model_inputs, 
                                   return_dict_in_generate=True, 
                                   output_hidden_states=True) 
    
    generated_ids = model_outputs.sequences[0][len(model_inputs.input_ids[0]) :]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return model_inputs, model_outputs.hidden_states[0], prediction


def generate_hidden_states_multitask(model, tokenizer, task, country_name):
    prompt = f"What is the {task} of {country_name}?"
    messages = [
        {"role": "system",
            "content": (
                f"You are an expert geographer. "
                f"Answer what you are asked with only one word and "
                f"without any other words or repetition of the question. Don't repeat the prompt neither. "
            )
        },
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([inputs], return_tensors="pt").to("cuda")
    model_outputs = model.generate(**model_inputs, 
                                   return_dict_in_generate=True, 
                                   output_hidden_states=True) 
    
    generated_ids = model_outputs.sequences[0][len(model_inputs.input_ids[0]) :]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return model_inputs, model_outputs.hidden_states[0], prediction


def compute_stackbar_region(model, tokenizer, df_path, task, top_k, proba_threshold, tail_save_path):
    df = pd.read_csv(df_path)
    results = []
    task_with_space = task.replace("_", " ")

    pbar = tqdm(df.iterrows(), total=len(df))
    for i, row in pbar:
        country_name = row['Country_Name']
        region_name = row["Continent"]
        task_value = row[task]

        pbar.set_description(f"[{i}] Task : {task_with_space}")
        pbar.set_postfix(value=country_name)

        inputs, hs, prediction = generate_hidden_states_multitask(model, tokenizer, task_with_space, country_name)

        layer_idx, details = first_layer_detection(model, tokenizer, hs, task_value, country_name, top_k, proba_threshold, 
                          inputs, True, True, True)
        
        # Not found
        if(layer_idx == -1):
            details = details + f" ({task_value})<br>"

        results.append({
            "Layer" : layer_idx,
            "Region" : region_name,
            "Country_Name" : country_name,
            task : task_value,
            "Prediction" : prediction,
            "Details" : details
        })

    df_results = pd.DataFrame(results)
    save_path = f"results/result_{tail_save_path}.csv"
    df_results.to_csv(save_path)
    print(f"First layer results saved at : {save_path}")
    return df_results


def plot_stackbar_region(df_results, tail_save_path):
    task_name = df_results.columns[3]

    df_valid = df_results[df_results["Layer"] >= 0]
    df_not_found = df_results[df_results["Layer"] == -1]

    def prepare_group(df):
        grouped = df.groupby(["Layer", "Region"]).agg({
            "Details": lambda x: "".join(x)
        }).reset_index()

        counts = df.groupby(["Layer", "Region"]).size().reset_index(name="Count")
        return grouped.merge(counts, on=["Layer", "Region"])

    grouped_valid = prepare_group(df_valid)
    grouped_not_found = prepare_group(df_not_found)

    color_map = {
        "Europe": "blue",
        "Asia": "red",
        "Africa": "green",
        "Europe/Asia": "yellow",
        "North America": "brown",
        "South America": "orange",
        "Oceania": "purple"
    }

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("", "Non trouvé(e)s"),
        shared_yaxes=True,
        column_widths=[0.95, 0.05]
    )

    # First subplot (trouvées)
    for region in grouped_valid["Region"].unique():
        df_r = grouped_valid[grouped_valid["Region"] == region]
        fig.add_trace(
            go.Bar(
                x=df_r["Layer"],
                y=df_r["Count"],
                name=region,
                marker_color=color_map.get(region, "gray"),
                hovertext=df_r["Details"],
                showlegend=True
            ),
            row=1, col=1
        )
    # Second subplot (Layer = -1, non trouvées)
    for region in grouped_not_found["Region"].unique():
        df_r = grouped_not_found[grouped_not_found["Region"] == region]
        fig.add_trace(
            go.Bar(
                x=df_r["Layer"],
                y=df_r["Count"],
                name=region,
                marker_color=color_map.get(region, "gray"),
                hovertext=df_r["Details"],
                showlegend=False  # no duplicate legend
            ),
            row=1, col=2
        )
    fig.update_layout(
        barmode="stack",
        template="plotly",
        title=f"{task_name} trouvé(e)s par couche (non cumulatif) par région du monde<br>parmis le top 10 des tokens."
    )
    fig.update_xaxes(title_text="Layer", row=1, col=1)

    save_path = f"results/stackedbar_{tail_save_path}.html"
    fig.write_html(save_path)
    print(f"Saved fig at : {save_path}")
    return fig


import argparse
import os
import sys

VALID_TASKS = ["ISO_Code", "Dialing_Code", "Continent", "Capital"]
VALID_MODELS = ["HuggingFaceTB/SmolLM3-3B"]

def main():
    parser = argparse.ArgumentParser(description="Process a dataframe with a given task.",
                                      usage="python residualstream_vizualisation.py <path> <task> <model_name>")
    parser.add_argument("--path", type=str, help="Path to the dataframe file")
    parser.add_argument("--task", type=str, help='Task to perform: one of ["ISO_Code", "Dialing_Code", "Continent", "Capital"]')
    parser.add_argument("--model_name", type=str, help="Hugging face model name")
    parser.add_argument("--top_k", type=int, default=10, help="Top k token to look at in the residual-stream")
    parser.add_argument("--min_prob", type=float, default=0.01, help="Top k token to look at in the residual-stream")

    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        parser.print_usage()
        print(f"Error: The path '{args.path}' does not exist.")
        sys.exit(1)

    if args.task not in VALID_TASKS:
        parser.print_usage()
        print(f"Error: Invalid task '{args.task}'. Must be one of {VALID_TASKS}.")
        sys.exit(1)
    
    if args.model_name not in VALID_MODELS:
        print(f"Warning: Invalid model name '{args.model_name}'. Must be one of {VALID_MODELS}.")
        print(f"Default model used : {VALID_MODELS[0]}")
        args.model_name = VALID_MODELS[0]

    # If everything is valid
    df_path = args.path
    task = args.task
    model_name = args.model_name
    top_k = args.top_k
    min_prob = args.min_prob
    model_str = args.model_name.replace("/","_").lower()

    save_path_tail = f"{model_str}_{task.lower()}_top{top_k}_p{str(min_prob).replace('.','')}"
    
    print(f"Path:  {args.path}")
    print(f"Task:  {args.task}")
    print(f"Model: {args.model_name}")
    
    model, tokenizer = load_model(model_name)

    res = compute_stackbar_region(model, tokenizer, df_path, task, top_k=top_k, proba_threshold=min_prob, tail_save_path=save_path_tail)
    plot_stackbar_region(res, save_path_tail)
    

if __name__ == "__main__":
    main()