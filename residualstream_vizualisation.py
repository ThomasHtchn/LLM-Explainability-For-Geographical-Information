from residualstream_utils import get_hidden_states, tokenize_prompt, remove_spaces
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import torch

def verify_second_token(model, inputs, first_token_id, second_token_id):
    new_input_ids = torch.cat(
        [inputs["input_ids"], torch.tensor([[first_token_id]], device=inputs["input_ids"].device)],
        dim=1
    )

    with torch.no_grad():
        outputs = model(input_ids=new_input_ids)

    logits = outputs.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)

    pred_token_id = torch.argmax(probs, dim=-1).item()

    return pred_token_id == second_token_id

# Returns the index of the first layer in which the label's token is detected or -1 if not found among top_k
def first_layer_detection(model, tokenizer, hidden_states, label, top_k, min_proba_threshold, 
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
                    info = f"<br>{label} [{decoded_token}] : {prob:.3f} | top {j+1}"
                    return i, info
                
            # Comparaison token ids
            if(comp_token):
                if(tid.item() == encoded_label):
                    prob = top_probs[0][j].item()

                    if prob < min_proba_threshold:
                        continue # filtre les proba trop faible
                    
                    if(len(label_ids) == 1 or not verif_sec_token):
                        info = f"<br>{label} [{decoded_token}] : {prob:.3f} | top {j+1}"
                        return i, info
                    # Multi token labels
                    elif(verif_sec_token and verify_second_token(model, inputs, label_ids[0], label_ids[1])):
                        info = f"<br>{label} [{decoded_token}, {tokenizer.decode(label_ids[1])}] : {prob:.3f} | top {j+1}"
                        return i, info

    return -1, f"[{label}] "

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

def compute_stackbar_region(model, tokenizer, df_path, top_k=10, proba_threshold=0.01):

    df = pd.read_csv(df_path)

    results = []

    for row in tqdm(df.itertuples(), total=len(df)):

        country_name = row._1
        region_name = row.Region
        capital_name = row.Capital

        inputs, hs, prediction = generate_hidden_states(model, tokenizer, country_name)

        layer_idx, details = first_layer_detection(model, tokenizer, hs, capital_name, top_k, proba_threshold, 
                          inputs, True, True, True)
        
        results.append({
            "Layer" : layer_idx,
            "Region" : region_name,
            "Capital" : capital_name,
            "Prediction" : prediction,
            "Details" : details
        })

    df_results = pd.DataFrame(results)
    
    return df_results

def plot_stackbar_region(df_results):

    df_valid = df_results[df_results["Layer"] >= 0]

    grouped = df_valid.groupby(["Layer", "Region"]).agg({
        "Details": lambda x: "".join(x)
    }).reset_index()

    # Add counts
    counts = df_valid.groupby(["Layer", "Region"]).size().reset_index(name="Count")

    grouped = grouped.merge(counts, on=["Layer", "Region"])

    color_map = {
        "Europe": "blue",
        "Asia": "red",
        "Africa": "green",
        "Europe/Asia": "yellow",
        "North America": "brown",
        "South America": "orange",
        "Oceania": "purple"
    }

    fig = px.bar(
        grouped,
        x="Layer",
        y="Count",
        color="Region",
        hover_data={
            "Details": True
        },
        color_discrete_map=color_map,
        title="Capitales détectées par couche (non cumulatif) par région du monde,<br>parmis le top 10 des tokens."
    )
    fig.update_layout(barmode="stack")
    fig.update_layout(template="plotly")
    fig.write_html("stackbar_first-layer-vizu_regions.html")
    return fig
