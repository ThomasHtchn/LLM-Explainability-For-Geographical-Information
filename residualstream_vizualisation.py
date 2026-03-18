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
    


