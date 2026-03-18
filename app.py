import streamlit as st
import pandas as pd
import torch

from residualstream_utils import load_model
from residualstream_vizualisation import plot_layer_analysis, only_plot_layer_analysis
# import your model + tokenizer loading here

st.set_page_config(layout="wide")

st.title("Layer Analysis Visualization")

# Sidebar controls
st.sidebar.header("Parameters")

top_k = st.sidebar.slider("Top-K", 1, 50, 10)
min_proba_threshold = st.sidebar.slider("Min Probability", 0.0, 1.0, 0.01)
nb_layers = st.sidebar.slider("Number of Layers", 1, 36, 36)

model_name_input = st.sidebar.text_input(
    "Model Name",
    value="HuggingFaceTB/SmolLM3-3B"
)

st.sidebar.header("Comparison Options")

# Slide toggles / checkboxes
compare_text = st.sidebar.checkbox("Compare by text", value=True)
compare_token = st.sidebar.checkbox("Compare by token ID", value=True)
verify_second_token = st.sidebar.checkbox("Verify second token for multi-token labels", value=True)

st.write("Selected comparison types:")
st.write(f"Compare text: {compare_text}")
st.write(f"Compare token: {compare_token}")
st.write(f"Verify second token: {verify_second_token}")

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.tokenizer = None

# Upload dataset
df = None

uploaded_vizu = st.file_uploader("Upload CSV to vizualise layer distributions", type=["csv"])
if uploaded_vizu:
    df = pd.read_csv(uploaded_vizu)
    fig = only_plot_layer_analysis(df)
    st.plotly_chart(fig, use_container_width=True)

uploaded_file = st.file_uploader("Upload CSV with columns: Prompt, Label", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of dataset:")
    st.dataframe(df.head())



# ------------------------
# Load Model Button
# ------------------------
if st.button("Load Model"):
    if not st.session_state.model_loaded:
        with st.spinner(f"Loading model '{model_name_input}'..."):
            model, tokenizer = load_model(model_name_input)
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_loaded = True
        st.success(f"Model '{model_name_input}' loaded!")
    else:
        st.info(f"Model '{model_name_input}' already loaded.")

# ------------------------
# Run Analysis Button
# ------------------------
if df is not None and st.button("Run Analysis & Plot"):
    if not st.session_state.model_loaded:
        st.warning("Please load the model first!")
    else:
        model = st.session_state.model
        tokenizer = st.session_state.tokenizer
        with st.spinner("Running layer analysis..."):
            fig = plot_layer_analysis(
                model,
                tokenizer,
                df,
                nb_layers=nb_layers,
                top_k=top_k,
                min_proba_threshold=min_proba_threshold,
                comp_text=compare_text,
                comp_token=compare_token,
                verif_sec_token=verify_second_token
            )

            st.plotly_chart(fig, use_container_width=True)
        st.success("Done!")