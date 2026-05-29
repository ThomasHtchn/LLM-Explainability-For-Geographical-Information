import argparse
import pandas as pd
#from geollm_scripts.geollm_utils import *
from geollm_utils import *
from scipy.stats import spearmanr
import os

def calculate_spearman_correlation(coordinates, predictions, groundtruth_tif):
    groundtruth = [extract_data(lat, lon, groundtruth_tif) for lat, lon in coordinates]
    corr, _ = spearmanr(predictions, groundtruth)
    return corr

def print_spearman_correl(predictions_csv, groundtruth_tif):
    df = pd.read_csv(predictions_csv)
    if 'Latitude' in df.columns and 'Longitude' in df.columns and 'Predictions' in df.columns:
        coordinates = list(zip(df['Latitude'], df['Longitude']))
        predictions = df['Predictions']
    else:
        raise ValueError("CSV file must contain 'Latitude', 'Longitude', and 'Predictions' columns.")

    corr = calculate_spearman_correlation(coordinates, predictions, groundtruth_tif)

    print(f"Spearman correlation: {corr:.2f}")


def multiple_layer_spearman(
    layers_output_dir,
    groundtruth_tif,
    start_layer,
    end_layer,
    file_prefix,
    min_n
):
    results = []

    for layer_idx in range(start_layer, end_layer + 1):
        # print(f'layers_output_dir, f"{file_prefix}_{layer_idx}.csv"')
        filepath = os.path.join(layers_output_dir, f"{file_prefix}_{layer_idx}.csv")

        if not os.path.exists(filepath):
            print(f"Missing {filepath}, skipping")
            continue

        df = pd.read_csv(filepath)

        preds = []
        gts = []

        for _, row in df.iterrows():
            lat = row["latitude"]
            lon = row["longitude"]
            pred = row["predicted_digit"]

            if pd.isna(pred):
                continue

            try:
                pred_val = float(pred)
                gt_val = extract_data(lat, lon, groundtruth_tif)
            except Exception as e:
                print(e)
                print(f"pred_val: {pred_val}, lat: {lat}")
                system.exit(1)
                continue

            if gt_val is None:
                continue

            preds.append(pred_val)
            gts.append(gt_val)

        if len(preds) < min_n:
            corr = None
        else:
            corr, _ = spearmanr(preds, gts)

        results.append({
            "layer": layer_idx,
            "spearman": corr,
            "n_samples": len(preds)
        })
        if corr:
            print(f"Layer {layer_idx}: Spearman = {corr:3f}, N = {len(preds)}")
        else :
            print(f"Layer {layer_idx}, N = {len(preds)}, corr = None")
    return pd.DataFrame(results)

import plotly.graph_objects as go

def plot_spearman_plotly(df_results, model_str):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_results["layer"],
        y=df_results["spearman"],
        mode="lines+markers+text",
        text=[f"{v:.3f}" for v in df_results["spearman"]],
        textposition="top center"
    ))

    fig.update_layout(
        title="Spearman Correlation Across Layers",
        xaxis_title="Layer",
        yaxis_title="Spearman Correlation",
        yaxis=dict(range=[df_results["spearman"].min(), 1]),
    )
    best_idx = df_results["spearman"].idxmax()
    best_layer = df_results.loc[best_idx, "layer"]
    best_value = df_results.loc[best_idx, "spearman"]

    fig.add_vline(x=best_layer, line_dash="dash")

    fig.write_html(f"spearman_plot_{model_str}.html")
    print("Plot saved !")

def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--pred_file", type=str, default=None, help="Path to the CSV file containing coordinates.")
    parser.add_argument("--tif", type=str, help="Path to the groundtruth tif file.")
    parser.add_argument("--input_dir", type=str, default=None, help="Multiple files ")
    parser.add_argument("--start_layer", type=int, default=1, help="Start layer")
    parser.add_argument("--end_layer", type=int, default=36, help="End layer")
    parser.add_argument("--prefix", type=str, default="layer", help="Prefix of the per layer result files")
    parser.add_argument("--N", type=int, help="Number of prompts")
    parser.add_argument("--model_name", type=str, help="Hugging face model name")

    args = parser.parse_args()

    model_str = args.model_name.replace("/","_").lower()

    pred_file = args.pred_file
    groundtruth_tif = args.tif
    dir = args.input_dir
    N = args.N
    MIN_N = N * 0.9

    if dir:
        df_results = multiple_layer_spearman(
            layers_output_dir=dir,
            groundtruth_tif=groundtruth_tif,
            start_layer=args.start_layer,
            end_layer=int(args.end_layer),
            file_prefix=args.prefix,
            min_n = MIN_N
        )

        print(df_results)
        plot_spearman_plotly(df_results, model_str)

    elif pred_file:
        df = pd.read_csv(pred_file)
        if 'Latitude' in df.columns and 'Longitude' in df.columns and 'Predictions' in df.columns:
            coordinates = list(zip(df['Latitude'], df['Longitude']))
            predictions = df['Predictions']
        else:
            raise ValueError("CSV file must contain 'Latitude', 'Longitude', and 'Predictions' columns.")

        corr = calculate_spearman_correlation(coordinates, predictions, groundtruth_tif)

        print(f"Spearman correlation: {corr:.2f}")

if __name__ == "__main__":
    main()