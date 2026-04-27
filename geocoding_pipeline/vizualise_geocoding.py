import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from datasets import load_dataset
from tqdm import tqdm
import torch
from plotly.subplots import make_subplots
from transformers import pipeline
import time
import numpy as np
import math

CONTINENT_COLORS = {
    "Africa":        "#E67E22",
    "Asia":          "#E74C3C",
    "Europe":        "#3498DB",
    "North America": "#2ECC71",
    "South America": "#9B59B6",
    "Oceania":       "#1ABC9C",
    "Antarctica":    "#95A5A6",
    "Unknown":       "#BDC3C7",
}

COUNTRY_TO_CONTINENT = {}

def load_countries(countries_path: str):
    print(f"[1/2] Loading countries from {countries_path}...")

    cols = ["ISO", "ISO3", "ISO_Numeric", "fips", "Country", "Capital", "Area", "Population", 
            "Continent", "tld", "CurrencyCode", "CurrencyName", "Phone", "Postal_Code_Format", 
            "Postal_Code_Regex", "Languages", "geonameid", "neighbours", "EquivalentFipsCode"]
    
    continent_names = {"na" : "North America",
                       "as" : "Asia",
                       "eu" : "Europe",
                       "af" : "Africa",
                       "sa" : "South America",
                       "oc" : "Oceania",
                       "an" : "Antarctica"}

    df = pd.read_csv(
        countries_path, sep="\t", header=None, names=cols,
        low_memory=False, encoding="utf-8", keep_default_na=False
    )

    for _, row in df.iterrows():
        country_code  = str(row["ISO"]).upper()
        if(country_code == ""):
            continue
        cc = str(row["Continent"]).strip().lower()
        continent = continent_names.get(cc, "Unknown")

        COUNTRY_TO_CONTINENT.update({country_code : continent})


def aggregate_and_plot(
    csv_path: str,
    output_path: str,
    n_docs: int,
    top_hover: int,
):
    """
    Reads a CSV with columns:
        entity, count, lat, lon, country, population, admin_level, ner_label, inference
    Aggregates mentions by continent and plots:
      - stacked bar: total entity mentions per continent, hover shows top-k entities
      - world bubble map: one dot per entity, sized by count
    """
    print("[2/2] Aggregating and plotting...")

    # Load CSV
    df = pd.read_csv(csv_path, sep=";")
    df.columns = df.columns.str.strip()

    required = {"entity", "count", "lat", "lon", "country", "population", "admin_level", "ner_label", "inference"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df["count"]       = pd.to_numeric(df["count"],       errors="coerce").fillna(0).astype(int)
    df["lat"]         = pd.to_numeric(df["lat"],         errors="coerce")
    df["lon"]         = pd.to_numeric(df["lon"],         errors="coerce")
    df["population"]  = pd.to_numeric(df["population"],  errors="coerce").fillna(0).astype(int)
    df["admin_level"] = pd.to_numeric(df["admin_level"], errors="coerce").fillna(0).astype(int)
    df["country"]     = df["country"].str.strip().str.upper()
    df["entity"]      = df["entity"].str.strip()


    # Attach continent
    df["continent"] = df["country"].map(
        lambda cc: COUNTRY_TO_CONTINENT.get(cc)
    )

    df = df[df["count"] > 0].dropna(subset=["lat", "lon", "continent"]).copy()
    df = df.sort_values("count", ascending=False).reset_index(drop=True)

    # Continent summary with top-k hover
    cont_rows = []
    for continent, grp in df.groupby("continent"):
        total = int(grp["count"].sum())
        top_k = grp.nlargest(top_hover, "count")

        hover_lines = [f"<b>Top {top_hover} entities</b>"]
        for rank, (_, r) in enumerate(top_k.iterrows(), 1):
            hover_lines.append(
                f"{rank}. {r['entity']} ({r['country']}, {r['ner_label']}) — {r['count']:,}"
            )

        cont_rows.append({
            "continent": continent,
            "mentions":  total,
            "color":     CONTINENT_COLORS.get(continent, "#999999"),
            "hover":     "<br>".join(hover_lines),
        })

    cont_df = (
        pd.DataFrame(cont_rows)
        .sort_values("mentions", ascending=False)
        .reset_index(drop=True)
    )

    # Build figure
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f"Entity Mentions by Continent  (hover for top-{top_hover} breakdown)",
            "Geographic Distribution of Entity Mentions",
        ],
        specs=[
            [{"type": "bar"}],
            [{"type": "scattergeo"}],
        ],
        vertical_spacing=0.10,
        row_heights=[0.35, 0.65],
    )

    # Continent bar
    fig.add_trace(
        go.Bar(
            name="Entity mentions",
            x=cont_df["continent"],
            y=cont_df["mentions"],
            marker_color=cont_df["color"].tolist(),
            marker_line=dict(width=0),
            text=cont_df["mentions"].apply(lambda x: f"{x:,}"),
            textposition="inside",
            insidetextanchor="middle",
            customdata=cont_df["hover"].tolist(),
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # World bubble map
    for continent, grp in df.groupby("continent"):
        fig.add_trace(
            go.Scattergeo(
                lat=grp["lat"],
                lon=grp["lon"],
                mode="markers",
                marker=dict(
                    size=grp["count"].apply(lambda x: max(2, min(15, x ** 0.45))),
                    color=CONTINENT_COLORS.get(continent, "#999"),
                    opacity=0.90,
                    line=dict(width=0.4, color="#ffffff"),
                ),
                text=grp.apply(
                    lambda r: (
                        f"<b>{r['entity']}</b> ({r['country']})<br>"
                        f"Label: {r['ner_label']}  |  Inference: {r['inference']}<br>"
                        f"Mentions: {r['count']:,}<br>"
                        f"Population: {r['population']:,}<br>"
                        f"Admin level: {r['admin_level']}"
                    ),
                    axis=1,
                ),
                hoverinfo="text",
                name=continent,
                showlegend=True,
            ),
            row=2, col=1,
        )

    # Layout
    total_mentions = int(df["count"].sum())
    n_entities     = len(df)

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Entity Mentions in HuggingFaceFW/fineweb ({n_docs:,} docs)</b><br>"
                f"<sup>{total_mentions:,} total mentions across {n_entities:,} distinct entities"
                f" — NER-based detection</sup>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=18, color="#111111"),
        ),
        legend=dict(
            title=dict(text="Continent"),
            orientation="v",
            yanchor="middle", y=0.25,
            xanchor="left",   x=1.01,
            font=dict(size=11),
        ),
        height=1100,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f7f7f7",
        font=dict(color="#222222", family="Inter, Arial, sans-serif"),
        geo=dict(
            showland=True,
            landcolor="#e8e8e8",
            showocean=True,
            oceancolor="#cfe2f3",
            showcountries=True,
            countrycolor="#aaaaaa",
            showcoastlines=True,
            coastlinecolor="#aaaaaa",
            projection_type="natural earth",
            bgcolor="#ffffff",
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#cccccc",
            font=dict(color="#111111", size=12),
            align="left",
        ),
        margin=dict(t=110, b=30, l=30, r=80),
    )
    fig.update_xaxes(gridcolor="#e0e0e0", zeroline=False, linecolor="#cccccc")
    fig.update_yaxes(gridcolor="#e0e0e0", zeroline=False, linecolor="#cccccc")

    fig.write_html(output_path)
    print(f"\n  Plot saved at: {output_path}")

    # Console summary
    print("\n-- Continent summary --")
    print(cont_df[["continent", "mentions"]].to_string(index=False))
    print("\n-- Top 20 entities --")
    print(df[["entity", "country", "continent", "ner_label", "count"]].head(20).to_string(index=False))

    return df, cont_df


def parse_args():
    p = argparse.ArgumentParser(
        description="Vizualise geocoded entities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_path",
                help="Path to the csv file containing the geocoded entities")
    p.add_argument("--n_docs", type=int,
                help="Number of docs from  the corpus (only used for the print)")
    p.add_argument("--output_path", default="plot_worldmap.html",
                   help="Path to save the plot as an html file")
    p.add_argument("--countries",    default="data/countryInfo.txt",
                    help="Path to GeoNames cities1000.txt")
    p.add_argument("--top_hover", type=int, default=10,
                   help="Number of entities that appear in the barplot hover")
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.countries).exists():
        print(f"ERROR: countryInfo file not found: {args.countries}")
        print("Download with:")
        print("  wget https://download.geonames.org/export/dump/countryInfo.txt")
        sys.exit(1)

    if not Path(args.input_path).exists():
        print(f"ERROR: Geocoded entities file not found: {args.input_path}")
        print("Create with:")
        print("     python geocode_entities.py --input_path <input path> --output_path <output path>")
        sys.exit(1)

    load_countries(args.countries)
    aggregate_and_plot(args.input_path, args.output_path, args.n_docs, args.top_hover)


if __name__ == "__main__":
    main()
