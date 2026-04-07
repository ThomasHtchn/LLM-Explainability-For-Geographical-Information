"""
City & Country Mentions in Corpus Analyzer
=====================================================
Loads a shuffled subset of HuggingFaceFW/fineweb-edu, detects city mentions
using NER, cross-references with GeoNames cities1000.txt (pop > min_pop),
and plots occurrence counts by continent.

Two NER backends :
    transformers - HuggingFace pipeline on GPU/CPU
                              model: dslim/bert-large-NER  (recommended)
    spacy (default) - spaCy GPU

    Download cities1000.txt and countryInfo.txt:
wget https://download.geonames.org/export/dump/cities1000.zip && unzip cities1000.zip
wget https://download.geonames.org/export/dump/countryInfo.txt


python occ_country_city.py --n_docs 20000

"""

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
COUNTRY_NAME_LOOKUP = {}

def load_countries(contries_path: str):
    print(f"[1/5] Loading countries from {contries_path}...")

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
        contries_path, sep="\t", header=None, names=cols,
        low_memory=False, encoding="utf-8", keep_default_na=False
    )

    for _, row in df.iterrows():
        country_code  = str(row["ISO"])
        if(country_code == ""):
            continue
        country_name = str(row["Country"]).strip().lower()
        cc = str(row["Continent"]).strip().lower()
        continent = continent_names.get(cc, "Unknown")

        COUNTRY_TO_CONTINENT.update({country_code : continent})
        COUNTRY_NAME_LOOKUP.update({country_name : country_code})


def load_cities(cities_path: str) -> tuple[dict, dict]:
    """
    Returns:
        city_lookup  : {ascii_name_lower → set of (geonameid, country_code, population)}
        city_meta    : {geonameid → {'name', 'country', 'continent', 'population', 'lat', 'lon'}}
    """
    print(f"[2/5] Loading cities from {cities_path}...")
    city_lookup: dict[str, list] = defaultdict(list)
    city_meta: dict[int, dict] = {}

    cols = [
        "geonameid","name","asciiname","alternatenames",
        "latitude","longitude","feature_class","feature_code",
        "country_code","cc2","admin1","admin2","admin3","admin4",
        "population","elevation","dem","timezone","modification_date"
    ]
    df = pd.read_csv(
        cities_path, sep="\t", header=None, names=cols,
        low_memory=False, encoding="utf-8"
    )
    df = df[df["population"] >= 1000].copy()
    df["asciiname"] = df["asciiname"].fillna("").str.strip()
    df["name"]      = df["name"].fillna("").str.strip()

    # Build alternate-name variants too
    for _, row in df.iterrows():
        gid  = int(row["geonameid"])
        cc   = str(row["country_code"]).strip().upper()
        cont = COUNTRY_TO_CONTINENT.get(cc, "Unknown")
        pop  = int(row["population"])

        city_meta[gid] = {
            "name":       row["name"],
            "country":    cc,
            "continent":  cont,
            "population": pop,
            "lat":        row["latitude"],
            "lon":        row["longitude"],
        }

        # Index by ascii name and original name (lower-cased)
        for variant in {row["asciiname"].lower(), row["name"].lower()}:
            if len(variant) >= 2:
                city_lookup[variant].append(gid)

        # Also index short alternate names (pipe-separated)
        alts = str(row.get("alternatenames", ""))
        for alt in alts.split(","):
            alt = alt.strip().lower()
            if 2 <= len(alt) <= 50 and not any(c.isdigit() for c in alt):
                city_lookup[alt].append(gid)

    print(f"    {len(df):,} cities indexed ({len(city_lookup):,} name variants)")
    return dict(city_lookup), city_meta

# Load dataset subet (shuffled)
def load_corpus(n_docs: int, seed: int = 42) -> list[str]:
    print(f"[3/5] Streaming {n_docs:,} shuffled docs from fineweb-edu...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    texts = []
    for doc in tqdm(ds, total=n_docs, unit="doc"):
        texts.append(doc["text"])
        if len(texts) >= n_docs:
            break
    print(f"    {len(texts):,} documents loaded")
    return texts

# Device resolution
def resolve_device(device_arg: str) -> tuple[str, int]:
    """
    Returns (device_str, torch_device_index).
    device_str  : 'cuda', 'mps', or 'cpu'
    torch_index : integer device id for transformers pipeline (-1 = cpu)
    """

    if device_arg == "cpu":
        return "cpu", -1

    if device_arg.startswith("cuda:"):
        idx = int(device_arg.split(":")[1])
        if not torch.cuda.is_available():
            print("     CUDA not available — falling back to CPU")
            return "cpu", -1
        return "cuda", idx

    # "auto" — pick best available
    if torch.cuda.is_available():
        print(f"    CUDA detected: {torch.cuda.get_device_name(0)}"
              f"  ({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")
        return "cuda", 0
    try:
        if torch.backends.mps.is_available():
            print("     Apple MPS detected")
            return "mps", 0
    except AttributeError:
        pass
    print("     No GPU found — using CPU")
    return "cpu", -1


# ---------------------------------------------------------------------------
# 3b. NER helpers
# ---------------------------------------------------------------------------
def _clean_entity(text: str) -> str:
    """Normalise a raw entity string for lookup."""
    text = re.sub(r"[''`]s?$", "", text)   # strip possessives
    text = text.strip(" .,;:\"'()[]")
    return text.lower()


def _match_country(token: str) -> str | None:
    """Return ISO-2 country code if token is a known country name/abbrev."""
    return COUNTRY_NAME_LOOKUP.get(token)


def _match_city(
    token: str,
    city_lookup: dict,
    city_meta: dict,
    min_pop: int = 0,
) -> int | None:
    """
    Return the best-matching city geonameid or None.
    Country names are excluded here (handled by _match_country).
    """
    # Skip tokens that are country names — they belong to the country counter
    if token in COUNTRY_NAME_LOOKUP:
        return None

    gids = city_lookup.get(token)
    if not gids:
        return None

    if min_pop > 0:
        gids = [g for g in gids if city_meta[g]["population"] >= min_pop]
    if not gids:
        return None

    return max(gids, key=lambda g: city_meta[g]["population"])


# ---------------------------------------------------------------------------
# 3c. Transformers backend  (default, best GPU utilisation)
# ---------------------------------------------------------------------------
_HF_LOC_LABELS = {"LOC", "GPE", "FAC", "B-LOC", "I-LOC", "B-GPE", "I-GPE"}

def _ner_transformers(
    texts: list[str],
    city_lookup: dict,
    city_meta: dict,
    model_name: str,
    device_index: int,
    batch_size: int,
    min_pop: int = 0,
) -> Counter:

    device_label = f"cuda:{device_index}" if device_index >= 0 else "cpu"
    print(f"  Loading '{model_name}' on {device_label} …")

    ner = pipeline(
        "ner",
        model=model_name,
        aggregation_strategy="simple",   # merge sub-tokens → full spans
        device=device_index,
        batch_size=batch_size,
    )

    city_counter: Counter = Counter()
    country_counter: Counter = Counter()

    # Truncate texts to avoid OOM on very long documents
    MAX_CHARS = 4096
    clipped = [t[:MAX_CHARS] for t in texts]

    for result in tqdm(
        ner(clipped, batch_size=batch_size),
        total=len(clipped),
        unit="doc",
        desc="NER (transformers)",
    ):
        for ent in result:
            label = ent.get("entity_group") or ent.get("entity", "")
            if label not in _HF_LOC_LABELS:
                continue
            token = _clean_entity(ent["word"])
            if len(token) < 2:
                continue
            cc = _match_country(token)
            if cc:
                country_counter[cc] += 1
                continue
            gid = _match_city(token, city_lookup, city_meta, min_pop)
            if gid:
                city_counter[gid] += 1

    return city_counter, country_counter


# ---------------------------------------------------------------------------
# 3d. spaCy backend  (GPU via spacy[cudaXXX] + en_core_web_trf)
# ---------------------------------------------------------------------------
def _ner_spacy(
    texts: list[str],
    city_lookup: dict,
    city_meta: dict,
    model_name: str,
    device_str: str,
    batch_size: int,
    min_pop: int = 0,
) -> Counter:
    try:
        import spacy
    except ImportError:
        print("spaCy not installed. Run: pip install spacy spacy[cuda12x]")
        sys.exit(1)

    if device_str == "cuda":
        activated = spacy.require_gpu()
        status = "GPU" if activated else "GPU requested but not available — using CPU"
        print(f"  spaCy device: {status}")
    else:
        print("  spaCy device: CPU")

    try:
        nlp = spacy.load(model_name, disable=["parser", "lemmatizer"])
    except OSError:
        print(f"  Model '{model_name}' not found. Run:")
        print(f"    python -m spacy download {model_name}")
        sys.exit(1)

    city_counter: Counter = Counter()
    country_counter: Counter = Counter()
    for doc in tqdm(
        nlp.pipe(texts, batch_size=batch_size),
        total=len(texts),
        unit="doc",
        desc="NER (spaCy)",
    ):
        for ent in doc.ents:
            if ent.label_ not in ("GPE", "LOC"):
                continue
            token = _clean_entity(ent.text)
            if len(token) < 2:
                continue
            cc = _match_country(token)
            if cc:
                country_counter[cc] += 1
                continue
            gid = _match_city(token, city_lookup, city_meta, min_pop)
            if gid:
                city_counter[gid] += 1

    return city_counter, country_counter


# ---------------------------------------------------------------------------
# 3e. Public entry point
# ---------------------------------------------------------------------------
def extract_city_counts(
    texts: list[str],
    city_lookup: dict,
    city_meta: dict,
    backend: str,
    model_name: str,
    device_str: str,
    device_index: int,
    batch_size: int,
    min_pop: int = 0,
) -> Counter:
    print(f"[4/5] Running NER  backend={backend}  model={model_name}  "
          f"device={'cpu' if device_index < 0 else f'cuda:{device_index}'}...")

    if backend == "transformers":
        city_counter, country_counter = _ner_transformers(
            texts, city_lookup, city_meta, model_name, device_index, batch_size, min_pop
        )
    elif backend == "spacy":
        city_counter, country_counter = _ner_spacy(
            texts, city_lookup, city_meta, model_name, device_str, batch_size, min_pop
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose: transformers | spacy")

    print(f"    -> {sum(city_counter.values()):,} city mentions across "
          f"{len(city_counter):,} distinct cities")
    print(f"    -> {sum(country_counter.values()):,} country mentions across "
          f"{len(country_counter):,} distinct countries")
    return city_counter, country_counter


# ---------------------------------------------------------------------------
# 4. Aggregate by continent & plot
# ---------------------------------------------------------------------------
def aggregate_and_plot(
    city_counter: Counter,
    country_counter: Counter,
    city_meta: dict,
    n_docs: int,
    output_path: str,
    top_hover: int = 10,
):
    print("[5/5] Aggregating and plotting...")

    # ── City dataframe ────────────────────────────────────────────────────────
    city_rows = []
    for gid, count in city_counter.items():
        meta = city_meta[gid]
        city_rows.append({
            "geonameid":  gid,
            "city":       meta["name"],
            "country":    meta["country"],
            "continent":  meta["continent"],
            "population": meta["population"],
            "lat":        meta["lat"],
            "lon":        meta["lon"],
            "mentions":   count,
        })
    city_df = pd.DataFrame(city_rows).sort_values("mentions", ascending=False)

    # ── Country dataframe ─────────────────────────────────────────────────────
    country_rows = []
    for cc, count in country_counter.items():
        continent = COUNTRY_TO_CONTINENT.get(cc, "Unknown")
        # Reverse-lookup display name from COUNTRY_NAME_LOOKUP
        display = next(
            (k.title() for k, v in COUNTRY_NAME_LOOKUP.items() if v == cc and len(k) > 3),
            cc,
        )
        country_rows.append({
            "country_code": cc,
            "country_name": display,
            "continent":    continent,
            "mentions":     count,
        })
    country_df = pd.DataFrame(country_rows).sort_values("mentions", ascending=False)

    # ── Continent summary: city + country mentions, with hover text ───────────
    cont_rows = []
    all_continents = set(city_df["continent"].unique()) | set(country_df["continent"].unique())
    for continent in all_continents:
        city_grp    = city_df[city_df["continent"] == continent]
        country_grp = country_df[country_df["continent"] == continent]

        city_total    = int(city_grp["mentions"].sum())
        country_total = int(country_grp["mentions"].sum())

        # City hover
        top_cities_k = city_grp.nlargest(top_hover, "mentions")
        city_hover = [f"<b>Top {top_hover} cities</b>"]
        for rank, (_, r) in enumerate(top_cities_k.iterrows(), 1):
            city_hover.append(f"{rank}. {r['city']} ({r['country']}) - {r['mentions']:,}")

        # Country hover
        top_countries_k = country_grp.nlargest(top_hover, "mentions")
        country_hover = [f"<b>Top {top_hover} countries</b>"]
        for rank, (_, r) in enumerate(top_countries_k.iterrows(), 1):
            country_hover.append(f"{rank}. {r['country_name']} - {r['mentions']:,}")

        cont_rows.append({
            "continent":      continent,
            "city_mentions":  city_total,
            "country_mentions": country_total,
            "color":          CONTINENT_COLORS.get(continent, "#999999"),
            "city_hover":     "<br>".join(city_hover),
            "country_hover":  "<br>".join(country_hover),
        })
    cont_df = (
        pd.DataFrame(cont_rows)
        .sort_values("city_mentions", ascending=False)
        .reset_index(drop=True)
    )

    # --- 2-panel layout: stacked bar (top) + world map (bottom) ---

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f"Mentions by Continent  (cities + countries, hover for top-{top_hover} breakdown)",
            f"Geographic Distribution of City Mentions",
        ],
        specs=[
            [{"type": "bar"}],
            [{"type": "scattergeo"}],
        ],
        vertical_spacing=0.10,
        row_heights=[0.38, 0.62],
    )

    # Lighten a hex color toward white
    def _lighten(hex_color, factor=0.45):
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        return f"#{r:02x}{g:02x}{b:02x}"

    light_colors = [_lighten(c) for c in cont_df["color"].tolist()]

    # City mentions bar (solid continent color)
    fig.add_trace(
        go.Bar(
            name="City mentions",
            x=cont_df["continent"],
            y=cont_df["city_mentions"],
            marker_color=cont_df["color"].tolist(),
            marker_line=dict(width=0),
            text=cont_df["city_mentions"].apply(lambda x: f"{x:,}"),
            textposition="inside",
            insidetextanchor="middle",
            customdata=cont_df["city_hover"].tolist(),
            hovertemplate="%{customdata}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Country mentions bar (lightened continent color, stacked on top)
    fig.add_trace(
        go.Bar(
            name="Country mentions",
            x=cont_df["continent"],
            y=cont_df["country_mentions"],
            marker_color=light_colors,
            marker_line=dict(width=0),
            text=cont_df["country_mentions"].apply(lambda x: f"{x:,}"),
            textposition="inside",
            insidetextanchor="middle",
            customdata=cont_df["country_hover"].tolist(),
            hovertemplate="%{customdata}<extra></extra>",
        ),
        row=1, col=1,
    )

    # World bubble map (cities only)
    for continent, grp in city_df.groupby("continent"):
        fig.add_trace(
            go.Scattergeo(
                lat=grp["lat"],
                lon=grp["lon"],
                mode="markers",
                marker=dict(
                    size=grp["mentions"].apply(lambda x: max(4, min(40, x**0.45))),
                    color=CONTINENT_COLORS.get(continent, "#999"),
                    opacity=0.80,
                    line=dict(width=0.4, color="#ffffff"),
                ),
                text=grp.apply(
                    lambda r: (
                        f"<b>{r['city']}</b> ({r['country']})<br>"
                        f"Mentions: {r['mentions']:,}<br>"
                        f"Population: {r['population']:,}"
                    ),
                    axis=1,
                ),
                hoverinfo="text",
                name=continent,
                showlegend=False,
            ),
            row=2, col=1,
        )

    total_city    = int(city_df["mentions"].sum())
    total_country = int(country_df["mentions"].sum()) if len(country_df) else 0
    n_cities      = len(city_df)
    n_countries = len(country_df)

    fig.update_layout(
        barmode="stack",
        title=dict(
            text=(
                f"<b>City & Country Mentions in HuggingFaceFW/fineweb-edu ({n_docs} samples)</b><br>"
                f"<sup>{total_city:,} city mentions ({n_cities:,} cities) + "
                f"{total_country:,} ({n_countries} countries) country mentions - NER-based detection</sup>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=18, color="#111111"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="center", x=0.5,
            font=dict(size=12),
        ),
        height=1050,
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
        ),
        margin=dict(t=110, b=30, l=30, r=30),
    )
    fig.update_xaxes(gridcolor="#e0e0e0", zeroline=False, linecolor="#cccccc")
    fig.update_yaxes(gridcolor="#e0e0e0", zeroline=False, linecolor="#cccccc")

    fig.write_html(output_path)
    print(f"\n  Plot saved to: {output_path}")

    # Console summary
    print("\n-- Continent summary --")
    print(cont_df[["continent", "city_mentions", "country_mentions"]].to_string(index=False))
    print("\n-- Top 20 cities --")
    print(city_df[["city", "country", "continent", "population", "mentions"]].head(20).to_string(index=False))
    if len(country_df):
        print("\n-- Top 20 countries --")
        print(country_df[["country_name", "continent", "mentions"]].head(20).to_string(index=False))

    return city_df, country_df, cont_df

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="City & country mentions in fineweb-edu by continent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cities",    default="data/cities1000.txt",
                   help="Path to GeoNames cities1000.txt")
    p.add_argument("--countries",    default="data/countryInfo.txt",
                    help="Path to GeoNames cities1000.txt")
    p.add_argument("--n_docs",    type=int, default=5_000,
                   help="Number of documents to sample")
    p.add_argument(
        "--ner", default="spacy", choices=["transformers", "spacy"],
        help=(
            "NER backend. 'transformers' runs a HuggingFace model "
            "'spacy' uses spaCy pipeline "
        ),
    )
    p.add_argument(
        "--model",
        default="en_core_web_trf",
        help=(
            "Model name. transformers : dslim/bert-large-NER, "
            "Defaults : spacy : en_core_web_trf"
        ),
    )
    p.add_argument(
        "--device", default="auto",
        help=(
            "Device to use. 'auto' picks best available GPU, "
            "'cpu' forces CPU, 'cuda:N' picks a specific GPU index"
        ),
    )
    p.add_argument("--batch_size", type=int, default=128,
                   help="Batch size fed to the NER model")
    p.add_argument("--seed",       type=int, default=42,
                   help="Shuffle seed")
    p.add_argument("--top_hover",  type=int, default=10,
                   help="Top K cities per continent shown in continent bar hover tooltip")
    p.add_argument(
        "--min_pop", type=int, default=50_000,
        help=(
            "Minimum city population for a NER match to be counted. "
            "Raises the bar so tiny towns named 'China' or 'England' are ignored. "
            "(default: 50000)"
        ),
    )
    p.add_argument("--output_dir",     default="results/",
                   help="Dir to output HTML file")
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.cities).exists():
        print(f"ERROR: cities file not found: {args.cities}")
        print("Download with:")
        print("  wget https://download.geonames.org/export/dump/cities1000.zip && unzip cities1000.zip")
        sys.exit(1)
    
    if not Path(args.countries).exists():
        print(f"ERROR: countryInfo file not found: {args.countries}")
        print("Download with:")
        print("  wget https://download.geonames.org/export/dump/countryInfo.txt")
        sys.exit(1)

    # Resolve device
    print("[0/5] Resolving compute device …")
    device_str, device_index = resolve_device(args.device)

    load_countries(args.countries)
    
    # Default models per backend
    model_name = args.model or (
        "dslim/bert-large-NER" if args.ner == "transformers"
        else "en_core_web_trf"
    )

    city_lookup, city_meta = load_cities(args.cities)
    texts                  = load_corpus(args.n_docs, seed=args.seed)

    start_time = time.time()
    city_counter, country_counter = extract_city_counts(
                                 texts, city_lookup, city_meta,
                                 backend=args.ner,
                                 model_name=model_name,
                                 device_str=device_str,
                                 device_index=device_index,
                                 batch_size=args.batch_size,
                                 min_pop=args.min_pop,
                             )
    exec_time = time.time() - start_time
    print(f"Total execution time on [{args.n_docs}] docs : {int(exec_time // 60)} m, {int(exec_time % 60)} s")

    output_path = f"{args.output_dir}/occ_city_country_ndocs_{args.n_docs}_min_pop_{args.min_pop}_{args.ner}.html"
    city_df, country_df, cont_df = aggregate_and_plot(city_counter, country_counter, city_meta, args.n_docs, output_path, args.top_hover)
    city_df.to_csv(f"{args.output_dir}/df_city.csv")
    country_df.to_csv(f"{args.output_dir}/df_country.csv")
    cont_df.to_csv(f"{args.output_dir}/df_cont_df.csv")

if __name__ == "__main__":
    main()