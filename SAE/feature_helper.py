from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

def inspect_neurons(model, df, hidden_dim, mute=True, device="cuda", top_k=20):
    xs = []

    for arr in df["mean_pooling"].values:
        x = torch.tensor(
            arr,
            dtype=torch.float32
        ).view(-1)
        xs.append(x)

    x = torch.stack(xs).to(device)
    model.eval()
    _, z = model(x)


    activation_sums = []
    for neuron_id in tqdm(range(0, hidden_dim)):
        
        scores = z[:, neuron_id]
        topk = torch.topk(scores, k=top_k)

        activation_sums.append(sum(topk.values))
        if not mute:
            if(sum(topk.values) > 10):
                print(f"    === Neuron {neuron_id} ===")

                print(f"total topk = {sum(topk.values):.4f}")
                for idx, score in zip(topk.indices, topk.values):
                    
                    idx = idx.item()
                    print(f"{score.item():.4f} -> {df.iloc[idx]['prompt']}")
        
    return activation_sums, z


def check_if_spare(latent_activation, threshold=1e-3):
    active = (latent_activation > threshold).float()
    print(f"Avg active neurons: {active.sum(dim=1).mean().item():.3f}")
    print(f"Density: {active.mean().item():.3f}")
    print(f"Dead neurons: {(active.sum(dim=0) == 0).float().mean().item():.3f}")
    print(f"Alive neurons: {((latent_activation.sum(dim=0) != 0).float().mean()).item():.3f}")

def plot_latent_distribution(latent_activations, bins=500):
    plt.hist(
        latent_activations.detach().cpu().numpy().flatten(),
        bins=bins,
        log=True
    )

    plt.title("Latent activation distribution")
    plt.xlabel('activation value')
    plt.show()


def plot_feature_density_histogram(
    z,
    threshold=0.0,
    bins=50,
    use_log_densities=True,
    use_non_zero=False
):
    """
    Plot histogram of SAE feature densities.

    Parameters
    ----------
    z : torch.Tensor
        Shape [num_samples, hidden_dim]
        Latent activations.

    threshold : float
        Activation threshold for considering a feature active.
        Use 0 for ReLU SAEs, e.g. 1e-3 for noisy activations.

    bins : int
        Number of histogram bins.
    """

    with torch.no_grad():

        # active[sample, feature]
        active = (z > threshold)

        # density per feature
        densities = active.float().mean(dim=0).cpu().numpy()

    # remove dead features to avoid log10(0)
    nonzero_densities = densities[densities > 0]

    if len(nonzero_densities) == 0:
        raise ValueError("All features have zero density.")

    log_densities = np.log10(nonzero_densities)

    if use_log_densities:
        plot_densities = log_densities
        x_label = "log10(feature density)"
    elif use_non_zero:
        plot_densities = nonzero_densities
        x_label = "feature density (nonzero)"
    else:
        plot_densities = densities
        x_label = "feature density"

    print(f"Features: {len(densities)}")
    print(f"Dead features: {(densities == 0).sum()} -> (%) {((densities == 0).sum() / len(densities)):.6f}")
    print(f"Mean density: {densities.mean():.6f}")
    print(f"Median density: {np.median(densities):.6f}")

    
    for q in [0.5, 0.9, 0.95, 0.99]:
        print(f"Q. {q} = {np.quantile(densities, q):.3f}")

    print(f"Nonzero mean density: {nonzero_densities.mean():.6f}")
    print(f"Nonzero median density: {np.median(nonzero_densities):.6f}")

    plt.figure(figsize=(8, 5))

    plt.hist(
        plot_densities,
        bins=bins,
    )

    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.title("Distribution of SAE Feature Densities")

    plt.grid(alpha=0.3)

    plt.show()


def print_topneurons_topk(df, z, neuron_ids, top_k=5,):
    for neuron_id in neuron_ids:
        scores = z[:, neuron_id]
        topk = torch.topk(scores, k=top_k)
        print(f"    === Neuron {neuron_id} ===")
        for i, (idx, score) in enumerate(zip(topk.indices, topk.values)): 
            idx = idx.item()
            print(f"top {i+1} -> {score.item():.4f} : {df.iloc[idx]['prompt']}")
            # if df.iloc[idx]['prompt'].split(",")[1] != " Brittany":
            #     print(f"{score.item():.4f} -> {df.iloc[idx]['prompt']}")

import sqlite3

def add_coords(df, db_path):
    conn = sqlite3.connect(db_path)
    query = """SELECT geonameid, latitude, longitude FROM geonames_gpe"""
    geo_df = pd.read_sql_query(query, conn)
    conn.close()
    df = df.merge(geo_df, on="geonameid", how="left")
    df = df.dropna(subset=["latitude", "longitude"])
    
    return df


import folium
from folium.plugins import HeatMap

def get_neuron_scores(z, neuron_id):
    scores = z[:, neuron_id].detach().cpu().numpy()
    return scores

def fetch_coordinates(conn, geonameid):
    cur = conn.cursor()
    cur.execute("""SELECT latitude, longitude FROM geonames_gpe WHERE geonameid = ?""",(int(geonameid),))
    result = cur.fetchone()
    if result is None:
        return None
    return result[0], result[1]


"""
=========   HEATMAP folium    =========
"""
def create_html_neuron_map(df, z, neuron_id, top_k=None):
    scores = get_neuron_scores(z, neuron_id)
    df = df.copy()
    df["score"] = scores
    if top_k is not None:
        df = df.sort_values("score", ascending=False).head(top_k)

    # France centered map
    m = folium.Map(
        location=[46.6, 2.5],
        zoom_start=6,
        tiles="CartoDB positron"
    )
    #heat_data = []
    for _, row in tqdm(df.iterrows(), total=len(df)):

        lat = row["latitude"]
        lon = row["longitude"]
        score = row["score"]

        color = plt_color(score)
        opacity = scale_opacity(score)
        folium.CircleMarker(
            location=[lat, lon],
            radius= 3 + 5 * abs(score),
            color=color,
            fill=True,
            #opacity=opacity,
            popup=f"{row['prompt']} | score={score:.4f}"
        ).add_to(m)
        #heat_data.append([lat, lon, float(score)])

    #HeatMap(heat_data, radius=10, blur=15, min_opacity=0.2, max_zoom=10).add_to(m)
    #save_path = f"heatmap_neuron{neuron_id}.html"
    save_path = f"plot_neuron{neuron_id}.html"
    m.save(save_path)
    print(f"Saved to {save_path}")

def plt_color(score):
    s = score
    if s > 0.5:
        return "red"
    elif s > 0.3:
        return "orange"
    elif s > 0.1:
        return "yellow"
    elif s > 0:
        return "lightgrey"
    else:
        return "blue"

def scale_opacity(score, min_opacity=0.03, max_opacity=0.9):
    s = score
    return min_opacity + s * (max_opacity - min_opacity)


"""
=========   PLOT FRANCE   =========
"""
def get_neuron_scores(z, neuron_id):
    return (z[:, neuron_id].detach().cpu().numpy())

def fetch_all_coordinates(db_path):
    conn = sqlite3.connect(db_path)
    query = """SELECT geonameid, latitude, longitude FROM geonames_gpe"""
    geo_df = pd.read_sql_query(query, conn)
    conn.close()

    return geo_df


def plot_neuron_france(df, z, neuron_id, figsize=(10, 10), top_k=None):
    scores = get_neuron_scores(z, neuron_id)

    work_df = df.copy()
    work_df["score"] = scores

    if top_k is not None:
        work_df = (work_df.sort_values("score", ascending=False).head(top_k))


    france = gpd.read_file(
        "https://france-geojson.gregoiredavid.fr/repo/regions.geojson"
    )

    fig, ax = plt.subplots(figsize=figsize)

    france.plot(
        ax=ax,
        edgecolor="black",
        color="white"
    )

    low_activations = work_df[work_df["score"] < 0.4]
    high_activations = work_df #[work_df["score"] > 0.4]

    # ax.scatter(low_activations["longitude"], low_activations["latitude"], color="lightgrey", 
    #            alpha=0.5, s=1)

    high_scores = high_activations["score"]
    sizes = 0.1 + 50 * np.log1p(high_scores)
    alphas = 0.05 + 0.5 * np.log1p(high_scores)

    scatter = ax.scatter(
        high_activations["longitude"],
        high_activations["latitude"],
        c=high_activations["score"],
        cmap="OrRd",
        alpha=alphas,
        s=sizes
    )

    plt.colorbar(
        scatter,
        ax=ax,
        label="Neuron activation",
        shrink=0.5
    )

    ax.set_title(f"Neuron {neuron_id} activations")
    
    ax.set_aspect("equal")
    
    ax.set_axis_off()
    ax.set_xlim(-5.5, 10)
    ax.set_ylim(41, 52)

    plt.show()


"""
=========   PLOT NEURON   =========
"""
def plot_neuron(df, z, neuron_id, figsize=(10, 10), top_k=None):
    scores = get_neuron_scores(z, neuron_id)

    work_df = df.copy()
    work_df["score"] = scores

    if top_k is not None:
        work_df = work_df.sort_values("score", ascending=False).head(top_k)

    # Convert activations to GeoDataFrame
    work_gdf = gpd.GeoDataFrame(
        work_df,
        geometry=gpd.points_from_xy(work_df["longitude"], work_df["latitude"]),
        crs="EPSG:4326"
    )
    work_gdf = work_gdf.to_crs("EPSG:3857")
    # Load France geometry (DOM-TOM included depending on dataset)
    # france = gpd.read_file(
    #     "https://france-geojson.gregoiredavid.fr/repo/regions.geojson"
    # ).to_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=figsize)

    # Base map
    # france.plot(
    #     ax=ax,
    #     edgecolor="black",
    #     color="white",
    #     linewidth=0.8
    # )

    scores = work_gdf["score"].values
    sizes = []
    alphas = []
    for score in scores:
        if score > 0.5:
            sizes.append(2)
            alphas.append(1)
        else:
            sizes.append(1)
            alphas.append(0.5)
    #sizes = 1 + 50 * np.log1p(scores)
    #alphas = np.clip(0.1 + 0.6 * np.log1p(scores), 0.1, 0.9)

    # Scatter plot
    scatter = ax.scatter(
        work_gdf.geometry.x,
        work_gdf.geometry.y,
        c=scores,
        cmap="viridis",
        alpha=alphas,
        s=sizes,
        linewidths=0
    )

    plt.colorbar(scatter, ax=ax, label="Neuron activation", shrink=0.5)

    ax.set_title(f"Neuron {neuron_id} activations")

    #ax.set_axis_off()
    # ax.set_aspect("equal")

    # ax.set_xlim(-12, 20)
    # ax.set_ylim(35, 60)

    plt.show()



from libpysal.weights import KNN
from esda.moran import Moran_Local

def plot_local_moran_clusters(
    df,
    scores,
    k=8,
    p_threshold=0.05,
    figsize=(10, 10),
):
    """
    Plot Local Moran's I cluster map.

    Parameters
    ----------
    df : DataFrame
        Must contain columns:
            lat
            lon

    scores : array-like
        Activation values for one neuron.
        Length must match len(df).

    k : int
        Number of nearest neighbors.

    p_threshold : float
        Significance threshold.

    figsize : tuple
    """

    work_df = df.copy()

    work_df["activation"] = scores

    # --------------------------------------------------
    # GeoDataFrame
    # --------------------------------------------------

    gdf = gpd.GeoDataFrame(
        work_df,
        geometry=gpd.points_from_xy(
            work_df["longitude"],
            work_df["latitude"]
        ),
        crs="EPSG:4326"
    )

    # France Lambert-93
    gdf = gdf.to_crs("EPSG:2154")

    # --------------------------------------------------
    # Spatial weights
    # --------------------------------------------------

    w = KNN.from_dataframe(
        gdf,
        k=k
    )

    w.transform = "R"

    # --------------------------------------------------
    # Local Moran
    # --------------------------------------------------

    lisa = Moran_Local(
        gdf["activation"].values,
        w
    )

    # --------------------------------------------------
    # Cluster labels
    # --------------------------------------------------

    sig = lisa.p_sim < p_threshold

    cluster = np.full(
        len(gdf),
        "Not significant",
        dtype=object
    )

    # Quadrants:
    # 1 HH
    # 2 LH
    # 3 LL
    # 4 HL

    cluster[(lisa.q == 1) & sig] = "High-High"
    cluster[(lisa.q == 2) & sig] = "Low-High"
    cluster[(lisa.q == 3) & sig] = "Low-Low"
    cluster[(lisa.q == 4) & sig] = "High-Low"

    gdf["cluster"] = cluster

    # --------------------------------------------------
    # Colors
    # --------------------------------------------------

    colors = {
        "High-High": "#d7191c",      # red
        "Low-Low": "#2c7bb6",        # blue
        "High-Low": "#fdae61",       # orange
        "Low-High": "#abd9e9",       # light blue
        "Not significant": "#d9d9d9"
    }

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=figsize
    )


    for name, color in colors.items():

        subset = gdf[
            gdf["cluster"] == name
        ]

        if len(subset) == 0:
            continue

        subset.plot(
            ax=ax,
            color=color,
            markersize=10,
            label=name,
            alpha=0.8
        )

    ax.legend(
        title="Local Moran Cluster"
    )

    ax.set_title(
        "Local Moran's I Cluster Map"
    )

    ax.set_axis_off()

    plt.show()

    return gdf