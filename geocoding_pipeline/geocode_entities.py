import sqlite3
import requests
import time
import pickle
from pathlib import Path
import argparse
import sys
from tqdm import tqdm
import pandas as pd
from create_geodb import clean_entity, db_lookup


API_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "spaCy-geocoder"

ent_not_found = {}
ent_found = []
proportions = [0,0,0]

def lookup_nominatim(name):
    try:
        params = {
            "q": name,
            "format": "json",
            "limit": 1
        }

        headers = {"User-Agent": USER_AGENT}

        r = requests.get(API_URL, params=params, headers=headers, timeout=5)
        data = r.json()

        if not data:
            return None

        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
        return (lat, lon, None, None)

    except Exception:
        return None

def lookup_photon(query):
    url = "https://photon.komoot.io/api"

    params = {
        "q": query,
        "limit": 1,
        "lang": "en"
    }

    headers = {"User-Agent": USER_AGENT}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=5)
        r.raise_for_status()
        data = r.json()

        if data["features"]:
            props = data["features"][0]["properties"]
            coords = data["features"][0]["geometry"]["coordinates"]

            return (
                coords[1],                 # lat
                coords[0],                 # lon
                props.get("countrycode"),  # country
                None,                      # Photon has no population
                None,                      # admin level
                None                       # geonames feature class
            )

    except Exception as e:
        print(f"[Photon ERROR] {query}: {e}")
        return None

    return None


def resolve_entity(cur, entity, label, count, cache):

    entity_norm = clean_entity(entity)
    
    res_cache = cache.get(entity_norm)
    if res_cache:
        proportions[0] += 1
        ent_found.append(res_cache)
        return res_cache, "cache"

    result = db_lookup(cur, entity_norm, label.lower())
    if result:
        cache[entity_norm] = result
        proportions[1] += 1
        ent_found.append(result)
        #print(f"res = {result}")
        return result, "db"

    # API FALLBACK
    result_api = lookup_photon(entity_norm)
    if result_api:
        cache[entity_norm] = result_api
        proportions[2] += 1
        ent_found.append(result_api)
        return result_api, "api"

    cache[entity_norm] = None
    ent_not_found[entity_norm] =  count

    return None, None


def geocode_entities(cur, entities, cache):
    results = {}
    success = 0

    pbar = tqdm(entities.items(), total=len(entities), unit="entity", desc="DB/API lookup")

    for i, (entity, attributs) in enumerate(pbar, 1):
        label = attributs["label"]
        count = attributs["count"]
        geo, found_with = resolve_entity(cur, entity, label, count, cache)

        if geo:
            success += 1
        
        success_rate = success / i * 100
        pbar.set_description(f"Success: {success_rate:.1f}% ({success}/{i})")

        results[entity] = {
            "count": count,
            "label": label,
            "inference": found_with,
            "geo": geo  # (lat, lon, country, population, admin_level, feature_class)
        }

    return results, success_rate


def save_df(dict, path):
    rows = []

    for entity, data in dict.items():
        geo = data["geo"]

        if geo:
            try:
                lat, lon, country, population, admin_level, feature_class = geo
            except Exception as e: 
                print(f"Exception [{e}] on : {entity} -> {geo}")
                sys.exit("FINI")
                lat, lon, country, population = geo
                admin_level = None
        else:
            lat, lon, country, population, admin_level, feature_class = None, None, None, None, None, None

        rows.append({
            "entity": entity,
            "count": data["count"],
            "ner_label": data["label"],
            "inference": data["inference"],
            "lat": lat,
            "lon": lon,
            "country": country,
            "population": population,
            "admin_level": admin_level,
            "feature_class": feature_class
        })

    df = pd.DataFrame(rows)
    #df.to_csv("entities_geocoded.csv", index=False, quoting=csv.QUOTE_ALL)
    df.to_csv(path, sep=";", index=False)
    print(f"Geocoding results saved at : {path}")


def save_cache(cache_path, cache):
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"Saved cache at : {cache_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Geocode entities from fineweb",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_path",
                help="Path to pickle file containing named entities.")
    p.add_argument("--cache_path",    default="data/geo_cache.pkl",
                help="Path to look-up cache")
    p.add_argument("--db_path",    default="data/geonames.db",
                help="Path to DB")
    p.add_argument("--output_path", default="results/geocoding/geocoded_entities.csv",
                   help="Path to save the geocoding csv")
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.input_path).exists():
        print(f"ERROR: Entity file not found: {args.input_path}")
        print("Create with:")
        print("     python extract_entities.py --output_path <output path> --n_docs <ndocs> --dataset <HuggingFaceFW/fineweb>")
        sys.exit(1)

    if not Path(args.db_path).exists():
        print(f"ERROR: DB not found: {args.db_path}")
        print("Create with:")
        print("     python create_geodb.py --path <data/allCountries.txt> --task <create>")
        sys.exit(1)

    try:
        with open(args.cache_path, "rb") as f:
            print(f"Using cache from : {args.cache_path}")
            cache = pickle.load(f)
    except FileNotFoundError:
        print("Using empty cache")
        cache = {}

    conn = sqlite3.connect(args.db_path)
    cur = conn.cursor()

    with open(args.input_path, "rb") as f:
        entities = pickle.load(f)

    start_time = time.time()
    results, sucess_rate = geocode_entities(cur, entities, cache)
    total_time = time.time() - start_time

    print(f"Total time to geocode dict : {int(total_time // 60)} m, {int(total_time % 60)} s")
    print(f"Sucess rate of geocoding : {sucess_rate:.1f} % on {len(entities):,} entities")
    print(f"Proportions :\n     CACHE : {proportions[0]:,}\n     DB    : {proportions[1]:,}\n     API   : {proportions[2]:,}")
    save_df(results, args.output_path)

    save_cache(args.cache_path, cache)
    

if __name__ == "__main__":
    main()
