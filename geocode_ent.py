import sqlite3
import requests
import time
import pickle
from pathlib import Path
import argparse
import sys
from tqdm import tqdm
from create_geodb import clean_entity, db_lookup


API_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "spaCy-geocoder"

CACHE_PATH = ""

cache = {}
ent_not_found = []


# ----------------------------
# API FALLBACK
# using OpenStreetMap Nominatim
# ----------------------------

def lookup_api(name):
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

# ----------------------------
# MAIN RESOLVER
# ----------------------------

def resolve_entity(cur, entity):
    key = clean_entity(entity)

    # # 1. CACHE HIT
    # if key in cache:
    #     return cache[key]

    # 2. SQLITE LOOKUP
    result = db_lookup(cur, key)

    if result:
        cache[key] = result
        return result

    # # 3. API FALLBACK
    # result = lookup_api(entity)

    # if result:
    #     cache[key] = result
    #     time.sleep(1)  # respect rate limits
    #     return result

    # 4. NOT FOUND
    # cache[key] = None
    ent_not_found.append(key)

    return None

# ----------------------------
# PROCESS COUNTER
# ----------------------------

def geocode_entities(cur, counter_dict):
    results = {}

    for entity, count in tqdm(counter_dict.items(), total=len(counter_dict)):
        geo = resolve_entity(cur, entity)

        results[entity] = {
            "count": count,
            "geo": geo  # (lat, lon, country, population)
        }

    return results

# ----------------------------
# SAVE CACHE
# ----------------------------

def save_cache():
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def parse_args():
    p = argparse.ArgumentParser(
        description="Geocode entities from fineweb",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--counter_path",
                help="Path to Counter containing named entities and their occurences from fineweb as a plk file.")
    p.add_argument("--cache_path",    default="data/geo_cache.pkl",
                help="Path to look-up cache")
    p.add_argument("--db_path",    default="data/geonames.db",
                help="Path to DB")
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.counter_path).exists():
        print(f"ERROR: counter file not found: {args.counter_path}")
        print("Create with:")
        print("     python extract_geoent.py --output_dir <output dir> --n_docs <ndocs> --dataset <HuggingFaceFW/fineweb>")
        sys.exit(1)

    if not Path(args.db_path).exists():
        print(f"ERROR: DB not found: {args.db_path}")
        print("Create with:")
        print("     python create_geodb.py --path <data/allCountries.txt> --task <create>")
        sys.exit(1)
    
    # CACHE_PATH = args.cache_path

    # try:
    #     with open(CACHE_PATH, "rb") as f:
    #         cache = pickle.load(f)
    # except FileNotFoundError:
    #     print("Using empty cache")

    conn = sqlite3.connect(args.db_path)
    cur = conn.cursor()

    with open(args.counter_path, "rb") as f:
        ent_counter = pickle.load(f)
    start_time = time.time()
    results = geocode_entities(cur, ent_counter)
    total_time = time.time() - start_time

    print(f"Total time to geocode dict : {int(total_time // 60)} m, {int(total_time % 60)} s")
    
    print(f"Entities Found : {len(results)}")
    print(f"Entities Not Found : {len(ent_not_found)}")
    # path_ents = f"results/geo_entity_counter_{dataset_str}_ndocs_{args}_{args.ner}.plk"
    # results = geocode_entities(example_counter)
    # for k, v in results.items():
    #     print(k, v)


if __name__ == "__main__":
    main()
