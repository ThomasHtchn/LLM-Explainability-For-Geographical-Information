import sqlite3
import pycountry
import pandas as pd
from tqdm import tqdm
import argparse

code_dict =  {
    'EG':'Egypt',
    'DZ':'Algeria',
    'MA':'Morocco',
    'LY':'Libya',
    'TN':'Tunisia',
    'EH':'Western Sahara',
    'SD':'Sudan'
}

def get_cities(cur, population, countries):
    placeholders = ",".join(["?"] * len(countries))

    query = (f"""
    SELECT geonameid, name, latitude, longitude, country_code, population, admin_level, admin1, admin2
    FROM geonames_gpe
    WHERE feature_class = "P" 
        and country_code in ({placeholders})
        and population > ?
    """)
    params = (*countries, population)
    cur.execute(query, params)

    return cur.fetchall()

def create_admin1_map(path):
    admin1_map = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")

            key = parts[0]
            name = parts[1]

            admin1_map[key] = name
    return admin1_map


def create_prompts(output_path, rows):#, fr, admin1_map):
    df = pd.DataFrame(columns=['geonameid', 'lat', 'lon', 'prompt'])
    prompts = []
    for line in tqdm(rows, desc="Place", total=len(rows)):
        id = line[0]
        name = line[1]
        lat = line[2]
        lon = line[3]
        cc = line[4]
        admin1 = line[7]
        # if fr:
        #     region = admin1_map.get(f"{cc}.{admin1}")
        # else:
        #     region = code_dict[cc]

        prompts.append({
            'geonameid': id,
            'lat': lat,
            'lon': lon,
            'prompt': f"{name}"#, {region}"
        })
        # sub = pd.DataFrame(prompts)
        # df = pd.concat([df, sub])

    df = pd.DataFrame(prompts)
    df.to_pickle(output_path)
    print(f"Prompts saved at '{output_path}' !")
    return prompts

def resolve_places(places_str):
    return places_str.split(",")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db_path", type=str, default="../geocoding_pipeline/geonames.db")
    parser.add_argument("--admin1_path", type=str, default="../data/admin1CodesASCII.txt")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--min_pop", type=int, required=True)
    parser.add_argument("--places", type=str, default="FR")

    args = parser.parse_args()

    db_path = args.db_path
    admin1_path = args.admin1_path
    output_path = args.output_path
    min_pop = args.min_pop

    global code_dict

    admin1_map = create_admin1_map(admin1_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # south_countires = ('EG','DZ','MA','LY','TN','EH','SD',)
    # fr_country = ('FR',)
    # fr_min_pop = 4000
    # south_min_pop = 1000

    countries = resolve_places(args.places)
    city_rows = get_cities(cur, min_pop, countries)
    print(f"Number of prompts : {len(city_rows)}")

    create_prompts(output_path, city_rows)

if __name__ == "__main__":
    main()