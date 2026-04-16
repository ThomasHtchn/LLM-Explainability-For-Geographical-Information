import sqlite3
import argparse
import re
import sys
from pathlib import Path
import time
import random
import numpy as np
import re

def init_db():
    print(f"[1/3] Creating tables geonames & alternatenames...")
    conn = sqlite3.connect("data/geonames.db")
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS geonames")
    cur.execute("DROP TABLE IF EXISTS alternatenames")

    cur.execute("""
    CREATE TABLE geonames (
        geonameid INTEGER PRIMARY KEY,
        name TEXT,
        name_norm TEXT,
        asciiname TEXT,
        latitude REAL,
        longitude REAL,
        feature_class TEXT,
        feature_code TEXT,
        country_code TEXT,
        admin1 TEXT,
        admin2 TEXT,
        admin3 TEXT,
        admin4 TEXT,
        population INTEGER
    )
    """)

    cur.execute("""
    CREATE TABLE alternatenames (
        geoname_id INTEGER,
        alt_name TEXT,
        alt_name_norm TEXT
    )
    """)

    conn.commit()
    return conn, cur


def clean_entity(text: str) -> str:
    text = text.lower()                     # lower case
    text = re.sub(r"(?:'s|’s)$", "", text)  # possessive removal
    text = re.sub(r"\bthe\b", "", text)     # remove "the" as a word only
    text = text.strip(" .,;:\"'()[]")       # remove any leading and tailing caracters present in the ("")
    text = re.sub(r"\s+", " ", text)        # normalize spaces
    return text


def fill_columns(conn, cur, data_path):
    print(f"[2/3] Filling tables with allCountries.txt data...")
    BATCH_SIZE = 10000
    geo_batch = []
    alt_batch = []

    cur.execute("PRAGMA journal_mode = OFF;")
    cur.execute("PRAGMA synchronous = OFF;")

    conn.execute("BEGIN TRANSACTION;") 

    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):

            parts = line.strip().split("\t")

            feature_class = parts[6]
            if feature_class not in ("P", "A"):
                continue

            geoname_id = int(parts[0])
            name = parts[1]
            name_norm = clean_entity(name)
            asciiname = parts[2]
            alternates = parts[3]
            lat = float(parts[4])
            lon = float(parts[5])
            feature_code = parts[7]
            country_code = parts[8]
            admin1 = parts[10]
            admin2 = parts[11]
            admin3 = parts[12]
            admin4 = parts[13]
            population = int(parts[14]) if parts[14] else 0

            # main table
            geo_batch.append((
                geoname_id,
                name,
                name_norm,
                asciiname,
                lat,
                lon,
                feature_class,
                feature_code,
                country_code,
                admin1,
                admin2,
                admin3,
                admin4,
                population
            ))

            # alternates name table
            if alternates:
                for alt in alternates.split(","):

                    alt = alt.strip()
                    if not alt:
                        continue
                    if not alt.isascii() or len(alt) < 2:
                        continue
                    if alt.lower() == name.lower():
                        continue

                    alt_norm = clean_entity(alt)

                    alt_batch.append((
                        geoname_id,
                        alt,
                        alt_norm
                    ))

            if len(geo_batch) >= BATCH_SIZE:
                cur.executemany("""
                    INSERT INTO geonames VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, geo_batch)
                geo_batch.clear()

            if len(alt_batch) >= BATCH_SIZE:
                cur.executemany("""
                    INSERT INTO alternatenames VALUES (?,?,?)
                """, alt_batch)
                alt_batch.clear()

            if i % 500000 == 0:
                print(f"    Processed {i} lines...")

    # Flush remaining
    if geo_batch:
        cur.executemany("""
            INSERT INTO geonames VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, geo_batch)

    if alt_batch:
        cur.executemany("""
            INSERT INTO alternatenames VALUES (?,?,?)
        """, alt_batch)

    conn.commit()


def create_indexes(conn, cur):
    print(f"[3/3] Creating indexes on geonames & alternatenames...")
    cur.execute("""
        CREATE INDEX idx_name_pop 
        ON geonames(name_norm, population DESC)
    """)

    cur.execute("""
        CREATE INDEX idx_alt_name_norm 
        ON alternatenames(alt_name_norm)
    """)

    conn.commit()


def get_coord_from_geonames(cur, entity:str):
    cur.execute("""
    SELECT latitude, longitude
    FROM geonames
    WHERE name_norm = ?
    ORDER BY population DESC
    LIMIT 1
    """, (entity,))

    return cur.fetchone()


def get_min_max_geoid(cur):
    cur.execute("""
    SELECT MIN(geonameid), MAX(geonameid)
    FROM geonames
    """)
    min_id, max_id = cur.fetchone()
    return min_id, max_id


def get_random_entities(cur, N):
    min_id, max_id = get_min_max_geoid(cur)
    ids = random.sample(range(min_id, max_id), N*3)

    cur.execute("""
    SELECT name_norm
    FROM geonames
    WHERE geonameid IN ({})
    """.format(",".join("?" * len(ids))), ids)

    rows = cur.fetchall()
    return rows[:N]


def db_lookup(cur, name):
    name_norm = clean_entity(name)

    cur.execute("""
        SELECT latitude, longitude, country_code, population
        FROM geonames
        WHERE name_norm = ?
        ORDER BY population DESC
        LIMIT 1
    """, (name_norm,))
    
    row = cur.fetchone()
    if row:
        return row

    cur.execute("""
        SELECT g.latitude, g.longitude, g.country_code, g.population
        FROM alternatenames a
        JOIN geonames g ON a.geoname_id = g.geonameid
        WHERE a.alt_name_norm = ?
        ORDER BY g.population DESC
        LIMIT 1
    """, (name_norm,))

    return cur.fetchone()


#nb_lines = 5738951
def test_speed_db(cur, N):
    print(f"Launching lookup speedtest on geonames [{N:,d}] instances...")
    exec_times = []

    BATCH_SIZE = 10000
    for _ in range(0, N, BATCH_SIZE):
        rows = get_random_entities(cur, BATCH_SIZE)
        for i in range(BATCH_SIZE):
            entity = str(rows[i]).strip("()'',")
            start_time = time.time()
            res = db_lookup(cur, entity)
            exec_time = time.time() - start_time
            exec_times.append(exec_time)
    
    mean_time = np.mean(exec_times)
    print(f"    -> Mean look-up time  :  {mean_time:.8f} s.")
    print(f"    -> Worst look-up time : {np.max(exec_times):.8f} s.")
    print(f"    -> Total look-up time : {np.sum(exec_times):.8f} s.")
    print(f"        ({int(np.sum(exec_times) // 60)} m, {int(np.sum(exec_times) % 60)} s)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Create and fill geonames DB with data from allCountries.txt dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--path",    default="data/allCountries.txt",
                   help="Path to GeoNames allCountries.txt")
    p.add_argument("--task", default="create", choices=["create", "speed"],
                   help="Task to be performe, create & fill tables / test lookup speed")
    p.add_argument("--N", type=int, default=1_000,
                   help="Number of instances used for the lookup speed test")
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.path).exists():
        print(f"ERROR: allCountries file not found: {args.cities}")
        print("Download with:")
        print("  wget https://download.geonames.org/export/dump/allCountries.zip && unzip allCountries.zip")
        sys.exit(1)
    
    if(args.task == "create"):
        start_time = time.time()

        conn, cur = init_db()
        fill_columns(conn, cur, args.path)
        create_indexes(conn, cur)

        create_time = time.time() - start_time
        print(f"Total time to create dbs : {int(create_time // 60)} m, {int(create_time % 60)} s")
        print(f"Number of rows : ")
        tables = ["geonames", "alternatenames"]
        for t in tables:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            count = cur.fetchone()[0]
            print(f"    {t}: {count:,d}")
    
    if(args.task == "speed"):
        conn = sqlite3.connect("data/geonames.db")
        cur = conn.cursor()

        test_speed_db(cur, args.N)

if __name__ == "__main__":
    main()