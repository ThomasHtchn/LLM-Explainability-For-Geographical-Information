import sqlite3
import argparse
import re
import sys
from pathlib import Path
import time
import random
import numpy as np
import re
from tqdm import tqdm

def init_db(output_path):
    print(f"[1/3] Creating tables geonames, alternatenames for GPE & LOC...")
    conn = sqlite3.connect(output_path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS geonames")
    cur.execute("DROP TABLE IF EXISTS alternatenames")
    cur.execute("DROP TABLE IF EXISTS geonames_gpe")
    cur.execute("DROP TABLE IF EXISTS geonames_loc")
    cur.execute("DROP TABLE IF EXISTS alternatenames_gpe")
    cur.execute("DROP TABLE IF EXISTS alternatenames_loc")

    cur.execute("""
    CREATE TABLE geonames_gpe (
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
        population INTEGER,
        admin_level INTEGER
    )
    """)
    cur.execute("""
    CREATE TABLE geonames_loc (
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
        population INTEGER,
        admin_level INTEGER
    )
    """)

    cur.execute("""
    CREATE TABLE alternatenames_gpe (
        geoname_id INTEGER,
        alt_name TEXT,
        alt_name_norm TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE alternatenames_loc (
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

def admin_level_helper(admins):
    for i, admin in enumerate(admins, 0):
        if(admin == ""):
            return i
    return 4

def fill_columns(conn, cur, data_path):
    print(f"[2/3] Filling tables with allCountries.txt data...")
    BATCH_SIZE = 10000
    gpe_geo_batch = []
    loc_geo_batch = []
    gpe_alt_batch = []
    loc_alt_batch = []

    cur.execute("PRAGMA journal_mode = OFF;")
    cur.execute("PRAGMA synchronous = OFF;")

    conn.execute("BEGIN TRANSACTION;") 

    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):

            parts = line.strip().split("\t")

            feature_class = parts[6]

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
            admin_level = admin_level_helper([admin1, admin2, admin3, admin4])
            population = int(parts[14]) if parts[14] else 0

            if feature_class in ("P", "A"):
                gpe_geo_batch.append((
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
                    population,
                    admin_level
                ))
            else:
                loc_geo_batch.append((
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
                    population,
                    admin_level
                ))
            # alternates name table
            if alternates:
                for alt in alternates.split(","):

                    alt = alt.strip()
                    if not alt or len(alt) < 2:
                        continue
                    # if not alt.isascii():
                    #     continue
                    if alt.lower() == name.lower():
                        continue

                    alt_norm = clean_entity(alt)

                    if feature_class in ("P", "A"):
                        gpe_alt_batch.append((
                            geoname_id,
                            alt,
                            alt_norm
                        ))
                    else:
                        loc_alt_batch.append((
                            geoname_id,
                            alt,
                            alt_norm
                        ))
            if len(gpe_geo_batch) >= BATCH_SIZE:
                cur.executemany("""
                    INSERT INTO geonames_gpe VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, gpe_geo_batch)
                gpe_geo_batch.clear()
            
            if len(loc_geo_batch) >= BATCH_SIZE:
                cur.executemany("""
                    INSERT INTO geonames_loc VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, loc_geo_batch)
                loc_geo_batch.clear()

            if len(gpe_alt_batch) >= BATCH_SIZE:
                cur.executemany("""
                    INSERT INTO alternatenames_gpe VALUES (?,?,?)
                """, gpe_alt_batch)
                gpe_alt_batch.clear()
            
            if len(loc_alt_batch) >= BATCH_SIZE:
                cur.executemany("""
                    INSERT INTO alternatenames_loc VALUES (?,?,?)
                """, loc_alt_batch)
                loc_alt_batch.clear()


            if i % 500000 == 0:
                print(f"    Processed {i:,} lines...")

    # Flush remaining
    if gpe_geo_batch:
        cur.executemany("""
            INSERT INTO geonames_gpe VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, gpe_geo_batch)

    if loc_geo_batch:
        cur.executemany("""
            INSERT INTO geonames_loc VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, loc_geo_batch)

    if gpe_alt_batch:
        cur.executemany("""
            INSERT INTO alternatenames_gpe VALUES (?,?,?)
        """, gpe_alt_batch)

    if loc_alt_batch:
        cur.executemany("""
            INSERT INTO alternatenames_loc VALUES (?,?,?)
        """, loc_alt_batch)

    conn.commit()


def create_indexes(conn, cur):
    print(f"[3/3] Creating indexes on geonames & alternatenames...")
    cur.execute("""
        CREATE INDEX idx_name_pop_gpe 
        ON geonames_gpe(name_norm, population DESC)
    """)
    cur.execute("""
        CREATE INDEX idx_name_pop_loc 
        ON geonames_loc(name_norm, population DESC)
    """)

    cur.execute("""
        CREATE INDEX idx_alt_name_norm_gpe 
        ON alternatenames_gpe(alt_name_norm)
    """)
    cur.execute("""
        CREATE INDEX idx_alt_name_norm_loc
        ON alternatenames_loc(alt_name_norm)
    """)

    cur.execute("""
        CREATE INDEX idx_alt_geoname_id_gpe
        ON alternatenames_gpe(geoname_id)
    """)
    cur.execute("""
        CREATE INDEX idx_alt_geoname_id_loc
        ON alternatenames_loc(geoname_id)
    """)

    conn.commit()


def get_coord_from_geonames(cur, table:str, entity:str):
    cur.execute(f"""
    SELECT latitude, longitude
    FROM {table}
    WHERE name_norm = ?
    ORDER BY population DESC
    LIMIT 1
    """, (entity,))

    return cur.fetchone()


def get_min_max_geoid(cur, talbe:str):
    cur.execute(f"""
    SELECT MIN(geonameid), MAX(geonameid)
    FROM {talbe}
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


def db_lookup(cur, name_norm, label):

    if label == "gpe":
        geo_table = "geonames_gpe"
        alt_table = "alternatenames_gpe"
    else:
        geo_table = "geonames_loc"
        alt_table = "alternatenames_loc"

    cur.execute(f"""
        SELECT latitude, longitude, country_code, population, admin_level, feature_class
        FROM {geo_table}
        WHERE name_norm = ?
        ORDER BY population DESC
        LIMIT 1
    """, (name_norm,))
    row1 = cur.fetchone()

    cur.execute(f"""
        SELECT g.latitude, g.longitude, g.country_code, g.population, g.admin_level, g.feature_class
        FROM {alt_table} a
        JOIN {geo_table} g ON a.geoname_id = g.geonameid
        WHERE a.alt_name_norm = ?
        ORDER BY g.population DESC
        LIMIT 1
    """, (name_norm,))
    row2 = cur.fetchone()

    # Return one with highest pop
    if row1 and row2:
        return row1 if row1[3] >= row2[3] else row2
    return row1 or row2


def get_file_nlines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
    return n_lines

def sample_window(file_path, n_lines, k):
    if n_lines <= k:
        raise ValueError("File smaller than k")

    # pick random end index
    end = random.randint(k, n_lines)
    start = end - k

    # read only the window
    lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            if i >= start and i < end:
                lines.append(clean_entity(parts[1]))
            elif i >= end:
                break

    return lines


def test_speed_db(cur, file_path, N):
    print(f"Launching lookup speedtest on geonames [{N:,d}] instances...")
    exec_times = []

    BATCH_SIZE = 1000
    n_lines = get_file_nlines(file_path)
    print(f"Number of lines in file : {n_lines:,}")
    for _ in tqdm(range(0, N, BATCH_SIZE)):
        rows = sample_window(file_path, n_lines, BATCH_SIZE)
        for i in range(BATCH_SIZE):
            entity = str(rows[i]).strip("()'',")
            label = random.choice(["gpe", "loc"])
            start_time = time.time()
            res = db_lookup(cur, entity, label)
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
    p.add_argument("--output_path", default="data/geonames.db",
                   help="Path for the database")
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.path).exists():
        print(f"ERROR: allCountries file not found: {args.path}")
        print("Download with:")
        print("  wget https://download.geonames.org/export/dump/allCountries.zip && unzip allCountries.zip")
        sys.exit(1)
    
    if(args.task == "create"):
        start_time = time.time()

        conn, cur = init_db(args.output_path)
        fill_columns(conn, cur, args.path)

        create_indexes(conn, cur)

        create_time = time.time() - start_time
        print(f"Total time to create dbs : {int(create_time // 60)} m, {int(create_time % 60)} s")
        print(f"Number of rows : ")
        tables = ["geonames_gpe", "geonames_loc", "alternatenames_gpe", "alternatenames_loc"]
        for t in tables:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            count = cur.fetchone()[0]
            print(f"    {t}: {count:,d}")
    
    if(args.task == "speed"):
        conn = sqlite3.connect("data/geonames.db")
        cur = conn.cursor()

        test_speed_db(cur, args.path, args.N)

if __name__ == "__main__":
    main()