#!/usr/bin/env bash
set -e

N_DOCS=${1:-100}

source $(conda info --base)/etc/profile.d/conda.sh
conda activate smollm

# Optional safety checks
[ -f geonames.db ] || { echo "Missing geonames.db. Run setup first."; exit 1; }

echo "Extracting entities..."
python extract_entities.py \
    --n_docs ${N_DOCS} \
    --dataset HuggingFaceFW/fineweb-edu \
    --output_path entities_output.pkl

echo "Geocoding..."
python geocode_entities.py \
    --input_path entities_output.pkl \
    --cache_path geo_cache.pkl \
    --db_path geonames.db \
    --output_path geocoding_output.csv

echo "Run complete."