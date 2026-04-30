#!/usr/bin/env bash
set -e

ENV_NAME="xai_env"
YAML_FILE="../env.yaml"

echo "Creating/updating env..."
conda env create -f ${YAML_FILE} -n ${ENV_NAME} || conda env update -f ${YAML_FILE}

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

echo "Downloading GeoNames..."
wget -nc https://download.geonames.org/export/dump/allCountries.zip
python -c "import zipfile; zipfile.ZipFile('allCountries.zip').extractall('.')"

echo "Creating database..."
python create_geodb.py \
    --path allCountries.txt \
    --task create \
    --output_path geonames.db

echo "Installing spaCy model..."
python -m spacy download en_core_web_trf

echo "Setup complete."