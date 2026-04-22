#!/usr/bin/env bash

set -e 

RUN_DB=false

for arg in "$@"; do
  case $arg in
    --db)
      RM_DB=TRUE
      shift
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

rm -rf geocoding_output.csv geo_cache.pkl allCountries.zip entities_output.pkl 
echo "Removed result files"

if [ "$RM_DB" = true ]; then
    rm -rf geonames.db
    echo "Removed DB"
fi