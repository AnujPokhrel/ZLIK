#!/bin/bash

# Stop the whole script if any command fails
set -e

echo "Starting DataGeneration" 

echo "Running Data Generation..."
python3 DataGeneration/random_walk_w_damage.py  #all the configs in data_gen.yaml

echo "Encoding Damage Embeddings..."
python3 encode_damages.py --folder {folder_where_you_saved_your_generated_data, check config/data_gen.yaml} 

echo "Generating Trajectories for training data..."
python3 DataGeneration/DataExtaction.py # Be mindful of the configs in data_extraction.yaml

python3 DataGeneration/generate_stats.py --data-dir {folder_with extracted data from prev step} --out {output_file}
echo "Pipeline execution complete."

