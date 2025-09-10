#!/bin/bash
#SBATCH --job-name=run_correlations_exclusions
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=./log/run_correlations_exclusions_%j.out
#SBATCH --error=./log/run_correlations_exclusions_%j.err
#SBATCH --partition=normal,russpold

# Script to run parcel-based correlation analysis

script=./scripts/run_corr.py

# Create log directory if it doesn't exist
mkdir -p ./log

# Run on subjects specified after --subjects flag (space-deliminated)
# Add --exclusions-file flag if you have an exclusions file
# Add --construct-contrast-map flag if you have a custom JSON mapping file
uv run python3 $script --output-dir "./output" \
    --exclusions-file "/scratch/users/logben/poldrack_glm/exclusions.json" \
    --subjects sub-s03 sub-s10

# Example with full subject list (commented out):
# uv run python3 $script --output-dir "./output" \
#     --exclusions-file "/scratch/users/logben/poldrack_glm/exclusions.json" \
#     --subjects sub-s03 sub-s10 sub-s19 sub-s29 sub-s43 \
#     sub-s76 sub-s180 sub-s216 sub-s247 sub-s286 sub-s295 sub-s300 sub-s320 sub-s321 sub-s336 \
#     sub-s373 sub-s394 sub-s415 sub-s480 sub-s599 sub-s645 sub-s874 sub-s956 sub-s1035 sub-s1057 \
#     sub-s1058 sub-s1127 sub-s1134 sub-s1175 sub-s1189 sub-s1258 sub-s1267 sub-s1270 sub-s1273 sub-s1292 \
#     sub-s1314 sub-s1326 sub-s1338 sub-s1351 sub-s1391 sub-s1399 sub-s1402 sub-s1408 sub-s1445 sub-s1481 \
#     sub-s1486