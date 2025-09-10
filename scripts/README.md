# Scripts

This directory contains the main analysis scripts for running parcel-based correlation analysis.

## Files

### `run_corr.py`

Main Python script that performs the parcel-based correlation analysis.

**Usage:**

```bash
uv run python3 scripts/run_corr.py --help
```

**Required arguments:**

- `--exclusions-file`: Path to JSON file containing exclusions

**Optional arguments:**

- `--subjects`: Subject IDs to analyze (default: sub-s03, sub-s10, sub-s19, sub-s29, sub-s43)
- `--input-dir`: Input directory containing subject data (default: /scratch/users/logben/poldrack_glm/level1/output)
- `--output-dir`: Output directory for results (default: /scratch/users/logben/poldrack_glm/correlations/output)
- `--atlas-parcels`: Number of Schaefer atlas parcels (default: 400)
- `--construct-contrast-map`: Path to JSON file containing construct-to-contrast mapping (default: uses built-in mapping)

**Example:**

```bash
uv run python3 scripts/run_corr.py \
    --output-dir "./output" \
    --exclusions-file "/path/to/exclusions.json" \
    --subjects sub-s03 sub-s10 sub-s19 \
    --construct-contrast-map "./example_construct_mapping.json"
```

### `run_corr.sh`

SLURM job script that runs the Python analysis script.

**Usage:**

```bash
sbatch scripts/run_corr.sh
```

Or run directly:

```bash
bash scripts/run_corr.sh
```

## Analysis Pipeline

The scripts perform the following steps:

1. **Atlas Loading**: Load Schaefer brain atlas (400 parcels by default) via TemplateFlow
2. **File Discovery**: Find all effect-size contrast files for specified subjects
3. **Data Extraction**: Extract voxel values from each parcel for each contrast
4. **HDF5 Storage**: Save organized data to `./output/all_contrasts.h5`
5. **Similarity Analysis**:
   - **Within-Subject**: Mean correlations across sessions within each subject
   - **Between-Subject**: Correlations between different subjects
   - **Across-Construct**: Correlations across contrasts within cognitive constructs
6. **Parcel Classification**:
   - **Variable**: `(within + between) < 0.1`
   - **Individual Fingerprint**: `(within - between) < 0.1`
   - **Canonical**: Default classification
7. **Results Storage**: Save correlations and classifications as HDF5 attributes

## Output

Results are saved to:

- `./output/all_contrasts.h5`: Complete HDF5 dataset with similarity values (within/between/across-construct) and classifications
- `./output/correlation_analysis.log`: Detailed analysis log
- SLURM logs (if using `run_corr.sh`): `./log/run_correlations_exclusions_*.out/err`

## Dependencies

All required dependencies are managed by `uv` and specified in the project's `pyproject.toml`.
