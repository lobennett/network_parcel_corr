# Network Parcel Correlation Analysis

A neuroimaging analysis pipeline for computing parcel-based correlation patterns in fMRI data. This package analyzes neural activation patterns across cognitive tasks and constructs by extracting activation data from brain parcels (using Schaefer atlas) and computing various correlation measures.

## Overview

This pipeline performs three distinct correlation analyses for each brain parcel:

1. **Within-Subject Similarity** - Computes mean correlations across sessions within each individual subject
2. **Between-Subject Similarity** - Calculates correlations between different subjects across all sessions
3. **Parcel Classification** - Classifies parcels as Variable, Individual Fingerprint, or Canonical based on correlation patterns

## Installation

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
git clone <repository-url>
cd network_parcel_corr
uv sync
```

## Quick Start

### Running the Analysis

#### Command Line Interface

```bash
uv run python3 scripts/run_corr.py \
    --output-dir "./output" \
    --exclusions-file "/path/to/exclusions.json" \
    --subjects sub-s03 sub-s10 sub-s19
```

#### SLURM Job Submission

```bash
sbatch scripts/run_corr.sh
```

#### Programmatic Usage

```python
from network_parcel_corr.main import run_analysis
from pathlib import Path

results = run_analysis(
    subjects=["sub-s01", "sub-s02"],
    input_dir=Path("/path/to/data"),
    output_dir=Path("/path/to/output"),
    exclusions_file="/path/to/exclusions.json",
    atlas_parcels=400
)
```

## Data Pipeline

### Stage 1: Data Acquisition and Preprocessing

- **Atlas Loading**: Loads Schaefer brain atlas (400 parcels by default) via TemplateFlow
- **File Discovery**: Scans subject directories for fMRI contrast files matching pattern `*/indiv_contrasts/*effect-size.nii.gz`
- **Exclusions**: Filters out excluded runs based on JSON exclusions file

### Stage 2: Feature Extraction

- **Parcel Extraction**: Extracts voxel values from each parcel for each contrast
- **Data Organization**: Groups data by contrast type and parcel
- **Quality Control**: Validates voxel count consistency within parcels

### Stage 3: Data Storage

- **HDF5 Format**: Saves all processed data to `./output/all_contrasts.h5`
- **Hierarchical Structure**: `contrast → parcel → subject_session_run → voxel_values`
- **Metadata**: Stores subject, session, and statistical information as attributes

### Stage 4: Similarity Analysis

- **Within-Subject**: Mean of upper triangle correlations across sessions for each subject
- **Between-Subject**: Mean of upper triangle correlations across all sessions, all subjects
- **Statistical Robustness**: Uses correlation matrices with proper upper-triangle extraction

### Stage 5: Parcel Classification

Each parcel is classified using sequential evaluation:

- **Variable Parcels**: `(within_correlation + between_correlation) < 0.1`
- **Individual Fingerprint**: `(within_correlation - between_correlation) < 0.1`
- **Canonical**: Default classification for remaining parcels

## Project Structure

```
network_parcel_corr/
├── src/network_parcel_corr/
│   ├── atlases/           # Atlas loading (Schaefer via TemplateFlow)
│   ├── core/              # Core similarity calculations
│   ├── io/                # File I/O operations
│   └── main.py            # Main pipeline orchestration
├── scripts/               # Analysis execution scripts
│   ├── run_corr.py        # Main analysis script
│   ├── run_corr.sh        # SLURM job script
│   └── README.md          # Scripts documentation
├── tests/                 # Comprehensive test suite
└── README.md              # This file
```

## Output

The analysis produces several output files:

### Primary Results

- **`./output/all_contrasts.h5`**: Complete dataset with:
  - Raw voxel data organized by contrast/parcel/session
  - Within/between-subject similarity values as HDF5 attributes
  - Parcel classifications
  - Global summary statistics

### Logs and Reports

- **`./output/correlation_analysis.log`**: Detailed analysis log with progress and results
- **`./log/run_correlations_exclusions_*.out`**: SLURM job stdout (if using job submission)
- **`./log/run_correlations_exclusions_*.err`**: SLURM job stderr (if using job submission)

### Example Results Structure

```python
# HDF5 file structure
/contrast_name/parcel_name/subject_session_run/
    ├── voxel_values (dataset)
    └── attributes:
        ├── subject
        ├── session
        ├── within_subject_similarity
        ├── between_subject_similarity
        └── parcel_classification
```

## Configuration

### Command Line Arguments

| Argument            | Description                       | Default                                                |
| ------------------- | --------------------------------- | ------------------------------------------------------ |
| `--subjects`        | Subject IDs to analyze            | sub-s03, sub-s10, sub-s19, sub-s29, sub-s43            |
| `--input-dir`       | Input directory with subject data | /scratch/users/logben/poldrack_glm/level1/output       |
| `--output-dir`      | Output directory for results      | /scratch/users/logben/poldrack_glm/correlations/output |
| `--atlas-parcels`   | Number of Schaefer atlas parcels  | 400                                                    |
| `--exclusions-file` | Path to JSON exclusions file      | **Required**                                           |

### Exclusions File Format

```json
{
  "fmriprep_exclusions": [
    {
      "subject": "sub-s01",
      "session": "ses-01",
      "task": "flanker",
      "run": "run-01"
    }
  ],
  "behavioral_exclusions": [
    {
      "subject": "sub-s02",
      "session": "ses-02",
      "task": "nBack",
      "run": "run-02"
    }
  ]
}
```

## Development

### Running Tests

```bash
uv run python -m pytest tests/ -v
```

### Key Development Principles

- **Test-Driven Development**: Comprehensive test suite with 21+ tests
- **Modular Design**: Clean separation of concerns across modules
- **Error Handling**: Robust error handling and logging throughout

### Testing Framework

- Uses `pytest` for all testing
- Mock datasets with realistic fMRI data structures
- Coverage for all major pipeline components
- Integration tests for end-to-end workflows

## Dependencies

Core dependencies managed by `uv`:

- `nibabel`: NIFTI file I/O
- `nilearn`: Neuroimaging data processing
- `numpy`: Numerical computations
- `scipy`: Statistical functions
- `pandas`: Data organization
- `h5py`: HDF5 file operations
- `templateflow`: Brain atlas access

## Citation

If you use this software in your research, please cite:

```bibtex
@software{network_parcel_corr,
  title={Network Parcel Correlation Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/lobennett/network_parcel_corr}
}
```

## License

[MIT](./LICENSE)

## Contributing

Contributions are welcome! Please ensure all tests pass and follow the existing code style.

## Support

For questions and issues, please [open an issue](https://github.com/lobennett/network_parcel_corr/issues) on GitHub.
