# Network Parcel Correlation Analysis

A neuroimaging analysis pipeline for computing parcel-based correlation patterns in fMRI data. This package analyzes neural activation patterns across cognitive tasks and constructs by extracting activation data from brain parcels (using Schaefer atlas) and computing various correlation measures.

## Overview

This pipeline performs four distinct correlation analyses for each brain parcel:

1. **Within-Subject Similarity** - Computes mean correlations across sessions within each individual subject
2. **Between-Subject Similarity** - Calculates correlations between different subjects across all sessions
3. **Across-Construct Similarity** - Analyzes correlations across contrasts within cognitive constructs
4. **Parcel Classification** - Classifies parcels as Variable, Individual Fingerprint, or Canonical based on correlation patterns

## Installation

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
git clone https://github.com/lobennett/network_parcel_corr
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
    --subjects sub-s03 sub-s10 sub-s19 \
    --construct-contrast-map "/path/to/custom_mapping.json"
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
- **Quality Control**: Validates voxel count within parcels

### Stage 3: Data Storage

- **HDF5 Format**: Saves all processed data to `./output/all_contrasts.h5`
- **Hierarchical Structure**: `contrast → parcel → subject_session_run → voxel_values`
- **Metadata**: Stores subject, session, and statistical information as attributes

### Stage 4: Similarity Analysis

- **Within-Subject**: Mean of upper triangle correlations across sessions for each subject
- **Between-Subject**: Mean of upper triangle correlations across all sessions, all subjects

### Stage 5: Parcel Classification

Each parcel is classified based on the following criteria:

- **Variable Parcels**: `(within_correlation + between_correlation) < 0.1`
- **Individual Fingerprint**: `NOT Variable AND (within_correlation - between_correlation) < 0.1`
- **Canonical**: Default classification for remaining parcels

## Project Structure

```
network_parcel_corr/
├── src/network_parcel_corr/
│   ├── atlases/           # Atlas loading (Schaefer via TemplateFlow)
│   ├── core/              # Core similarity calculations
│   ├── data/              # Default construct-contrast mappings
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
  - Across-construct similarity values as HDF5 attributes
  - Parcel classifications
  - Global summary statistics

### CSV Reports

The analysis automatically generates comprehensive CSV reports for detailed analysis:

- **`parcel_classifications.csv`**: Detailed classifications for all contrast-parcel combinations
- **`classification_summary.csv`**: Summary statistics by contrast and overall totals
- **`most_fingerprint_parcels.csv`**: Top parcels ranked by individual fingerprint strength 
- **`most_variable_parcels.csv`**: Top parcels ranked by variability (instability)
- **`most_canonical_parcels.csv`**: Top parcels ranked by canonical activation patterns
- **`cross_contrast_consistency.csv`**: Classification consistency across contrasts

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
        ├── across_construct_similarity_[construct_name]
        └── parcel_classification
```

## Configuration

### Command Line Arguments

| Argument                   | Description                             | Default                                                |
| -------------------------- | --------------------------------------- | ------------------------------------------------------ |
| `--subjects`               | Subject IDs to analyze                  | sub-s03, sub-s10, sub-s19, sub-s29, sub-s43            |
| `--input-dir`              | Input directory with subject data       | /scratch/users/logben/poldrack_glm/level1/output       |
| `--output-dir`             | Output directory for results            | /scratch/users/logben/poldrack_glm/correlations/output |
| `--atlas-parcels`          | Number of Schaefer atlas parcels        | 400                                                    |
| `--exclusions-file`        | Path to JSON exclusions file            | **Required**                                           |
| `--construct-contrast-map` | Path to JSON construct-contrast mapping | Uses default mapping                                   |

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

### Construct-Contrast Mapping Format

The construct-contrast mapping defines which contrasts belong to each cognitive construct for across-construct similarity analysis:

```json
{
  "Working Memory": [
    "task-nBack_contrast-match-mismatch",
    "task-nBack_contrast-twoBack-oneBack"
  ],
  "Cognitive Control": [
    "task-flanker_contrast-incongruent-congruent",
    "task-stopSignal_contrast-go"
  ],
  "Task Switching": [
    "task-cuedTS_contrast-cue_switch_cost",
    "task-spatialTS_contrast-cue_switch_cost"
  ]
}
```

If no custom mapping is provided via `--construct-contrast-map`, the system uses a comprehensive default mapping defined in `src/network_parcel_corr/data/construct_mappings.py` with 11 cognitive constructs including Active Maintenance, Flexible Updating, Monitoring, Interference Control, and others.

## Detailed Metrics and Classifications

### Parcel Classifications

Each brain parcel is classified into one of three categories based on its correlation patterns:

#### Variable Parcels
- **Criteria**: `(within_correlation + between_correlation) < 0.1`
- **Interpretation**: Parcels showing low reliability both within and across subjects
- **Meaning**: These parcels have inconsistent activation patterns that may reflect noise, measurement artifacts, or highly variable neural responses

#### Individual Fingerprint Parcels  
- **Criteria**: `NOT variable AND (within_correlation - between_correlation) > 0.1`
- **Interpretation**: Parcels with high within-subject consistency but low between-subject similarity
- **Meaning**: These parcels show reliable individual differences - they activate consistently within a person but differently across people, making them useful for individual identification

#### Canonical Parcels
- **Criteria**: All remaining parcels (neither Variable nor Individual Fingerprint)
- **Interpretation**: Parcels with similar activation patterns both within and across subjects
- **Meaning**: These parcels show consistent, generalizable activation patterns that are similar across the population

### CSV Report Details

#### 1. Parcel Classifications (`parcel_classifications.csv`)
**Columns**:
- `contrast`: Task contrast name (e.g., "task-nBack_contrast-twoBack-oneBack")
- `parcel`: Brain parcel name (e.g., "7Networks_LH_Vis_1")
- `within_similarity`: Mean correlation across sessions within each subject, then averaged across subjects
- `between_similarity`: Mean correlation between different subjects across all sessions
- `fingerprint_strength`: `within_similarity - between_similarity` (higher = more individual-specific)
- `variability_score`: `within_similarity + between_similarity` (lower = more variable/unreliable)
- `classification`: Final parcel classification (Variable/Individual Fingerprint/Canonical)

**Computation**:
- Within-subject similarity: For each subject, compute correlations between all session pairs, extract upper triangle, take mean. Average across all subjects.
- Between-subject similarity: Compute correlations between all subject-session pairs from different subjects, extract upper triangle, take mean.

#### 2. Classification Summary (`classification_summary.csv`)
**Columns**:
- `contrast`: Task contrast name (or "Overall" for global statistics)
- `canonical_count`: Number of canonical parcels
- `fingerprint_count`: Number of individual fingerprint parcels  
- `variable_count`: Number of variable parcels
- `total_parcels`: Total parcels analyzed
- `canonical_percentage`: Percentage of parcels classified as canonical
- `fingerprint_percentage`: Percentage of parcels classified as individual fingerprint
- `variable_percentage`: Percentage of parcels classified as variable

#### 3. Fingerprint Ranking (`most_fingerprint_parcels.csv`)
**Columns**:
- `rank`: Ranking position (1 = strongest fingerprint)
- `contrast`: Task contrast name
- `parcel`: Brain parcel name
- `fingerprint_strength`: `within_similarity - between_similarity`
- `classification`: Parcel classification
- `within_similarity`: Within-subject correlation
- `between_similarity`: Between-subject correlation

**Purpose**: Identifies parcels that are most reliable for individual identification

#### 4. Variability Ranking (`most_variable_parcels.csv`)
**Columns**:
- `rank`: Ranking position (1 = most variable/unreliable)
- `contrast`: Task contrast name  
- `parcel`: Brain parcel name
- `variability_score`: `within_similarity + between_similarity` (lower = more variable/unreliable)
- `classification`: Parcel classification
- `within_similarity`: Within-subject correlation
- `between_similarity`: Between-subject correlation

**Purpose**: Identifies parcels with the most unstable/unreliable activation patterns (lowest `within + between` correlations)

#### 5. Canonicality Ranking (`most_canonical_parcels.csv`)
**Columns**:
- `rank`: Ranking position (1 = most canonical)
- `contrast`: Task contrast name
- `parcel`: Brain parcel name
- `canonicality_score`: `within_similarity - |within_similarity - between_similarity|`
- `classification`: Parcel classification
- `within_similarity`: Within-subject correlation
- `between_similarity`: Between-subject correlation

**Purpose**: Identifies parcels with the most consistent, generalizable activation patterns

#### 6. Cross-Contrast Consistency (`cross_contrast_consistency.csv`)
**Columns**:
- `parcel`: Brain parcel name
- `canonical_proportion`: Proportion of contrasts where parcel was classified as canonical
- `fingerprint_proportion`: Proportion of contrasts where parcel was classified as individual fingerprint
- `variable_proportion`: Proportion of contrasts where parcel was classified as variable
- `total_contrasts`: Total number of contrasts this parcel appeared in
- `most_common_classification`: Most frequent classification across contrasts
- `consistency_score`: Maximum proportion (higher = more consistent classification)

**Purpose**: Evaluates how consistently each parcel is classified across different task contrasts

### Interpretation Guidelines

**Fingerprint Strength:**
- **Strong Individual Differences**: > 0.3 (good for personalization/individual differences studies)
- **Moderate Individual Differences**: 0.1 - 0.3 (some individual specificity)
- **Weak Individual Differences**: < 0.1 (similar patterns across individuals)

**Variability Score (within + between correlations):**
- **Highly Variable/Unreliable**: < 0.2 (consider excluding from analyses)
- **Moderately Stable**: 0.2 - 1.0 (typical range for many brain regions)
- **Highly Stable**: > 1.0 (very consistent activation patterns)

**Cross-Contrast Consistency:**
- **Highly Consistent**: > 0.8 (stable classification across tasks)
- **Moderately Consistent**: 0.6 - 0.8 (mostly stable with some variation)
- **Inconsistent**: < 0.6 (classification varies substantially across tasks)

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

## License

[MIT](./LICENSE)

## Contributing

Contributions are welcome! Please ensure all tests pass and follow the existing code style.

## Support

For questions and issues, please [open an issue](https://github.com/lobennett/network_parcel_corr/issues) on GitHub.