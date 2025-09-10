"""Main pipeline for parcel-based correlation analysis."""

from pathlib import Path
from typing import List, Dict

from .atlases.load import load_schaefer_atlas
from .io.readers import find_all_contrast_files
from .io.writers import extract_and_group_by_parcel, save_to_hdf5
from .core.similarity import (
    compute_within_subject_similarity,
    compute_between_subject_similarity,
    classify_parcels,
)


def run_analysis(
    subjects: List[str],
    input_dir: Path,
    output_dir: Path,
    exclusions_file: str,
    atlas_parcels: int = 400,
) -> Dict:
    """
    Run the complete parcel-based correlation analysis pipeline.

    Parameters
    ----------
    subjects : List[str]
        List of subject IDs to analyze
    input_dir : Path
        Directory containing subject data
    output_dir : Path
        Directory for output files
    exclusions_file : str
        Path to exclusions JSON file
    atlas_parcels : int, optional
        Number of atlas parcels to use (default: 400)

    Returns
    -------
    Dict
        Results containing within/between similarities and classifications
    """
    print(f'Starting analysis with {len(subjects)} subjects...')

    # Load atlas
    print(f'Loading Schaefer {atlas_parcels}-parcel atlas...')
    atlas_data, atlas_labels = load_schaefer_atlas(atlas_parcels)

    # Find contrast files
    print('Discovering contrast files...')
    contrast_files = find_all_contrast_files(subjects, input_dir, exclusions_file)
    print(f'Found {len(contrast_files)} contrasts')

    # Extract and group data
    print('Extracting parcel data...')
    grouped_by_contrast = {}
    for contrast_name, files in contrast_files.items():
        print(f'Processing {contrast_name} ({len(files)} files)...')
        parcel_data = extract_and_group_by_parcel(files, atlas_data, atlas_labels)
        grouped_by_contrast[contrast_name] = parcel_data

    # Save to HDF5
    print('Saving data to HDF5...')
    hdf5_path = save_to_hdf5(grouped_by_contrast, output_dir)

    # Compute similarities
    print('Computing similarities...')
    within_similarities = compute_within_subject_similarity(hdf5_path)
    between_similarities = compute_between_subject_similarity(hdf5_path)

    # Classify parcels
    print('Classifying parcels...')
    classifications = classify_parcels(within_similarities, between_similarities)

    results = {
        'hdf5_path': hdf5_path,
        'within_similarities': within_similarities,
        'between_similarities': between_similarities,
        'classifications': classifications,
        'n_contrasts': len(contrast_files),
        'n_subjects': len(subjects),
    }

    print('Analysis complete!')
    return results


def main():
    """Main function for smoke testing."""
    print('Network Parcel Correlation Analysis')
    print('This is a smoke test - use run_analysis() for actual analysis')

    # Example usage would be:
    # results = run_analysis(
    #     subjects=["sub-s01", "sub-s02"],
    #     input_dir=Path("/path/to/data"),
    #     output_dir=Path("/path/to/output"),
    #     exclusions_file="/path/to/exclusions.json"
    # )


if __name__ == '__main__':
    main()
