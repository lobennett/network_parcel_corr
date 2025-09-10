"""Modular main pipeline for parcel-based correlation analysis."""

from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

from .atlases.load import load_schaefer_atlas
from .io.readers import find_all_contrast_files
from .io.writers import extract_and_group_by_parcel, save_to_hdf5
from .core.similarity import (
    compute_within_subject_similarity,
    compute_between_subject_similarity,
    classify_parcels,
)


def load_atlas_data(atlas_parcels: int) -> Tuple[np.ndarray, List[str]]:
    """
    Load atlas data and labels.
    
    Parameters
    ----------
    atlas_parcels : int
        Number of atlas parcels to use
        
    Returns
    -------
    Tuple[np.ndarray, List[str]]
        Atlas data array and list of parcel labels
    """
    print(f'Loading Schaefer {atlas_parcels}-parcel atlas...')
    return load_schaefer_atlas(atlas_parcels)


def discover_contrast_files(
    subjects: List[str], input_dir: Path, exclusions_file: str
) -> Dict[str, List[Path]]:
    """
    Discover and filter contrast files.
    
    Parameters
    ----------
    subjects : List[str]
        List of subject IDs to analyze
    input_dir : Path
        Directory containing subject data
    exclusions_file : str
        Path to exclusions JSON file
        
    Returns
    -------
    Dict[str, List[Path]]
        Dictionary mapping contrast names to file lists
    """
    print('Discovering contrast files...')
    contrast_files = find_all_contrast_files(subjects, input_dir, exclusions_file)
    print(f'Found {len(contrast_files)} contrasts')
    return contrast_files


def extract_parcel_data(
    contrast_files: Dict[str, List[Path]], 
    atlas_data: np.ndarray, 
    atlas_labels: List[str]
) -> Dict:
    """
    Extract and group parcel data from contrast files.
    
    Parameters
    ----------
    contrast_files : Dict[str, List[Path]]
        Dictionary mapping contrast names to file lists
    atlas_data : np.ndarray
        Atlas data array
    atlas_labels : List[str]
        List of parcel labels
        
    Returns
    -------
    Dict
        Dictionary mapping contrast names to grouped parcel data
    """
    print('Extracting parcel data...')
    grouped_by_contrast = {}
    
    total_contrasts = len(contrast_files)
    for i, (contrast_name, files) in enumerate(contrast_files.items(), 1):
        print(f'Processing {contrast_name} ({len(files)} files) [{i}/{total_contrasts}]...')
        try:
            parcel_data = extract_and_group_by_parcel(files, atlas_data, atlas_labels)
            grouped_by_contrast[contrast_name] = parcel_data
            print(f'✓ Completed {contrast_name} - found {len(parcel_data)} parcels with data')
        except Exception as e:
            print(f'✗ Failed {contrast_name}: {e}')
            raise
        
    return grouped_by_contrast


def compute_all_similarities(hdf5_path: Path) -> Tuple[Dict, Dict]:
    """
    Compute within and between subject similarities.
    
    Parameters
    ----------
    hdf5_path : Path
        Path to HDF5 file containing parcel data
        
    Returns
    -------
    Tuple[Dict, Dict]
        Within and between subject similarities
    """
    print('Computing similarities...')
    within_similarities = compute_within_subject_similarity(hdf5_path)
    between_similarities = compute_between_subject_similarity(hdf5_path)
    return within_similarities, between_similarities


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
    atlas_data, atlas_labels = load_atlas_data(atlas_parcels)

    # Find contrast files
    contrast_files = discover_contrast_files(subjects, input_dir, exclusions_file)

    # Extract and group data
    grouped_by_contrast = extract_parcel_data(contrast_files, atlas_data, atlas_labels)

    # Save to HDF5
    print('Saving data to HDF5...')
    hdf5_path = save_to_hdf5(grouped_by_contrast, output_dir)

    # Compute similarities
    within_similarities, between_similarities = compute_all_similarities(hdf5_path)

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
