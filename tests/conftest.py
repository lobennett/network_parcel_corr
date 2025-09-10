"""Pytest configuration and fixtures for network_parcel_corr tests."""

import tempfile
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def synthetic_3d_data():
    """Create synthetic 3D neuroimaging data."""
    data_3d = np.random.randn(10, 10, 10).astype(np.float32)
    return data_3d


@pytest.fixture
def synthetic_3d_nifti(synthetic_3d_data, temp_dir):
    """Create a synthetic 3D Nifti file."""
    affine = np.eye(4)
    affine[0, 0] = 3.0
    affine[1, 1] = 3.0
    affine[2, 2] = 3.0

    img = nib.Nifti1Image(synthetic_3d_data, affine)
    filepath = temp_dir / 'synthetic_3d.nii.gz'
    nib.save(img, filepath)
    return filepath


@pytest.fixture
def test_atlas_data():
    """Create a simple test atlas with 5 parcels."""
    atlas = np.zeros((10, 10, 10), dtype=np.int32)
    # Create 5 distinct parcels
    atlas[0:2, 0:2, 0:2] = 1  # Parcel 1
    atlas[3:5, 3:5, 3:5] = 2  # Parcel 2
    atlas[6:8, 6:8, 6:8] = 3  # Parcel 3
    atlas[0:2, 8:10, 0:2] = 4  # Parcel 4
    atlas[8:10, 0:2, 8:10] = 5  # Parcel 5
    return atlas


@pytest.fixture
def test_atlas_nifti(test_atlas_data, temp_dir):
    """Create a test atlas Nifti file."""
    affine = np.eye(4)
    affine[0, 0] = 3.0
    affine[1, 1] = 3.0
    affine[2, 2] = 3.0

    img = nib.Nifti1Image(test_atlas_data, affine)
    filepath = temp_dir / 'test_atlas.nii.gz'
    nib.save(img, filepath)
    return filepath


@pytest.fixture
def non_existent_file(temp_dir):
    """Return path to a non-existent file."""
    return temp_dir / 'non_existent.nii.gz'


@pytest.fixture
def invalid_nifti_file(temp_dir):
    """Create an invalid file that's not a valid Nifti."""
    filepath = temp_dir / 'invalid.nii.gz'
    filepath.write_text('This is not a valid nifti file')
    return filepath


@pytest.fixture
def sample_dataset(temp_dir):
    """Create a sample dataset

    Dataset structure:
    - Files follow pattern: */indiv_contrasts/*effect-size.nii.gz
    - Naming: sub-s01_ses-01_run-01_task-flanker_contrast-incongruent-congruent_rtmodel-rt_centered_stat-effect-size.nii.gz
    """
    dataset = {
        'base_dir': temp_dir,
        'subjects': ['sub-s01', 'sub-s02'],
        'file_paths': [],
        'exclusions_file': temp_dir / 'exclusions.json',
    }

    # Create exclusions file
    exclusions = {'fmriprep_exclusions': [], 'behavioral_exclusions': []}
    with open(dataset['exclusions_file'], 'w') as f:
        json.dump(exclusions, f)

    # Define test contrasts
    contrasts = [
        'task-flanker_contrast-incongruent-congruent',
        'task-nBack_contrast-high-load-low-load',
        'task-shapeMatching_contrast-mainvars',
    ]

    # Create synthetic contrast map data
    np.random.seed(42)
    base_data = np.random.randn(10, 10, 10).astype(np.float32)
    affine = np.eye(4)

    # Generate files for 2 subjects, 2 sessions, 3 contrasts each
    for subject_id in dataset['subjects']:
        subject_dir = temp_dir / subject_id

        for ses_idx in range(1, 3):  # 2 sessions each
            session_id = f'ses-{ses_idx:02d}'
            session_dir = subject_dir / session_id / 'indiv_contrasts'
            session_dir.mkdir(parents=True, exist_ok=True)

            for run_idx, contrast in enumerate(contrasts, 1):
                run_id = f'run-{run_idx:02d}'

                # Create filename
                filename = f'{subject_id}_{session_id}_{run_id}_{contrast}_rtmodel-rt_centered_stat-effect-size.nii.gz'
                filepath = session_dir / filename

                # Create unique data for each file
                seed_val = hash(f'{subject_id}_{session_id}_{run_id}_{contrast}') % 1000
                np.random.seed(seed_val)
                data = base_data + np.random.randn(10, 10, 10) * 0.1

                img = nib.Nifti1Image(data, affine)
                nib.save(img, filepath)
                dataset['file_paths'].append(filepath)

    return dataset


@pytest.fixture
def sample_dataset_flat_structure(temp_dir):
    """Create a sample dataset with flat directory structure (all files in one directory).

    Same dataset as sample_dataset but with all contrast maps in a single directory
    for testing different organizational patterns.
    """
    dataset = {'base_dir': temp_dir, 'file_paths': []}

    # Define the three contrasts
    contrasts = [
        'flanker_incongruent-congruent',
        'nBack_high-load-low-load',
        'shapeMatching_mainvars',
    ]

    # Create synthetic contrast map data
    np.random.seed(42)  # For reproducible test data
    contrast_data = np.random.randn(64, 64, 30).astype(np.float32)
    affine = np.eye(4)
    affine[0, 0] = 3.0
    affine[1, 1] = 3.0
    affine[2, 2] = 3.0

    # Generate all contrast maps in flat structure
    for sub_idx in range(1, 3):  # 2 subjects
        for ses_idx in range(1, 6):  # 5 sessions
            for run_idx, contrast in enumerate(contrasts, 1):  # 3 contrasts
                subject_id = f'sub-s{sub_idx:02d}'
                session_id = f'ses-{ses_idx:02d}'
                run_id = f'run-{run_idx:02d}'
                contrast_name = f'{subject_id}_{session_id}_{run_id}_{contrast}'

                # Create contrast map file in flat structure
                contrast_filename = f'{contrast_name}.nii.gz'
                contrast_filepath = temp_dir / contrast_filename

                # Add variation to contrast data with consistent seeding
                seed_val = hash(contrast_name) % 1000
                np.random.seed(seed_val)
                varied_data = contrast_data + np.random.randn(64, 64, 30) * 0.1
                img = nib.Nifti1Image(varied_data, affine)
                nib.save(img, contrast_filepath)

                dataset['file_paths'].append(contrast_filepath)

    return dataset
