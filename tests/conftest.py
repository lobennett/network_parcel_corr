"""Pytest configuration and fixtures for network_parcel_corr tests."""

import tempfile
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
def empty_parcel_atlas():
    """Create an atlas with an empty parcel (no voxels)."""
    atlas = np.zeros((10, 10, 10), dtype=np.int32)
    atlas[0:3, 0:3, 0:3] = 1  # Parcel 1
    atlas[5:8, 5:8, 5:8] = 2  # Parcel 2
    # Parcel 3 doesn't exist (empty)
    atlas[0:2, 8:10, 0:2] = 4  # Parcel 4 (skip 3)
    return atlas


@pytest.fixture
def single_voxel_parcel_atlas():
    """Create an atlas with single-voxel parcels."""
    atlas = np.zeros((10, 10, 10), dtype=np.int32)
    atlas[0, 0, 0] = 1  # Single voxel parcel
    atlas[5:7, 5:7, 5:7] = 2  # Regular parcel
    atlas[9, 9, 9] = 3  # Another single voxel
    return atlas


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
