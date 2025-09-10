"""Test input reading functionality."""

import numpy as np
import nibabel as nib
import pytest
from pathlib import Path
import tempfile

from network_parcel_corr.io import load_nifti, InvalidNiftiError


class TestNIFTILoader:
    """Test NIFTI loader functionality."""

    def test_load_nifti_valid_file(self):
        """Test loading a valid NIfTI file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a sample 3D NIfTI image
            data = np.random.rand(64, 64, 30)
            img = nib.Nifti1Image(data, affine=np.eye(4))

            # Save to temporary file
            nifti_path = Path(temp_dir) / 'test_image.nii.gz'
            nib.save(img, nifti_path)

            # Test loading
            loaded_img = load_nifti(nifti_path)

            # Verify the loaded image
            assert isinstance(loaded_img, nib.Nifti1Image)
            np.testing.assert_array_equal(loaded_img.get_fdata(), data)
            np.testing.assert_array_equal(loaded_img.affine, np.eye(4))

    def test_load_nifti_string_path(self):
        """Test loading NIfTI file with string path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a sample NIfTI image
            data = np.random.rand(32, 32, 20)
            img = nib.Nifti1Image(data, affine=np.eye(4))

            # Save to temporary file
            nifti_path = Path(temp_dir) / 'test_string.nii'
            nib.save(img, nifti_path)

            # Test loading with string path
            loaded_img = load_nifti(str(nifti_path))

            # Verify the loaded image
            assert isinstance(loaded_img, nib.Nifti1Image)
            np.testing.assert_array_equal(loaded_img.get_fdata(), data)

    def test_load_nifti_nonexistent_file(self):
        """Test loading a non-existent NIfTI file raises error."""
        nonexistent_path = Path('nonexistent_file.nii.gz')

        with pytest.raises(InvalidNiftiError, match='File does not exist'):
            load_nifti(nonexistent_path)

    def test_load_nifti_none_path(self):
        """Test loading with None path raises error."""
        with pytest.raises(InvalidNiftiError, match='File path cannot be None'):
            load_nifti(None)

    def test_load_nifti_invalid_file(self):
        """Test loading an invalid file raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a text file instead of NIfTI
            invalid_path = Path(temp_dir) / 'invalid.nii.gz'
            invalid_path.write_text('This is not a NIfTI file')

            with pytest.raises(InvalidNiftiError, match='Failed to load Nifti file'):
                load_nifti(invalid_path)
