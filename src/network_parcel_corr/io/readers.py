"""File reading utilities for neuroimaging data."""

from pathlib import Path
from typing import Union

import nibabel as nib


class InvalidNiftiError(Exception):
    """Custom exception for invalid Nifti files."""

    pass


def load_nifti(filepath: Union[str, Path, None]) -> nib.Nifti1Image:
    """
    Load a Nifti file from disk.

    Parameters
    ----------
    filepath : str, Path, or None
        Path to the Nifti file (.nii or .nii.gz).

    Returns
    -------
    nibabel.Nifti1Image
        Loaded Nifti image.

    Raises
    ------
    InvalidNiftiError
        If the file path is invalid or the file cannot be loaded.
    """
    if filepath is None:
        raise InvalidNiftiError('File path cannot be None')

    file_path = Path(filepath)

    if not file_path.exists():
        raise InvalidNiftiError(f'File does not exist: {file_path}')

    try:
        img = nib.load(file_path)
    except Exception as e:
        raise InvalidNiftiError(f'Failed to load Nifti file {file_path}: {e}')

    return img
