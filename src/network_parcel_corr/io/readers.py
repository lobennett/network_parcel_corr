"""File reading utilities for neuroimaging data."""

import re
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, List, Set
from collections import defaultdict
import json
import os

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


def extract_contrast_info(
    filepath: Path,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Extract subject, session, run, and task contrast information from filepath."""
    filename = filepath.name

    # Extract subject
    subj_match = re.search(r'(sub-s\d+)', filename)
    subject = subj_match.group(1) if subj_match else None

    # Extract session
    sess_match = re.search(r'(ses-\d+)', filename)
    session = sess_match.group(1) if sess_match else None

    # Extract contrast using the pattern: everything between run-X_ and _rtmodel
    contrast_match = re.search(r'run-\d+_(.+?)_rtmodel', filename)
    contrast = contrast_match.group(1) if contrast_match else None

    # Extract run
    run_match = re.search(r'(run-\d+)', filename)
    run = run_match.group(1) if run_match else None

    return subject, session, contrast, run


def load_exclusions(exclusions_file: str) -> Set[str]:
    """Load exclusions from JSON file and return a set of exclusion keys."""
    if not os.path.exists(exclusions_file):
        raise FileNotFoundError(f'Exclusions file {exclusions_file} does not exist.')

    try:
        with open(exclusions_file, 'r') as f:
            exclusions_data = json.load(f)

        excluded_keys = set()

        # Process fMRIPrep exclusions
        for exclusion in exclusions_data.get('fmriprep_exclusions', []):
            key = f'{exclusion["subject"]}_{exclusion["session"]}_{exclusion["task"]}_{exclusion["run"]}'
            excluded_keys.add(key)

        # Process behavioral exclusions
        for exclusion in exclusions_data.get('behavioral_exclusions', []):
            key = f'{exclusion["subject"]}_{exclusion["session"]}_{exclusion["task"]}_{exclusion["run"]}'
            excluded_keys.add(key)

        return excluded_keys

    except Exception:
        return set()


def find_all_contrast_files(
    subjects: List[str], input_dir: Path, exclusions_file: str
) -> Dict[str, List[Path]]:
    """Find all effect-size contrast files for given subjects."""
    contrast_files = defaultdict(list)
    exclusions = load_exclusions(exclusions_file)

    for subject in subjects:
        subject_dir = input_dir / subject
        if not subject_dir.exists():
            continue

        # Find all effect-size files
        pattern = '*/indiv_contrasts/*effect-size.nii.gz'
        files = list(subject_dir.glob(pattern))

        for filepath in files:
            subj, session, contrast, run = extract_contrast_info(filepath)

            if not all([subj, session, run, contrast]):
                continue

            task = contrast.split('_')[0]
            exclusion_key = f'{subj}_{session}_{task}_{run}'

            if exclusion_key in exclusions:
                continue

            contrast_files[contrast].append(filepath)

    return dict(contrast_files)
