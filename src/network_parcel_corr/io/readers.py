"""Modular file reading utilities for neuroimaging data."""

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


def extract_subject_id(filename: str) -> Optional[str]:
    """
    Extract subject ID from filename.
    
    Parameters
    ----------
    filename : str
        Filename to parse
        
    Returns
    -------
    Optional[str]
        Subject ID or None if not found
    """
    match = re.search(r'(sub-s\d+)', filename)
    return match.group(1) if match else None


def extract_session_id(filename: str) -> Optional[str]:
    """
    Extract session ID from filename.
    
    Parameters
    ----------
    filename : str
        Filename to parse
        
    Returns
    -------
    Optional[str]
        Session ID or None if not found
    """
    match = re.search(r'(ses-\d+)', filename)
    return match.group(1) if match else None


def extract_contrast_name(filename: str) -> Optional[str]:
    """
    Extract contrast name from filename.
    
    Parameters
    ----------
    filename : str
        Filename to parse
        
    Returns
    -------
    Optional[str]
        Contrast name or None if not found
    """
    # Extract task name
    task_match = re.search(r'task-([^_]+)', filename)
    if not task_match:
        return None
    task = task_match.group(1)
    
    # Extract contrast name  
    contrast_match = re.search(r'contrast-([^_]+(?:-[^_]+)*)', filename)
    if not contrast_match:
        return None
    contrast = contrast_match.group(1)
    
    return f'task-{task}_contrast-{contrast}'


def extract_run_id(filename: str) -> Optional[str]:
    """
    Extract run ID from filename.
    
    Parameters
    ----------
    filename : str
        Filename to parse
        
    Returns
    -------
    Optional[str]
        Run ID or None if not found
    """
    match = re.search(r'run-(\d+)', filename)
    if match:
        run_num = match.group(1)
        # Ensure it's zero-padded to 2 digits
        return f'run-{run_num.zfill(2)}'
    return None


def extract_contrast_info(
    filepath: Path,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Extract subject, session, run, and task contrast information from filepath.
    
    Parameters
    ----------
    filepath : Path
        Path to the contrast file
        
    Returns
    -------
    Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]
        (subject, session, contrast, run) or None values if not found
    """
    filename = filepath.name
    
    return (
        extract_subject_id(filename),
        extract_session_id(filename), 
        extract_contrast_name(filename),
        extract_run_id(filename)
    )


def create_exclusion_key(subject: str, session: str, task: str, run: str) -> str:
    """
    Create standardized exclusion key from components.
    
    Parameters
    ----------
    subject : str
        Subject ID
    session : str
        Session ID
    task : str
        Task name
    run : str
        Run ID
        
    Returns
    -------
    str
        Formatted exclusion key
    """
    return f'{subject}_{session}_{task}_{run}'


def parse_exclusion_entry(exclusion: Dict) -> str:
    """
    Parse a single exclusion entry into a standardized key.
    
    Parameters
    ----------
    exclusion : Dict
        Exclusion dictionary with subject, session, task, run keys
        
    Returns
    -------
    str
        Formatted exclusion key
    """
    return create_exclusion_key(
        exclusion["subject"], 
        exclusion["session"], 
        exclusion["task"], 
        exclusion["run"]
    )


def load_exclusions(exclusions_file: str) -> Set[str]:
    """
    Load exclusions from JSON file and return a set of exclusion keys.
    
    Parameters
    ----------
    exclusions_file : str
        Path to exclusions JSON file
        
    Returns
    -------
    Set[str]
        Set of exclusion keys
        
    Raises
    ------
    FileNotFoundError
        If exclusions file does not exist
    """
    if not os.path.exists(exclusions_file):
        raise FileNotFoundError(f'Exclusions file {exclusions_file} does not exist.')

    try:
        with open(exclusions_file, 'r') as f:
            exclusions_data = json.load(f)

        excluded_keys = set()

        # Process fMRIPrep exclusions
        for exclusion in exclusions_data.get('fmriprep_exclusions', []):
            excluded_keys.add(parse_exclusion_entry(exclusion))

        # Process behavioral exclusions
        for exclusion in exclusions_data.get('behavioral_exclusions', []):
            excluded_keys.add(parse_exclusion_entry(exclusion))

        return excluded_keys

    except Exception:
        return set()


def find_subject_contrast_files(subject_dir: Path, exclusions: Set[str]) -> List[Tuple[str, Path]]:
    """
    Find contrast files for a single subject.
    
    Parameters
    ----------
    subject_dir : Path
        Path to subject directory
    exclusions : Set[str]
        Set of exclusion keys
        
    Returns
    -------
    List[Tuple[str, Path]]
        List of (contrast_name, filepath) tuples
    """
    if not subject_dir.exists():
        return []
        
    pattern = '*/indiv_contrasts/*effect-size.nii.gz'
    files = list(subject_dir.glob(pattern))
    
    valid_files = []
    for filepath in files:
        subj, session, contrast, run = extract_contrast_info(filepath)
        
        if not all([subj, session, run, contrast]):
            continue
            
        task = contrast.split('_')[0]
        exclusion_key = create_exclusion_key(subj, session, task, run)
        
        if exclusion_key not in exclusions:
            valid_files.append((contrast, filepath))
            
    return valid_files


def find_all_contrast_files(
    subjects: List[str], input_dir: Path, exclusions_file: str
) -> Dict[str, List[Path]]:
    """
    Find all effect-size contrast files for given subjects.
    
    Parameters
    ----------
    subjects : List[str]
        List of subject IDs
    input_dir : Path
        Directory containing subject data
    exclusions_file : str
        Path to exclusions JSON file
        
    Returns
    -------
    Dict[str, List[Path]]
        Dictionary mapping contrast names to lists of file paths
    """
    contrast_files = defaultdict(list)
    exclusions = load_exclusions(exclusions_file)

    for subject in subjects:
        subject_dir = input_dir / subject
        subject_files = find_subject_contrast_files(subject_dir, exclusions)
        
        for contrast, filepath in subject_files:
            contrast_files[contrast].append(filepath)

    return dict(contrast_files)
