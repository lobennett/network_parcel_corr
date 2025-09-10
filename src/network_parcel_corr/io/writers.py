"""Modular file writing utilities for HDF5 results."""

from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import h5py
from nilearn import image


def create_label_to_name_mapping(atlas_labels: List[str]) -> Dict[int, str]:
    """
    Create mapping from atlas indices to parcel names.
    
    Parameters
    ----------
    atlas_labels : List[str]
        List of parcel names
        
    Returns
    -------
    Dict[int, str]
        Mapping from 1-based indices to parcel names
    """
    return {i: name for i, name in enumerate(atlas_labels, start=1)}


def extract_parcel_voxels(
    img_data: np.ndarray, atlas_data: np.ndarray, parcel_idx: int
) -> np.ndarray:
    """
    Extract voxel values for a specific parcel.
    
    Parameters
    ----------
    img_data : np.ndarray
        Image data array
    atlas_data : np.ndarray
        Atlas data with parcel labels
    parcel_idx : int
        Parcel index to extract
        
    Returns
    -------
    np.ndarray
        Voxel values for the parcel
    """
    parcel_mask = atlas_data == parcel_idx
    return img_data[parcel_mask]


def process_single_contrast_file(
    filepath: Path, 
    atlas_data: np.ndarray, 
    label_to_name_map: Dict[int, str]
) -> Dict[str, Tuple[str, str, str, str, np.ndarray]]:
    """
    Process a single contrast file and extract parcel data.
    
    Parameters
    ----------
    filepath : Path
        Path to contrast file
    atlas_data : np.ndarray
        Atlas data with parcel labels
    label_to_name_map : Dict[int, str]
        Mapping from parcel indices to names
        
    Returns
    -------
    Dict[str, Tuple[str, str, str, str, np.ndarray]]
        Dictionary mapping parcel names to (subject, session, contrast, run, voxel_values)
    """
    from .readers import extract_contrast_info
    
    parcel_data = {}
    
    try:
        subj, session, contrast, run = extract_contrast_info(filepath)
        
        if not all([subj, session, contrast, run]):
            return parcel_data
            
        img_data = image.load_img(filepath).get_fdata()
        
        for parcel_idx, parcel_name in label_to_name_map.items():
            voxel_values = extract_parcel_voxels(img_data, atlas_data, parcel_idx)
            
            if voxel_values.size > 0:
                parcel_data[parcel_name] = (subj, session, contrast, run, voxel_values)
                
    except Exception:
        pass
        
    return parcel_data


def extract_and_group_by_parcel(
    filepaths: List[Path], atlas_data: np.ndarray, atlas_labels: List[str]
) -> Dict[str, List[Tuple[str, str, str, str, np.ndarray]]]:
    """
    Extract voxel values from contrast files and group them by parcel.

    Parameters
    ----------
    filepaths : List[Path]
        List of contrast file paths
    atlas_data : np.ndarray
        Atlas data with parcel labels
    atlas_labels : List[str]
        List of parcel names

    Returns
    -------
    Dict[str, List[Tuple[str, str, str, str, np.ndarray]]]
        Dictionary mapping parcel names to lists of (subject, session, contrast, run, voxel_values)
    """
    grouped_by_parcel = defaultdict(list)
    label_to_name_map = create_label_to_name_mapping(atlas_labels)

    for filepath in filepaths:
        parcel_data = process_single_contrast_file(filepath, atlas_data, label_to_name_map)
        
        for parcel_name, record in parcel_data.items():
            grouped_by_parcel[parcel_name].append(record)

    return dict(grouped_by_parcel)


def validate_parcel_voxel_consistency(
    parcel_name: str, records: List[Tuple], contrast_name: str
) -> None:
    """
    Validate that all records in a parcel have consistent voxel counts.
    
    Parameters
    ----------
    parcel_name : str
        Name of the parcel
    records : List[Tuple]
        List of (subject, session, contrast, run, voxel_values) tuples
    contrast_name : str
        Name of the contrast
        
    Raises
    ------
    ValueError
        If voxel counts are inconsistent
    """
    voxel_lengths = [len(record[4]) for record in records]
    if len(set(voxel_lengths)) > 1:
        raise ValueError(
            f"Inconsistent voxel counts in parcel '{parcel_name}' for contrast '{contrast_name}': {voxel_lengths}"
        )


def create_record_name(subject: str, session: str, run: str) -> str:
    """
    Create standardized record name from components.
    
    Parameters
    ----------
    subject : str
        Subject ID
    session : str
        Session ID
    run : str
        Run ID
        
    Returns
    -------
    str
        Formatted record name
    """
    return f'{subject}_{session}_{run}'


def create_hdf5_record_group(
    parcel_group, subject: str, session: str, contrast: str, run: str, parcel_data: np.ndarray
):
    """
    Create an individual record group in HDF5 parcel group.
    
    Parameters
    ----------
    parcel_group : h5py.Group
        Parent parcel group
    subject : str
        Subject ID
    session : str
        Session ID
    contrast : str
        Contrast name
    run : str
        Run ID
    parcel_data : np.ndarray
        Voxel data for this record
    """
    record_name = create_record_name(subject, session, run)
    record_group = parcel_group.create_group(record_name)
    record_group.attrs['subject'] = subject
    record_group.attrs['session'] = session
    record_group.attrs['mean_voxel_value'] = np.mean(parcel_data)
    record_group.create_dataset('voxel_values', data=parcel_data)


def create_hdf5_parcel_group(
    contrast_group, parcel_name: str, records: List[Tuple]
):
    """
    Create a parcel group in HDF5 contrast group.
    
    Parameters
    ----------
    contrast_group : h5py.Group
        Parent contrast group
    parcel_name : str
        Name of the parcel
    records : List[Tuple]
        List of (subject, session, contrast, run, voxel_values) tuples
    """
    parcel_group = contrast_group.create_group(parcel_name)
    parcel_group.attrs['n_records'] = len(records)
    parcel_group.attrs['n_voxels'] = len(records[0][4])

    for subject, session, contrast, run, parcel_data in records:
        create_hdf5_record_group(parcel_group, subject, session, contrast, run, parcel_data)


def create_hdf5_contrast_group(
    hdf5_file, contrast_name: str, grouped_by_parcel: Dict
):
    """
    Create a contrast group in HDF5 file.
    
    Parameters
    ----------
    hdf5_file : h5py.File
        HDF5 file object
    contrast_name : str
        Name of the contrast
    grouped_by_parcel : Dict
        Dictionary mapping parcel names to record lists
    """
    # Validate voxel consistency
    for parcel_name, records in grouped_by_parcel.items():
        validate_parcel_voxel_consistency(parcel_name, records, contrast_name)

    # Create contrast group
    contrast_group = hdf5_file.create_group(contrast_name)
    contrast_group.attrs['contrast_name'] = contrast_name
    contrast_group.attrs['n_parcels'] = len(grouped_by_parcel)

    for parcel_name, records in grouped_by_parcel.items():
        create_hdf5_parcel_group(contrast_group, parcel_name, records)


def save_to_hdf5(grouped_by_contrast: Dict, output_dir: Path) -> Path:
    """
    Save grouped contrast data to a single combined HDF5 file.
    
    Parameters
    ----------
    grouped_by_contrast : Dict
        Dictionary mapping contrast names to parcel data
    output_dir : Path
        Output directory path
        
    Returns
    -------
    Path
        Path to created HDF5 file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_hdf5_path = output_dir / 'all_contrasts.h5'
    
    # Remove existing file if it exists to ensure clean start
    if combined_hdf5_path.exists():
        combined_hdf5_path.unlink()

    with h5py.File(combined_hdf5_path, 'w') as f:
        # Add top-level metadata
        f.attrs['n_contrasts'] = len(grouped_by_contrast)
        f.attrs['contrast_names'] = list(grouped_by_contrast.keys())

        for contrast_name, grouped_by_parcel in grouped_by_contrast.items():
            create_hdf5_contrast_group(f, contrast_name, grouped_by_parcel)

    return combined_hdf5_path
