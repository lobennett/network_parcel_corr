"""Core similarity calculation functions with modular design."""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import h5py


def compute_correlation_matrix_upper_triangle(data_matrix: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix and extract upper triangle values.
    
    Parameters
    ----------
    data_matrix : np.ndarray
        Matrix with observations as rows
        
    Returns
    -------
    np.ndarray
        Upper triangle correlation values (excluding diagonal)
    """
    if data_matrix.shape[0] < 2:
        return np.array([])
        
    corr_matrix = np.corrcoef(data_matrix)
    upper_tri = np.triu(corr_matrix, k=1)
    return upper_tri[upper_tri != 0]


def extract_subject_sessions_from_parcel(parcel_group) -> Dict[str, List[np.ndarray]]:
    """
    Extract voxel values organized by subject from HDF5 parcel group.
    
    Parameters
    ----------
    parcel_group : h5py.Group
        HDF5 group containing parcel data
        
    Returns
    -------
    Dict[str, List[np.ndarray]]
        Subject ID mapped to list of session voxel values
    """
    subject_data = defaultdict(list)
    for record_name in parcel_group.keys():
        record = parcel_group[record_name]
        subject = record.attrs['subject']
        voxel_values = record['voxel_values'][:]
        subject_data[subject].append(voxel_values)
    return dict(subject_data)


def compute_within_subject_correlation(sessions: List[np.ndarray]) -> Optional[float]:
    """
    Compute mean correlation within a single subject's sessions.
    
    Parameters
    ----------
    sessions : List[np.ndarray]
        List of voxel value arrays for different sessions
        
    Returns
    -------
    Optional[float]
        Mean within-subject correlation, None if insufficient data
    """
    if len(sessions) < 2:
        return None
        
    session_matrix = np.column_stack(sessions)
    upper_tri_values = compute_correlation_matrix_upper_triangle(session_matrix.T)
    
    return np.mean(upper_tri_values) if len(upper_tri_values) > 0 else None


def compute_within_subject_similarity(hdf5_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Compute within-subject mean correlations for each contrast-parcel.
    
    Parameters
    ----------
    hdf5_path : Path
        Path to HDF5 file containing parcel data
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dict: {contrast_name: {parcel_name: correlation}}
    """
    results = {}

    with h5py.File(hdf5_path, 'r') as f:
        for contrast_name in f.keys():
            contrast_group = f[contrast_name]
            results[contrast_name] = {}

            for parcel_name in contrast_group.keys():
                parcel_group = contrast_group[parcel_name]
                subject_data = extract_subject_sessions_from_parcel(parcel_group)

                subject_correlations = []
                for subject, sessions in subject_data.items():
                    correlation = compute_within_subject_correlation(sessions)
                    if correlation is not None:
                        subject_correlations.append(correlation)

                if subject_correlations:
                    results[contrast_name][parcel_name] = np.mean(subject_correlations)

    return results


def extract_session_info_from_parcel(parcel_group) -> List[Tuple[np.ndarray, str]]:
    """
    Extract session information from HDF5 parcel group.
    
    Parameters
    ----------
    parcel_group : h5py.Group
        HDF5 group containing parcel data
        
    Returns
    -------
    List[Tuple[np.ndarray, str]]
        List of (voxel_values, subject_id) tuples
    """
    session_info = []
    for record_name in parcel_group.keys():
        record = parcel_group[record_name]
        subject = record.attrs['subject']
        voxel_values = record['voxel_values'][:]
        session_info.append((voxel_values, subject))
    return session_info


def compute_between_subject_correlations(session_info: List[Tuple[np.ndarray, str]]) -> List[float]:
    """
    Compute correlations between sessions from different subjects only.
    
    Parameters
    ----------
    session_info : List[Tuple[np.ndarray, str]]
        List of (voxel_values, subject_id) tuples
        
    Returns
    -------
    List[float]
        List of between-subject correlations
    """
    correlations = []
    
    for i in range(len(session_info)):
        for j in range(i + 1, len(session_info)):
            voxels_i, subject_i = session_info[i]
            voxels_j, subject_j = session_info[j]

            if subject_i != subject_j:
                correlation = np.corrcoef(voxels_i, voxels_j)[0, 1]
                if not np.isnan(correlation):
                    correlations.append(correlation)
                    
    return correlations


def compute_between_subject_similarity(hdf5_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Compute between-subject correlations for each contrast-parcel.
    
    Parameters
    ----------
    hdf5_path : Path
        Path to HDF5 file containing parcel data
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dict: {contrast_name: {parcel_name: correlation}}
    """
    results = {}

    with h5py.File(hdf5_path, 'r') as f:
        for contrast_name in f.keys():
            contrast_group = f[contrast_name]
            results[contrast_name] = {}

            for parcel_name in contrast_group.keys():
                parcel_group = contrast_group[parcel_name]
                session_info = extract_session_info_from_parcel(parcel_group)
                
                # Check if we have subjects from at least 2 different subjects
                unique_subjects = set(subject for _, subject in session_info)
                if len(unique_subjects) < 2:
                    continue

                correlations = compute_between_subject_correlations(session_info)
                if correlations:
                    results[contrast_name][parcel_name] = np.mean(correlations)

    return results


def find_constructs_for_contrast(
    contrast_name: str, construct_to_contrast_map: Dict[str, List[str]]
) -> List[str]:
    """
    Find all constructs that contain the given contrast.
    
    Parameters
    ----------
    contrast_name : str
        Name of the contrast
    construct_to_contrast_map : Dict[str, List[str]]
        Mapping from construct names to contrast lists
        
    Returns
    -------
    List[str]
        List of construct names containing the contrast
    """
    constructs = []
    for construct, contrasts in construct_to_contrast_map.items():
        if contrast_name in contrasts:
            constructs.append(construct)
    return constructs


def is_parcel_variable(
    contrast_name: str, 
    parcel_name: str, 
    parcel_classifications: Optional[Dict[str, Dict[str, str]]]
) -> bool:
    """
    Check if a parcel is classified as 'variable'.
    
    Parameters
    ----------
    contrast_name : str
        Name of the contrast
    parcel_name : str
        Name of the parcel
    parcel_classifications : Dict[str, Dict[str, str]], optional
        Parcel classifications
        
    Returns
    -------
    bool
        True if parcel is classified as 'variable'
    """
    if not parcel_classifications:
        return False
        
    return (
        contrast_name in parcel_classifications
        and parcel_name in parcel_classifications[contrast_name]
        and parcel_classifications[contrast_name][parcel_name] == 'variable'
    )


def collect_construct_voxel_data(
    construct_contrasts: List[str], 
    parcel_name: str, 
    hdf5_file
) -> List[np.ndarray]:
    """
    Collect voxel data across contrasts for a specific parcel.
    
    Parameters
    ----------
    construct_contrasts : List[str]
        List of contrast names in the construct
    parcel_name : str
        Name of the parcel
    hdf5_file : h5py.File
        Open HDF5 file
        
    Returns
    -------
    List[np.ndarray]
        List of concatenated voxel values for each contrast
    """
    all_contrast_voxels = []
    
    for sc_contrast in construct_contrasts:
        if sc_contrast not in hdf5_file or parcel_name not in hdf5_file[sc_contrast]:
            continue
            
        sc_parcel_group = hdf5_file[sc_contrast][parcel_name]
        contrast_voxels = []
        
        for record_name in sc_parcel_group.keys():
            record = sc_parcel_group[record_name]
            voxel_values = record['voxel_values'][:]
            contrast_voxels.extend(voxel_values)
            
        if contrast_voxels:
            all_contrast_voxels.append(np.array(contrast_voxels))
            
    return all_contrast_voxels


def compute_across_construct_correlation(voxel_data: List[np.ndarray]) -> Optional[float]:
    """
    Compute mean correlation across contrasts within a construct.
    
    Parameters
    ----------
    voxel_data : List[np.ndarray]
        List of voxel value arrays for different contrasts
        
    Returns
    -------
    Optional[float]
        Mean across-construct correlation, None if insufficient data
    """
    if len(voxel_data) < 2:
        return None
        
    contrast_matrix = np.column_stack(voxel_data)
    upper_tri_values = compute_correlation_matrix_upper_triangle(contrast_matrix.T)
    
    return np.mean(upper_tri_values) if upper_tri_values.size > 0 else None


def compute_across_construct_similarity(
    hdf5_path: Path,
    construct_to_contrast_map: Dict[str, List[str]],
    parcel_classifications: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute across-construct correlations for each contrast-parcel.
    
    Parameters
    ----------
    hdf5_path : Path
        Path to the HDF5 file containing the data
    construct_to_contrast_map : Dict[str, List[str]]
        Mapping from construct names to contrast lists
    parcel_classifications : Dict[str, Dict[str, str]], optional
        Parcel classifications to exclude 'variable' parcels

    Returns
    -------
    Dict[str, Dict[str, Dict[str, float]]]
        Nested dict: {contrast_name: {parcel_name: {construct_name: correlation}}}
    """
    results = {}

    with h5py.File(hdf5_path, 'r') as f:
        available_contrasts = list(f.keys())

        for contrast_name in available_contrasts:
            contrast_group = f[contrast_name]
            results[contrast_name] = {}
            
            constructs = find_constructs_for_contrast(contrast_name, construct_to_contrast_map)

            for parcel_name in contrast_group.keys():
                if is_parcel_variable(contrast_name, parcel_name, parcel_classifications):
                    continue

                results[contrast_name][parcel_name] = {}

                for construct in constructs:
                    construct_contrasts = [
                        c for c in construct_to_contrast_map[construct]
                        if c in available_contrasts
                    ]

                    if len(construct_contrasts) < 2:
                        continue

                    voxel_data = collect_construct_voxel_data(
                        construct_contrasts, parcel_name, f
                    )
                    
                    correlation = compute_across_construct_correlation(voxel_data)
                    if correlation is not None:
                        results[contrast_name][parcel_name][construct] = correlation

    return results


def classify_single_parcel(
    within_val: float, 
    between_val: float, 
    threshold: float = 0.1
) -> str:
    """
    Classify a single parcel based on within and between subject correlations.
    
    Parameters
    ----------
    within_val : float
        Within-subject correlation value
    between_val : float
        Between-subject correlation value
    threshold : float
        Classification threshold
        
    Returns
    -------
    str
        Classification: 'variable', 'indiv_fingerprint', or 'canonical'
    """
    if (within_val + between_val) < threshold:
        return 'variable'
    elif (within_val - between_val) < threshold:
        return 'indiv_fingerprint'
    else:
        return 'canonical'


def classify_parcels(
    within_correlations: Dict[str, Dict[str, float]],
    between_correlations: Dict[str, Dict[str, float]],
    threshold: float = 0.1,
) -> Dict[str, Dict[str, str]]:
    """
    Classify parcels based on within and between subject correlations.
    
    Parameters
    ----------
    within_correlations : Dict[str, Dict[str, float]]
        Within-subject correlations by contrast and parcel
    between_correlations : Dict[str, Dict[str, float]]
        Between-subject correlations by contrast and parcel
    threshold : float
        Classification threshold
        
    Returns
    -------
    Dict[str, Dict[str, str]]
        Classifications by contrast and parcel
    """
    results = {}

    for contrast_name in within_correlations.keys():
        if contrast_name not in between_correlations:
            continue

        results[contrast_name] = {}

        for parcel_name in within_correlations[contrast_name].keys():
            if parcel_name not in between_correlations[contrast_name]:
                continue

            within_val = within_correlations[contrast_name][parcel_name]
            between_val = between_correlations[contrast_name][parcel_name]
            
            classification = classify_single_parcel(within_val, between_val, threshold)
            results[contrast_name][parcel_name] = classification

    return results
