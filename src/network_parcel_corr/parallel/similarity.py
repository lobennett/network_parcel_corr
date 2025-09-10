"""Parallel optimized similarity computation functions."""

from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import h5py
import logging

from .optimization import get_optimal_worker_count, parallel_compute_parcel_similarities
from ..core.similarity import (
    extract_subject_sessions_from_parcel,
    extract_session_info_from_parcel,
    compute_within_subject_correlation,
    compute_between_subject_correlations,
    classify_single_parcel,
)


def parallel_compute_within_subject_similarity(
    hdf5_path: Path, max_workers: int = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute within-subject similarities in parallel.
    
    Parameters
    ----------
    hdf5_path : Path
        Path to HDF5 file containing parcel data
    max_workers : int, optional
        Maximum number of worker threads
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dict: {contrast_name: {parcel_name: correlation}}
    """
    max_workers = get_optimal_worker_count(max_workers)
    logger = logging.getLogger(__name__)
    
    print(f'Computing within-subject similarities using {max_workers} workers...')
    
    def compute_within_similarity_for_parcel(parcel_data):
        """Compute within-subject similarity for a single parcel."""
        try:
            subject_sessions = extract_subject_sessions_from_parcel(parcel_data)
            subject_correlations = []
            
            for subject, sessions in subject_sessions.items():
                correlation = compute_within_subject_correlation(sessions)
                if correlation is not None:
                    subject_correlations.append(correlation)
            
            return np.mean(subject_correlations) if subject_correlations else None
            
        except Exception as exc:
            logger.error(f'Error in within-subject similarity computation: {exc}')
            return None
    
    results = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        # Prepare all parcel data for parallel processing
        parcel_work_items = []
        for contrast_name in f.keys():
            contrast_group = f[contrast_name]
            for parcel_name in contrast_group.keys():
                parcel_group = contrast_group[parcel_name]
                parcel_work_items.append((contrast_name, parcel_name, parcel_group))
        
        print(f'Processing {len(parcel_work_items)} contrast-parcel combinations...')
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all parcel processing tasks
            future_to_parcel = {
                executor.submit(compute_within_similarity_for_parcel, parcel_data): (contrast_name, parcel_name)
                for contrast_name, parcel_name, parcel_data in parcel_work_items
            }
            
            completed = 0
            total = len(parcel_work_items)
            
            # Collect results
            for future in as_completed(future_to_parcel):
                contrast_name, parcel_name = future_to_parcel[future]
                try:
                    similarity = future.result()
                    
                    if contrast_name not in results:
                        results[contrast_name] = {}
                    
                    if similarity is not None:
                        results[contrast_name][parcel_name] = similarity
                    
                    completed += 1
                    if completed % 25 == 0 or completed == total:
                        print(f'✓ Within-subject: {completed}/{total} parcels')
                        
                except Exception as exc:
                    logger.error(f'Error processing {contrast_name}-{parcel_name}: {exc}')
    
    return results


def parallel_compute_between_subject_similarity(
    hdf5_path: Path, max_workers: int = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute between-subject similarities in parallel.
    
    Parameters
    ----------
    hdf5_path : Path
        Path to HDF5 file containing parcel data
    max_workers : int, optional
        Maximum number of worker threads
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dict: {contrast_name: {parcel_name: correlation}}
    """
    max_workers = get_optimal_worker_count(max_workers)
    logger = logging.getLogger(__name__)
    
    print(f'Computing between-subject similarities using {max_workers} workers...')
    
    def compute_between_similarity_for_parcel(parcel_data):
        """Compute between-subject similarity for a single parcel."""
        try:
            session_info = extract_session_info_from_parcel(parcel_data)
            
            # Check if we have subjects from at least 2 different subjects
            unique_subjects = set(subject for _, subject in session_info)
            if len(unique_subjects) < 2:
                return None
            
            correlations = compute_between_subject_correlations(session_info)
            return np.mean(correlations) if correlations else None
            
        except Exception as exc:
            logger.error(f'Error in between-subject similarity computation: {exc}')
            return None
    
    results = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        # Prepare all parcel data for parallel processing
        parcel_work_items = []
        for contrast_name in f.keys():
            contrast_group = f[contrast_name]
            for parcel_name in contrast_group.keys():
                parcel_group = contrast_group[parcel_name]
                parcel_work_items.append((contrast_name, parcel_name, parcel_group))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all parcel processing tasks
            future_to_parcel = {
                executor.submit(compute_between_similarity_for_parcel, parcel_data): (contrast_name, parcel_name)
                for contrast_name, parcel_name, parcel_data in parcel_work_items
            }
            
            completed = 0
            total = len(parcel_work_items)
            
            # Collect results
            for future in as_completed(future_to_parcel):
                contrast_name, parcel_name = future_to_parcel[future]
                try:
                    similarity = future.result()
                    
                    if contrast_name not in results:
                        results[contrast_name] = {}
                    
                    if similarity is not None:
                        results[contrast_name][parcel_name] = similarity
                    
                    completed += 1
                    if completed % 25 == 0 or completed == total:
                        print(f'✓ Between-subject: {completed}/{total} parcels')
                        
                except Exception as exc:
                    logger.error(f'Error processing {contrast_name}-{parcel_name}: {exc}')
    
    return results


def parallel_classify_parcels(
    within_correlations: Dict[str, Dict[str, float]],
    between_correlations: Dict[str, Dict[str, float]],
    threshold: float = 0.1,
    max_workers: int = None
) -> Dict[str, Dict[str, str]]:
    """
    Classify parcels in parallel based on within and between subject correlations.
    
    Parameters
    ----------
    within_correlations : Dict[str, Dict[str, float]]
        Within-subject correlations by contrast and parcel
    between_correlations : Dict[str, Dict[str, float]]
        Between-subject correlations by contrast and parcel
    threshold : float
        Classification threshold
    max_workers : int, optional
        Maximum number of worker threads
        
    Returns
    -------
    Dict[str, Dict[str, str]]
        Classifications by contrast and parcel
    """
    max_workers = get_optimal_worker_count(max_workers)
    
    print(f'Classifying parcels using {max_workers} workers...')
    
    # Prepare work items
    work_items = []
    for contrast_name in within_correlations.keys():
        if contrast_name not in between_correlations:
            continue
            
        for parcel_name in within_correlations[contrast_name].keys():
            if parcel_name not in between_correlations[contrast_name]:
                continue
                
            within_val = within_correlations[contrast_name][parcel_name]
            between_val = between_correlations[contrast_name][parcel_name]
            work_items.append((contrast_name, parcel_name, within_val, between_val))
    
    def classify_parcel_item(item):
        """Classify a single parcel."""
        contrast_name, parcel_name, within_val, between_val = item
        classification = classify_single_parcel(within_val, between_val, threshold)
        return contrast_name, parcel_name, classification
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all classification tasks
        futures = [executor.submit(classify_parcel_item, item) for item in work_items]
        
        # Collect results
        for future in as_completed(futures):
            try:
                contrast_name, parcel_name, classification = future.result()
                
                if contrast_name not in results:
                    results[contrast_name] = {}
                    
                results[contrast_name][parcel_name] = classification
                
            except Exception as exc:
                logging.getLogger(__name__).error(f'Error in parcel classification: {exc}')
    
    print(f'✓ Classified {len(work_items)} parcels')
    return results


def parallel_compute_all_similarities(
    hdf5_path: Path, max_workers: int = None
) -> tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Compute both within and between subject similarities in parallel.
    
    This version runs both computations concurrently for maximum efficiency.
    
    Parameters
    ----------
    hdf5_path : Path
        Path to HDF5 file containing parcel data
    max_workers : int, optional
        Maximum number of worker threads per similarity type
        
    Returns
    -------
    Tuple[Dict, Dict]
        Within and between subject similarities
    """
    print('Computing within and between subject similarities concurrently...')
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both similarity computations concurrently
        within_future = executor.submit(
            parallel_compute_within_subject_similarity, hdf5_path, max_workers
        )
        between_future = executor.submit(
            parallel_compute_between_subject_similarity, hdf5_path, max_workers
        )
        
        # Wait for both to complete
        within_similarities = within_future.result()
        between_similarities = between_future.result()
    
    print('✓ Completed all similarity computations')
    return within_similarities, between_similarities