"""Parallel optimization utilities for correlation analysis."""

import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Callable, Any
from pathlib import Path
from functools import partial
import numpy as np
import logging

from ..io.readers import extract_contrast_info
from ..io.writers import process_single_contrast_file, extract_and_group_by_parcel


def get_optimal_worker_count(max_workers: int = None) -> int:
    """
    Get optimal number of workers based on available CPUs.
    
    Parameters
    ----------
    max_workers : int, optional
        Maximum number of workers to use. If None, uses all available CPUs.
        
    Returns
    -------
    int
        Optimal number of workers
    """
    if max_workers is None:
        # Use all available CPUs, but cap at 16 for memory considerations
        max_workers = min(os.cpu_count() or 1, 16)
    
    return max_workers


def parallel_extract_contrast_files(
    contrast_files: Dict[str, List[Path]], 
    atlas_data: np.ndarray, 
    atlas_labels: List[str],
    max_workers: int = None
) -> Dict:
    """
    Extract parcel data from contrast files in parallel.
    
    Parameters
    ----------
    contrast_files : Dict[str, List[Path]]
        Dictionary mapping contrast names to file lists
    atlas_data : np.ndarray
        Atlas data array
    atlas_labels : List[str]
        List of parcel labels
    max_workers : int, optional
        Maximum number of worker threads
        
    Returns
    -------
    Dict
        Dictionary mapping contrast names to grouped parcel data
    """
    max_workers = get_optimal_worker_count(max_workers)
    logger = logging.getLogger(__name__)
    
    print(f'Extracting parcel data using {max_workers} workers...')
    
    def process_contrast(contrast_item):
        """Process a single contrast."""
        contrast_name, files = contrast_item
        logger.info(f'Processing {contrast_name} ({len(files)} files)...')
        parcel_data = extract_and_group_by_parcel(files, atlas_data, atlas_labels)
        return contrast_name, parcel_data
    
    grouped_by_contrast = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all contrast processing tasks
        future_to_contrast = {
            executor.submit(process_contrast, item): item[0] 
            for item in contrast_files.items()
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_contrast):
            contrast_name = future_to_contrast[future]
            try:
                contrast_name, parcel_data = future.result()
                grouped_by_contrast[contrast_name] = parcel_data
                print(f'✓ Completed {contrast_name}')
            except Exception as exc:
                logger.error(f'Error processing {contrast_name}: {exc}')
                raise exc
    
    return grouped_by_contrast


def parallel_extract_single_files(
    filepaths: List[Path], 
    atlas_data: np.ndarray, 
    atlas_labels: List[str],
    max_workers: int = None
) -> Dict:
    """
    Extract parcel data from individual files in parallel.
    
    Parameters
    ----------
    filepaths : List[Path]
        List of contrast file paths
    atlas_data : np.ndarray
        Atlas data with parcel labels
    atlas_labels : List[str]
        List of parcel names
    max_workers : int, optional
        Maximum number of worker threads
        
    Returns
    -------
    Dict
        Dictionary mapping parcel names to lists of records
    """
    from collections import defaultdict
    
    max_workers = get_optimal_worker_count(max_workers)
    logger = logging.getLogger(__name__)
    
    print(f'Processing {len(filepaths)} files using {max_workers} workers...')
    
    # Create label mapping once
    label_to_name_map = {i: name for i, name in enumerate(atlas_labels, start=1)}
    
    def process_file(filepath):
        """Process a single contrast file."""
        try:
            return process_single_contrast_file(filepath, atlas_data, label_to_name_map)
        except Exception as exc:
            logger.warning(f'Failed to process {filepath}: {exc}')
            return {}
    
    grouped_by_parcel = defaultdict(list)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all file processing tasks
        future_to_file = {
            executor.submit(process_file, filepath): filepath 
            for filepath in filepaths
        }
        
        completed = 0
        total = len(filepaths)
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            filepath = future_to_file[future]
            try:
                parcel_data = future.result()
                
                # Merge results into grouped_by_parcel
                for parcel_name, record in parcel_data.items():
                    grouped_by_parcel[parcel_name].append(record)
                    
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f'✓ Processed {completed}/{total} files')
                    
            except Exception as exc:
                logger.error(f'Error processing {filepath}: {exc}')
                
    return dict(grouped_by_parcel)


def parallel_compute_correlations(
    subject_sessions: Dict[str, List[np.ndarray]], 
    correlation_func: Callable,
    max_workers: int = None
) -> Dict[str, float]:
    """
    Compute correlations for multiple subjects in parallel.
    
    Parameters
    ----------
    subject_sessions : Dict[str, List[np.ndarray]]
        Dictionary mapping subject IDs to session data
    correlation_func : Callable
        Function to compute correlation for a single subject
    max_workers : int, optional
        Maximum number of worker threads
        
    Returns
    -------
    Dict[str, float]
        Dictionary mapping subject IDs to correlation values
    """
    max_workers = get_optimal_worker_count(max_workers)
    
    def compute_subject_correlation(subject_item):
        """Compute correlation for a single subject."""
        subject_id, sessions = subject_item
        correlation = correlation_func(sessions)
        return subject_id, correlation
    
    correlations = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all subject correlation tasks
        future_to_subject = {
            executor.submit(compute_subject_correlation, item): item[0]
            for item in subject_sessions.items()
        }
        
        # Collect results
        for future in as_completed(future_to_subject):
            subject_id = future_to_subject[future]
            try:
                subject_id, correlation = future.result()
                if correlation is not None:
                    correlations[subject_id] = correlation
            except Exception as exc:
                logging.getLogger(__name__).error(f'Error computing correlation for {subject_id}: {exc}')
    
    return correlations


def parallel_compute_parcel_similarities(
    hdf5_data: Dict, 
    similarity_func: Callable,
    max_workers: int = None
) -> Dict:
    """
    Compute similarities for multiple parcels in parallel.
    
    Parameters
    ----------
    hdf5_data : Dict
        HDF5 data organized by contrast and parcel
    similarity_func : Callable
        Function to compute similarity for a single parcel
    max_workers : int, optional
        Maximum number of worker threads
        
    Returns
    -------
    Dict
        Results organized by contrast and parcel
    """
    max_workers = get_optimal_worker_count(max_workers)
    
    def compute_parcel_similarity(parcel_item):
        """Compute similarity for a single contrast-parcel combination."""
        contrast_name, parcel_name, parcel_data = parcel_item
        try:
            similarity = similarity_func(parcel_data)
            return contrast_name, parcel_name, similarity
        except Exception as exc:
            logging.getLogger(__name__).error(
                f'Error computing similarity for {contrast_name}-{parcel_name}: {exc}'
            )
            return contrast_name, parcel_name, None
    
    # Prepare work items
    work_items = []
    for contrast_name, contrast_data in hdf5_data.items():
        for parcel_name, parcel_data in contrast_data.items():
            work_items.append((contrast_name, parcel_name, parcel_data))
    
    print(f'Computing similarities for {len(work_items)} parcels using {max_workers} workers...')
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all parcel similarity tasks
        future_to_parcel = {
            executor.submit(compute_parcel_similarity, item): item
            for item in work_items
        }
        
        completed = 0
        total = len(work_items)
        
        # Collect results
        for future in as_completed(future_to_parcel):
            try:
                contrast_name, parcel_name, similarity = future.result()
                
                if contrast_name not in results:
                    results[contrast_name] = {}
                    
                if similarity is not None:
                    results[contrast_name][parcel_name] = similarity
                
                completed += 1
                if completed % 50 == 0 or completed == total:
                    print(f'✓ Computed {completed}/{total} similarities')
                    
            except Exception as exc:
                logging.getLogger(__name__).error(f'Error in similarity computation: {exc}')
    
    return results


def optimize_numpy_performance():
    """
    Optimize NumPy for multi-threaded performance.
    
    This function sets optimal NumPy threading settings for correlation computations.
    Note: For best results, these environment variables should be set before importing numpy.
    """
    # Set optimal BLAS threading for correlation computations
    optimal_threads = str(get_optimal_worker_count())
    
    # Only set if not already configured (to respect SLURM/user settings)
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = optimal_threads
    if 'MKL_NUM_THREADS' not in os.environ:
        os.environ['MKL_NUM_THREADS'] = optimal_threads
    if 'OPENBLAS_NUM_THREADS' not in os.environ:
        os.environ['OPENBLAS_NUM_THREADS'] = optimal_threads
    if 'VECLIB_MAXIMUM_THREADS' not in os.environ:
        os.environ['VECLIB_MAXIMUM_THREADS'] = optimal_threads
    if 'NUMEXPR_NUM_THREADS' not in os.environ:
        os.environ['NUMEXPR_NUM_THREADS'] = optimal_threads
    
    current_threads = os.environ.get('OMP_NUM_THREADS', optimal_threads)
    print(f'NumPy configured to use {current_threads} threads for BLAS operations')


def batch_process_with_memory_management(
    items: List[Any],
    process_func: Callable,
    batch_size: int = None,
    max_workers: int = None
) -> List[Any]:
    """
    Process items in batches to manage memory usage.
    
    Parameters
    ----------
    items : List[Any]
        Items to process
    process_func : Callable
        Function to process each item
    batch_size : int, optional
        Size of each batch. If None, auto-determined based on available memory.
    max_workers : int, optional
        Maximum number of worker threads
        
    Returns
    -------
    List[Any]
        Processed results
    """
    if batch_size is None:
        # Auto-determine batch size based on available workers
        max_workers = get_optimal_worker_count(max_workers)
        batch_size = max(len(items) // max_workers, 1)
    
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        print(f'Processing batch {i//batch_size + 1}/{(len(items) + batch_size - 1)//batch_size} ({len(batch)} items)...')
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_futures = [executor.submit(process_func, item) for item in batch]
            batch_results = [future.result() for future in as_completed(batch_futures)]
            results.extend(batch_results)
    
    return results