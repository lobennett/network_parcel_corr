"""Parallel-optimized main pipeline for parcel-based correlation analysis."""

from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

from .optimization import (
    optimize_numpy_performance,
    parallel_extract_contrast_files,
    parallel_extract_single_files,
    get_optimal_worker_count,
)
from .similarity import (
    parallel_compute_all_similarities,
    parallel_classify_parcels,
)
from ..main import load_atlas_data, discover_contrast_files
from ..io.writers import save_to_hdf5


def parallel_extract_parcel_data(
    contrast_files: Dict[str, List[Path]], 
    atlas_data: np.ndarray, 
    atlas_labels: List[str],
    max_workers: int = None
) -> Dict:
    """
    Extract and group parcel data from contrast files in parallel.
    
    This version processes either by contrast (if few contrasts with many files each)
    or by individual files (if many contrasts with few files each).
    
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
    
    total_files = sum(len(files) for files in contrast_files.values())
    num_contrasts = len(contrast_files)
    
    print(f'Processing {total_files} files across {num_contrasts} contrasts...')
    
    # Choose strategy based on data distribution
    avg_files_per_contrast = total_files / num_contrasts if num_contrasts > 0 else 0
    
    if num_contrasts <= max_workers and avg_files_per_contrast >= 10:
        # Few contrasts with many files each - parallelize by contrast
        print('Using contrast-level parallelization')
        return parallel_extract_contrast_files(contrast_files, atlas_data, atlas_labels, max_workers)
    else:
        # Many contrasts or few files per contrast - parallelize by individual files
        print('Using file-level parallelization')
        from collections import defaultdict
        
        # Flatten all files and process in parallel
        all_files = []
        for files in contrast_files.values():
            all_files.extend(files)
        
        # Process all files in parallel
        all_parcel_data = parallel_extract_single_files(all_files, atlas_data, atlas_labels, max_workers)
        
        # Group results back by contrast
        grouped_by_contrast = {}
        for contrast_name, files in contrast_files.items():
            grouped_by_contrast[contrast_name] = defaultdict(list)
            
            for filepath in files:
                # Extract contrast info to match records
                from ..io.readers import extract_contrast_info
                subj, session, contrast, run = extract_contrast_info(filepath)
                
                if not all([subj, session, contrast, run]):
                    continue
                
                # Find matching records in all_parcel_data
                for parcel_name, records in all_parcel_data.items():
                    matching_records = [
                        record for record in records
                        if (record[0] == subj and record[1] == session and 
                            record[2] == contrast and record[3] == run)
                    ]
                    grouped_by_contrast[contrast_name][parcel_name].extend(matching_records)
            
            # Convert to dict
            grouped_by_contrast[contrast_name] = dict(grouped_by_contrast[contrast_name])
    
    return grouped_by_contrast


def parallel_run_analysis(
    subjects: List[str],
    input_dir: Path,
    output_dir: Path,
    exclusions_file: str,
    atlas_parcels: int = 400,
    max_workers: int = None,
) -> Dict:
    """
    Run the complete parcel-based correlation analysis pipeline with parallel optimization.

    Parameters
    ----------
    subjects : List[str]
        List of subject IDs to analyze
    input_dir : Path
        Directory containing subject data
    output_dir : Path
        Directory for output files
    exclusions_file : str
        Path to exclusions JSON file
    atlas_parcels : int, optional
        Number of atlas parcels to use (default: 400)
    max_workers : int, optional
        Maximum number of worker threads (default: all available CPUs, max 16)

    Returns
    -------
    Dict
        Results containing within/between similarities and classifications
    """
    max_workers = get_optimal_worker_count(max_workers)
    
    print(f'Starting PARALLEL analysis with {len(subjects)} subjects using {max_workers} workers...')
    
    # Optimize NumPy performance for multi-threading
    optimize_numpy_performance()

    # Load atlas (not parallelizable, but fast)
    atlas_data, atlas_labels = load_atlas_data(atlas_parcels)

    # Find contrast files (I/O bound, but typically fast)
    contrast_files = discover_contrast_files(subjects, input_dir, exclusions_file)

    # Extract and group data (MAJOR BOTTLENECK - parallelize this)
    grouped_by_contrast = parallel_extract_parcel_data(
        contrast_files, atlas_data, atlas_labels, max_workers
    )
    
    # Debug: Check for duplicate contrast names
    all_contrast_names = list(grouped_by_contrast.keys())
    unique_contrast_names = set(all_contrast_names)
    if len(all_contrast_names) != len(unique_contrast_names):
        print(f'WARNING: Found {len(all_contrast_names)} total contrasts but only {len(unique_contrast_names)} unique names')
        print(f'All contrast names: {all_contrast_names}')
    else:
        print(f'✓ All {len(all_contrast_names)} contrast names are unique')

    # Save to HDF5 (I/O bound, not easily parallelizable)
    print('Saving data to HDF5...')
    hdf5_path = save_to_hdf5(grouped_by_contrast, output_dir)

    # Compute similarities (MAJOR BOTTLENECK - parallelize this)
    within_similarities, between_similarities = parallel_compute_all_similarities(
        hdf5_path, max_workers
    )

    # Classify parcels (CPU bound, parallelize this)
    classifications = parallel_classify_parcels(
        within_similarities, between_similarities, threshold=0.1, max_workers=max_workers
    )

    results = {
        'hdf5_path': hdf5_path,
        'within_similarities': within_similarities,
        'between_similarities': between_similarities,
        'classifications': classifications,
        'n_contrasts': len(contrast_files),
        'n_subjects': len(subjects),
    }

    print('PARALLEL analysis complete!')
    return results


def benchmark_parallel_vs_serial(
    subjects: List[str],
    input_dir: Path,
    output_dir: Path,
    exclusions_file: str,
    atlas_parcels: int = 400,
) -> Dict:
    """
    Benchmark parallel vs serial performance.
    
    Parameters
    ----------
    subjects : List[str]
        List of subject IDs to analyze
    input_dir : Path
        Directory containing subject data
    output_dir : Path
        Directory for output files
    exclusions_file : str
        Path to exclusions JSON file
    atlas_parcels : int, optional
        Number of atlas parcels to use
        
    Returns
    -------
    Dict
        Timing comparison results
    """
    import time
    
    print("=== PERFORMANCE BENCHMARK ===")
    
    # Test with different worker counts
    worker_counts = [1, 2, 4, 8, 16]
    results = {}
    
    for workers in worker_counts:
        if workers > get_optimal_worker_count():
            continue
            
        print(f"\n--- Testing with {workers} worker(s) ---")
        start_time = time.time()
        
        try:
            analysis_results = parallel_run_analysis(
                subjects, input_dir, output_dir, exclusions_file, 
                atlas_parcels, max_workers=workers
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            results[workers] = {
                'duration': duration,
                'n_contrasts': analysis_results['n_contrasts'],
                'n_subjects': analysis_results['n_subjects'],
                'success': True
            }
            
            print(f"✓ Completed in {duration:.2f} seconds")
            
        except Exception as e:
            results[workers] = {
                'duration': float('inf'),
                'error': str(e),
                'success': False
            }
            print(f"✗ Failed: {e}")
    
    # Print summary
    print("\n=== PERFORMANCE SUMMARY ===")
    successful_results = {k: v for k, v in results.items() if v['success']}
    
    if successful_results:
        fastest = min(successful_results.keys(), key=lambda k: successful_results[k]['duration'])
        baseline = successful_results.get(1, {}).get('duration', float('inf'))
        
        for workers in sorted(successful_results.keys()):
            duration = successful_results[workers]['duration']
            speedup = baseline / duration if baseline != float('inf') else 1.0
            print(f"{workers:2d} workers: {duration:6.2f}s (speedup: {speedup:.2f}x)")
        
        print(f"\nOptimal configuration: {fastest} workers ({successful_results[fastest]['duration']:.2f}s)")
    
    return results