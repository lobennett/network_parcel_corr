"""Core similarity calculation functions."""

from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
import h5py


def compute_within_subject_similarity(hdf5_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Compute within-subject mean correlations for each contrast-parcel.
    Returns mean of upper triangle correlations across sessions for each subject.
    """
    results = {}

    with h5py.File(hdf5_path, 'r') as f:
        for contrast_name in f.keys():
            contrast_group = f[contrast_name]
            results[contrast_name] = {}

            for parcel_name in contrast_group.keys():
                parcel_group = contrast_group[parcel_name]

                # Build subject-session mapping
                subject_data = defaultdict(list)
                for record_name in parcel_group.keys():
                    record = parcel_group[record_name]
                    subject = record.attrs['subject']
                    voxel_values = record['voxel_values'][:]
                    subject_data[subject].append(voxel_values)

                # Compute within-subject correlations
                subject_correlations = []
                for subject, sessions in subject_data.items():
                    if len(sessions) < 2:
                        continue

                    # Stack sessions as columns for efficient correlation
                    session_matrix = np.column_stack(sessions)
                    corr_matrix = np.corrcoef(session_matrix.T)

                    # Extract upper triangle (excluding diagonal)
                    upper_tri = np.triu(corr_matrix, k=1)
                    upper_tri_values = upper_tri[upper_tri != 0]

                    if len(upper_tri_values) > 0:
                        subject_correlations.append(np.mean(upper_tri_values))

                # Mean across subjects
                if subject_correlations:
                    results[contrast_name][parcel_name] = np.mean(subject_correlations)

    return results


def compute_between_subject_similarity(hdf5_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Compute between-subject correlations for each contrast-parcel.
    Only includes correlations between sessions from DIFFERENT subjects.
    Excludes all within-subject correlations.
    """
    results = {}

    with h5py.File(hdf5_path, 'r') as f:
        for contrast_name in f.keys():
            contrast_group = f[contrast_name]
            results[contrast_name] = {}

            for parcel_name in contrast_group.keys():
                parcel_group = contrast_group[parcel_name]

                # Collect voxel values organized by subject
                subject_sessions = defaultdict(list)
                session_info = []  # Track (voxel_values, subject) for each session

                for record_name in parcel_group.keys():
                    record = parcel_group[record_name]
                    subject = record.attrs['subject']
                    voxel_values = record['voxel_values'][:]

                    subject_sessions[subject].append(voxel_values)
                    session_info.append((voxel_values, subject))

                if len(subject_sessions) < 2:
                    # Need at least 2 subjects for between-subject comparison
                    continue

                # Compute only between-subject correlations
                between_subject_correlations = []

                for i in range(len(session_info)):
                    for j in range(i + 1, len(session_info)):
                        voxels_i, subject_i = session_info[i]
                        voxels_j, subject_j = session_info[j]

                        # Only correlate if sessions are from different subjects
                        if subject_i != subject_j:
                            correlation = np.corrcoef(voxels_i, voxels_j)[0, 1]
                            if not np.isnan(correlation):
                                between_subject_correlations.append(correlation)

                # Mean between-subject correlation
                if between_subject_correlations:
                    results[contrast_name][parcel_name] = np.mean(
                        between_subject_correlations
                    )

    return results


def compute_across_construct_similarity(
    hdf5_path: Path,
    construct_to_contrast_map: Dict[str, List[str]],
    parcel_classifications: Dict[str, Dict[str, str]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute across-construct correlations for each contrast-parcel.
    For each contrast, compute correlations across all contrasts within the same constructs.
    Excludes parcels classified as 'variable' from the computation.

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

    def find_constructs_for_contrast(contrast_name: str) -> List[str]:
        """Find all constructs that contain the given contrast."""
        constructs = []
        for construct, contrasts in construct_to_contrast_map.items():
            if contrast_name in contrasts:
                constructs.append(construct)
        return constructs

    with h5py.File(hdf5_path, 'r') as f:
        available_contrasts = list(f.keys())

        for contrast_name in available_contrasts:
            contrast_group = f[contrast_name]
            results[contrast_name] = {}

            # Find which constructs this contrast belongs to
            constructs = find_constructs_for_contrast(contrast_name)

            for parcel_name in contrast_group.keys():
                # Skip variable parcels if classifications are provided
                if (
                    parcel_classifications
                    and contrast_name in parcel_classifications
                    and parcel_name in parcel_classifications[contrast_name]
                    and parcel_classifications[contrast_name][parcel_name] == 'variable'
                ):
                    continue

                results[contrast_name][parcel_name] = {}

                for construct in constructs:
                    # Get all contrasts in this construct that are available in the HDF5
                    construct_contrasts = [
                        c
                        for c in construct_to_contrast_map[construct]
                        if c in available_contrasts
                    ]

                    if len(construct_contrasts) < 2:
                        # Skip if less than 2 contrasts available for correlation
                        continue

                    # Collect all voxel values across contrasts in this construct
                    all_contrast_voxels = []
                    for sc_contrast in construct_contrasts:
                        sc_contrast_group = f[sc_contrast]
                        if parcel_name not in sc_contrast_group:
                            continue

                        sc_parcel_group = sc_contrast_group[parcel_name]

                        # Collect all voxel values for this contrast-parcel combination
                        contrast_voxels = []
                        for record_name in sc_parcel_group.keys():
                            record = sc_parcel_group[record_name]
                            voxel_values = record['voxel_values'][:]
                            contrast_voxels.extend(voxel_values)

                        if contrast_voxels:
                            all_contrast_voxels.append(np.array(contrast_voxels))

                    if len(all_contrast_voxels) < 2:
                        continue

                    # Stack contrasts as columns and compute correlation matrix for upper triangle extraction
                    contrast_matrix = np.column_stack(all_contrast_voxels)
                    corr_matrix = np.corrcoef(contrast_matrix.T)

                    # Extract upper triangle (excluding diagonal)
                    upper_tri = np.triu(corr_matrix, k=1)
                    upper_tri_values = upper_tri[upper_tri != 0]

                    # Mean across-contrast correlation for this construct
                    if upper_tri_values.size > 0:
                        results[contrast_name][parcel_name][construct] = np.mean(
                            upper_tri_values
                        )

    return results


def classify_parcels(
    within_correlations: Dict[str, Dict[str, float]],
    between_correlations: Dict[str, Dict[str, float]],
    threshold: float = 0.1,
) -> Dict[str, Dict[str, str]]:
    """
    Classify parcels based on within and between subject correlations.

    Classification logic:
    - Variable: (within + between) < threshold
    - Individual Fingerprint: (within - between) < threshold
    - Canonical: Default for parcels not meeting above criteria
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

            # Classification logic with sequential evaluation
            if (within_val + between_val) < threshold:
                classification = 'variable'
            elif (within_val - between_val) < threshold:
                classification = 'indiv_fingerprint'
            else:
                classification = 'canonical'

            results[contrast_name][parcel_name] = classification

    return results
