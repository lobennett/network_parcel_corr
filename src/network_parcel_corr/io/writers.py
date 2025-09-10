"""File writing utilities for HDF5 results."""

from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import h5py
from nilearn import image


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
    from .readers import extract_contrast_info

    grouped_by_parcel = defaultdict(list)
    label_to_name_map = {i: name for i, name in enumerate(atlas_labels, start=1)}

    for filepath in filepaths:
        try:
            subj, session, contrast, run = extract_contrast_info(filepath)

            if not all([subj, session, contrast, run]):
                continue

            img_data = image.load_img(filepath).get_fdata()

            for parcel_idx, parcel_name in label_to_name_map.items():
                parcel_mask = atlas_data == parcel_idx
                voxel_values = img_data[parcel_mask]

                if voxel_values.size == 0:
                    continue

                # Append tuple of metadata and voxel values
                grouped_by_parcel[parcel_name].append(
                    (subj, session, contrast, run, voxel_values)
                )

        except Exception:
            continue

    return dict(grouped_by_parcel)


def save_to_hdf5(grouped_by_contrast: Dict, output_dir: Path) -> Path:
    """Save grouped contrast data to a single combined HDF5 file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_hdf5_path = output_dir / 'all_contrasts.h5'

    with h5py.File(combined_hdf5_path, 'w') as f:
        # Add top-level metadata
        f.attrs['n_contrasts'] = len(grouped_by_contrast)
        f.attrs['contrast_names'] = list(grouped_by_contrast.keys())

        for contrast_name, grouped_by_parcel in grouped_by_contrast.items():
            # Validate voxel value consistency within each parcel
            for parcel_name, records in grouped_by_parcel.items():
                voxel_lengths = [len(record[4]) for record in records]
                if len(set(voxel_lengths)) > 1:
                    raise ValueError(
                        f"Inconsistent voxel counts in parcel '{parcel_name}' for contrast '{contrast_name}': {voxel_lengths}"
                    )

            # Create contrast group
            contrast_group = f.create_group(contrast_name)
            contrast_group.attrs['contrast_name'] = contrast_name
            contrast_group.attrs['n_parcels'] = len(grouped_by_parcel)

            for parcel_name, records in grouped_by_parcel.items():
                parcel_group = contrast_group.create_group(parcel_name)
                parcel_group.attrs['n_records'] = len(records)
                parcel_group.attrs['n_voxels'] = len(records[0][4])

                for subject, session, contrast, run, parcel_data in records:
                    record_name = f'{subject}_{session}_{run}'
                    record_group = parcel_group.create_group(record_name)
                    record_group.attrs['subject'] = subject
                    record_group.attrs['session'] = session
                    record_group.attrs['mean_voxel_value'] = np.mean(parcel_data)
                    record_group.create_dataset('voxel_values', data=parcel_data)

    return combined_hdf5_path
