import templateflow.api as tf
import numpy as np
import pandas as pd
from nilearn import image
from typing import List, Tuple


def load_schaefer_atlas(n_parcels: int = 400) -> Tuple[np.ndarray, List[str]]:
    """Load Schaefer atlas from TemplateFlow."""
    print(f'\nLoading Schaefer {n_parcels}-parcel atlas...')

    atlas_path = tf.get(
        'MNI152NLin2009cAsym',
        resolution=2,
        atlas='Schaefer2018',
        desc=f'{n_parcels}Parcels7Networks',
        suffix='dseg',
        extension='nii.gz',
    )
    labels_path = tf.get(
        'MNI152NLin2009cAsym',
        atlas='Schaefer2018',
        desc=f'{n_parcels}Parcels7Networks',
        suffix='dseg',
        extension='tsv',
    )

    atlas_img = image.load_img(atlas_path)
    atlas_data = atlas_img.get_fdata()

    labels_df = pd.read_csv(labels_path, sep='\t')
    atlas_labels = labels_df['name'].tolist()

    print(f'✓ Loaded {len(atlas_labels)} parcels')
    return atlas_data, atlas_labels
