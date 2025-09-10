"""Test atlas loading functionality."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from network_parcel_corr.atlases.load import load_schaefer_atlas


class TestAtlasLoading:
    """Test atlas loading functionality."""

    @patch('network_parcel_corr.atlases.load.tf')
    @patch('network_parcel_corr.atlases.load.image')
    @patch('network_parcel_corr.atlases.load.pd')
    def test_load_schaefer_atlas_default_400_parcels(
        self, mock_pd, mock_image, mock_tf
    ):
        """Test loading Schaefer atlas with default 400 parcels."""
        # Mock the atlas data (64x64x30 with 400 unique parcels)
        mock_atlas_data = np.zeros((64, 64, 30), dtype=np.int32)
        # Fill with parcel labels 1-400
        for i in range(400):
            x, y, z = np.unravel_index(i, (64, 64, 30))
            if x < 64 and y < 64 and z < 30:
                mock_atlas_data[x, y, z] = i + 1

        # Mock the image loading
        mock_img = MagicMock()
        mock_img.get_fdata.return_value = mock_atlas_data
        mock_image.load_img.return_value = mock_img

        # Mock the labels DataFrame
        mock_labels_df = MagicMock()
        mock_labels_df.__getitem__.return_value.tolist.return_value = [
            f'7Networks_LH_Parcel{i}' for i in range(1, 401)
        ]
        mock_pd.read_csv.return_value = mock_labels_df

        # Mock templateflow paths
        mock_tf.get.side_effect = [
            '/path/to/atlas.nii.gz',  # atlas file
            '/path/to/labels.tsv',  # labels file
        ]

        # Test the function
        atlas_data, atlas_labels = load_schaefer_atlas()

        # Verify templateflow was called correctly for atlas
        mock_tf.get.assert_any_call(
            'MNI152NLin2009cAsym',
            resolution=2,
            atlas='Schaefer2018',
            desc='400Parcels7Networks',
            suffix='dseg',
            extension='nii.gz',
        )

        # Verify templateflow was called correctly for labels
        mock_tf.get.assert_any_call(
            'MNI152NLin2009cAsym',
            atlas='Schaefer2018',
            desc='400Parcels7Networks',
            suffix='dseg',
            extension='tsv',
        )

        # Verify results
        assert isinstance(atlas_data, np.ndarray)
        assert isinstance(atlas_labels, list)
        assert len(atlas_labels) == 400
        np.testing.assert_array_equal(atlas_data, mock_atlas_data)

    @patch('network_parcel_corr.atlases.load.tf')
    @patch('network_parcel_corr.atlases.load.image')
    @patch('network_parcel_corr.atlases.load.pd')
    def test_load_schaefer_atlas_custom_parcels(self, mock_pd, mock_image, mock_tf):
        """Test loading Schaefer atlas with custom number of parcels."""
        n_parcels = 100

        # Mock the atlas data
        mock_atlas_data = np.zeros((64, 64, 30), dtype=np.int32)
        for i in range(n_parcels):
            x, y, z = np.unravel_index(i, (64, 64, 30))
            if x < 64 and y < 64 and z < 30:
                mock_atlas_data[x, y, z] = i + 1

        # Mock the image loading
        mock_img = MagicMock()
        mock_img.get_fdata.return_value = mock_atlas_data
        mock_image.load_img.return_value = mock_img

        # Mock the labels DataFrame
        mock_labels_df = MagicMock()
        mock_labels_df.__getitem__.return_value.tolist.return_value = [
            f'7Networks_LH_Parcel{i}' for i in range(1, n_parcels + 1)
        ]
        mock_pd.read_csv.return_value = mock_labels_df

        # Mock templateflow paths
        mock_tf.get.side_effect = ['/path/to/atlas100.nii.gz', '/path/to/labels100.tsv']

        # Test the function
        atlas_data, atlas_labels = load_schaefer_atlas(n_parcels=n_parcels)

        # Verify templateflow was called with correct parameters
        mock_tf.get.assert_any_call(
            'MNI152NLin2009cAsym',
            resolution=2,
            atlas='Schaefer2018',
            desc=f'{n_parcels}Parcels7Networks',
            suffix='dseg',
            extension='nii.gz',
        )

        # Verify results
        assert len(atlas_labels) == n_parcels
        np.testing.assert_array_equal(atlas_data, mock_atlas_data)

    @patch('network_parcel_corr.atlases.load.tf')
    def test_load_schaefer_atlas_templateflow_error(self, mock_tf):
        """Test that errors from templateflow are properly propagated."""
        # Mock templateflow to raise an exception
        mock_tf.get.side_effect = Exception('TemplateFlow download failed')

        # Test that the exception is propagated
        with pytest.raises(Exception, match='TemplateFlow download failed'):
            load_schaefer_atlas()

    @patch('network_parcel_corr.atlases.load.tf')
    @patch('network_parcel_corr.atlases.load.image')
    @patch('network_parcel_corr.atlases.load.pd')
    def test_load_schaefer_atlas_image_loading_error(
        self, mock_pd, mock_image, mock_tf
    ):
        """Test that image loading errors are properly handled."""
        # Mock templateflow paths
        mock_tf.get.side_effect = ['/path/to/atlas.nii.gz', '/path/to/labels.tsv']

        # Mock image loading to fail
        mock_image.load_img.side_effect = Exception('Failed to load image')

        # Test that the exception is propagated
        with pytest.raises(Exception, match='Failed to load image'):
            load_schaefer_atlas()

    @patch('network_parcel_corr.atlases.load.tf')
    @patch('network_parcel_corr.atlases.load.image')
    @patch('network_parcel_corr.atlases.load.pd')
    def test_load_schaefer_atlas_labels_loading_error(
        self, mock_pd, mock_image, mock_tf
    ):
        """Test that label loading errors are properly handled."""
        # Mock atlas data and image loading
        mock_atlas_data = np.zeros((64, 64, 30), dtype=np.int32)
        mock_img = MagicMock()
        mock_img.get_fdata.return_value = mock_atlas_data
        mock_image.load_img.return_value = mock_img

        # Mock templateflow paths
        mock_tf.get.side_effect = ['/path/to/atlas.nii.gz', '/path/to/labels.tsv']

        # Mock labels loading to fail
        mock_pd.read_csv.side_effect = Exception('Failed to load labels')

        # Test that the exception is propagated
        with pytest.raises(Exception, match='Failed to load labels'):
            load_schaefer_atlas()
