"""Test sample dataset functionality."""

import numpy as np
import h5py
from pathlib import Path

from network_parcel_corr.io.readers import (
    find_all_contrast_files,
    extract_contrast_info,
    load_exclusions,
)
from network_parcel_corr.io.writers import extract_and_group_by_parcel, save_to_hdf5
from network_parcel_corr.core.similarity import (
    compute_within_subject_similarity,
    compute_between_subject_similarity,
    compute_across_construct_similarity,
    classify_parcels,
)


class TestContrastFileDiscovery:
    """Test contrast file discovery."""

    def test_extract_contrast_info(self):
        """Test extracting information from filenames."""
        filepath = Path(
            'sub-s01_ses-01_task-nBack_run-1_contrast-twoBack-oneBack_rtmodel-rt_centered_stat-effect-size.nii.gz'
        )
        subject, session, contrast, run = extract_contrast_info(filepath)

        assert subject == 'sub-s01'
        assert session == 'ses-01'
        assert contrast == 'task-nBack_contrast-twoBack-oneBack'
        assert run == 'run-01'

    def test_find_all_contrast_files(self, sample_dataset):
        """Test finding contrast files with exclusions."""
        contrast_files = find_all_contrast_files(
            sample_dataset['subjects'],
            sample_dataset['base_dir'],
            str(sample_dataset['exclusions_file']),
        )

        # Should find 3 contrasts
        assert len(contrast_files) == 3

        # Each contrast should have files from both subjects and sessions
        for contrast, files in contrast_files.items():
            assert len(files) == 4  # 2 subjects × 2 sessions

    def test_load_exclusions_empty_file(self, sample_dataset):
        """Test loading exclusions from empty file."""
        exclusions = load_exclusions(str(sample_dataset['exclusions_file']))
        assert len(exclusions) == 0


class TestDataExtractionAndStorage:
    """Test data extraction and HDF5 storage."""

    def test_extract_and_group_by_parcel(self, sample_dataset, test_atlas_data):
        """Test extracting voxel data and grouping by parcel."""
        atlas_labels = [f'Parcel_{i}' for i in range(1, 6)]

        # Get first few files for testing
        test_files = sample_dataset['file_paths'][:2]

        grouped_data = extract_and_group_by_parcel(
            test_files, test_atlas_data, atlas_labels
        )

        # Should have data for each parcel
        assert len(grouped_data) == 5

        # Each parcel should have records
        for parcel_name, records in grouped_data.items():
            assert len(records) > 0

            # Check record structure
            for record in records:
                assert len(record) == 5  # subject, session, contrast, run, voxel_values
                assert isinstance(record[4], np.ndarray)  # voxel_values is numpy array

    def test_save_to_hdf5(self, sample_dataset, test_atlas_data, temp_dir):
        """Test saving grouped data to HDF5."""
        atlas_labels = [f'Parcel_{i}' for i in range(1, 6)]

        # Extract data from a few files
        test_files = sample_dataset['file_paths'][:2]
        grouped_by_parcel = extract_and_group_by_parcel(
            test_files, test_atlas_data, atlas_labels
        )

        # Group by contrast (simulating the full pipeline)
        grouped_by_contrast = {'test_contrast': grouped_by_parcel}

        # Save to HDF5
        hdf5_path = save_to_hdf5(grouped_by_contrast, temp_dir)

        assert hdf5_path.exists()

        # Verify structure
        with h5py.File(hdf5_path, 'r') as f:
            assert 'test_contrast' in f
            contrast_group = f['test_contrast']

            # Should have parcels
            assert len(contrast_group.keys()) > 0

            # Check a parcel structure
            first_parcel = list(contrast_group.keys())[0]
            parcel_group = contrast_group[first_parcel]

            # Should have records
            assert len(parcel_group.keys()) > 0

            # Check a record structure
            first_record = list(parcel_group.keys())[0]
            record_group = parcel_group[first_record]

            assert 'subject' in record_group.attrs
            assert 'session' in record_group.attrs
            assert 'voxel_values' in record_group


class TestSimilarityCalculations:
    """Test similarity calculations using HDF5 data."""

    def test_compute_within_subject_similarity(self, temp_dir):
        """Test computing within-subject similarity from HDF5."""
        # Create simple test HDF5 data
        hdf5_path = temp_dir / 'test_data.h5'

        with h5py.File(hdf5_path, 'w') as f:
            # Create test data structure
            contrast_group = f.create_group('test_contrast')
            parcel_group = contrast_group.create_group('test_parcel')

            # Create records for same subject, different sessions
            np.random.seed(42)
            base_data = np.random.randn(100).astype(np.float32)

            for session in [1, 2, 3]:
                record_name = f'sub-s01_ses-{session:02d}_run-01'
                record_group = parcel_group.create_group(record_name)
                record_group.attrs['subject'] = 'sub-s01'
                record_group.attrs['session'] = f'ses-{session:02d}'

                # Add some session-specific variation
                session_data = base_data + np.random.randn(100) * 0.1
                record_group.create_dataset('voxel_values', data=session_data)

        # Compute within-subject similarity
        results = compute_within_subject_similarity(hdf5_path)

        assert 'test_contrast' in results
        assert 'test_parcel' in results['test_contrast']

        # Should have a correlation value
        similarity = results['test_contrast']['test_parcel']
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1  # Correlation should be positive and <= 1

    def test_compute_between_subject_similarity(self, temp_dir):
        """Test computing between-subject similarity from HDF5."""
        hdf5_path = temp_dir / 'test_data.h5'

        with h5py.File(hdf5_path, 'w') as f:
            contrast_group = f.create_group('test_contrast')
            parcel_group = contrast_group.create_group('test_parcel')

            # Create records for different subjects with distinct patterns
            np.random.seed(42)

            # Subject 1 sessions - similar pattern
            subj1_base = np.random.randn(100).astype(np.float32)
            for session in [1, 2]:
                record_name = f'sub-s01_ses-{session:02d}_run-01'
                record_group = parcel_group.create_group(record_name)
                record_group.attrs['subject'] = 'sub-s01'
                record_group.attrs['session'] = f'ses-{session:02d}'

                # Subject 1 sessions are similar to each other
                session_data = subj1_base + np.random.randn(100) * 0.1
                record_group.create_dataset('voxel_values', data=session_data)

            # Subject 2 sessions - different pattern
            subj2_base = (
                np.random.randn(100).astype(np.float32) + 5.0
            )  # Different from subject 1
            for session in [1, 2]:
                record_name = f'sub-s02_ses-{session:02d}_run-01'
                record_group = parcel_group.create_group(record_name)
                record_group.attrs['subject'] = 'sub-s02'
                record_group.attrs['session'] = f'ses-{session:02d}'

                # Subject 2 sessions are similar to each other but different from subject 1
                session_data = subj2_base + np.random.randn(100) * 0.1
                record_group.create_dataset('voxel_values', data=session_data)

        # Compute between-subject similarity
        results = compute_between_subject_similarity(hdf5_path)

        assert 'test_contrast' in results
        assert 'test_parcel' in results['test_contrast']

        similarity = results['test_contrast']['test_parcel']
        assert isinstance(similarity, float)

        # Between-subject similarity should be lower than within-subject
        # (since we created different patterns for each subject)

    def test_classify_parcels(self):
        """Test parcel classification based on similarity values."""
        # Create test similarity data
        within_correlations = {
            'test_contrast': {
                'variable_parcel': 0.05,  # Low within + between = variable
                'fingerprint_parcel': 0.8,  # High within, high difference = fingerprint  
                'canonical_parcel': 0.9,  # High within, low difference = canonical
            }
        }

        between_correlations = {
            'test_contrast': {
                'variable_parcel': 0.04,  # Low within + between = variable
                'fingerprint_parcel': 0.2,  # High within, high difference = fingerprint
                'canonical_parcel': 0.85,  # High within, low difference = canonical
            }
        }

        classifications = classify_parcels(within_correlations, between_correlations)

        assert classifications['test_contrast']['variable_parcel'] == 'variable'
        assert (
            classifications['test_contrast']['fingerprint_parcel']
            == 'indiv_fingerprint'
        )
        assert classifications['test_contrast']['canonical_parcel'] == 'canonical'

    def test_across_construct_similarity_excludes_variable_parcels(self, temp_dir):
        """Test that across-construct similarity excludes variable parcels."""
        hdf5_path = temp_dir / 'test_data.h5'

        # Create test data with multiple contrasts in same construct
        with h5py.File(hdf5_path, 'w') as f:
            np.random.seed(42)
            base_data = np.random.randn(100).astype(np.float32)

            # Create two contrasts in the same construct
            for contrast_name in ['task-test_contrast-1', 'task-test_contrast-2']:
                contrast_group = f.create_group(contrast_name)

                # Create parcels with different classifications
                for parcel_name in ['variable_parcel', 'canonical_parcel']:
                    parcel_group = contrast_group.create_group(parcel_name)

                    # Add some test data
                    record_group = parcel_group.create_group('sub-s01_ses-01_run-01')
                    record_group.attrs['subject'] = 'sub-s01'
                    record_group.attrs['session'] = 'ses-01'

                    # Make contrasts correlated for testing
                    if contrast_name == 'task-test_contrast-1':
                        data = base_data + np.random.randn(100) * 0.1
                    else:
                        data = (
                            base_data + np.random.randn(100) * 0.1
                        )  # Similar to first

                    record_group.create_dataset('voxel_values', data=data)

        # Create construct mapping
        construct_map = {
            'Test Construct': ['task-test_contrast-1', 'task-test_contrast-2']
        }

        # Create parcel classifications (variable_parcel should be excluded)
        classifications = {
            'task-test_contrast-1': {
                'variable_parcel': 'variable',
                'canonical_parcel': 'canonical',
            },
            'task-test_contrast-2': {
                'variable_parcel': 'variable',
                'canonical_parcel': 'canonical',
            },
        }

        # Test with classifications (should exclude variable parcels)
        results_with_exclusion = compute_across_construct_similarity(
            hdf5_path, construct_map, classifications
        )

        # Test without classifications (should include all parcels)
        results_without_exclusion = compute_across_construct_similarity(
            hdf5_path, construct_map, None
        )

        # Verify variable parcels are excluded when classifications provided
        for contrast_name in ['task-test_contrast-1', 'task-test_contrast-2']:
            assert 'variable_parcel' not in results_with_exclusion[contrast_name]
            assert 'canonical_parcel' in results_with_exclusion[contrast_name]

            # Without exclusion, both parcels should be present
            assert 'variable_parcel' in results_without_exclusion[contrast_name]
            assert 'canonical_parcel' in results_without_exclusion[contrast_name]
