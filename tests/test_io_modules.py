"""Tests for modular I/O functions."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import json
import tempfile
import h5py

from src.network_parcel_corr.io.readers import (
    extract_subject_id,
    extract_session_id,
    extract_contrast_name,
    extract_run_id,
    create_exclusion_key,
    parse_exclusion_entry,
    find_subject_contrast_files,
)

from src.network_parcel_corr.io.writers import (
    create_label_to_name_mapping,
    extract_parcel_voxels,
    process_single_contrast_file,
    validate_parcel_voxel_consistency,
    create_record_name,
    create_hdf5_record_group,
)


class TestFilenameExtraction:
    """Test filename parsing functions."""
    
    def test_extract_subject_id(self):
        """Test subject ID extraction."""
        assert extract_subject_id('sub-s01_ses-01_task-nBack_run-1_contrast-twoBack-oneBack_rtmodel-rt_centered_stat-effect-size.nii.gz') == 'sub-s01'
        assert extract_subject_id('sub-s123_other_stuff.nii.gz') == 'sub-s123'
        assert extract_subject_id('no_subject_here.nii.gz') is None
        
    def test_extract_session_id(self):
        """Test session ID extraction."""
        assert extract_session_id('sub-s01_ses-01_task-nBack_run-1_contrast-twoBack-oneBack_rtmodel-rt_centered_stat-effect-size.nii.gz') == 'ses-01'
        assert extract_session_id('ses-123_other_stuff.nii.gz') == 'ses-123'
        assert extract_session_id('no_session_here.nii.gz') is None
        
    def test_extract_contrast_name(self):
        """Test contrast name extraction."""
        filename = 'sub-s01_ses-01_task-nBack_run-1_contrast-twoBack-oneBack_rtmodel-rt_centered_stat-effect-size.nii.gz'
        assert extract_contrast_name(filename) == 'task-nBack_contrast-twoBack-oneBack'
        
        filename2 = 'sub-s01_ses-02_task-flanker_run-02_contrast-incongruent-congruent_rtmodel-rt_stat-effect-size.nii.gz'
        assert extract_contrast_name(filename2) == 'task-flanker_contrast-incongruent-congruent'
        
        # Test contrast names with underscores (the bug case)
        filename3 = 'sub-s03_ses-01_task-spatialTS_run-1_contrast-task_switch_cost_rtmodel-rt_centered_stat-effect-size.nii.gz'
        assert extract_contrast_name(filename3) == 'task-spatialTS_contrast-task_switch_cost'
        
        filename4 = 'sub-s03_ses-01_task-spatialTS_run-1_contrast-task_switch_cue_switch-task_stay_cue_stay_rtmodel-rt_centered_stat-effect-size.nii.gz'
        assert extract_contrast_name(filename4) == 'task-spatialTS_contrast-task_switch_cue_switch-task_stay_cue_stay'
        
        assert extract_contrast_name('no_task_or_contrast_here.nii.gz') is None
        
    def test_extract_run_id(self):
        """Test run ID extraction."""
        assert extract_run_id('sub-s01_ses-01_task-nBack_run-1_contrast-twoBack-oneBack_rtmodel-rt_centered_stat-effect-size.nii.gz') == 'run-01'
        assert extract_run_id('sub-s01_ses-02_task-flanker_run-123_contrast-incongruent-congruent_effect-size.nii.gz') == 'run-123'
        assert extract_run_id('no_run_here.nii.gz') is None


class TestExclusionHandling:
    """Test exclusion key creation and parsing."""
    
    def test_create_exclusion_key(self):
        """Test exclusion key creation."""
        key = create_exclusion_key('sub-s01', 'ses-02', 'faces', 'run-01')
        assert key == 'sub-s01_ses-02_faces_run-01'
        
    def test_parse_exclusion_entry(self):
        """Test exclusion entry parsing."""
        entry = {
            "subject": "sub-s01",
            "session": "ses-02", 
            "task": "faces",
            "run": "run-01"
        }
        result = parse_exclusion_entry(entry)
        assert result == 'sub-s01_ses-02_faces_run-01'


class TestContrastFileFinding:
    """Test contrast file discovery."""
    
    @patch('src.network_parcel_corr.io.readers.extract_contrast_info')
    def test_find_subject_contrast_files(self, mock_extract):
        """Test finding contrast files for single subject."""
        # Mock directory structure
        mock_subject_dir = Mock(spec=Path)
        mock_subject_dir.exists.return_value = True
        
        # Mock file paths
        mock_files = [Path('file1.nii.gz'), Path('file2.nii.gz')]
        mock_subject_dir.glob.return_value = mock_files
        
        # Mock extraction results
        mock_extract.side_effect = [
            ('sub-s01', 'ses-01', 'faces_vs_fixation', 'run-01'),
            ('sub-s01', 'ses-01', 'math_vs_story', 'run-01')
        ]
        
        exclusions = set()
        result = find_subject_contrast_files(mock_subject_dir, exclusions)
        
        assert len(result) == 2
        assert ('faces_vs_fixation', mock_files[0]) in result
        assert ('math_vs_story', mock_files[1]) in result
        
    @patch('src.network_parcel_corr.io.readers.extract_contrast_info')
    def test_find_subject_contrast_files_with_exclusions(self, mock_extract):
        """Test finding contrast files with exclusions."""
        mock_subject_dir = Mock(spec=Path)
        mock_subject_dir.exists.return_value = True
        mock_files = [Path('file1.nii.gz')]
        mock_subject_dir.glob.return_value = mock_files
        
        mock_extract.return_value = ('sub-s01', 'ses-01', 'faces_vs_fixation', 'run-01')
        
        exclusions = {'sub-s01_ses-01_faces_run-01'}  # This should be excluded
        result = find_subject_contrast_files(mock_subject_dir, exclusions)
        
        assert len(result) == 0  # File should be excluded
        
    def test_find_subject_contrast_files_nonexistent_dir(self):
        """Test with nonexistent subject directory."""
        mock_subject_dir = Mock(spec=Path)
        mock_subject_dir.exists.return_value = False
        
        result = find_subject_contrast_files(mock_subject_dir, set())
        assert len(result) == 0


class TestLabelMapping:
    """Test atlas label mapping functions."""
    
    def test_create_label_to_name_mapping(self):
        """Test label to name mapping creation."""
        labels = ['parcel1', 'parcel2', 'parcel3']
        mapping = create_label_to_name_mapping(labels)
        
        expected = {1: 'parcel1', 2: 'parcel2', 3: 'parcel3'}
        assert mapping == expected
        
    def test_empty_labels(self):
        """Test with empty label list."""
        mapping = create_label_to_name_mapping([])
        assert mapping == {}


class TestParcelVoxelExtraction:
    """Test parcel voxel extraction functions."""
    
    def test_extract_parcel_voxels(self):
        """Test voxel extraction for specific parcel."""
        # Create test data
        img_data = np.random.rand(10, 10, 10)
        atlas_data = np.ones((10, 10, 10))
        atlas_data[0:5, 0:5, 0:5] = 2  # Parcel 2 region
        
        voxels = extract_parcel_voxels(img_data, atlas_data, parcel_idx=2)
        
        assert len(voxels) == 125  # 5x5x5 region
        
    def test_extract_empty_parcel(self):
        """Test extraction for non-existent parcel."""
        img_data = np.random.rand(10, 10, 10)
        atlas_data = np.ones((10, 10, 10))  # Only parcel 1 exists
        
        voxels = extract_parcel_voxels(img_data, atlas_data, parcel_idx=2)
        assert len(voxels) == 0


class TestContrastFileProcessing:
    """Test single contrast file processing."""
    
    @patch('src.network_parcel_corr.io.writers.image.load_img')
    @patch('src.network_parcel_corr.io.readers.extract_contrast_info')
    def test_process_single_contrast_file(self, mock_extract, mock_load_img):
        """Test processing single contrast file."""
        # Mock file processing
        mock_extract.return_value = ('sub-s01', 'ses-01', 'faces_vs_fixation', 'run-01')
        
        mock_img = Mock()
        mock_img_data = np.random.rand(10, 10, 10)
        mock_img.get_fdata.return_value = mock_img_data
        mock_load_img.return_value = mock_img
        
        # Create test atlas
        atlas_data = np.ones((10, 10, 10))
        atlas_data[0:5, 0:5, 0:5] = 2
        label_mapping = {1: 'parcel1', 2: 'parcel2'}
        
        filepath = Path('test_file.nii.gz')
        result = process_single_contrast_file(filepath, atlas_data, label_mapping)
        
        assert len(result) == 2  # Should have data for both parcels
        assert 'parcel1' in result
        assert 'parcel2' in result
        
        # Check record structure
        parcel1_record = result['parcel1']
        assert parcel1_record[0] == 'sub-s01'  # subject
        assert parcel1_record[1] == 'ses-01'   # session
        assert parcel1_record[2] == 'faces_vs_fixation'  # contrast
        assert parcel1_record[3] == 'run-01'   # run
        assert isinstance(parcel1_record[4], np.ndarray)  # voxel_values
        
    @patch('src.network_parcel_corr.io.readers.extract_contrast_info')
    def test_process_single_contrast_file_invalid_info(self, mock_extract):
        """Test processing file with invalid contrast info."""
        mock_extract.return_value = (None, 'ses-01', 'faces_vs_fixation', 'run-01')  # Invalid subject
        
        filepath = Path('test_file.nii.gz')
        atlas_data = np.ones((10, 10, 10))
        label_mapping = {1: 'parcel1'}
        
        result = process_single_contrast_file(filepath, atlas_data, label_mapping)
        assert len(result) == 0  # Should return empty dict for invalid info


class TestHDF5Utilities:
    """Test HDF5 utility functions."""
    
    def test_validate_parcel_voxel_consistency_valid(self):
        """Test voxel consistency validation with valid data."""
        records = [
            ('sub-s01', 'ses-01', 'contrast1', 'run-01', np.array([1, 2, 3])),
            ('sub-s02', 'ses-01', 'contrast1', 'run-01', np.array([4, 5, 6])),
        ]
        
        # Should not raise exception
        validate_parcel_voxel_consistency('parcel1', records, 'contrast1')
        
    def test_validate_parcel_voxel_consistency_invalid(self):
        """Test voxel consistency validation with inconsistent data."""
        records = [
            ('sub-s01', 'ses-01', 'contrast1', 'run-01', np.array([1, 2, 3])),
            ('sub-s02', 'ses-01', 'contrast1', 'run-01', np.array([4, 5])),  # Different length
        ]
        
        with pytest.raises(ValueError, match="Inconsistent voxel counts"):
            validate_parcel_voxel_consistency('parcel1', records, 'contrast1')
            
    def test_create_record_name(self):
        """Test record name creation."""
        name = create_record_name('sub-s01', 'ses-02', 'run-01')
        assert name == 'sub-s01_ses-02_run-01'
        
    def test_create_hdf5_record_group(self):
        """Test HDF5 record group creation."""
        # Create mock parcel group
        mock_parcel_group = Mock()
        mock_record_group = Mock()
        mock_parcel_group.create_group.return_value = mock_record_group
        mock_record_group.attrs = {}
        mock_dataset = Mock()
        mock_record_group.create_dataset.return_value = mock_dataset
        
        parcel_data = np.array([1.0, 2.0, 3.0])
        
        create_hdf5_record_group(
            mock_parcel_group, 'sub-s01', 'ses-02', 'contrast1', 'run-01', parcel_data
        )
        
        # Verify group creation and attributes
        mock_parcel_group.create_group.assert_called_once_with('sub-s01_ses-02_run-01')
        assert mock_record_group.attrs['subject'] == 'sub-s01'
        assert mock_record_group.attrs['session'] == 'ses-02'
        assert mock_record_group.attrs['mean_voxel_value'] == 2.0
        mock_record_group.create_dataset.assert_called_once_with('voxel_values', data=parcel_data)