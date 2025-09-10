"""Tests for modular main pipeline functions."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from src.network_parcel_corr.main import (
    load_atlas_data,
    discover_contrast_files,
    extract_parcel_data,
    compute_all_similarities,
)


class TestLoadAtlasData:
    """Test atlas data loading."""
    
    @patch('src.network_parcel_corr.main.load_schaefer_atlas')
    @patch('builtins.print')
    def test_load_atlas_data(self, mock_print, mock_load_atlas):
        """Test atlas data loading with proper logging."""
        # Mock return values
        mock_atlas_data = np.random.rand(91, 109, 91)
        mock_atlas_labels = ['parcel1', 'parcel2', 'parcel3']
        mock_load_atlas.return_value = (mock_atlas_data, mock_atlas_labels)
        
        atlas_data, atlas_labels = load_atlas_data(400)
        
        # Verify function calls
        mock_load_atlas.assert_called_once_with(400)
        mock_print.assert_called_once_with('Loading Schaefer 400-parcel atlas...')
        
        # Verify return values
        np.testing.assert_array_equal(atlas_data, mock_atlas_data)
        assert atlas_labels == mock_atlas_labels
        
    @patch('src.network_parcel_corr.main.load_schaefer_atlas')
    @patch('builtins.print')
    def test_load_atlas_data_different_parcels(self, mock_print, mock_load_atlas):
        """Test loading different parcel counts."""
        mock_load_atlas.return_value = (np.array([]), [])
        
        load_atlas_data(200)
        mock_load_atlas.assert_called_with(200)
        mock_print.assert_called_with('Loading Schaefer 200-parcel atlas...')


class TestDiscoverContrastFiles:
    """Test contrast file discovery."""
    
    @patch('src.network_parcel_corr.main.find_all_contrast_files')
    @patch('builtins.print')
    def test_discover_contrast_files(self, mock_print, mock_find_files):
        """Test contrast file discovery with proper logging."""
        # Mock return values
        mock_contrast_files = {
            'faces_vs_fixation': [Path('file1.nii.gz'), Path('file2.nii.gz')],
            'math_vs_story': [Path('file3.nii.gz')]
        }
        mock_find_files.return_value = mock_contrast_files
        
        subjects = ['sub-s01', 'sub-s02']
        input_dir = Path('/test/input')
        exclusions_file = 'exclusions.json'
        
        result = discover_contrast_files(subjects, input_dir, exclusions_file)
        
        # Verify function calls
        mock_find_files.assert_called_once_with(subjects, input_dir, exclusions_file)
        
        # Verify logging
        assert mock_print.call_count == 2
        mock_print.assert_any_call('Discovering contrast files...')
        mock_print.assert_any_call('Found 2 contrasts')
        
        # Verify return values
        assert result == mock_contrast_files
        
    @patch('src.network_parcel_corr.main.find_all_contrast_files')
    @patch('builtins.print')
    def test_discover_contrast_files_empty(self, mock_print, mock_find_files):
        """Test with no contrast files found."""
        mock_find_files.return_value = {}
        
        result = discover_contrast_files([], Path('/test'), 'exclusions.json')
        
        mock_print.assert_any_call('Found 0 contrasts')
        assert result == {}


class TestExtractParcelData:
    """Test parcel data extraction."""
    
    @patch('src.network_parcel_corr.main.extract_and_group_by_parcel')
    @patch('builtins.print')
    def test_extract_parcel_data(self, mock_print, mock_extract):
        """Test parcel data extraction with proper logging."""
        # Mock inputs
        contrast_files = {
            'faces_vs_fixation': [Path('file1.nii.gz'), Path('file2.nii.gz')],
            'math_vs_story': [Path('file3.nii.gz')]
        }
        atlas_data = np.random.rand(91, 109, 91)
        atlas_labels = ['parcel1', 'parcel2']
        
        # Mock return values
        mock_parcel_data1 = {'parcel1': [('sub-s01', 'ses-01', 'faces_vs_fixation', 'run-01', np.array([1, 2, 3]))]}
        mock_parcel_data2 = {'parcel1': [('sub-s01', 'ses-01', 'math_vs_story', 'run-01', np.array([4, 5, 6]))]}
        mock_extract.side_effect = [mock_parcel_data1, mock_parcel_data2]
        
        result = extract_parcel_data(contrast_files, atlas_data, atlas_labels)
        
        # Verify function calls
        assert mock_extract.call_count == 2
        mock_extract.assert_any_call([Path('file1.nii.gz'), Path('file2.nii.gz')], atlas_data, atlas_labels)
        mock_extract.assert_any_call([Path('file3.nii.gz')], atlas_data, atlas_labels)
        
        # Verify logging (updated for enhanced progress reporting)
        assert mock_print.call_count == 5  # Original 3 + 2 completion messages
        mock_print.assert_any_call('Extracting parcel data...')
        mock_print.assert_any_call('Processing faces_vs_fixation (2 files) [1/2]...')
        mock_print.assert_any_call('✓ Completed faces_vs_fixation - found 1 parcels with data')
        mock_print.assert_any_call('Processing math_vs_story (1 files) [2/2]...')
        mock_print.assert_any_call('✓ Completed math_vs_story - found 1 parcels with data')
        
        # Verify return values
        expected = {
            'faces_vs_fixation': mock_parcel_data1,
            'math_vs_story': mock_parcel_data2
        }
        assert result == expected
        
    @patch('src.network_parcel_corr.main.extract_and_group_by_parcel')
    @patch('builtins.print')
    def test_extract_parcel_data_empty(self, mock_print, mock_extract):
        """Test with empty contrast files."""
        result = extract_parcel_data({}, np.array([]), [])
        
        mock_print.assert_called_once_with('Extracting parcel data...')
        assert result == {}
        assert mock_extract.call_count == 0


class TestComputeAllSimilarities:
    """Test similarity computation."""
    
    @patch('src.network_parcel_corr.main.compute_between_subject_similarity')
    @patch('src.network_parcel_corr.main.compute_within_subject_similarity')
    @patch('builtins.print')
    def test_compute_all_similarities(self, mock_print, mock_within, mock_between):
        """Test computing all similarities with proper logging."""
        # Mock return values
        mock_within_sim = {'contrast1': {'parcel1': 0.8}}
        mock_between_sim = {'contrast1': {'parcel1': 0.3}}
        mock_within.return_value = mock_within_sim
        mock_between.return_value = mock_between_sim
        
        hdf5_path = Path('/test/data.h5')
        
        within, between = compute_all_similarities(hdf5_path)
        
        # Verify function calls
        mock_within.assert_called_once_with(hdf5_path)
        mock_between.assert_called_once_with(hdf5_path)
        mock_print.assert_called_once_with('Computing similarities...')
        
        # Verify return values
        assert within == mock_within_sim
        assert between == mock_between_sim
        
    @patch('src.network_parcel_corr.main.compute_between_subject_similarity')
    @patch('src.network_parcel_corr.main.compute_within_subject_similarity')
    @patch('builtins.print')
    def test_compute_all_similarities_empty(self, mock_print, mock_within, mock_between):
        """Test with empty similarity results."""
        mock_within.return_value = {}
        mock_between.return_value = {}
        
        within, between = compute_all_similarities(Path('/test/empty.h5'))
        
        assert within == {}
        assert between == {}


class TestPipelineIntegration:
    """Test integration between pipeline components."""
    
    @patch('src.network_parcel_corr.main.classify_parcels')
    @patch('src.network_parcel_corr.main.compute_all_similarities')
    @patch('src.network_parcel_corr.main.save_to_hdf5')
    @patch('src.network_parcel_corr.main.extract_parcel_data')
    @patch('src.network_parcel_corr.main.discover_contrast_files')
    @patch('src.network_parcel_corr.main.load_atlas_data')
    @patch('builtins.print')
    def test_pipeline_data_flow(self, mock_print, mock_load_atlas, mock_discover, 
                               mock_extract, mock_save, mock_similarities, mock_classify):
        """Test data flow through pipeline components."""
        # Mock all pipeline components
        mock_load_atlas.return_value = (np.array([1, 2, 3]), ['parcel1'])
        mock_discover.return_value = {'contrast1': [Path('file1.nii.gz')]}
        mock_extract.return_value = {'contrast1': {'parcel1': [('sub-s01', 'ses-01', 'contrast1', 'run-01', np.array([1, 2, 3]))]}}
        mock_save.return_value = Path('/output/data.h5')
        mock_similarities.return_value = ({'contrast1': {'parcel1': 0.8}}, {'contrast1': {'parcel1': 0.3}})
        mock_classify.return_value = {'contrast1': {'parcel1': 'canonical'}}
        
        # Import and test the full pipeline
        from src.network_parcel_corr.main import run_analysis
        
        result = run_analysis(
            subjects=['sub-s01'],
            input_dir=Path('/input'),
            output_dir=Path('/output'),
            exclusions_file='exclusions.json',
            atlas_parcels=400
        )
        
        # Verify pipeline execution
        mock_load_atlas.assert_called_once_with(400)
        mock_discover.assert_called_once()
        mock_extract.assert_called_once()
        mock_save.assert_called_once()
        mock_similarities.assert_called_once()
        mock_classify.assert_called_once()
        
        # Verify result structure
        assert 'hdf5_path' in result
        assert 'within_similarities' in result
        assert 'between_similarities' in result
        assert 'classifications' in result
        assert 'n_contrasts' in result
        assert 'n_subjects' in result
        
        assert result['n_subjects'] == 1
        assert result['n_contrasts'] == 1
        
    def test_pipeline_error_handling(self):
        """Test pipeline error handling."""
        from src.network_parcel_corr.main import run_analysis
        
        # Test with invalid inputs - should raise appropriate errors
        with pytest.raises((FileNotFoundError, ValueError, TypeError)):
            run_analysis(
                subjects=[],  # Empty subjects list
                input_dir=Path('/nonexistent'),
                output_dir=Path('/nonexistent'),
                exclusions_file='nonexistent.json'
            )