"""Tests for modular similarity calculation functions."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from src.network_parcel_corr.core.similarity import (
    compute_correlation_matrix_upper_triangle,
    extract_subject_sessions_from_parcel,
    compute_within_subject_correlation,
    compute_between_subject_correlations,
    extract_session_info_from_parcel,
    find_constructs_for_contrast,
    is_parcel_variable,
    collect_construct_voxel_data,
    compute_across_construct_correlation,
    classify_single_parcel,
)


class TestCorrelationMatrixUpperTriangle:
    """Test correlation matrix upper triangle extraction."""
    
    def test_basic_correlation_matrix(self):
        """Test basic correlation matrix computation."""
        # Create simple test data
        data = np.array([[1, 2, 3], [2, 4, 6], [1.1, 2.1, 3.1]])
        
        result = compute_correlation_matrix_upper_triangle(data)
        
        assert len(result) == 3  # 3 choose 2 upper triangle values
        assert all(-1 <= val <= 1 for val in result)  # Valid correlation range
        
    def test_insufficient_data(self):
        """Test with insufficient data (less than 2 rows)."""
        data = np.array([[1, 2, 3]])
        result = compute_correlation_matrix_upper_triangle(data)
        assert len(result) == 0
        
    def test_empty_data(self):
        """Test with empty data."""
        data = np.array([])
        result = compute_correlation_matrix_upper_triangle(data)
        assert len(result) == 0
        
    def test_perfect_correlation(self):
        """Test with perfectly correlated data."""
        data = np.array([[1, 2, 3], [2, 4, 6]])
        result = compute_correlation_matrix_upper_triangle(data)
        assert len(result) == 1
        assert np.isclose(result[0], 1.0)


class TestExtractSubjectSessions:
    """Test subject session extraction from HDF5 groups."""
    
    def test_extract_subject_sessions(self):
        """Test extracting subject sessions from mock parcel group."""
        # Mock HDF5 group structure
        mock_parcel_group = Mock()
        mock_parcel_group.keys.return_value = ['sub-s01_ses-01_run-01', 'sub-s01_ses-02_run-01', 'sub-s02_ses-01_run-01']
        
        # Mock records with proper __getitem__ setup
        mock_record1 = Mock()
        mock_record1.attrs = {'subject': 'sub-s01'}
        mock_record1.__getitem__ = Mock(return_value=np.array([1, 2, 3]))
        
        mock_record2 = Mock()
        mock_record2.attrs = {'subject': 'sub-s01'}
        mock_record2.__getitem__ = Mock(return_value=np.array([1.1, 2.1, 3.1]))
        
        mock_record3 = Mock()
        mock_record3.attrs = {'subject': 'sub-s02'}
        mock_record3.__getitem__ = Mock(return_value=np.array([4, 5, 6]))
        
        # Mock the parcel group's __getitem__ properly
        mock_parcel_group.__getitem__ = Mock(side_effect=[mock_record1, mock_record2, mock_record3])
        
        result = extract_subject_sessions_from_parcel(mock_parcel_group)
        
        assert len(result) == 2  # Two subjects
        assert 'sub-s01' in result and 'sub-s02' in result
        assert len(result['sub-s01']) == 2  # Two sessions for sub-s01
        assert len(result['sub-s02']) == 1  # One session for sub-s02


class TestWithinSubjectCorrelation:
    """Test within-subject correlation computation."""
    
    def test_sufficient_sessions(self):
        """Test with sufficient sessions for correlation."""
        sessions = [
            np.array([1, 2, 3, 4, 5]),
            np.array([1.1, 2.1, 3.1, 4.1, 5.1]),
            np.array([0.9, 1.9, 2.9, 3.9, 4.9])
        ]
        
        result = compute_within_subject_correlation(sessions)
        
        assert result is not None
        assert -1 <= result <= 1
        assert result > 0.9  # Should be highly correlated
        
    def test_insufficient_sessions(self):
        """Test with insufficient sessions."""
        sessions = [np.array([1, 2, 3, 4, 5])]
        result = compute_within_subject_correlation(sessions)
        assert result is None
        
    def test_empty_sessions(self):
        """Test with empty sessions list."""
        sessions = []
        result = compute_within_subject_correlation(sessions)
        assert result is None


class TestBetweenSubjectCorrelations:
    """Test between-subject correlation computation."""
    
    def test_multiple_subjects(self):
        """Test with multiple subjects."""
        session_info = [
            (np.array([1, 2, 3]), 'sub-s01'),
            (np.array([1.1, 2.1, 3.1]), 'sub-s01'),  # Same subject - should be excluded
            (np.array([4, 5, 6]), 'sub-s02'),
            (np.array([7, 8, 9]), 'sub-s03')
        ]
        
        result = compute_between_subject_correlations(session_info)
        
        # Should have 4 between-subject correlations: 
        # (s01-sess1, s02), (s01-sess1, s03), (s01-sess2, s02), (s01-sess2, s03), (s02, s03)
        assert len(result) == 5  # Actually 5 correlations, not 4
        assert all(-1 <= val <= 1 for val in result)
        
    def test_same_subject_only(self):
        """Test with sessions from same subject only."""
        session_info = [
            (np.array([1, 2, 3]), 'sub-s01'),
            (np.array([1.1, 2.1, 3.1]), 'sub-s01')
        ]
        
        result = compute_between_subject_correlations(session_info)
        assert len(result) == 0  # No between-subject correlations


class TestExtractSessionInfo:
    """Test session info extraction."""
    
    def test_extract_session_info(self):
        """Test extracting session info from mock parcel group."""
        mock_parcel_group = Mock()
        mock_parcel_group.keys.return_value = ['record1', 'record2']
        
        mock_record1 = Mock()
        mock_record1.attrs = {'subject': 'sub-s01'}
        mock_record1.__getitem__ = Mock(return_value=np.array([1, 2, 3]))
        
        mock_record2 = Mock()
        mock_record2.attrs = {'subject': 'sub-s02'}
        mock_record2.__getitem__ = Mock(return_value=np.array([4, 5, 6]))
        
        mock_parcel_group.__getitem__ = Mock(side_effect=[mock_record1, mock_record2])
        
        result = extract_session_info_from_parcel(mock_parcel_group)
        
        assert len(result) == 2
        assert result[0][1] == 'sub-s01'
        assert result[1][1] == 'sub-s02'
        np.testing.assert_array_equal(result[0][0], np.array([1, 2, 3]))
        np.testing.assert_array_equal(result[1][0], np.array([4, 5, 6]))


class TestFindConstructsForContrast:
    """Test finding constructs for contrasts."""
    
    def test_single_construct(self):
        """Test contrast in single construct."""
        construct_map = {
            'language': ['contrast1', 'contrast2'],
            'faces': ['contrast3', 'contrast4']
        }
        
        result = find_constructs_for_contrast('contrast1', construct_map)
        assert result == ['language']
        
    def test_multiple_constructs(self):
        """Test contrast in multiple constructs."""
        construct_map = {
            'language': ['contrast1', 'contrast2'],
            'general': ['contrast1', 'contrast3']
        }
        
        result = find_constructs_for_contrast('contrast1', construct_map)
        assert set(result) == {'language', 'general'}
        
    def test_no_constructs(self):
        """Test contrast not in any construct."""
        construct_map = {
            'language': ['contrast1', 'contrast2'],
            'faces': ['contrast3', 'contrast4']
        }
        
        result = find_constructs_for_contrast('contrast5', construct_map)
        assert result == []


class TestIsParcelVariable:
    """Test parcel variability checking."""
    
    def test_variable_parcel(self):
        """Test variable parcel classification."""
        classifications = {
            'contrast1': {'parcel1': 'variable', 'parcel2': 'canonical'}
        }
        
        result = is_parcel_variable('contrast1', 'parcel1', classifications)
        assert result is True
        
    def test_non_variable_parcel(self):
        """Test non-variable parcel classification."""
        classifications = {
            'contrast1': {'parcel1': 'canonical', 'parcel2': 'variable'}
        }
        
        result = is_parcel_variable('contrast1', 'parcel1', classifications)
        assert result is False
        
    def test_no_classifications(self):
        """Test with no classifications provided."""
        result = is_parcel_variable('contrast1', 'parcel1', None)
        assert result is False
        
    def test_missing_contrast(self):
        """Test with missing contrast in classifications."""
        classifications = {'contrast2': {'parcel1': 'variable'}}
        
        result = is_parcel_variable('contrast1', 'parcel1', classifications)
        assert result is False


class TestAcrossConstructCorrelation:
    """Test across-construct correlation computation."""
    
    def test_sufficient_data(self):
        """Test with sufficient contrasts for correlation."""
        voxel_data = [
            np.array([1, 2, 3, 4, 5]),
            np.array([1.1, 2.1, 3.1, 4.1, 5.1]),
            np.array([0.9, 1.9, 2.9, 3.9, 4.9])
        ]
        
        result = compute_across_construct_correlation(voxel_data)
        
        assert result is not None
        assert -1 <= result <= 1
        assert result > 0.9  # Should be highly correlated
        
    def test_insufficient_data(self):
        """Test with insufficient contrasts."""
        voxel_data = [np.array([1, 2, 3, 4, 5])]
        result = compute_across_construct_correlation(voxel_data)
        assert result is None
        
    def test_empty_data(self):
        """Test with empty data."""
        voxel_data = []
        result = compute_across_construct_correlation(voxel_data)
        assert result is None


class TestClassifySingleParcel:
    """Test single parcel classification."""
    
    def test_variable_classification(self):
        """Test variable parcel classification."""
        result = classify_single_parcel(0.02, 0.03, threshold=0.1)
        assert result == 'variable'
        
    def test_individual_fingerprint_classification(self):
        """Test individual fingerprint classification."""
        result = classify_single_parcel(0.8, 0.2, threshold=0.1)
        assert result == 'indiv_fingerprint'
        
    def test_canonical_classification(self):
        """Test canonical parcel classification."""
        result = classify_single_parcel(0.15, 0.12, threshold=0.1)
        assert result == 'canonical'
        
    def test_edge_cases(self):
        """Test edge cases near threshold."""
        # Exactly at threshold for variable
        result = classify_single_parcel(0.05, 0.05, threshold=0.1)
        assert result == 'canonical'  # sum equals threshold, so not < threshold, hence not variable
        
        # Exactly at threshold for individual fingerprint
        result = classify_single_parcel(0.2, 0.1, threshold=0.1)
        assert result == 'canonical'  # difference equals threshold, so not > threshold, hence canonical