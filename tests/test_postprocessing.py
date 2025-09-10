"""Tests for postprocessing module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.network_parcel_corr.postprocessing.analysis import (
    compute_classification_summary,
    compute_parcel_statistics,
    rank_parcels_by_fingerprint_strength,
    rank_parcels_by_variability,
    rank_parcels_by_canonicality,
    compute_cross_contrast_consistency,
)

from src.network_parcel_corr.postprocessing.export import (
    export_parcel_classifications_csv,
    export_summary_statistics_csv,
    export_ranked_parcels_csv,
    export_cross_contrast_consistency_csv,
    export_all_postprocessing_results,
)


@pytest.fixture
def sample_analysis_data():
    """Create sample analysis data for testing."""
    within_similarities = {
        'task-nBack_contrast-twoBack-oneBack': {
            'parcel1': 0.8,
            'parcel2': 0.4,
            'parcel3': 0.05,
        },
        'task-flanker_contrast-incongruent-congruent': {
            'parcel1': 0.85,
            'parcel2': 0.6,
            'parcel3': 0.03,
        }
    }
    
    between_similarities = {
        'task-nBack_contrast-twoBack-oneBack': {
            'parcel1': 0.2,
            'parcel2': 0.55,
            'parcel3': 0.02,
        },
        'task-flanker_contrast-incongruent-congruent': {
            'parcel1': 0.15,
            'parcel2': 0.58,
            'parcel3': 0.01,
        }
    }
    
    classifications = {
        'task-nBack_contrast-twoBack-oneBack': {
            'parcel1': 'canonical',
            'parcel2': 'indiv_fingerprint',
            'parcel3': 'variable',
        },
        'task-flanker_contrast-incongruent-congruent': {
            'parcel1': 'canonical', 
            'parcel2': 'indiv_fingerprint',
            'parcel3': 'variable',
        }
    }
    
    return within_similarities, between_similarities, classifications


class TestAnalysisFunctions:
    """Test analysis functions."""
    
    def test_compute_classification_summary(self, sample_analysis_data):
        """Test classification summary computation."""
        _, _, classifications = sample_analysis_data
        
        summary = compute_classification_summary(classifications)
        
        assert len(summary) == 2  # Two contrasts
        
        # Check each contrast has correct counts
        for contrast_name in classifications.keys():
            assert summary[contrast_name]['canonical'] == 1
            assert summary[contrast_name]['indiv_fingerprint'] == 1
            assert summary[contrast_name]['variable'] == 1
    
    def test_compute_parcel_statistics(self, sample_analysis_data):
        """Test parcel statistics computation."""
        within, between, classifications = sample_analysis_data
        
        statistics = compute_parcel_statistics(within, between, classifications)
        
        assert len(statistics) == 2  # Two contrasts
        
        # Check first contrast, first parcel
        contrast1 = 'task-nBack_contrast-twoBack-oneBack'
        parcel1_stats = statistics[contrast1]['parcel1']
        
        assert parcel1_stats['within_subject_similarity'] == 0.8
        assert parcel1_stats['between_subject_similarity'] == 0.2
        assert parcel1_stats['similarity_difference'] == pytest.approx(0.6)
        assert parcel1_stats['similarity_sum'] == pytest.approx(1.0)
        assert parcel1_stats['similarity_ratio'] == pytest.approx(4.0)
        assert parcel1_stats['classification'] == 'canonical'
    
    def test_rank_parcels_by_fingerprint_strength(self, sample_analysis_data):
        """Test fingerprint strength ranking."""
        within, between, classifications = sample_analysis_data
        
        ranking = rank_parcels_by_fingerprint_strength(within, between, classifications)
        
        # Should have 6 parcels total (3 parcels × 2 contrasts)
        assert len(ranking) == 6
        
        # First should be highest fingerprint strength (within - between)
        top_parcel = ranking[0]
        assert top_parcel[2] == pytest.approx(0.7)  # parcel1 from flanker contrast (0.85 - 0.15)
        assert top_parcel[1] == 'parcel1'
        
        # Last should be lowest fingerprint strength
        bottom_parcel = ranking[-1]
        assert bottom_parcel[2] == pytest.approx(-0.15)  # parcel2 from nBack contrast (0.4 - 0.55)
        assert bottom_parcel[1] == 'parcel2'
    
    def test_rank_parcels_by_variability(self, sample_analysis_data):
        """Test variability ranking."""
        within, between, classifications = sample_analysis_data
        
        ranking = rank_parcels_by_variability(within, between, classifications)
        
        assert len(ranking) == 6
        
        # Most variable should have lowest within + between similarity
        most_variable = ranking[0]
        # Actually parcel2 from flanker has highest variability score (highest sum)
        assert most_variable[1] == 'parcel2'
        assert most_variable[3] == 'indiv_fingerprint'
    
    def test_rank_parcels_by_canonicality(self, sample_analysis_data):
        """Test canonicality ranking."""
        within, between, classifications = sample_analysis_data
        
        ranking = rank_parcels_by_canonicality(within, between, classifications)
        
        assert len(ranking) == 6
        
        # Most canonical should have high within * (within - between)
        most_canonical = ranking[0]
        assert most_canonical[1] == 'parcel1'
        assert most_canonical[3] == 'canonical'
    
    def test_compute_cross_contrast_consistency(self, sample_analysis_data):
        """Test cross-contrast consistency computation."""
        _, _, classifications = sample_analysis_data
        
        consistency = compute_cross_contrast_consistency(classifications)
        
        assert len(consistency) == 3  # Three unique parcels
        
        # All parcels should have perfect consistency (same classification across contrasts)
        for parcel_name, consistency_info in consistency.items():
            assert consistency_info['consistency_score'] == 1.0
            assert consistency_info['n_contrasts'] == 2
            
            if parcel_name == 'parcel1':
                assert consistency_info['most_common_classification'] == 'canonical'
                assert consistency_info['canonical'] == 1.0
            elif parcel_name == 'parcel2':
                assert consistency_info['most_common_classification'] == 'indiv_fingerprint'
                assert consistency_info['indiv_fingerprint'] == 1.0
            elif parcel_name == 'parcel3':
                assert consistency_info['most_common_classification'] == 'variable'
                assert consistency_info['variable'] == 1.0


class TestExportFunctions:
    """Test export functions."""
    
    def test_export_parcel_classifications_csv(self, sample_analysis_data):
        """Test parcel classifications CSV export."""
        within, between, classifications = sample_analysis_data
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            output_path = export_parcel_classifications_csv(
                within, between, classifications, output_dir
            )
            
            assert output_path.exists()
            
            # Read and verify CSV content
            df = pd.read_csv(output_path)
            
            assert len(df) == 6  # 3 parcels × 2 contrasts
            assert set(df.columns) == {
                'contrast', 'parcel', 'classification',
                'within_subject_similarity', 'between_subject_similarity',
                'similarity_difference', 'similarity_sum', 'similarity_ratio'
            }
            
            # Check specific values
            parcel1_nback = df[(df['contrast'] == 'task-nBack_contrast-twoBack-oneBack') & 
                              (df['parcel'] == 'parcel1')]
            assert len(parcel1_nback) == 1
            assert parcel1_nback.iloc[0]['within_subject_similarity'] == 0.8
            assert parcel1_nback.iloc[0]['between_subject_similarity'] == 0.2
            assert parcel1_nback.iloc[0]['classification'] == 'canonical'
    
    def test_export_summary_statistics_csv(self, sample_analysis_data):
        """Test summary statistics CSV export."""
        within, between, classifications = sample_analysis_data
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            output_path = export_summary_statistics_csv(
                within, between, classifications, output_dir
            )
            
            assert output_path.exists()
            
            # Read and verify CSV content
            df = pd.read_csv(output_path)
            
            # Should have 2 contrasts + 1 overall row
            assert len(df) == 3
            
            # Check column structure
            expected_columns = {
                'contrast', 'total_parcels',
                'canonical_count', 'canonical_percentage',
                'indiv_fingerprint_count', 'indiv_fingerprint_percentage',
                'variable_count', 'variable_percentage'
            }
            assert set(df.columns) == expected_columns
            
            # Check individual contrast rows
            contrast_rows = df[df['contrast'] != 'OVERALL']
            for _, row in contrast_rows.iterrows():
                assert row['total_parcels'] == 3
                assert row['canonical_count'] == 1
                assert row['canonical_percentage'] == pytest.approx(33.33, rel=1e-2)
                assert row['indiv_fingerprint_count'] == 1
                assert row['variable_count'] == 1
            
            # Check overall row
            overall_row = df[df['contrast'] == 'OVERALL'].iloc[0]
            assert overall_row['total_parcels'] == 6
            assert overall_row['canonical_count'] == 2
            assert overall_row['canonical_percentage'] == pytest.approx(33.33, rel=1e-2)
    
    def test_export_ranked_parcels_csv(self, sample_analysis_data):
        """Test ranked parcels CSV export."""
        within, between, classifications = sample_analysis_data
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            output_paths = export_ranked_parcels_csv(
                within, between, classifications, output_dir, top_n=10
            )
            
            assert len(output_paths) == 3  # fingerprint, variability, canonicality
            assert 'fingerprint' in output_paths
            assert 'variability' in output_paths
            assert 'canonicality' in output_paths
            
            # Check fingerprint ranking
            fingerprint_df = pd.read_csv(output_paths['fingerprint'])
            assert len(fingerprint_df) == 6  # All 6 parcels
            assert set(fingerprint_df.columns) == {
                'rank', 'contrast', 'parcel', 'fingerprint_strength', 'classification'
            }
            
            # Top parcel should have highest fingerprint strength
            top_row = fingerprint_df.iloc[0]
            assert top_row['rank'] == 1
            assert top_row['parcel'] == 'parcel1'
            assert top_row['fingerprint_strength'] == pytest.approx(0.7, rel=1e-2)
            
            # Check variability ranking
            variability_df = pd.read_csv(output_paths['variability'])
            assert len(variability_df) == 6
            
            # Most variable is actually parcel2 with highest sum
            top_variable = variability_df.iloc[0]
            assert top_variable['parcel'] == 'parcel2'
            assert top_variable['classification'] == 'indiv_fingerprint'
    
    def test_export_cross_contrast_consistency_csv(self, sample_analysis_data):
        """Test cross-contrast consistency CSV export."""
        _, _, classifications = sample_analysis_data
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            output_path = export_cross_contrast_consistency_csv(
                classifications, output_dir
            )
            
            assert output_path.exists()
            
            # Read and verify CSV content
            df = pd.read_csv(output_path)
            
            assert len(df) == 3  # Three unique parcels
            assert set(df.columns) == {
                'parcel', 'most_common_classification', 'consistency_score', 'n_contrasts',
                'canonical_proportion', 'indiv_fingerprint_proportion', 'variable_proportion'
            }
            
            # All parcels should have perfect consistency
            for _, row in df.iterrows():
                assert row['consistency_score'] == 1.0
                assert row['n_contrasts'] == 2
    
    def test_export_all_postprocessing_results(self, sample_analysis_data):
        """Test exporting all postprocessing results."""
        within, between, classifications = sample_analysis_data
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            output_paths = export_all_postprocessing_results(
                within, between, classifications, output_dir, top_n=10
            )
            
            # Should have all expected files
            expected_keys = {
                'classifications', 'summary', 'consistency',
                'fingerprint', 'variability', 'canonicality'
            }
            assert set(output_paths.keys()) == expected_keys
            
            # All files should exist
            for path in output_paths.values():
                assert path.exists()
                assert path.suffix == '.csv'
                
            # Verify content is not empty
            for path in output_paths.values():
                df = pd.read_csv(path)
                assert len(df) > 0