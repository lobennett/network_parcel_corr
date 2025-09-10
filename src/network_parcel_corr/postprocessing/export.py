"""Export functions for parcel classification results to CSV format."""

from typing import Dict, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from .analysis import (
    compute_classification_summary,
    compute_parcel_statistics,
    rank_parcels_by_fingerprint_strength,
    rank_parcels_by_variability,
    rank_parcels_by_canonicality,
    compute_cross_contrast_consistency,
)


def export_parcel_classifications_csv(
    within_similarities: Dict[str, Dict[str, float]],
    between_similarities: Dict[str, Dict[str, float]],
    classifications: Dict[str, Dict[str, str]],
    output_dir: Path,
    filename: str = "parcel_classifications.csv"
) -> Path:
    """
    Export detailed parcel classification results to CSV.
    
    Parameters
    ----------
    within_similarities : Dict[str, Dict[str, float]]
        Within-subject similarities by contrast and parcel
    between_similarities : Dict[str, Dict[str, float]]
        Between-subject similarities by contrast and parcel
    classifications : Dict[str, Dict[str, str]]
        Parcel classifications by contrast and parcel
    output_dir : Path
        Output directory for CSV file
    filename : str
        Output filename (default: "parcel_classifications.csv")
        
    Returns
    -------
    Path
        Path to created CSV file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    # Compute detailed statistics
    statistics = compute_parcel_statistics(
        within_similarities, between_similarities, classifications
    )
    
    # Convert to DataFrame format
    rows = []
    for contrast_name, parcel_stats in statistics.items():
        for parcel_name, stats in parcel_stats.items():
            row = {
                'contrast': contrast_name,
                'parcel': parcel_name,
                'classification': stats['classification'],
                'within_subject_similarity': stats['within_subject_similarity'],
                'between_subject_similarity': stats['between_subject_similarity'],
                'similarity_difference': stats['similarity_difference'],
                'similarity_sum': stats['similarity_sum'],
                'similarity_ratio': stats['similarity_ratio'],
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by contrast and parcel for consistency
    df = df.sort_values(['contrast', 'parcel']).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False, float_format='%.6f')
    
    print(f"Exported parcel classifications to: {output_path}")
    return output_path


def export_summary_statistics_csv(
    within_similarities: Dict[str, Dict[str, float]],
    between_similarities: Dict[str, Dict[str, float]],
    classifications: Dict[str, Dict[str, str]],
    output_dir: Path,
    filename: str = "classification_summary.csv"
) -> Path:
    """
    Export summary statistics for classifications to CSV.
    
    Parameters
    ----------
    within_similarities : Dict[str, Dict[str, float]]
        Within-subject similarities by contrast and parcel
    between_similarities : Dict[str, Dict[str, float]]
        Between-subject similarities by contrast and parcel
    classifications : Dict[str, Dict[str, str]]
        Parcel classifications by contrast and parcel
    output_dir : Path
        Output directory for CSV file
    filename : str
        Output filename (default: "classification_summary.csv")
        
    Returns
    -------
    Path
        Path to created CSV file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    # Compute classification summary
    summary = compute_classification_summary(classifications)
    
    # Convert to DataFrame format
    rows = []
    for contrast_name, classification_counts in summary.items():
        total_parcels = sum(classification_counts.values())
        
        row = {
            'contrast': contrast_name,
            'total_parcels': total_parcels,
            'canonical_count': classification_counts.get('canonical', 0),
            'canonical_percentage': (classification_counts.get('canonical', 0) / total_parcels) * 100,
            'indiv_fingerprint_count': classification_counts.get('indiv_fingerprint', 0),
            'indiv_fingerprint_percentage': (classification_counts.get('indiv_fingerprint', 0) / total_parcels) * 100,
            'variable_count': classification_counts.get('variable', 0),
            'variable_percentage': (classification_counts.get('variable', 0) / total_parcels) * 100,
        }
        rows.append(row)
    
    # Add overall summary row
    total_classifications = {}
    total_count = 0
    
    for classification_counts in summary.values():
        for classification, count in classification_counts.items():
            total_classifications[classification] = total_classifications.get(classification, 0) + count
            total_count += count
    
    overall_row = {
        'contrast': 'OVERALL',
        'total_parcels': total_count,
        'canonical_count': total_classifications.get('canonical', 0),
        'canonical_percentage': (total_classifications.get('canonical', 0) / total_count) * 100,
        'indiv_fingerprint_count': total_classifications.get('indiv_fingerprint', 0),
        'indiv_fingerprint_percentage': (total_classifications.get('indiv_fingerprint', 0) / total_count) * 100,
        'variable_count': total_classifications.get('variable', 0),
        'variable_percentage': (total_classifications.get('variable', 0) / total_count) * 100,
    }
    rows.append(overall_row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    df.to_csv(output_path, index=False, float_format='%.2f')
    
    print(f"Exported classification summary to: {output_path}")
    return output_path


def export_ranked_parcels_csv(
    within_similarities: Dict[str, Dict[str, float]],
    between_similarities: Dict[str, Dict[str, float]],
    classifications: Dict[str, Dict[str, str]],
    output_dir: Path,
    top_n: int = 50
) -> Dict[str, Path]:
    """
    Export ranked parcel lists to separate CSV files.
    
    Parameters
    ----------
    within_similarities : Dict[str, Dict[str, float]]
        Within-subject similarities by contrast and parcel
    between_similarities : Dict[str, Dict[str, float]]
        Between-subject similarities by contrast and parcel
    classifications : Dict[str, Dict[str, str]]
        Parcel classifications by contrast and parcel
    output_dir : Path
        Output directory for CSV files
    top_n : int
        Number of top parcels to include in each ranking (default: 50)
        
    Returns
    -------
    Dict[str, Path]
        Paths to created CSV files by ranking type
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    
    # Rank by fingerprint strength
    fingerprint_ranking = rank_parcels_by_fingerprint_strength(
        within_similarities, between_similarities, classifications
    )
    
    df_fingerprint = pd.DataFrame(
        fingerprint_ranking[:top_n],
        columns=['contrast', 'parcel', 'fingerprint_strength', 'classification']
    )
    df_fingerprint['rank'] = range(1, len(df_fingerprint) + 1)
    df_fingerprint = df_fingerprint[['rank', 'contrast', 'parcel', 'fingerprint_strength', 'classification']]
    
    fingerprint_path = output_dir / 'most_fingerprint_parcels.csv'
    df_fingerprint.to_csv(fingerprint_path, index=False, float_format='%.6f')
    output_paths['fingerprint'] = fingerprint_path
    
    # Rank by variability
    variability_ranking = rank_parcels_by_variability(
        within_similarities, between_similarities, classifications
    )
    
    df_variability = pd.DataFrame(
        [(contrast, parcel, score, classification) for contrast, parcel, score, classification in variability_ranking[:top_n]],
        columns=['contrast', 'parcel', 'variability_score', 'classification']
    )
    df_variability['rank'] = range(1, len(df_variability) + 1)
    df_variability = df_variability[['rank', 'contrast', 'parcel', 'variability_score', 'classification']]
    
    variability_path = output_dir / 'most_variable_parcels.csv'
    df_variability.to_csv(variability_path, index=False, float_format='%.6f')
    output_paths['variability'] = variability_path
    
    # Rank by canonicality
    canonicality_ranking = rank_parcels_by_canonicality(
        within_similarities, between_similarities, classifications
    )
    
    df_canonicality = pd.DataFrame(
        canonicality_ranking[:top_n],
        columns=['contrast', 'parcel', 'canonicality_score', 'classification']
    )
    df_canonicality['rank'] = range(1, len(df_canonicality) + 1)
    df_canonicality = df_canonicality[['rank', 'contrast', 'parcel', 'canonicality_score', 'classification']]
    
    canonicality_path = output_dir / 'most_canonical_parcels.csv'
    df_canonicality.to_csv(canonicality_path, index=False, float_format='%.6f')
    output_paths['canonicality'] = canonicality_path
    
    print(f"Exported ranked parcel lists to:")
    for ranking_type, path in output_paths.items():
        print(f"  {ranking_type}: {path}")
    
    return output_paths


def export_cross_contrast_consistency_csv(
    classifications: Dict[str, Dict[str, str]],
    output_dir: Path,
    filename: str = "cross_contrast_consistency.csv"
) -> Path:
    """
    Export cross-contrast consistency analysis to CSV.
    
    Parameters
    ----------
    classifications : Dict[str, Dict[str, str]]
        Parcel classifications by contrast and parcel
    output_dir : Path
        Output directory for CSV file
    filename : str
        Output filename (default: "cross_contrast_consistency.csv")
        
    Returns
    -------
    Path
        Path to created CSV file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    # Compute cross-contrast consistency
    consistency_data = compute_cross_contrast_consistency(classifications)
    
    # Convert to DataFrame format
    rows = []
    for parcel_name, consistency_info in consistency_data.items():
        row = {
            'parcel': parcel_name,
            'most_common_classification': consistency_info.get('most_common_classification'),
            'consistency_score': consistency_info.get('consistency_score'),
            'n_contrasts': consistency_info.get('n_contrasts'),
            'canonical_proportion': consistency_info.get('canonical', 0),
            'indiv_fingerprint_proportion': consistency_info.get('indiv_fingerprint', 0),
            'variable_proportion': consistency_info.get('variable', 0),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by consistency score (descending) then by parcel name
    df = df.sort_values(['consistency_score', 'parcel'], ascending=[False, True]).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False, float_format='%.4f')
    
    print(f"Exported cross-contrast consistency to: {output_path}")
    return output_path


def export_all_postprocessing_results(
    within_similarities: Dict[str, Dict[str, float]],
    between_similarities: Dict[str, Dict[str, float]],
    classifications: Dict[str, Dict[str, str]],
    output_dir: Path,
    top_n: int = 50
) -> Dict[str, Path]:
    """
    Export all postprocessing results to CSV files.
    
    Parameters
    ----------
    within_similarities : Dict[str, Dict[str, float]]
        Within-subject similarities by contrast and parcel
    between_similarities : Dict[str, Dict[str, float]]
        Between-subject similarities by contrast and parcel
    classifications : Dict[str, Dict[str, str]]
        Parcel classifications by contrast and parcel
    output_dir : Path
        Output directory for CSV files
    top_n : int
        Number of top parcels to include in rankings (default: 50)
        
    Returns
    -------
    Dict[str, Path]
        Paths to all created CSV files
    """
    output_paths = {}
    
    print("Exporting postprocessing results...")
    
    # Export detailed classifications
    output_paths['classifications'] = export_parcel_classifications_csv(
        within_similarities, between_similarities, classifications, output_dir
    )
    
    # Export summary statistics
    output_paths['summary'] = export_summary_statistics_csv(
        within_similarities, between_similarities, classifications, output_dir
    )
    
    # Export ranked parcels
    ranked_paths = export_ranked_parcels_csv(
        within_similarities, between_similarities, classifications, output_dir, top_n
    )
    output_paths.update(ranked_paths)
    
    # Export cross-contrast consistency
    output_paths['consistency'] = export_cross_contrast_consistency_csv(
        classifications, output_dir
    )
    
    print(f"\nAll postprocessing results exported to: {output_dir}")
    return output_paths