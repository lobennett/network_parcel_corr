"""Analysis functions for parcel classification results."""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np
import pandas as pd


def compute_classification_summary(
    classifications: Dict[str, Dict[str, str]]
) -> Dict[str, Dict[str, int]]:
    """
    Compute summary statistics for parcel classifications.
    
    Parameters
    ----------
    classifications : Dict[str, Dict[str, str]]
        Nested dict: {contrast_name: {parcel_name: classification}}
        
    Returns
    -------
    Dict[str, Dict[str, int]]
        Summary statistics: {contrast_name: {classification: count}}
    """
    summary = {}
    
    for contrast_name, parcel_classifications in classifications.items():
        classification_counts = Counter(parcel_classifications.values())
        summary[contrast_name] = dict(classification_counts)
    
    return summary


def compute_parcel_statistics(
    within_similarities: Dict[str, Dict[str, float]],
    between_similarities: Dict[str, Dict[str, float]],
    classifications: Dict[str, Dict[str, str]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute detailed statistics for each parcel across contrasts.
    
    Parameters
    ----------
    within_similarities : Dict[str, Dict[str, float]]
        Within-subject similarities by contrast and parcel
    between_similarities : Dict[str, Dict[str, float]]
        Between-subject similarities by contrast and parcel
    classifications : Dict[str, Dict[str, str]]
        Parcel classifications by contrast and parcel
        
    Returns
    -------
    Dict[str, Dict[str, Dict[str, float]]]
        Statistics: {contrast_name: {parcel_name: {stat_name: value}}}
    """
    statistics = {}
    
    for contrast_name in classifications.keys():
        if contrast_name not in within_similarities or contrast_name not in between_similarities:
            continue
            
        statistics[contrast_name] = {}
        
        for parcel_name in classifications[contrast_name].keys():
            if (parcel_name not in within_similarities[contrast_name] or 
                parcel_name not in between_similarities[contrast_name]):
                continue
                
            within_val = within_similarities[contrast_name][parcel_name]
            between_val = between_similarities[contrast_name][parcel_name]
            classification = classifications[contrast_name][parcel_name]
            
            statistics[contrast_name][parcel_name] = {
                'within_subject_similarity': within_val,
                'between_subject_similarity': between_val,
                'similarity_difference': within_val - between_val,
                'similarity_sum': within_val + between_val,
                'similarity_ratio': within_val / between_val if between_val != 0 else np.inf,
                'classification': classification
            }
    
    return statistics


def rank_parcels_by_fingerprint_strength(
    within_similarities: Dict[str, Dict[str, float]],
    between_similarities: Dict[str, Dict[str, float]],
    classifications: Dict[str, Dict[str, str]]
) -> List[Tuple[str, str, float, str]]:
    """
    Rank parcels by individual fingerprint strength (within - between).
    
    Parameters
    ----------
    within_similarities : Dict[str, Dict[str, float]]
        Within-subject similarities by contrast and parcel
    between_similarities : Dict[str, Dict[str, float]]
        Between-subject similarities by contrast and parcel
    classifications : Dict[str, Dict[str, str]]
        Parcel classifications by contrast and parcel
        
    Returns
    -------
    List[Tuple[str, str, float, str]]
        Ranked list of (contrast, parcel, fingerprint_strength, classification)
    """
    fingerprint_scores = []
    
    for contrast_name in classifications.keys():
        if contrast_name not in within_similarities or contrast_name not in between_similarities:
            continue
            
        for parcel_name in classifications[contrast_name].keys():
            if (parcel_name not in within_similarities[contrast_name] or 
                parcel_name not in between_similarities[contrast_name]):
                continue
                
            within_val = within_similarities[contrast_name][parcel_name]
            between_val = between_similarities[contrast_name][parcel_name]
            fingerprint_strength = within_val - between_val
            classification = classifications[contrast_name][parcel_name]
            
            fingerprint_scores.append((
                contrast_name, parcel_name, fingerprint_strength, classification
            ))
    
    # Sort by fingerprint strength (descending - highest first)
    return sorted(fingerprint_scores, key=lambda x: x[2], reverse=True)


def rank_parcels_by_variability(
    within_similarities: Dict[str, Dict[str, float]], 
    between_similarities: Dict[str, Dict[str, float]],
    classifications: Dict[str, Dict[str, str]]
) -> List[Tuple[str, str, float, str]]:
    """
    Rank parcels by variability (low within + between similarity).
    
    Parameters
    ----------
    within_similarities : Dict[str, Dict[str, float]]
        Within-subject similarities by contrast and parcel
    between_similarities : Dict[str, Dict[str, float]]
        Between-subject similarities by contrast and parcel
    classifications : Dict[str, Dict[str, str]]
        Parcel classifications by contrast and parcel
        
    Returns
    -------
    List[Tuple[str, str, float, str]]
        Ranked list of (contrast, parcel, variability_score, classification)
    """
    variability_scores = []
    
    for contrast_name in classifications.keys():
        if contrast_name not in within_similarities or contrast_name not in between_similarities:
            continue
            
        for parcel_name in classifications[contrast_name].keys():
            if (parcel_name not in within_similarities[contrast_name] or 
                parcel_name not in between_similarities[contrast_name]):
                continue
                
            within_val = within_similarities[contrast_name][parcel_name]
            between_val = between_similarities[contrast_name][parcel_name]
            variability_score = -(within_val + between_val)  # Negative for ascending sort (most variable first)
            classification = classifications[contrast_name][parcel_name]
            
            variability_scores.append((
                contrast_name, parcel_name, variability_score, classification
            ))
    
    # Sort by variability score (ascending - most variable first)
    return sorted(variability_scores, key=lambda x: x[2])


def rank_parcels_by_canonicality(
    within_similarities: Dict[str, Dict[str, float]],
    between_similarities: Dict[str, Dict[str, float]],
    classifications: Dict[str, Dict[str, str]]
) -> List[Tuple[str, str, float, str]]:
    """
    Rank parcels by canonicality (high within, low between similarity).
    
    Parameters
    ----------
    within_similarities : Dict[str, Dict[str, float]]
        Within-subject similarities by contrast and parcel
    between_similarities : Dict[str, Dict[str, float]]
        Between-subject similarities by contrast and parcel
    classifications : Dict[str, Dict[str, str]]
        Parcel classifications by contrast and parcel
        
    Returns
    -------
    List[Tuple[str, str, float, str]]
        Ranked list of (contrast, parcel, canonicality_score, classification)
    """
    canonicality_scores = []
    
    for contrast_name in classifications.keys():
        if contrast_name not in within_similarities or contrast_name not in between_similarities:
            continue
            
        for parcel_name in classifications[contrast_name].keys():
            if (parcel_name not in within_similarities[contrast_name] or 
                parcel_name not in between_similarities[contrast_name]):
                continue
                
            within_val = within_similarities[contrast_name][parcel_name]
            between_val = between_similarities[contrast_name][parcel_name]
            # Canonicality score: high within similarity with large difference from between
            canonicality_score = within_val * (within_val - between_val)
            classification = classifications[contrast_name][parcel_name]
            
            canonicality_scores.append((
                contrast_name, parcel_name, canonicality_score, classification
            ))
    
    # Sort by canonicality score (descending - most canonical first)
    return sorted(canonicality_scores, key=lambda x: x[2], reverse=True)


def compute_cross_contrast_consistency(
    classifications: Dict[str, Dict[str, str]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute how consistently each parcel is classified across contrasts.
    
    Parameters
    ----------
    classifications : Dict[str, Dict[str, str]]
        Parcel classifications by contrast and parcel
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Consistency scores: {parcel_name: {classification: proportion}}
    """
    # Collect all unique parcels
    all_parcels = set()
    for contrast_classifications in classifications.values():
        all_parcels.update(contrast_classifications.keys())
    
    consistency_scores = {}
    
    for parcel_name in all_parcels:
        parcel_classifications = []
        
        # Collect classifications for this parcel across all contrasts
        for contrast_name, contrast_classifications in classifications.items():
            if parcel_name in contrast_classifications:
                parcel_classifications.append(contrast_classifications[parcel_name])
        
        # Compute proportion of each classification
        if parcel_classifications:
            classification_counts = Counter(parcel_classifications)
            total_contrasts = len(parcel_classifications)
            
            consistency_scores[parcel_name] = {
                classification: count / total_contrasts
                for classification, count in classification_counts.items()
            }
            
            # Add consistency metrics
            consistency_scores[parcel_name]['most_common_classification'] = classification_counts.most_common(1)[0][0]
            consistency_scores[parcel_name]['consistency_score'] = classification_counts.most_common(1)[0][1] / total_contrasts
            consistency_scores[parcel_name]['n_contrasts'] = total_contrasts
    
    return consistency_scores