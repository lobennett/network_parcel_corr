#!/usr/bin/env python3
"""
Simple Parcel-based Correlation Analysis for fMRI Data

Computes simple upper triangle correlations with averaging:
1. Within-subject: Mean correlations across sessions within each subject
2. Between-subjects: Mean correlations across subjects
3. Across-subjects: Mean correlations across contrast in each construct

"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import h5py

from network_parcel_corr.main import run_analysis
from network_parcel_corr.core.similarity import compute_across_construct_similarity
from network_parcel_corr.data.construct_mappings import CONSTRUCT_TO_CONTRAST_MAP


def get_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Simple parcel-based correlation analysis for fMRI contrast maps'
    )
    parser.add_argument(
        '--subjects',
        nargs='+',
        default=['sub-s03', 'sub-s10', 'sub-s19', 'sub-s29', 'sub-s43'],
        help='Subject IDs to analyze',
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('/scratch/users/logben/poldrack_glm/level1/output'),
        help='Input directory containing subject data',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('/scratch/users/logben/poldrack_glm/correlations/output'),
        help='Output directory for results',
    )
    parser.add_argument(
        '--atlas-parcels',
        type=int,
        default=400,
        help='Number of Schaefer atlas parcels (default: 400)',
    )
    parser.add_argument(
        '--exclusions-file',
        type=str,
        required=True,
        help='Path to JSON file containing exclusions.',
    )
    parser.add_argument(
        '--construct-contrast-map',
        type=str,
        help='Path to JSON file containing construct-to-contrast mapping. If not provided, uses default mapping.',
    )
    return parser


def load_construct_contrast_map(map_file: str = None) -> dict:
    """Load construct-to-contrast mapping from JSON file or use default."""
    if map_file:
        with open(map_file, 'r') as f:
            return json.load(f)
    else:
        return CONSTRUCT_TO_CONTRAST_MAP


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'correlation_analysis.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return logging.getLogger(__name__)


def save_results_to_hdf5(results: dict, hdf5_path: Path) -> None:
    """Save analysis results as HDF5 attributes."""
    with h5py.File(hdf5_path, 'a') as f:
        all_within_values = []
        all_between_values = []

        # Add similarity results as parcel-level attributes
        for contrast_name in f.keys():
            contrast_group = f[contrast_name]

            for parcel_name in contrast_group.keys():
                parcel_group = contrast_group[parcel_name]

                # Add within and between subject similarity as parcel attributes
                if (
                    contrast_name in results['within_similarities']
                    and parcel_name in results['within_similarities'][contrast_name]
                ):
                    within_val = results['within_similarities'][contrast_name][
                        parcel_name
                    ]
                    parcel_group.attrs['within_subject_similarity'] = within_val
                    all_within_values.append(within_val)

                if (
                    contrast_name in results['between_similarities']
                    and parcel_name in results['between_similarities'][contrast_name]
                ):
                    between_val = results['between_similarities'][contrast_name][
                        parcel_name
                    ]
                    parcel_group.attrs['between_subject_similarity'] = between_val
                    all_between_values.append(between_val)

                # Add parcel classification
                if (
                    contrast_name in results['classifications']
                    and parcel_name in results['classifications'][contrast_name]
                ):
                    classification = results['classifications'][contrast_name][
                        parcel_name
                    ]
                    parcel_group.attrs['parcel_classification'] = classification

                # Add across-construct similarity results
                if (
                    'across_construct_similarities' in results
                    and contrast_name in results['across_construct_similarities']
                    and parcel_name
                    in results['across_construct_similarities'][contrast_name]
                ):
                    construct_similarities = results['across_construct_similarities'][
                        contrast_name
                    ][parcel_name]
                    for construct, similarity in construct_similarities.items():
                        attr_name = (
                            f'across_construct_similarity_{construct.replace(" ", "_")}'
                        )
                        parcel_group.attrs[attr_name] = similarity

        # Add global means as top-level attributes
        if all_within_values:
            f.attrs['mean_within_subject_similarity'] = np.mean(all_within_values)
        if all_between_values:
            f.attrs['mean_between_subject_similarity'] = np.mean(all_between_values)


def main():
    """Main analysis pipeline."""
    args = get_parser().parse_args()

    # Setup logging
    logger = setup_logging(args.output_dir)

    logger.info('Simple Parcel Correlation Analysis')
    logger.info(f'Subjects: {args.subjects}')
    logger.info(f'Atlas: Schaefer {args.atlas_parcels} parcels')
    logger.info(f'Exclusions file: {args.exclusions_file}')

    # Load construct-to-contrast mapping
    construct_map = load_construct_contrast_map(args.construct_contrast_map)
    if args.construct_contrast_map:
        logger.info(f'Using construct-contrast map from: {args.construct_contrast_map}')
    else:
        logger.info('Using default construct-contrast mapping')
    logger.info(f'Number of constructs: {len(construct_map)}')

    try:
        # Run the analysis using our modular package
        results = run_analysis(
            subjects=args.subjects,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            exclusions_file=args.exclusions_file,
            atlas_parcels=args.atlas_parcels,
        )

        # Count variable parcels for logging
        variable_parcel_count = 0
        total_parcel_count = 0
        for contrast, parcels in results['classifications'].items():
            for parcel, classification in parcels.items():
                total_parcel_count += 1
                if classification == 'variable':
                    variable_parcel_count += 1

        logger.info(
            f'Excluding {variable_parcel_count} variable parcels from across-construct similarity (out of {total_parcel_count} total)'
        )

        # Compute across-construct similarity (excluding variable parcels)
        logger.info(
            'Computing across-construct similarity (excluding variable parcels)...'
        )
        across_construct_similarities = compute_across_construct_similarity(
            results['hdf5_path'], construct_map, results['classifications']
        )
        results['across_construct_similarities'] = across_construct_similarities

        # Log results summary
        logger.info('\n--- Analysis Summary ---')
        logger.info(
            f'Processed {results["n_contrasts"]} contrasts for {results["n_subjects"]} subjects'
        )

        # Log within-subject similarity summary
        within_means = []
        for contrast, parcels in results['within_similarities'].items():
            contrast_mean = np.mean(list(parcels.values()))
            within_means.append(contrast_mean)
            logger.info(
                f'Within-subject similarity for {contrast}: {contrast_mean:.3f}'
            )

        if within_means:
            logger.info(
                f'Overall mean within-subject similarity: {np.mean(within_means):.3f}'
            )

        # Log between-subject similarity summary
        between_means = []
        for contrast, parcels in results['between_similarities'].items():
            contrast_mean = np.mean(list(parcels.values()))
            between_means.append(contrast_mean)
            logger.info(
                f'Between-subject similarity for {contrast}: {contrast_mean:.3f}'
            )

        if between_means:
            logger.info(
                f'Overall mean between-subject similarity: {np.mean(between_means):.3f}'
            )

        # Log across-construct similarity summary
        if 'across_construct_similarities' in results:
            logger.info('\n--- Across-Construct Similarity Summary ---')
            for contrast, parcels in results['across_construct_similarities'].items():
                for parcel, constructs in parcels.items():
                    for construct, similarity in constructs.items():
                        logger.info(
                            f'{contrast} - {parcel} - {construct}: {similarity:.3f}'
                        )

        # Log parcel classifications summary
        all_classifications = {}
        for contrast, parcels in results['classifications'].items():
            for parcel, classification in parcels.items():
                all_classifications[classification] = (
                    all_classifications.get(classification, 0) + 1
                )

        logger.info('\n--- Parcel Classifications ---')
        for classification, count in all_classifications.items():
            logger.info(f'{classification}: {count} parcels')

        # Save results to HDF5 file
        save_results_to_hdf5(results, results['hdf5_path'])

        logger.info(f'\nResults saved to: {results["hdf5_path"]}')
        logger.info('Analysis completed successfully!')

    except Exception as e:
        logger.error(f'Error in correlation analysis pipeline: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
