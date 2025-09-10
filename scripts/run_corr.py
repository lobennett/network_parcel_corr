#!/usr/bin/env python3
"""
Simple Parcel-based Correlation Analysis for fMRI Data

Computes simple upper triangle correlations with averaging:
1. Within-subject: Mean correlations across sessions within each subject
2. Between-subjects: Mean correlations across subjects
3. Across-subjects: Mean correlations across contrast in each construct

"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import h5py

from network_parcel_corr.main import run_analysis


# Subconstruct mapping for cognitive constructs
CONTRAST_TO_SUBCONSTRUCT_MAP = {
    'Active Maintenance': [
        'task-nBack_contrast-match-mismatch',
        'task-nBack_contrast-twoBack-oneBack',
    ],
    'Flexible Updating': [
        'task-cuedTS_contrast-cue_switch_cost',
        'task-cuedTS_contrast-task_switch_cost',
        'task-cuedTS_contrast-task_switch_cue_switch-task_stay_cue_stay',
        'task-spatialTS_contrast-cue_switch_cost',
        'task-spatialTS_contrast-task_switch_cost',
        'task-spatialTS_contrast-task_switch_cue_switch-task_stay_cue_stay',
    ],
    'Limited Capacity': [
        'task-nBack_contrast-twoBack-oneBack',
    ],
    'Interference Control': [
        'task-flanker_contrast-incongruent-congruent',
        'task-directedForgetting_contrast-neg-con',
    ],
    'Goal Selection': [
        'task-cuedTS_contrast-cue_switch_cost',
        'task-spatialTS_contrast-cue_switch_cost',
        'task-stopSignal_contrast-go',
        'task-goNogo_contrast-go',
    ],
    'Updating Representation and Maintenance': ['task-nBack_contrast-match-mismatch'],
    'Response Selection': [
        'task-flanker_contrast-incongruent-congruent',
        'task-stopSignal_contrast-go',
        'task-goNogo_contrast-go',
    ],
    'Inhibition Suppression': [
        'task-stopSignal_contrast-stop_success',
        'task-stopSignal_contrast-stop_success-go',
        'task-stopSignal_contrast-stop_success-stop_failure',
        'task-goNogo_contrast-nogo_success',
        'task-goNogo_contrast-nogo_success-go',
    ],
    'Performance Monitoring': [
        'task-stopSignal_contrast-stop_failure',
        'task-stopSignal_contrast-stop_failure-go',
        'task-stopSignal_contrast-stop_failure-stop_success',
    ],
    'Attention': [
        'task-flanker_contrast-incongruent-congruent',
        'task-cuedTS_contrast-task_switch_cue_switch-task_stay_cue_stay',
        'task-spatialTS_contrast-task_switch_cue_switch-task_stay_cue_stay',
    ],
    'Task Baseline': [
        'task-cuedTS_contrast-task-baseline',
        'task-directedForgetting_contrast-task-baseline',
        'task-flanker_contrast-task-baseline',
        'task-goNogo_contrast-task-baseline',
        'task-nBack_contrast-task-baseline',
        'task-shapeMatching_contrast-task-baseline',
        'task-spatialTS_contrast-task-baseline',
        'task-stopSignal_contrast-task-baseline',
    ],
    'Response Time': [
        'task-cuedTS_contrast-response_time',
        'task-directedForgetting_contrast-response_time',
        'task-flanker_contrast-response_time',
        'task-goNogo_contrast-response_time',
        'task-nBack_contrast-response_time',
        'task-shapeMatching_contrast-response_time',
        'task-spatialTS_contrast-response_time',
        'task-stopSignal_contrast-response_time',
    ],
}


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
    return parser


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

    try:
        # Run the analysis using our modular package
        results = run_analysis(
            subjects=args.subjects,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            exclusions_file=args.exclusions_file,
            atlas_parcels=args.atlas_parcels,
        )

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
