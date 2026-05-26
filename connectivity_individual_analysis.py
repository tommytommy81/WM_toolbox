"""
Connectivity Analysis - Individual Level (Refactored)
=====================================================

This script performs all-to-all source-space connectivity analysis on MEG data
for individual subjects using the DICS beamformer (conpy).

All parameters are loaded from a configuration file (config.yaml) and the
heavy lifting is delegated to the functions in
STWM_functions_for_connectivity.py.

Prerequisites (run once across the group, see example_usage_connectivity.py):
  - Cross-spectral density (CSD) files from the sensor-space analysis
  - fsaverage template source space (src_average)
  - Connectivity pairs and common-vertex forward models (pairs_identification)

Requires: conpy package (pip install conpy)

Usage:
------
python connectivity_individual_analysis.py --config config.yaml --subject S1

@author: Nikita Otstavnov, 2023 (refactored 2026)
"""

import os
import argparse
import yaml

from STWM_functions_for_connectivity import (
    new_morphing,
    new_morphed_forward_model,
    connectivity_estimation,
    connectivity_vizualization
)


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_connectivity_analysis(config):
    """
    Run complete individual-level connectivity analysis.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    """
    # Extract parameters
    folder        = config['paths']['data_folder']
    output_folder = config['paths'].get('output_folder', folder)
    subject_name  = config['subject']['subject_name']

    condition_1   = config['conditions']['condition_1']['name']
    condition_2   = config['conditions']['condition_2']['name']

    spacing       = config['connectivity']['spacing']
    freq_min      = config['connectivity']['freq_min']
    freq_max      = config['connectivity']['freq_max']

    print(f"\n{'='*60}")
    print(f"Starting Connectivity Analysis for Subject: {subject_name}")
    print(f"{'='*60}\n")

    os.chdir(folder)

    # ========================================
    # STEP 1: Morph source space to subject
    # ========================================
    print("Step 1: Morphing fsaverage source space to subject...")
    try:
        subject_src = new_morphing(config)
        print(f"  ✓ Morphed source space created with spacing: {spacing}")
    except Exception as e:
        print(f"  ✗ Error morphing source space: {e}")
        raise

    # ========================================
    # STEP 2: Create morphed forward model
    # ========================================
    print("\nStep 2: Creating morphed forward model...")
    try:
        fwd = new_morphed_forward_model(config)
        print(f"  ✓ Forward model created and restricted to sensor range")
    except Exception as e:
        print(f"  ✗ Error creating forward model: {e}")
        raise

    # ========================================
    # STEP 3: Estimate connectivity
    # ========================================
    print("\nStep 3: Estimating DICS connectivity...")
    print(f"  Frequency band: {freq_min}-{freq_max} Hz")
    try:
        connectivity_1, connectivity_2 = connectivity_estimation(config)
        print(f"  ✓ Connectivity computed for '{condition_1}' and '{condition_2}'")
    except Exception as e:
        print(f"  ✗ Error estimating connectivity: {e}")
        raise

    # ========================================
    # STEP 4: Visualize connectivity contrast
    # ========================================
    print("\nStep 4: Visualizing connectivity contrast...")
    try:
        p, brain = connectivity_vizualization(config)
        print(f"  ✓ Connectivity visualization created")
    except Exception as e:
        print(f"  ⚠ Warning: Visualization failed: {e}")

    # ========================================
    # Analysis Complete
    # ========================================
    print(f"\n{'='*60}")
    print(f"Connectivity Analysis Complete for: {subject_name}")
    print(f"{'='*60}\n")
    print(f"Output files saved to: {output_folder}")
    print(f"\nGenerated files:")
    print(f"  - Morphed source space: {subject_name}_for_con-morph-src.fif")
    print(f"  - Forward model: {subject_name}-for_con-morphed-fwd.fif")
    print(f"  - Connectivity: {subject_name}-connectivity for band from {freq_min} to {freq_max}_*")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run connectivity analysis for individual MEG subjects')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--subject', type=str, default=None,
                       help='Subject name (overrides config file)')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override subject name if provided
    if args.subject is not None:
        config['subject']['subject_name'] = args.subject

    # Run analysis
    try:
        run_connectivity_analysis(config)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
