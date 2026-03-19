"""
Source Space Analysis - Individual Level (Refactored)
=====================================================

This script performs source-level analysis on MEG data for individual subjects.
Includes source space creation, forward modeling, and source estimation.

All parameters are loaded from a configuration file (config.yaml).

Usage:
------
python source_space_individual_analysis.py --config config.yaml --subject S1

@author: Nikita Otstavnov, 2023 (refactored 2026)
"""

import os
import argparse
import yaml
import numpy as np
import mne
from STWM_functions_core import (
    creating_source_space_object,
    creating_forward_model,
    creating_source_estimate_object,
    source_estimate_morphing_to_average,
    source_estimate_visualization,
    source_estimate_visualization_morph
)


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_source_analysis(config):
    """
    Run complete source-level analysis for individual subject.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    """
    # Extract parameters
    folder = config['paths']['data_folder']
    output_folder = config['paths'].get('output_folder', folder)
    subjects_dir = config['paths']['subjects_dir']
    subject_name = config['subject']['subject_name']
    file_name    = config['subject']['file_name']
    
    # Conditions
    condition_1 = config['conditions']['condition_1']['name']
    condition_2 = config['conditions']['condition_2']['name']
    
    # Source space parameters
    spacing = config['source_space']['spacing']
    brain_surfaces = config['source_space']['surfaces']
    orientation = config['source_space']['orientation']
    
    # Forward model parameters
    conductivity = tuple(config['forward_model']['conductivity'])
    ico = config['forward_model']['ico']
    mindist = config['forward_model']['mindist']
    surfaces = config['forward_model']['surfaces']
    coord_frame = config['forward_model']['coord_frame']
    
    # Source estimate parameters
    freq_min = config['source_estimate']['freq_min']
    freq_max = config['source_estimate']['freq_max']
    reg = config['source_estimate']['reg']
    depth = config['source_estimate']['depth']
    inversion = config['source_estimate']['method']
    orient_fix = config['source_estimate']['orientation']
    surf_ori = config['source_estimate']['surf_ori']
    force_fixed = config['source_estimate']['force_fixed']
    
    # Processing
    n_jobs = config['processing']['n_jobs']
    
    # Visualization
    viz_config = config.get('visualization', {})
    hemi = viz_config.get('hemi', 'both')
    surface = viz_config.get('surface', 'inflated')
    views = viz_config.get('views', ['dorsal', 'lateral', 'medial', 'ventral'])
    
    print(f"\n{'='*60}")
    print(f"Starting Source Space Analysis for Subject: {subject_name}")
    print(f"{'='*60}\n")
    
    os.chdir(folder)
    
    # ========================================
    # STEP 1: Create Source Space
    # ========================================
    print("Step 1: Creating source space...")
    try:
        source_space = creating_source_space_object(file_name,
            subject_name, subjects_dir, spacing, brain_surfaces,
            folder, n_jobs, orientation)
        print(f"  ✓ Source space created with spacing: {spacing}")
    except Exception as e:
        print(f"  ✗ Error creating source space: {e}")
        raise
    
    # ========================================
    # STEP 2: Create Forward Model
    # ========================================
    print("\nStep 2: Creating forward model...")
    try:
        forward_model = creating_forward_model(file_name,
            folder, subject_name, spacing, conductivity,
            subjects_dir, mindist, n_jobs, ico, surfaces, coord_frame)
        print(f"  ✓ Forward model created")
        print(f"     Conductivity: {conductivity}")
        print(f"     Minimum distance: {mindist} mm")
    except Exception as e:
        print(f"  ✗ Error creating forward model: {e}")
        raise
    
    # ========================================
    # STEP 3: Compute Source Estimates
    # ========================================
    print("\nStep 3: Computing source estimates...")
    try:
        # Note: condition_3 might be baseline - check config
        condition_3 = config['conditions'].get('baseline', {}).get('name', 'Baseline')
        
        stc_1, stc_2, stc_baseline, src = creating_source_estimate_object(
            folder, subject_name, spacing,
            condition_1, condition_2, condition_3,
            freq_min, freq_max, orient_fix,
            surf_ori, force_fixed, reg, depth, inversion)
        
        print(f"  ✓ Source estimates computed")
        print(f"     Condition 1 ({condition_1}): {stc_1.data.shape}")
        print(f"     Condition 2 ({condition_2}): {stc_2.data.shape}")
        print(f"     Frequency range: {freq_min}-{freq_max} Hz")
    except Exception as e:
        print(f"  ✗ Error computing source estimates: {e}")
        raise
    
    # ========================================
    # STEP 4: Visualize Source Estimate
    # ========================================
    print("\nStep 4: Visualizing source estimates...")
    try:
        # Visualize condition 1
        brain = source_estimate_visualization(
            stc_1, subject_name, freq_min, freq_max, spacing,
            hemi, 'auto', 'auto', 0.5, 'auto',
            views, 1.0, 'vertical', surface,
            'aparc.a2009s', 'rgb', subjects_dir, 'auto', condition_1)
        
        # Save image
        output_img = os.path.join(output_folder,
                                 f'{subject_name}_from_{freq_min}_to_{freq_max}_{condition_1}.png')
        brain.save_image(output_img)
        print(f"  ✓ Visualization saved: {os.path.basename(output_img)}")
    except Exception as e:
        print(f"  ⚠ Warning: Visualization failed: {e}")
    
    # ========================================
    # STEP 5: Morph to Average Brain
    # ========================================
    print("\nStep 5: Morphing to fsaverage...")
    try:
        stc_1_morph, stc_2_morph, stc_baseline_morph, src_subj, src_avg = \
            source_estimate_morphing_to_average(
                folder, subject_name, spacing, freq_min, freq_max,
                orient_fix, condition_1, condition_2, condition_3, subjects_dir)
        
        print(f"  ✓ Morphed to fsaverage brain")
        
        # Visualize contrast on average brain
        stc_contrast = (stc_1_morph - stc_2_morph) / stc_baseline_morph
        
        brain_avg = source_estimate_visualization_morph(
            stc_contrast, subject_name, freq_min, freq_max, spacing,
            hemi, 'auto', 'auto', 0.5, 'auto',
            views, 1.0, 'vertical', surface,
            'aparc.a2009s', 'rgb', subjects_dir, 'auto', 'Contrast')
        
        output_contrast = os.path.join(output_folder,
                                      f'{subject_name}_contrast_{freq_min}_to_{freq_max}.png')
        brain_avg.save_image(output_contrast)
        print(f"  ✓ Contrast visualization saved")
    except Exception as e:
        print(f"  ⚠ Warning: Morphing/contrast visualization failed: {e}")
    
    # ========================================
    # Analysis Complete
    # ========================================
    print(f"\n{'='*60}")
    print(f"Source Space Analysis Complete for: {subject_name}")
    print(f"{'='*60}\n")
    print(f"Output files saved to: {output_folder}")
    print(f"\nGenerated files:")
    print(f"  - Source space: *-src.fif")
    print(f"  - Forward model: *-fwd.fif")
    print(f"  - Source estimates: *-stc files")
    print(f"  - Visualizations: *.png")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run source space analysis for individual MEG subjects')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--subject', type=str, default=None,
                       help='Subject name (overrides config file)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override subject name if provided
    if args.subject is not None:
        config['subject']['name'] = args.subject
    
    # Run analysis
    try:
        run_source_analysis(config)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
