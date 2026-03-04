"""
Connectivity Analysis - Individual Level (Refactored)
====================================================

This script performs connectivity analysis on MEG data for individual subjects
using DICS beamformer approach with source space connectivity estimation.

Requires: conpy package for connectivity analysis

All parameters are loaded from a configuration file (config.yaml).

Usage:
------
python connectivity_individual_analysis.py --config config.yaml --subject S1

@author: Nikita Otstavnov, 2023 (refactored 2026)
"""

import os
import argparse
import yaml
import numpy as np
import mne

try:
    import conpy
    CONPY_AVAILABLE = True
except ImportError:
    CONPY_AVAILABLE = False
    print("Warning: conpy package not available. Install with: pip install conpy")


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_connectivity_analysis(config):
    """
    Run complete connectivity analysis for individual subject.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    """
    if not CONPY_AVAILABLE:
        raise ImportError("conpy package is required for connectivity analysis")
    
    # Extract parameters
    folder = config['paths']['data_folder']
    output_folder = config['paths'].get('output_folder', folder)
    file_name = config['subject']['file_name']
    subject_name = config['subject']['name']
    subjects_dir = config['paths']['subjects_dir']
    
    # Conditions
    condition_1 = config['conditions']['condition_1']['name']
    condition_2 = config['conditions']['condition_2']['name']
    
    # Connectivity parameters
    spacing = config['connectivity']['spacing']
    max_sensor_dist = config['connectivity']['max_sensor_dist']
    min_dist = config['connectivity']['min_dist']
    reg = config['connectivity']['regularization']
    
    # Frequency parameters
    freq_min = config['connectivity']['freq_min']
    freq_max = config['connectivity']['freq_max']
    
    # Forward model parameters
    conductivity = tuple(config['forward_model']['conductivity'])
    ico = config['forward_model']['ico']
    mindist = config['forward_model']['mindist']
    
    # Processing
    n_jobs = config['processing']['n_jobs']
    
    # Visualization
    atlas = config['connectivity'].get('atlas', 'aparc')
    n_lines = config['connectivity'].get('n_lines', 1000)
    
    print(f"\n{'='*60}")
    print(f"Starting Connectivity Analysis for Subject: {subject_name}")
    print(f"{'='*60}\n")
    
    os.chdir(folder)
    
    # ========================================
    # STEP 1: Setup Average Source Space (fsaverage)
    # ========================================
    print("Step 1: Setting up fsaverage source space...")
    try:
        fsaverage_src_file = f'Sub_for_con_Avg-{spacing}-src.fif'
        
        if not os.path.exists(fsaverage_src_file):
            print("  Creating fsaverage source space...")
            fsaverage = mne.setup_source_space(
                'fsaverage', spacing=spacing,
                subjects_dir=subjects_dir,
                n_jobs=n_jobs, add_dist=False)
            mne.write_source_spaces(fsaverage_src_file, fsaverage, overwrite=True)
            print(f"  ✓ Created fsaverage source space: {spacing}")
        else:
            fsaverage = mne.read_source_spaces(fsaverage_src_file)
            print(f"  ✓ Loaded existing fsaverage source space")
        
    except Exception as e:
        print(f"  ✗ Error setting up fsaverage: {e}")
        raise
    
    # ========================================
    # STEP 2: Morph Source Space to Subject
    # ========================================
    print(f"\nStep 2: Morphing source space to {subject_name}...")
    try:
        subject_src_file = f'{subject_name}_for_con-morph-src.fif'
        
        if not os.path.exists(subject_src_file):
            print(f"  Morphing fsaverage to {subject_name}...")
            subject_src = mne.morph_source_spaces(
                fsaverage, subject_name,
                subjects_dir=subjects_dir)
            mne.write_source_spaces(subject_src_file, subject_src, overwrite=True)
            print(f"  ✓ Morphed source space created")
        else:
            subject_src = mne.read_source_spaces(subject_src_file)
            print(f"  ✓ Loaded existing morphed source space")
        
    except Exception as e:
        print(f"  ✗ Error morphing source space: {e}")
        raise
    
    # ========================================
    # STEP 3: Create Morphed Forward Model
    # ========================================
    print(f"\nStep 3: Creating morphed forward model...")
    try:
        fwd_file = f'{subject_name}-for_con-morphed-fwd.fif'
        
        if not os.path.exists(fwd_file):
            print("  Loading MEG data for info...")
            raw_data = mne.io.read_raw_fif(
                os.path.join(folder, file_name),
                preload=False, verbose=False)
            info = raw_data.info
            
            # Load transformation
            trans = os.path.join(folder, f'{subject_name}-trans.fif')
            if not os.path.exists(trans):
                print(f"  ⚠ Warning: Transformation file not found: {trans}")
                print("     You may need to run coregistration first")
                raise FileNotFoundError(f"Transformation file not found: {trans}")
            
            # Select vertices in sensor range
            print("  Selecting vertices in sensor range...")
            verts = conpy.select_vertices_in_sensor_range(
                subject_src, dist=max_sensor_dist,
                info=info, trans=trans)
            src_sub = conpy.restrict_src_to_vertices(subject_src, verts)
            
            # Create BEM model
            print("  Creating BEM model...")
            bem_model = mne.make_bem_model(
                subject_name, ico=ico,
                subjects_dir=subjects_dir,
                conductivity=conductivity)
            bem = mne.make_bem_solution(bem_model)
            
            # Create forward solution
            print("  Computing forward solution...")
            fwd = mne.make_forward_solution(
                info, trans=trans, src=src_sub, bem=bem,
                meg=True, eeg=False,
                mindist=mindist, n_jobs=n_jobs)
            
            mne.write_forward_solution(fwd_file, fwd, overwrite=True)
            print(f"  ✓ Forward model created and saved")
        else:
            fwd = mne.read_forward_solution(fwd_file)
            print(f"  ✓ Loaded existing forward model")
        
    except Exception as e:
        print(f"  ✗ Error creating forward model: {e}")
        raise
    
    # ========================================
    # STEP 4: Identify Connectivity Pairs
    # ========================================
    print(f"\nStep 4: Identifying connectivity pairs...")
    try:
        pairs_file = 'Average-pairs.npy'
        
        # Note: This step typically needs to be done once across all subjects
        # For individual analysis, pairs should already exist
        if os.path.exists(pairs_file):
            pairs = np.load(pairs_file)
            print(f"  ✓ Loaded existing connectivity pairs: {len(pairs[0])} pairs")
        else:
            print("  ⚠ Connectivity pairs file not found")
            print("     Run group-level pair identification first")
            print("     Or this is the first subject - pairs will be computed")
            # Would need to compute pairs here if this is first subject
            raise FileNotFoundError("Connectivity pairs not found. Run pair identification first.")
        
    except Exception as e:
        print(f"  ⚠ Warning: {e}")
        pairs = None
    
    # ========================================
    # STEP 5: Estimate Connectivity
    # ========================================
    print(f"\nStep 5: Estimating connectivity...")
    try:
        # Load CSD matrices
        csd_1_file = f'{subject_name}_{condition_1}_csd.h5'
        csd_2_file = f'{subject_name}_{condition_2}_csd.h5'
        
        if not os.path.exists(csd_1_file) or not os.path.exists(csd_2_file):
            print("  ✗ CSD files not found. Run sensor space analysis first.")
            raise FileNotFoundError("CSD files not found")
        
        print(f"  Loading CSD for {condition_1}...")
        csd_1 = mne.time_frequency.read_csd(csd_1_file)
        print(f"  Loading CSD for {condition_2}...")
        csd_2 = mne.time_frequency.read_csd(csd_2_file)
        
        # Average over frequency band
        print(f"  Averaging over frequency band: {freq_min}-{freq_max} Hz")
        csd_1_avg = csd_1.mean(fmin=freq_min, fmax=freq_max)
        csd_2_avg = csd_2.mean(fmin=freq_min, fmax=freq_max)
        
        if pairs is not None:
            # Convert forward to tangential
            fwd_tan = conpy.forward_to_tangential(fwd)
            
            # Map pairs from fsaverage to subject
            fsaverage_to_subj = conpy.utils.get_morph_src_mapping(
                fsaverage, fwd['src'],
                indices=True,
                subjects_dir=subjects_dir)[0]
            
            pairs_subj = [[fsaverage_to_subj[v] for v in pairs[0]],
                         [fsaverage_to_subj[v] for v in pairs[1]]]
            
            # Compute connectivity
            print(f"  Computing DICS connectivity for {condition_1}...")
            con_1 = conpy.dics_connectivity(
                vertex_pairs=pairs_subj,
                fwd=fwd_tan,
                data_csd=csd_1_avg,
                reg=reg,
                n_jobs=n_jobs)
            
            print(f"  Computing DICS connectivity for {condition_2}...")
            con_2 = conpy.dics_connectivity(
                vertex_pairs=pairs_subj,
                fwd=fwd_tan,
                data_csd=csd_2_avg,
                reg=reg,
                n_jobs=n_jobs)
            
            # Save connectivity matrices
            con_1_file = os.path.join(output_folder,
                                     f'{subject_name}_{condition_1}_connectivity.npy')
            con_2_file = os.path.join(output_folder,
                                     f'{subject_name}_{condition_2}_connectivity.npy')
            
            np.save(con_1_file, con_1)
            np.save(con_2_file, con_2)
            
            print(f"  ✓ Connectivity computed and saved")
            print(f"     Condition 1: {con_1_file}")
            print(f"     Condition 2: {con_2_file}")
        else:
            print("  ⚠ Skipping connectivity computation (no pairs)")
        
    except Exception as e:
        print(f"  ✗ Error computing connectivity: {e}")
        raise
    
    # ========================================
    # Analysis Complete
    # ========================================
    print(f"\n{'='*60}")
    print(f"Connectivity Analysis Complete for: {subject_name}")
    print(f"{'='*60}\n")
    print(f"Output files saved to: {output_folder}")
    print(f"\nGenerated files:")
    print(f"  - Morphed source space: *_for_con-morph-src.fif")
    print(f"  - Forward model: *-for_con-morphed-fwd.fif")
    print(f"  - Connectivity matrices: *_connectivity.npy")


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
        config['subject']['name'] = args.subject
    
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
