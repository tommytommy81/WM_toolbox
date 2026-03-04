"""
Example Usage of the Refactored MEG Analysis Pipeline
======================================================

This script demonstrates how to use the refactored analysis pipeline
for MEG sensor space analysis.

It shows:
1. Loading configuration
2. Running individual analysis
3. Running group statistics
4. Visual inspection

Author: 2026
"""

import os
import yaml
import numpy as np
from sensor_space_individual_analysis import run_individual_analysis
from sensor_space_group_statistics import run_group_statistics
from visual_inspection import VisualInspector


def example_single_subject_analysis():
    """
    Example: Analyze a single subject
    """
    print("\n" + "="*60)
    print("Example 1: Single Subject Analysis")
    print("="*60 + "\n")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for specific subject
    # config['subject']['name'] = 'S1'
    # config['subject']['file_name'] = '1_test_1_tsss_mc_trans.fif'
    
    # Run analysis
    print("Running individual analysis for subject S1...")
    run_individual_analysis(config)
    print("\n✓ Analysis complete!")


def example_batch_subjects():
    """
    Example: Process multiple subjects
    """
    print("\n" + "="*60)
    print("Example 2: Batch Processing Multiple Subjects")
    print("="*60 + "\n")
    
    # Load base configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Define subjects
    subjects = [
        {'name': 'S1', 'file': '1_test_1_tsss_mc_trans.fif'},
        {'name': 'S2', 'file': '2_test_1_tsss_mc_trans.fif'},
        {'name': 'S3', 'file': '3_test_1_tsss_mc_trans.fif'},
    ]
    
    # Process each subject
    for subj in subjects:
        print(f"\nProcessing {subj['name']}...")
        config['subject']['name'] = subj['name']
        config['subject']['file_name'] = subj['file']
        
        try:
            run_individual_analysis(config)
            print(f"✓ {subj['name']} completed")
        except Exception as e:
            print(f"✗ Error processing {subj['name']}: {e}")


def example_group_statistics():
    """
    Example: Run group-level statistics
    """
    print("\n" + "="*60)
    print("Example 3: Group-Level Statistics")
    print("="*60 + "\n")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set number of subjects
    config['statistics']['num_subjects'] = 3
    
    # Run group analysis
    print("Running group statistics...")
    T_obs, T_obs_plot, clusters, cluster_p_values = run_group_statistics(config)
    
    # Display results
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"T-values shape: {T_obs.shape}")
    print(f"Number of clusters: {len(clusters)}")
    print(f"Significant clusters (p < 0.05): {np.sum(cluster_p_values < 0.05)}")
    
    if np.any(cluster_p_values < 0.05):
        print("\nSignificant clusters:")
        for i, p in enumerate(cluster_p_values):
            if p < 0.05:
                print(f"  Cluster {i}: p = {p:.4f}")


def example_visual_inspection():
    """
    Example: Visual inspection of results
    """
    print("\n" + "="*60)
    print("Example 4: Visual Inspection")
    print("="*60 + "\n")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create inspector
    inspector = VisualInspector(config)
    
    # Load and inspect data
    print("Loading epochs...")
    epochs_S = inspector.load_and_inspect_epochs('S')
    epochs_T = inspector.load_and_inspect_epochs('T')
    
    print("\nLoading time-frequency data...")
    power_S = inspector.load_and_inspect_power('S')
    power_T = inspector.load_and_inspect_power('T')
    
    print("\nCreating comparison report...")
    inspector.create_comparison_report()


def example_custom_analysis():
    """
    Example: Custom analysis using core functions
    """
    print("\n" + "="*60)
    print("Example 5: Custom Analysis with Core Functions")
    print("="*60 + "\n")
    
    import mne
    from STWM_functions_core import (
        load_preprocessed_data,
        create_epochs,
        compute_time_frequency,
        apply_fooof_multi_channel
    )
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    print("Loading data...")
    folder = config['paths']['data_folder']
    file_name = config['subject']['file_name']
    file_path = os.path.join(folder, file_name)
    
    raw_data = load_preprocessed_data(file_path, meg_type='grad')
    print(f"✓ Loaded {len(raw_data.ch_names)} channels")
    
    # Extract events
    print("Extracting events...")
    events = mne.find_events(raw_data, stim_channel='STI101')
    print(f"✓ Found {len(events)} events")
    
    # Create epochs
    print("Creating epochs...")
    epochs = create_epochs(
        raw_data, events, event_id=155,
        tmin=-8, tmax=8, picks='grad'
    )
    print(f"✓ Created {len(epochs)} epochs")
    
    # Compute time-frequency
    print("Computing time-frequency...")
    freqs = np.logspace(np.log10(4), np.log10(80), num=30)
    power, itc = compute_time_frequency(
        epochs, freqs, n_cycles=5, decim=20, n_jobs=4
    )
    print(f"✓ TFR computed: {power.data.shape}")
    
    # Apply FOOOF
    print("Applying FOOOF decomposition...")
    spectrum = np.mean(power.data, axis=2)  # Average over time
    periodic, aperiodic = apply_fooof_multi_channel(spectrum, freqs)
    print(f"✓ FOOOF completed: periodic shape = {periodic.shape}")
    
    print("\n✓ Custom analysis complete!")


def main():
    """
    Main function to run examples
    """
    print("\n" + "="*60)
    print("MEG Analysis Pipeline - Example Usage")
    print("="*60)
    
    print("\nAvailable examples:")
    print("  1. Single subject analysis")
    print("  2. Batch processing multiple subjects")
    print("  3. Group-level statistics")
    print("  4. Visual inspection")
    print("  5. Custom analysis with core functions")
    
    print("\nNote: Make sure config.yaml is properly configured")
    print("      and data files are available before running.")
    
    # Uncomment the example you want to run:
    
    example_single_subject_analysis()
    # example_batch_subjects()
    # example_group_statistics()
    # example_visual_inspection()
    # example_custom_analysis()
    
    print("\nTo run an example, uncomment it in the main() function.")


if __name__ == "__main__":
    main()
