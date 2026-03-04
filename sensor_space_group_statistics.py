"""
Sensor Space Analysis - Group Level Statistics (Refactored)
============================================================

This script performs group-level statistical analysis on frequency-domain
MEG data, comparing two experimental conditions across multiple subjects.

All parameters are loaded from a configuration file (config.yaml).

Usage:
------
python sensor_space_group_statistics.py --config config.yaml

@author: Nikita Otstavnov, 2023 (refactored 2026)
"""

import os
import argparse
import yaml
import numpy as np
import mne
from scipy import stats


def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
        
    Returns
    -------
    config : dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_fooof_results(num_subjects, folder, condition_1, condition_2):
    """
    Merge FOOOF results from all subjects.
    
    Parameters
    ----------
    num_subjects : int
        Number of subjects
    folder : str
        Data folder path
    condition_1 : str
        First condition name
    condition_2 : str
        Second condition name
        
    Returns
    -------
    list_1_ped : array
        Periodic component for condition 1, shape (n_subjects, n_channels, n_freqs)
    list_1_aper : array
        Aperiodic component for condition 1
    list_2_ped : array
        Periodic component for condition 2
    list_2_aper : array
        Aperiodic component for condition 2
    """
    os.chdir(folder)
    
    list_1_ped = []
    list_1_aper = []
    list_2_ped = []
    list_2_aper = []
    
    print(f"Loading FOOOF results for {num_subjects} subjects...")
    
    for i in range(num_subjects):
        subject_id = i + 1
        
        # Load condition 1
        ped_1 = np.load(f'S{subject_id}_{condition_1}_ped_crop.npy')
        aper_1 = np.load(f'S{subject_id}_{condition_1}_aper_crop.npy')
        list_1_ped.append(ped_1)
        list_1_aper.append(aper_1)
        
        # Load condition 2
        ped_2 = np.load(f'S{subject_id}_{condition_2}_ped_crop.npy')
        aper_2 = np.load(f'S{subject_id}_{condition_2}_aper_crop.npy')
        list_2_ped.append(ped_2)
        list_2_aper.append(aper_2)
        
        print(f"  ✓ Subject S{subject_id} loaded")
    
    # Convert to arrays
    list_1_ped_array = np.array(list_1_ped)
    list_1_aper_array = np.array(list_1_aper)
    list_2_ped_array = np.array(list_2_ped)
    list_2_aper_array = np.array(list_2_aper)
    
    print(f"\nData shapes:")
    print(f"  Condition '{condition_1}' periodic: {list_1_ped_array.shape}")
    print(f"  Condition '{condition_2}' periodic: {list_2_ped_array.shape}")
    
    # Save merged results
    np.save(f'list_{condition_1}_ped_crop.npy', list_1_ped_array)
    np.save(f'list_{condition_1}_aper_crop.npy', list_1_aper_array)
    np.save(f'list_{condition_2}_ped_crop.npy', list_2_ped_array)
    np.save(f'list_{condition_2}_aper_crop.npy', list_2_aper_array)
    print(f"  ✓ Merged results saved")
    
    return list_1_ped_array, list_1_aper_array, list_2_ped_array, list_2_aper_array


def perform_cluster_statistics(data_1, data_2, epochs_file, ch_type, 
                               alpha, p_threshold, n_permutations, 
                               tail, out_type):
    """
    Perform cluster-based permutation test on sensor-space data.
    
    Parameters
    ----------
    data_1 : array
        Data for condition 1, shape (n_subjects, n_channels, n_freqs)
    data_2 : array
        Data for condition 2, shape (n_subjects, n_channels, n_freqs)
    epochs_file : str
        Path to epochs file (used to get channel adjacency)
    ch_type : str
        Channel type
    alpha : float
        Significance level
    p_threshold : float
        P-value threshold for cluster formation
    n_permutations : int
        Number of permutations
    tail : int
        Test tail (-1, 0, or 1)
    out_type : str
        Output type ('mask' or 'indices')
        
    Returns
    -------
    T_obs : array
        Observed T-values
    T_obs_plot : array
        T-values with only significant clusters
    clusters : list
        List of cluster arrays
    cluster_p_values : array
        P-values for each cluster
    """
    print("\nPerforming cluster-based permutation test...")
    
    # Load epochs to get channel adjacency
    epochs = mne.read_epochs(epochs_file, preload=False, verbose=False)
    info = epochs.info
    adj, ch_names = mne.channels.find_ch_adjacency(info, ch_type=ch_type)
    print(f"  ✓ Channel adjacency computed for {len(ch_names)} channels")
    
    # Prepare data
    print(f"  ✓ Data shape: {data_1.shape}")
    
    # Transpose to (n_subjects, n_freqs, n_channels) for MNE
    data_1_trans = np.transpose(data_1, (0, 2, 1))
    data_2_trans = np.transpose(data_2, (0, 2, 1))
    
    # Compute difference
    diff = data_2_trans - data_1_trans
    
    # Calculate threshold
    df = len(data_1) - 1
    t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)
    print(f"  ✓ T-threshold: {t_threshold:.3f} (df={df}, p={p_threshold})")
    
    # Run cluster test
    print(f"  ✓ Running permutation test with {n_permutations} permutations...")
    T_obs, clusters, cluster_p_values, H0 = mne.stats.spatio_temporal_cluster_1samp_test(
        diff, out_type=out_type, adjacency=adj,
        n_permutations=n_permutations, threshold=t_threshold, 
        tail=tail, verbose=False)
    
    print(f"\n  Results:")
    print(f"    T-obs shape: {T_obs.shape}")
    print(f"    Number of clusters: {len(clusters)}")
    print(f"    Cluster p-values: {cluster_p_values}")
    
    # Identify significant clusters
    significant_clusters = np.where(cluster_p_values < alpha)[0]
    print(f"    Significant clusters (p < {alpha}): {len(significant_clusters)}")
    
    if len(significant_clusters) > 0:
        print(f"    Significant cluster indices: {significant_clusters}")
        for idx in significant_clusters:
            print(f"      Cluster {idx}: p = {cluster_p_values[idx]:.4f}")
    
    # Create masked T-values (only significant clusters)
    T_obs_plot = np.zeros_like(T_obs)
    for cluster, p_val in zip(clusters, cluster_p_values):
        if p_val <= alpha:
            T_obs_plot[cluster] = T_obs[cluster]
    
    return T_obs, T_obs_plot, clusters, cluster_p_values


def save_statistics_results(T_obs, T_obs_plot, clusters, cluster_p_values,
                            output_folder, condition_1, condition_2):
    """
    Save statistical results to files.
    
    Parameters
    ----------
    T_obs : array
        Observed T-values
    T_obs_plot : array
        Masked T-values (significant only)
    clusters : list
        List of clusters
    cluster_p_values : array
        P-values for clusters
    output_folder : str
        Output directory
    condition_1, condition_2 : str
        Condition names
    """
    os.chdir(output_folder)
    
    # Save T-values
    np.save(f'T_obs_{condition_1}_vs_{condition_2}.npy', T_obs)
    np.save(f'T_obs_significant_{condition_1}_vs_{condition_2}.npy', T_obs_plot)
    
    # Save cluster information
    np.save(f'cluster_p_values_{condition_1}_vs_{condition_2}.npy', cluster_p_values)
    
    # Save summary text file
    with open(f'statistics_summary_{condition_1}_vs_{condition_2}.txt', 'w') as f:
        f.write(f"Statistical Analysis Summary\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Comparison: {condition_2} vs {condition_1}\n\n")
        f.write(f"T-obs shape: {T_obs.shape}\n")
        f.write(f"Number of clusters: {len(clusters)}\n\n")
        f.write(f"Cluster p-values:\n")
        for i, p in enumerate(cluster_p_values):
            f.write(f"  Cluster {i}: p = {p:.6f}\n")
    
    print(f"  ✓ Statistical results saved to {output_folder}")


def run_group_statistics(config):
    """
    Run complete group-level statistical analysis.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    """
    # Extract parameters
    folder = config['paths']['data_folder']
    output_folder = config['paths'].get('output_folder', folder)
    
    condition_1 = config['conditions']['condition_1']['name']
    condition_2 = config['conditions']['condition_2']['name']
    
    num_subjects = config['statistics']['num_subjects']
    ch_type = config['statistics']['ch_type']
    alpha = config['statistics']['alpha']
    p_threshold = config['statistics']['threshold']
    n_permutations = config['statistics']['n_permutations']
    tail = config['statistics']['tail']
    out_type = config['statistics']['out_type']
    
    # Get a representative subject for channel info
    subject_name = config['subject']['name']
    
    print(f"\n{'='*60}")
    print(f"Starting Group-Level Statistical Analysis")
    print(f"{'='*60}\n")
    print(f"Number of subjects: {num_subjects}")
    print(f"Conditions: '{condition_1}' vs '{condition_2}'")
    print(f"Significance level: {alpha}")
    print(f"Permutations: {n_permutations}\n")
    
    # ========================================
    # STEP 1: Merge FOOOF results
    # ========================================
    print("Step 1: Merging FOOOF results across subjects...")
    list_1_ped, list_1_aper, list_2_ped, list_2_aper = merge_fooof_results(
        num_subjects, folder, condition_1, condition_2)
    
    # ========================================
    # STEP 2: Perform statistical analysis
    # ========================================
    print("\nStep 2: Performing statistical analysis...")
    
    # Get epochs file for channel information
    epochs_file = os.path.join(folder, f'{subject_name}_{condition_1}_epochs-epo.fif')
    
    # Run statistics on periodic component
    T_obs, T_obs_plot, clusters, cluster_p_values = perform_cluster_statistics(
        list_1_ped, list_2_ped, epochs_file, ch_type,
        alpha, p_threshold, n_permutations, tail, out_type)
    
    # ========================================
    # STEP 3: Save results
    # ========================================
    print("\nStep 3: Saving results...")
    save_statistics_results(T_obs, T_obs_plot, clusters, cluster_p_values,
                           output_folder, condition_1, condition_2)
    
    # ========================================
    # Analysis Complete
    # ========================================
    print(f"\n{'='*60}")
    print(f"Group-Level Statistical Analysis Complete")
    print(f"{'='*60}\n")
    print(f"Output files saved to: {output_folder}")
    print(f"\nGenerated files:")
    print(f"  - T_obs_{condition_1}_vs_{condition_2}.npy")
    print(f"  - T_obs_significant_{condition_1}_vs_{condition_2}.npy")
    print(f"  - cluster_p_values_{condition_1}_vs_{condition_2}.npy")
    print(f"  - statistics_summary_{condition_1}_vs_{condition_2}.txt")
    print(f"\nTo visualize results, use the visual_inspection module or")
    print(f"Script_for_Figure_generating.py")
    
    return T_obs, T_obs_plot, clusters, cluster_p_values


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(
        description='Run group-level sensor space statistics for MEG data')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run analysis
    try:
        run_group_statistics(config)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
