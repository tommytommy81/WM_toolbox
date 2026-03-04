"""
Sensor Space Analysis - Individual Level (Refactored)
======================================================

This script performs frequency-domain analysis on preprocessed MEG data
for individual subjects, comparing two experimental conditions.

The script assumes that the input data is already preprocessed 
(filtered, ICA applied, bad channels marked).

All parameters are loaded from a configuration file (config.yaml).

Usage:
------
python sensor_space_individual_analysis.py --config config.yaml --subject S1

@author: Nikita Otstavnov, 2023 (refactored 2026)
"""

import os
import argparse
import yaml
import numpy as np
import mne
from STWM_functions_core import (
    load_preprocessed_data,
    extract_events,
    create_epochs,
    compute_time_frequency,
    compute_csd
)


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


def run_individual_analysis(config):
    """
    Run complete individual-level sensor space analysis.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    """
    
    # Extract parameters from config
    folder = config['paths']['data_folder']
    output_folder = config['paths'].get('output_folder', folder)
    subject_name = config['subject']['name']
    file_name = config['subject']['file_name']
    
    condition_1 = config['conditions']['condition_1']['name']
    condition_2 = config['conditions']['condition_2']['name']
    event_id_1 = config['conditions']['condition_1']['event_id']
    event_id_2 = config['conditions']['condition_2']['event_id']
    
    stim_channel = config['events']['stim_channel']
    ch_type = config['channels']['meg_type']
    
    # Epoching parameters
    tmin = config['epoching']['tmin']
    tmax = config['epoching']['tmax']
    reject_criteria = config['epoching']['reject_criteria']
    flat_criteria = config['epoching']['flat_criteria']
    
    # Convert reject and flat criteria values to float if they're strings
    if reject_criteria is not None:
        reject_criteria = {k: float(v) if isinstance(v, str) else v 
                          for k, v in reject_criteria.items()}
    if flat_criteria is not None:
        flat_criteria = {k: float(v) if isinstance(v, str) else v 
                        for k, v in flat_criteria.items()}
    
    # Time-frequency parameters
    min_freq_log = config['time_frequency']['min_freq_log']
    max_freq_log = config['time_frequency']['max_freq_log']
    freq_res = config['time_frequency']['freq_resolution']
    n_cycles = config['time_frequency']['n_cycles']
    decim = config['time_frequency']['decim']
    n_jobs = config['processing']['n_jobs']
    
    # Time window of interest
    t_min_interest = config['time_window']['tmin']
    t_max_interest = config['time_window']['tmax']
    t_min_baseline = config['csd']['baseline_tmin']
    t_max_baseline = config['csd']['baseline_tmax']
    
    print(f"\n{'='*60}")
    print(f"Starting Sensor Space Analysis for Subject: {subject_name}")
    print(f"{'='*60}\n")
    
    # Change to data folder
    os.chdir(folder)
    
    # ========================================
    # STEP 1: Load preprocessed data
    # ========================================
    print("Step 1: Loading preprocessed data...")
    file_path = os.path.join(folder, file_name)
    raw_data = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
    print(f"  ✓ Loaded: {file_name}")
    print(f"  ✓ Duration: {raw_data.times[-1]:.2f} seconds")
    print(f"  ✓ Channels: {len(raw_data.ch_names)} total")
    
    # Pick only MEG channels
    # raw_data = raw_data.pick_types(meg=ch_type, exclude=[])
    # print(f"  ✓ Selected {ch_type} channels: {len(raw_data.ch_names)}")
    
    # ========================================
    # STEP 2: Extract and process events
    # ========================================
    print("\nStep 2: Extracting events...")
    events = mne.find_events(raw_data, stim_channel=stim_channel, 
                            shortest_event=1, verbose=False)
    print(f"  ✓ Found {len(events)} events")
    
    # Count events per condition
    events_cond1 = np.sum(events[:, 2] == event_id_1)
    events_cond2 = np.sum(events[:, 2] == event_id_2)
    print(f"  ✓ Condition '{condition_1}' (ID={event_id_1}): {events_cond1} events")
    print(f"  ✓ Condition '{condition_2}' (ID={event_id_2}): {events_cond2} events")
    
    # ========================================
    # STEP 3: Create epochs for both conditions
    # ========================================
    print("\nStep 3: Creating epochs...")
    
    # Condition 1
    epochs_1 = mne.Epochs(raw_data, events, event_id=event_id_1,
                         tmin=tmin, tmax=tmax,
                         reject=reject_criteria, flat=flat_criteria,
                         preload=True, picks=ch_type,
                         baseline=None, verbose=False)
    epochs_1.save(os.path.join(output_folder, f'{subject_name}_{condition_1}_epochs-epo.fif'),
                 overwrite=True)
    print(f"  ✓ Condition '{condition_1}': {len(epochs_1)} epochs")
    print(f"     Saved: {subject_name}_{condition_1}_epochs-epo.fif")
    
    # Condition 2
    epochs_2 = mne.Epochs(raw_data, events, event_id=event_id_2,
                         tmin=tmin, tmax=tmax,
                         reject=reject_criteria, flat=flat_criteria,
                         preload=True, picks=ch_type,
                         baseline=None, verbose=False)
    epochs_2.save(os.path.join(output_folder, f'{subject_name}_{condition_2}_epochs-epo.fif'),
                 overwrite=True)
    print(f"  ✓ Condition '{condition_2}': {len(epochs_2)} epochs")
    print(f"     Saved: {subject_name}_{condition_2}_epochs-epo.fif")
    
    # Combined epochs
    epochs_full = mne.Epochs(raw_data, events, event_id=[event_id_1, event_id_2],
                            tmin=tmin, tmax=tmax,
                            reject=reject_criteria, flat=flat_criteria,
                            preload=True, picks=ch_type,
                            baseline=None, verbose=False)
    epochs_full.save(os.path.join(output_folder, f'{subject_name}_ave_epochs-epo.fif'),
                    overwrite=True)
    print(f"  ✓ Combined epochs: {len(epochs_full)} epochs")
    
    # ========================================
    # STEP 4: Time-frequency analysis
    # ========================================
    print("\nStep 4: Computing time-frequency representations...")
    
    frequencies = np.logspace(min_freq_log, max_freq_log, num=freq_res)
    print(f"  ✓ Frequency range: {frequencies[0]:.2f} - {frequencies[-1]:.2f} Hz")
    print(f"  ✓ Number of frequencies: {freq_res}")
    print(f"  ✓ Computing with {n_cycles} cycles...")
    
    # Condition 1
    power_1, itc_1 = epochs_1.compute_tfr(
        method="morlet", freqs=frequencies, n_cycles=n_cycles,
        decim=decim, n_jobs=n_jobs, return_itc=True, average=True, verbose=False)
    power_1.save(os.path.join(output_folder, f'{subject_name}_power_{condition_1}-tfr.h5'),
                overwrite=True)
    itc_1.save(os.path.join(output_folder, f'{subject_name}_itc_{condition_1}-tfr.h5'),
              overwrite=True)
    print(f"  ✓ Condition '{condition_1}' TFR computed")
    
    # Condition 2
    power_2, itc_2 = epochs_2.compute_tfr(
        method="morlet", freqs=frequencies, n_cycles=n_cycles,
        decim=decim, n_jobs=n_jobs, return_itc=True, average=True, verbose=False)
    power_2.save(os.path.join(output_folder, f'{subject_name}_power_{condition_2}-tfr.h5'),
                overwrite=True)
    itc_2.save(os.path.join(output_folder, f'{subject_name}_itc_{condition_2}-tfr.h5'),
              overwrite=True)
    print(f"  ✓ Condition '{condition_2}' TFR computed")
    
    # ========================================
    # STEP 5: FOOOF decomposition
    # ========================================
    print("\nStep 5: Applying FOOOF decomposition...")
    
    # Crop to time window of interest
    power_1_crop = power_1.copy().crop(t_min_interest, t_max_interest)
    power_2_crop = power_2.copy().crop(t_min_interest, t_max_interest)
    print(f"  ✓ Cropped to time window: {t_min_interest} - {t_max_interest} s")
    
    # Apply FOOOF
    from fooof import FOOOF
    from fooof.sim.gen import gen_aperiodic
    
    fm = FOOOF()
    
    # Process condition 1
    print(f"  ✓ Processing condition '{condition_1}'...")
    spectrum_1 = np.mean(power_1_crop.data, axis=2)  # Average over time
    spectrum_peak_1, spectrum_aper_1 = process_fooof(
        spectrum_1, frequencies, fm, subject_name, condition_1, output_folder)
    print(f"     Processed {spectrum_peak_1.shape[0]} channels")
    
    # Process condition 2
    print(f"  ✓ Processing condition '{condition_2}'...")
    spectrum_2 = np.mean(power_2_crop.data, axis=2)  # Average over time
    spectrum_peak_2, spectrum_aper_2 = process_fooof(
        spectrum_2, frequencies, fm, subject_name, condition_2, output_folder)
    print(f"     Processed {spectrum_peak_2.shape[0]} channels")
    
    # Save FOOOF results
    np.save(os.path.join(output_folder, f'{subject_name}_{condition_1}_ped_crop.npy'),
           spectrum_peak_1)
    np.save(os.path.join(output_folder, f'{subject_name}_{condition_1}_aper_crop.npy'),
           spectrum_aper_1)
    np.save(os.path.join(output_folder, f'{subject_name}_{condition_2}_ped_crop.npy'),
           spectrum_peak_2)
    np.save(os.path.join(output_folder, f'{subject_name}_{condition_2}_aper_crop.npy'),
           spectrum_aper_2)
    print(f"  ✓ FOOOF results saved")
    
    # ========================================
    # STEP 6: Cross-Spectral Density (CSD)
    # ========================================
    print("\nStep 6: Computing Cross-Spectral Density...")
    
    csd_1 = mne.time_frequency.csd_morlet(
        epochs_1, frequencies, tmin=t_min_interest, tmax=t_max_interest,
        n_cycles=n_cycles, decim=decim, n_jobs=n_jobs, verbose=False)
    csd_1.save(os.path.join(output_folder, f'{subject_name}_{condition_1}_csd.h5'),
              overwrite=True)
    print(f"  ✓ Condition '{condition_1}' CSD computed")
    
    csd_2 = mne.time_frequency.csd_morlet(
        epochs_2, frequencies, tmin=t_min_interest, tmax=t_max_interest,
        n_cycles=n_cycles, decim=decim, n_jobs=n_jobs, verbose=False)
    csd_2.save(os.path.join(output_folder, f'{subject_name}_{condition_2}_csd.h5'),
              overwrite=True)
    print(f"  ✓ Condition '{condition_2}' CSD computed")
    
    csd_base = mne.time_frequency.csd_morlet(
        epochs_full, frequencies, tmin=t_min_baseline, tmax=t_max_baseline,
        n_cycles=n_cycles, decim=decim, n_jobs=n_jobs, verbose=False)
    csd_base.save(os.path.join(output_folder, f'{subject_name}_baseline_csd.h5'),
              overwrite=True)
    print(f"  ✓ Baseline CSD computed")
    
    # ========================================
    # Analysis Complete
    # ========================================
    print(f"\n{'='*60}")
    print(f"Analysis Complete for Subject: {subject_name}")
    print(f"{'='*60}\n")
    print(f"Output files saved to: {output_folder}")
    print(f"\nGenerated files:")
    print(f"  - Epochs: *_epochs-epo.fif")
    print(f"  - Time-Frequency: *_power_*-tfr.h5, *_itc_*-tfr.h5")
    print(f"  - FOOOF: *_ped_crop.npy, *_aper_crop.npy")
    print(f"  - CSD: *_csd.h5")


def process_fooof(spectrum, frequencies, fm, subject_name, condition, output_folder):
    """
    Apply FOOOF decomposition to separate periodic and aperiodic components.
    
    Parameters
    ----------
    spectrum : array, shape (n_channels, n_freqs)
        Power spectrum for each channel
    frequencies : array
        Frequency values (can be log-spaced)
    fm : FOOOF
        FOOOF model object
    subject_name : str
        Subject identifier
    condition : str
        Condition name
    output_folder : str
        Output directory
        
    Returns
    -------
    spectrum_peak : array
        Periodic (peak) component (interpolated back to original frequencies)
    spectrum_aper : array
        Aperiodic component (interpolated back to original frequencies)
    """
    from fooof.sim.gen import gen_aperiodic
    from scipy.interpolate import interp1d
    
    n_channels = spectrum.shape[0]
    n_freqs = len(frequencies)
    
    # Create linear frequency spacing for FOOOF
    freq_min = frequencies[0]
    freq_max = frequencies[-1]
    freqs_linear = np.linspace(freq_min, freq_max, n_freqs)
    
    # Calculate frequency resolution and set appropriate peak width bounds
    freq_res = freqs_linear[1] - freqs_linear[0]
    min_peak_width = 2.0 * freq_res  # 2x frequency resolution as recommended
    max_peak_width = 12.0  # Reasonable upper limit for neural oscillations
    
    # Configure FOOOF with appropriate settings
    fm.peak_width_limits = (min_peak_width, max_peak_width)
    
    spectrum_peak = np.zeros((n_channels, n_freqs))
    spectrum_aper = np.zeros((n_channels, n_freqs))
    
    for ch in range(n_channels):
        spec = spectrum[ch, :]
        
        # Interpolate spectrum to linear frequency spacing
        interp_func = interp1d(frequencies, spec, kind='cubic', fill_value='extrapolate')
        spec_linear = interp_func(freqs_linear)
        
        # Fit FOOOF on linear frequencies
        fm.fit(freqs_linear, spec_linear)
        
        # Save individual FOOOF results (optional)
        # fm.save(os.path.join(output_folder, 
        #                     f'FOOOF_{subject_name}_{condition}_{ch}'),
        #        save_results=True, save_settings=True, save_data=True)
        
        # Extract aperiodic component
        init_ap_fit = gen_aperiodic(fm.freqs, 
                                    fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
        
        # Extract periodic component (flattened spectrum)
        init_flat_spec = fm.power_spectrum - init_ap_fit
        
        # Interpolate results back to original frequency spacing
        interp_peak = interp1d(freqs_linear, init_flat_spec, kind='cubic', fill_value='extrapolate')
        interp_aper = interp1d(freqs_linear, init_ap_fit, kind='cubic', fill_value='extrapolate')
        
        spectrum_peak[ch, :] = interp_peak(frequencies)
        spectrum_aper[ch, :] = interp_aper(frequencies)
    
    return spectrum_peak, spectrum_aper


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(
        description='Run sensor space analysis for individual MEG subjects')
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
        run_individual_analysis(config)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
