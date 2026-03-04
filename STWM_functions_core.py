"""
Core Functions for MEG Sensor Space Analysis (Refactored)
===========================================================

This module contains core data processing functions WITHOUT visualization.
All plotting/visual inspection has been moved to visual_inspection.py.

These functions focus on:
- Data loading and preprocessing
- Epoching
- Time-frequency analysis
- FOOOF decomposition
- Cross-spectral density computation
- Statistical analysis

@author: Nikita Otstavnov, 2023 (refactored 2026)
"""

import os
import numpy as np
import mne
from scipy import stats


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_preprocessed_data(file_path, meg_type='grad', exclude_channels=None):
    """
    Load preprocessed MEG data.
    
    Parameters
    ----------
    file_path : str
        Path to FIF file
    meg_type : str
        Type of MEG channels to load ('grad', 'mag', or 'all')
    exclude_channels : list or None
        List of channels to exclude
        
    Returns
    -------
    raw_data : mne.io.Raw
        Loaded raw data
    """
    raw_data = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
    
    if exclude_channels is None:
        exclude_channels = []
    
    raw_data = raw_data.pick_types(meg=meg_type, exclude=exclude_channels)
    
    return raw_data


def apply_filtering(raw_data, l_freq, h_freq, notch_freq=None, n_jobs=1):
    """
    Apply bandpass and notch filtering to raw data.
    
    Parameters
    ----------
    raw_data : mne.io.Raw
        Raw MEG data
    l_freq : float
        Low frequency cutoff (Hz)
    h_freq : float
        High frequency cutoff (Hz)
    notch_freq : float or None
        Notch filter frequency (Hz) for line noise
    n_jobs : int
        Number of parallel jobs
        
    Returns
    -------
    raw_data : mne.io.Raw
        Filtered raw data
    """
    # Bandpass filter
    raw_data = raw_data.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs)
    
    # Notch filter
    if notch_freq is not None:
        raw_data = raw_data.notch_filter(freqs=notch_freq, 
                                         method='spectrum_fit',
                                         filter_length='10s',
                                         n_jobs=n_jobs)
    
    return raw_data


def apply_ica(raw_data, ica):
    """
    Apply ICA transformation to raw data.
    
    Parameters
    ----------
    raw_data : mne.io.Raw
        Raw MEG data
    ica : mne.preprocessing.ICA
        Fitted ICA object
        
    Returns
    -------
    raw_data : mne.io.Raw
        ICA-corrected raw data
    """
    ica.apply(raw_data)
    return raw_data


# ============================================================================
# EVENT PROCESSING
# ============================================================================

def extract_events(raw_data, stim_channel='STI101', min_duration=0.001):
    """
    Extract events from raw data.
    
    Parameters
    ----------
    raw_data : mne.io.Raw
        Raw MEG data
    stim_channel : str
        Name of stimulus channel
    min_duration : float
        Minimum event duration (seconds)
        
    Returns
    -------
    events : array
        Events array (n_events, 3)
    """
    events = mne.find_events(raw_data, stim_channel=stim_channel,
                            shortest_event=1, verbose=False)
    return events


def rename_events(events, mapping_dict):
    """
    Rename event IDs according to a mapping dictionary.
    
    Parameters
    ----------
    events : array
        Events array
    mapping_dict : dict
        Dictionary mapping old event IDs to new ones
        
    Returns
    -------
    events : array
        Modified events array
    """
    events_copy = events.copy()
    for old_id, new_id in mapping_dict.items():
        events_copy[events_copy[:, 2] == old_id, 2] = new_id
    return events_copy


# ============================================================================
# EPOCHING
# ============================================================================

def create_epochs(raw_data, events, event_id, tmin, tmax,
                 reject=None, flat=None, baseline=None, 
                 picks=None, preload=True):
    """
    Create epochs from raw data.
    
    Parameters
    ----------
    raw_data : mne.io.Raw
        Raw MEG data
    events : array
        Events array
    event_id : int or list
        Event ID(s) to epoch
    tmin, tmax : float
        Start and end time relative to event (seconds)
    reject : dict or None
        Rejection criteria
    flat : dict or None
        Flat detection criteria
    baseline : tuple or None
        Baseline period (tmin, tmax)
    picks : str, list, or None
        Channels to pick
    preload : bool
        Whether to preload data
        
    Returns
    -------
    epochs : mne.Epochs
        Epochs object
    """
    epochs = mne.Epochs(raw_data, events, event_id=event_id,
                       tmin=tmin, tmax=tmax,
                       reject=reject, flat=flat,
                       baseline=baseline, picks=picks,
                       preload=preload, verbose=False)
    return epochs


def save_epochs(epochs, file_path, overwrite=True):
    """
    Save epochs to file.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    file_path : str
        Output file path
    overwrite : bool
        Whether to overwrite existing file
    """
    epochs.save(file_path, overwrite=overwrite)


# ============================================================================
# TIME-FREQUENCY ANALYSIS
# ============================================================================

def compute_time_frequency(epochs, freqs, n_cycles=5, decim=1, 
                          n_jobs=1, return_itc=True):
    """
    Compute time-frequency representation using Morlet wavelets.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    freqs : array
        Frequencies of interest (Hz)
    n_cycles : float or array
        Number of cycles for Morlet wavelets
    decim : int
        Decimation factor
    n_jobs : int
        Number of parallel jobs
    return_itc : bool
        Whether to return inter-trial coherence
        
    Returns
    -------
    power : mne.time_frequency.AverageTFR
        Time-frequency power
    itc : mne.time_frequency.AverageTFR or None
        Inter-trial coherence (if return_itc=True)
    """
    power, itc = mne.time_frequency.tfr_morlet(
        epochs, n_cycles=n_cycles, return_itc=return_itc,
        freqs=freqs, decim=decim, n_jobs=n_jobs, verbose=False)
    
    if return_itc:
        return power, itc
    else:
        return power


def save_tfr(tfr, file_path, overwrite=True):
    """
    Save time-frequency representation to file.
    
    Parameters
    ----------
    tfr : mne.time_frequency.AverageTFR
        Time-frequency object
    file_path : str
        Output file path
    overwrite : bool
        Whether to overwrite existing file
    """
    tfr.save(file_path, overwrite=overwrite)


def load_tfr(file_path):
    """
    Load time-frequency representation from file.
    
    Parameters
    ----------
    file_path : str
        Path to TFR file
        
    Returns
    -------
    tfr : mne.time_frequency.AverageTFR
        Time-frequency object
    """
    tfr = mne.time_frequency.read_tfrs(file_path)[0]
    return tfr


# ============================================================================
# FOOOF DECOMPOSITION
# ============================================================================

def apply_fooof_single_channel(spectrum, freqs, fm_settings=None):
    """
    Apply FOOOF to a single channel spectrum.
    
    Parameters
    ----------
    spectrum : array
        Power spectrum (n_freqs,)
    freqs : array
        Frequency values
    fm_settings : dict or None
        FOOOF settings
        
    Returns
    -------
    periodic : array
        Periodic (peak) component
    aperiodic : array
        Aperiodic component
    fm : FOOOF
        Fitted FOOOF model
    """
    from fooof import FOOOF
    from fooof.sim.gen import gen_aperiodic
    
    # Initialize FOOOF model
    if fm_settings is None:
        fm = FOOOF()
    else:
        fm = FOOOF(**fm_settings)
    
    # Fit model
    fm.fit(freqs, spectrum)
    
    # Extract components
    aperiodic = gen_aperiodic(fm.freqs, 
                             fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
    periodic = fm.power_spectrum - aperiodic
    
    return periodic, aperiodic, fm


def apply_fooof_multi_channel(spectrum, freqs, fm_settings=None):
    """
    Apply FOOOF to multi-channel spectra.
    
    Parameters
    ----------
    spectrum : array
        Power spectra (n_channels, n_freqs)
    freqs : array
        Frequency values
    fm_settings : dict or None
        FOOOF settings
        
    Returns
    -------
    periodic : array
        Periodic components (n_channels, n_freqs)
    aperiodic : array
        Aperiodic components (n_channels, n_freqs)
    """
    n_channels, n_freqs = spectrum.shape
    periodic = np.zeros((n_channels, n_freqs))
    aperiodic = np.zeros((n_channels, n_freqs))
    
    for ch in range(n_channels):
        per, aper, _ = apply_fooof_single_channel(spectrum[ch, :], freqs, 
                                                  fm_settings)
        periodic[ch, :] = per
        aperiodic[ch, :] = aper
    
    return periodic, aperiodic


# ============================================================================
# CROSS-SPECTRAL DENSITY
# ============================================================================

def compute_csd(epochs, freqs, tmin=None, tmax=None, 
               n_cycles=5, decim=1, n_jobs=1):
    """
    Compute cross-spectral density using Morlet wavelets.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    freqs : array
        Frequencies of interest (Hz)
    tmin, tmax : float or None
        Time window for CSD computation
    n_cycles : float or array
        Number of cycles for Morlet wavelets
    decim : int
        Decimation factor
    n_jobs : int
        Number of parallel jobs
        
    Returns
    -------
    csd : mne.time_frequency.CrossSpectralDensity
        CSD object
    """
    csd = mne.time_frequency.csd_morlet(
        epochs, freqs, tmin=tmin, tmax=tmax,
        n_cycles=n_cycles, decim=decim, n_jobs=n_jobs, verbose=False)
    return csd


def save_csd(csd, file_path, overwrite=True):
    """
    Save cross-spectral density to file.
    
    Parameters
    ----------
    csd : mne.time_frequency.CrossSpectralDensity
        CSD object
    file_path : str
        Output file path
    overwrite : bool
        Whether to overwrite existing file
    """
    csd.save(file_path, overwrite=overwrite, verbose=None)


def load_csd(file_path):
    """
    Load cross-spectral density from file.
    
    Parameters
    ----------
    file_path : str
        Path to CSD file
        
    Returns
    -------
    csd : mne.time_frequency.CrossSpectralDensity
        CSD object
    """
    csd = mne.time_frequency.read_csd(file_path)
    return csd


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_channel_adjacency(info, ch_type='grad'):
    """
    Compute spatial adjacency matrix for channels.
    
    Parameters
    ----------
    info : mne.Info
        MNE info object
    ch_type : str
        Channel type
        
    Returns
    -------
    adjacency : sparse matrix
        Adjacency matrix
    ch_names : list
        Channel names
    """
    adjacency, ch_names = mne.channels.find_ch_adjacency(info, ch_type=ch_type)
    return adjacency, ch_names


def cluster_permutation_test(data, adjacency, n_permutations=5000,
                             threshold=None, tail=0, out_type='mask'):
    """
    Perform cluster-based permutation test on sensor-space data.
    
    Parameters
    ----------
    data : array
        Data array (n_subjects, n_times/freqs, n_channels)
    adjacency : sparse matrix
        Channel adjacency matrix
    n_permutations : int
        Number of permutations
    threshold : float or None
        T-value threshold for cluster formation
    tail : int
        Test tail: 0 (two-tailed), 1 (greater), -1 (less)
    out_type : str
        Output type: 'mask' or 'indices'
        
    Returns
    -------
    T_obs : array
        Observed T-values
    clusters : list
        List of cluster masks/indices
    cluster_p_values : array
        P-values for each cluster
    H0 : array
        Null distribution
    """
    if threshold is None:
        # Use default threshold based on sample size
        df = len(data) - 1
        threshold = stats.distributions.t.ppf(0.975, df=df)
    
    T_obs, clusters, cluster_p_values, H0 = \
        mne.stats.spatio_temporal_cluster_1samp_test(
            data, out_type=out_type, adjacency=adjacency,
            n_permutations=n_permutations, threshold=threshold,
            tail=tail, verbose=False)
    
    return T_obs, clusters, cluster_p_values, H0


def extract_significant_clusters(T_obs, clusters, cluster_p_values, alpha=0.05):
    """
    Extract only significant clusters from test results.
    
    Parameters
    ----------
    T_obs : array
        Observed T-values
    clusters : list
        List of cluster masks/indices
    cluster_p_values : array
        P-values for each cluster
    alpha : float
        Significance level
        
    Returns
    -------
    T_obs_masked : array
        T-values with only significant clusters
    significant_indices : array
        Indices of significant clusters
    """
    T_obs_masked = np.zeros_like(T_obs)
    significant_indices = np.where(cluster_p_values < alpha)[0]
    
    for idx in significant_indices:
        T_obs_masked[clusters[idx]] = T_obs[clusters[idx]]
    
    return T_obs_masked, significant_indices


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def average_over_time(tfr_data, axis=-1):
    """
    Average time-frequency data over time dimension.
    
    Parameters
    ----------
    tfr_data : array
        Time-frequency data
    axis : int
        Time axis
        
    Returns
    -------
    averaged : array
        Time-averaged data
    """
    return np.mean(tfr_data, axis=axis)


def crop_time_window(tfr, tmin, tmax):
    """
    Crop time-frequency data to specific time window.
    
    Parameters
    ----------
    tfr : mne.time_frequency.AverageTFR
        Time-frequency object
    tmin, tmax : float
        Start and end time (seconds)
        
    Returns
    -------
    tfr_crop : mne.time_frequency.AverageTFR
        Cropped time-frequency object
    """
    return tfr.copy().crop(tmin, tmax)


def merge_subjects_data(file_list, load_func=np.load):
    """
    Load and merge data from multiple subjects.
    
    Parameters
    ----------
    file_list : list
        List of file paths
    load_func : callable
        Function to load individual files
        
    Returns
    -------
    merged : array
        Merged data array
    """
    data_list = [load_func(f) for f in file_list]
    merged = np.array(data_list)
    return merged


def get_frequency_bands(freqs, bands={'theta': (4, 8), 'alpha': (8, 13),
                                      'beta': (13, 30), 'gamma': (30, 80)}):
    """
    Get frequency band indices from frequency array.
    
    Parameters
    ----------
    freqs : array
        Frequency values
    bands : dict
        Dictionary of band names and (fmin, fmax) tuples
        
    Returns
    -------
    band_indices : dict
        Dictionary of band names and corresponding frequency indices
    """
    band_indices = {}
    for band_name, (fmin, fmax) in bands.items():
        indices = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        band_indices[band_name] = indices
    return band_indices
