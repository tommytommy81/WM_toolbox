"""
Sensor-space utilities
======================

Used by notebooks/example_usage_sensor.ipynb and
notebooks/example_usage_sensor_complete.ipynb, and by `cli/stwm.py sensor*`.

Contents
--------
- Preprocessing (optional, data is normally already preprocessed)
- Low-level helpers: loading, epoching, TFR, CSD, FOOOF
- Individual-level pipeline: run_individual_analysis
- Group-level statistics: run_group_statistics
- Visual inspection: VisualInspector
- Publication figures: fig_3 (behaviour), fig_4 (sensor statistics)

@author: Nikita Otstavnov, 2023 (refactored 2026)
"""

import os
import os.path as op

import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import stats


# ============================================================================
# PREPROCESSING (optional)
# ============================================================================

def perform_initial_analysis(n_components, random_state, max_iter,
                             stim_channel, list_channels,
                             meg, exclude_channels, folder,
                             file_name, subject_name, min_freqs,
                             max_freqs, notch_freq, n_jobs):
    """Band-pass + notch filter the raw data and fit ICA (EOG/ECG auto-exclusion)."""
    raw_data = mne.io.read_raw_fif(os.path.join(folder, file_name),
                                   allow_maxshield=False, preload=False,
                                   on_split_missing='raise', verbose=None)
    raw_data.plot(title='Raw data')

    raw_data.load_data()
    raw_data = raw_data.filter(l_freq=min_freqs, h_freq=max_freqs, n_jobs=n_jobs)
    raw_data = raw_data.notch_filter(freqs=notch_freq, method='spectrum_fit',
                                     filter_length='10s', n_jobs=n_jobs)
    raw_data.plot_psd(fmax=max_freqs, average=True, n_jobs=n_jobs)
    raw_data.plot(title='Notch + Bandpass data')

    ica = mne.preprocessing.ICA(n_components=n_components,
                                random_state=random_state, max_iter=max_iter)
    ica.fit(raw_data)
    ica.plot_components(sensors=True, colorbar=True,
                        title='ICA components', outlines='head')

    eog_indices, _ = ica.find_bads_eog(raw_data)
    ecg_indices, _ = ica.find_bads_ecg(raw_data, method='correlation',
                                       threshold='auto')
    ica.exclude = eog_indices + ecg_indices

    ica.save(op.join(folder, '{}-ica.fif'.format(subject_name)), overwrite=True)

    return raw_data, ica, ica.exclude


def ica_apply(data, ica, subject_name):
    """Apply a fitted ICA to the data and plot the result."""
    ica.apply(data)
    data.plot()
    return data


def event_renaming(data, stim_channel, list_channels, subject_name, meg):
    """Recode event 155 to 255 when it is surrounded by events > 180 (temporal condition)."""
    event = mne.find_events(data, shortest_event=1, stim_channel=stim_channel)
    data.plot(events=event, title='Original events')

    for i in range(len(event[:, 2])):
        if event[i, 2] == 155 and event[i - 1, 2] > 180 and event[i + 1, 2] > 180:
            event[i, 2] = 255
    data.plot(events=event, title='New events')

    return data, event


def raw_data_saver(folder, data, subject_name, meg):
    """Save gradiometer-only filtered data and its annotations."""
    data_2 = data.copy().pick_types(meg="grad", exclude=[])
    data_2.save(op.join(folder, '{}_filtered.fif'.format(subject_name)), overwrite=True)
    data_2.annotations.save(op.join(folder, '{}_-annotations.csv'.format(subject_name)),
                            overwrite=True)
    return data_2


# ============================================================================
# LOW-LEVEL HELPERS
# ============================================================================

def load_preprocessed_data(file_path, meg_type='grad', exclude_channels=None):
    """Load preprocessed MEG data and keep only `meg_type` channels."""
    raw_data = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
    return raw_data.pick_types(meg=meg_type, exclude=exclude_channels or [])


def extract_events(raw_data, stim_channel='STI101', min_duration=0.001):
    """Extract events (n_events, 3) from the stimulus channel."""
    return mne.find_events(raw_data, stim_channel=stim_channel,
                           shortest_event=1, verbose=False)


def create_epochs(raw_data, events, event_id, tmin, tmax,
                  reject=None, flat=None, baseline=None,
                  picks=None, preload=True):
    """Create epochs from raw data."""
    return mne.Epochs(raw_data, events, event_id=event_id,
                      tmin=tmin, tmax=tmax, reject=reject, flat=flat,
                      baseline=baseline, picks=picks,
                      preload=preload, verbose=False)


def compute_time_frequency(epochs, freqs, n_cycles=5, decim=1,
                           n_jobs=1, return_itc=True):
    """Morlet-wavelet TFR. Returns (power, itc) or power if return_itc=False."""
    out = mne.time_frequency.tfr_morlet(
        epochs, n_cycles=n_cycles, return_itc=return_itc,
        freqs=freqs, decim=decim, n_jobs=n_jobs, verbose=False)
    return out


def compute_csd(epochs, freqs, tmin=None, tmax=None,
                n_cycles=5, decim=1, n_jobs=1):
    """Morlet-wavelet cross-spectral density."""
    return mne.time_frequency.csd_morlet(
        epochs, freqs, tmin=tmin, tmax=tmax,
        n_cycles=n_cycles, decim=decim, n_jobs=n_jobs, verbose=False)


def apply_fooof_single_channel(spectrum, freqs, fm_settings=None):
    """FOOOF on one spectrum. Returns (periodic, aperiodic, fm)."""
    from fooof import FOOOF
    from fooof.sim.gen import gen_aperiodic

    fm = FOOOF(**(fm_settings or {}))
    fm.fit(freqs, spectrum)
    aperiodic = gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
    periodic = fm.power_spectrum - aperiodic
    return periodic, aperiodic, fm


def apply_fooof_multi_channel(spectrum, freqs, fm_settings=None):
    """FOOOF on (n_channels, n_freqs) spectra. Returns (periodic, aperiodic)."""
    periodic = np.zeros(spectrum.shape)
    aperiodic = np.zeros(spectrum.shape)
    for ch in range(spectrum.shape[0]):
        periodic[ch], aperiodic[ch], _ = apply_fooof_single_channel(
            spectrum[ch], freqs, fm_settings)
    return periodic, aperiodic


def process_fooof(spectrum, frequencies, fm, subject_name, condition, output_folder):
    """
    FOOOF on log-spaced spectra: interpolate to a linear grid, fit, and
    interpolate the periodic / aperiodic components back to `frequencies`.

    Returns
    -------
    spectrum_peak, spectrum_aper : array, shape (n_channels, n_freqs)
    """
    from fooof.sim.gen import gen_aperiodic
    from scipy.interpolate import interp1d

    n_channels = spectrum.shape[0]
    n_freqs = len(frequencies)

    freqs_linear = np.linspace(frequencies[0], frequencies[-1], n_freqs)
    freq_res = freqs_linear[1] - freqs_linear[0]
    # 2x frequency resolution as recommended; 12 Hz upper limit for neural oscillations
    fm.peak_width_limits = (2.0 * freq_res, 12.0)

    spectrum_peak = np.zeros((n_channels, n_freqs))
    spectrum_aper = np.zeros((n_channels, n_freqs))

    for ch in range(n_channels):
        spec_linear = interp1d(frequencies, spectrum[ch], kind='cubic',
                               fill_value='extrapolate')(freqs_linear)
        fm.fit(freqs_linear, spec_linear)

        init_ap_fit = gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
        init_flat_spec = fm.power_spectrum - init_ap_fit

        spectrum_peak[ch] = interp1d(freqs_linear, init_flat_spec, kind='cubic',
                                     fill_value='extrapolate')(frequencies)
        spectrum_aper[ch] = interp1d(freqs_linear, init_ap_fit, kind='cubic',
                                     fill_value='extrapolate')(frequencies)

    return spectrum_peak, spectrum_aper


# ============================================================================
# INDIVIDUAL-LEVEL PIPELINE
# ============================================================================

def run_individual_analysis(config):
    """
    Individual sensor-space pipeline: epochs -> TFR -> FOOOF -> CSD.
    Outputs are written to <output_folder>/<subject_name>/.
    """
    from fooof import FOOOF

    folder         = config['paths']['data_folder']
    output_folder  = config['paths'].get('output_folder', folder)
    subject_name   = config['subject']['subject_name']
    file_name      = config['subject']['file_name']
    results_folder = os.path.join(output_folder, subject_name)

    if os.path.exists(results_folder):
        print(f"Warning: Output folder for subject '{subject_name}' already exists. Files may be overwritten.")
    os.makedirs(results_folder, exist_ok=True)

    condition_1 = config['conditions']['condition_1']['name']
    condition_2 = config['conditions']['condition_2']['name']
    event_id_1  = config['conditions']['condition_1']['event_id']
    event_id_2  = config['conditions']['condition_2']['event_id']

    stim_channel = config['events']['stim_channel']
    ch_type      = config['channels']['meg_type']

    tmin = config['epoching']['tmin']
    tmax = config['epoching']['tmax']
    # YAML may parse values like 3000e-13 as strings
    reject_criteria = {k: float(v) for k, v in (config['epoching']['reject_criteria'] or {}).items()} or None
    flat_criteria   = {k: float(v) for k, v in (config['epoching']['flat_criteria'] or {}).items()} or None

    tf       = config['time_frequency']
    n_cycles = tf['n_cycles']
    decim    = tf['decim']
    n_jobs   = config['processing']['n_jobs']

    t_min_interest = config['time_window']['tmin']
    t_max_interest = config['time_window']['tmax']
    t_min_baseline = config['csd']['baseline_tmin']
    t_max_baseline = config['csd']['baseline_tmax']

    def out(name):
        return os.path.join(results_folder, f'{subject_name}_{name}')

    print(f"\n{'='*60}\nStarting Sensor Space Analysis for Subject: {subject_name}\n{'='*60}\n")
    os.chdir(folder)

    # STEP 1: Load preprocessed data
    print("Step 1: Loading preprocessed data...")
    raw_data = mne.io.read_raw_fif(os.path.join(folder, subject_name, file_name),
                                   preload=True, verbose=False)
    print(f"  ✓ Loaded: {file_name}")
    print(f"  ✓ Duration: {raw_data.times[-1]:.2f} seconds")
    print(f"  ✓ Channels: {len(raw_data.ch_names)} total")

    # STEP 2: Events
    print("\nStep 2: Extracting events...")
    events = mne.find_events(raw_data, stim_channel=stim_channel,
                             shortest_event=1, verbose=False)
    print(f"  ✓ Found {len(events)} events")
    print(f"  ✓ Condition '{condition_1}' (ID={event_id_1}): {np.sum(events[:, 2] == event_id_1)} events")
    print(f"  ✓ Condition '{condition_2}' (ID={event_id_2}): {np.sum(events[:, 2] == event_id_2)} events")

    # STEP 3: Epochs
    print("\nStep 3: Creating epochs...")
    epoch_kw = dict(tmin=tmin, tmax=tmax, reject=reject_criteria, flat=flat_criteria,
                    preload=True, picks=ch_type, baseline=None, verbose=False)
    epochs_1    = mne.Epochs(raw_data, events, event_id=event_id_1, **epoch_kw)
    epochs_2    = mne.Epochs(raw_data, events, event_id=event_id_2, **epoch_kw)
    epochs_full = mne.Epochs(raw_data, events, event_id=[event_id_1, event_id_2], **epoch_kw)
    epochs_1.save(out(f'{condition_1}_epochs-epo.fif'), overwrite=True)
    epochs_2.save(out(f'{condition_2}_epochs-epo.fif'), overwrite=True)
    epochs_full.save(out('ave_epochs-epo.fif'), overwrite=True)
    print(f"  ✓ Condition '{condition_1}': {len(epochs_1)} epochs")
    print(f"  ✓ Condition '{condition_2}': {len(epochs_2)} epochs")
    print(f"  ✓ Combined epochs: {len(epochs_full)} epochs")

    # STEP 4: Time-frequency
    print("\nStep 4: Computing time-frequency representations...")
    frequencies = np.logspace(tf['min_freq_log'], tf['max_freq_log'], num=tf['freq_resolution'])
    print(f"  ✓ Frequency range: {frequencies[0]:.2f} - {frequencies[-1]:.2f} Hz")
    powers = {}
    for cond, epochs in ((condition_1, epochs_1), (condition_2, epochs_2)):
        power, itc = epochs.compute_tfr(
            method="morlet", freqs=frequencies, n_cycles=n_cycles,
            decim=decim, n_jobs=n_jobs, return_itc=True, average=True, verbose=False)
        power.save(out(f'power_{cond}-tfr.h5'), overwrite=True)
        itc.save(out(f'itc_{cond}-tfr.h5'), overwrite=True)
        powers[cond] = power
        print(f"  ✓ Condition '{cond}' TFR computed")

    # STEP 5: FOOOF on the time window of interest
    print("\nStep 5: Applying FOOOF decomposition...")
    fm = FOOOF()
    for cond, power in powers.items():
        spectrum = np.mean(power.copy().crop(t_min_interest, t_max_interest).data, axis=2)
        peak, aper = process_fooof(spectrum, frequencies, fm, subject_name, cond, output_folder)
        np.save(out(f'{cond}_ped_crop.npy'), peak)
        np.save(out(f'{cond}_aper_crop.npy'), aper)
        print(f"  ✓ Condition '{cond}': {peak.shape[0]} channels")

    # STEP 6: CSD
    print("\nStep 6: Computing Cross-Spectral Density...")
    csd_kw = dict(n_cycles=n_cycles, decim=decim, n_jobs=n_jobs, verbose=False)
    for name, epochs, t0, t1 in ((condition_1, epochs_1, t_min_interest, t_max_interest),
                                 (condition_2, epochs_2, t_min_interest, t_max_interest),
                                 ('baseline', epochs_full, t_min_baseline, t_max_baseline)):
        csd = mne.time_frequency.csd_morlet(epochs, frequencies, tmin=t0, tmax=t1, **csd_kw)
        csd.save(out(f'{name}_csd.h5'), overwrite=True)
        print(f"  ✓ '{name}' CSD computed")

    print(f"\n{'='*60}\nAnalysis Complete for Subject: {subject_name}\n{'='*60}")
    print(f"Output files saved to: {results_folder}")


# ============================================================================
# GROUP-LEVEL STATISTICS
# ============================================================================

def merge_fooof_results(num_subjects, folder, condition_1, condition_2):
    """
    Stack per-subject FOOOF results (<folder>/S<i>/S<i>_<cond>_{ped,aper}_crop.npy).

    Returns
    -------
    ped_1, aper_1, ped_2, aper_2 : array, shape (n_subjects, n_channels, n_freqs)
    """
    print(f"Loading FOOOF results for {num_subjects} subjects...")
    data = {}
    for cond in (condition_1, condition_2):
        for kind in ('ped', 'aper'):
            data[cond, kind] = np.array([
                np.load(os.path.join(folder, f'S{i}', f'S{i}_{cond}_{kind}_crop.npy'))
                for i in range(1, num_subjects + 1)])
            np.save(os.path.join(folder, f'list_{cond}_{kind}_crop.npy'), data[cond, kind])

    print(f"  Condition '{condition_1}' periodic: {data[condition_1, 'ped'].shape}")
    print(f"  Condition '{condition_2}' periodic: {data[condition_2, 'ped'].shape}")
    return (data[condition_1, 'ped'], data[condition_1, 'aper'],
            data[condition_2, 'ped'], data[condition_2, 'aper'])


def perform_cluster_statistics(data_1, data_2, epochs_file, ch_type,
                               alpha, p_threshold, n_permutations,
                               tail, out_type):
    """
    Paired cluster permutation test (condition 2 - condition 1) over
    channels x frequencies.

    Returns
    -------
    T_obs, T_obs_plot (significant clusters only), clusters, cluster_p_values
    """
    print("\nPerforming cluster-based permutation test...")
    info = mne.read_epochs(epochs_file, preload=False, verbose=False).info
    adj, ch_names = mne.channels.find_ch_adjacency(info, ch_type=ch_type)
    print(f"  ✓ Channel adjacency computed for {len(ch_names)} channels")

    # (n_subjects, n_freqs, n_channels)
    diff = np.transpose(data_2, (0, 2, 1)) - np.transpose(data_1, (0, 2, 1))
    df = len(data_1) - 1
    t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)
    print(f"  ✓ T-threshold: {t_threshold:.3f} (df={df}, p={p_threshold})")

    T_obs, clusters, cluster_p_values, H0 = mne.stats.spatio_temporal_cluster_1samp_test(
        diff, out_type=out_type, adjacency=adj, n_permutations=n_permutations,
        threshold=t_threshold, tail=tail, verbose=False)

    significant = np.where(cluster_p_values < alpha)[0]
    print(f"  Clusters: {len(clusters)} | significant (p < {alpha}): {len(significant)}")
    for idx in significant:
        print(f"    Cluster {idx}: p = {cluster_p_values[idx]:.4f}")

    T_obs_plot = np.zeros_like(T_obs)
    for cluster, p_val in zip(clusters, cluster_p_values):
        if p_val <= alpha:
            T_obs_plot[cluster] = T_obs[cluster]

    return T_obs, T_obs_plot, clusters, cluster_p_values


def save_statistics_results(T_obs, T_obs_plot, clusters, cluster_p_values,
                            output_folder, condition_1, condition_2):
    """Save T maps, cluster p-values and a text summary to `output_folder`."""
    tag = f'{condition_1}_vs_{condition_2}'
    np.save(os.path.join(output_folder, f'T_obs_{tag}.npy'), T_obs)
    np.save(os.path.join(output_folder, f'T_obs_significant_{tag}.npy'), T_obs_plot)
    np.save(os.path.join(output_folder, f'cluster_p_values_{tag}.npy'), cluster_p_values)

    with open(os.path.join(output_folder, f'statistics_summary_{tag}.txt'), 'w') as f:
        f.write("Statistical Analysis Summary\n" + "=" * 50 + "\n\n")
        f.write(f"Comparison: {condition_2} vs {condition_1}\n\n")
        f.write(f"T-obs shape: {T_obs.shape}\n")
        f.write(f"Number of clusters: {len(clusters)}\n\nCluster p-values:\n")
        for i, p in enumerate(cluster_p_values):
            f.write(f"  Cluster {i}: p = {p:.6f}\n")

    print(f"  ✓ Statistical results saved to {output_folder}")


def run_group_statistics(config):
    """Group-level sensor statistics on the FOOOF periodic component."""
    folder        = config['paths']['data_folder']
    output_folder = config['paths'].get('output_folder', folder)
    condition_1   = config['conditions']['condition_1']['name']
    condition_2   = config['conditions']['condition_2']['name']
    subject_name  = config['subject']['subject_name']   # representative subject for channel info
    st            = config['statistics']

    print(f"\n{'='*60}\nStarting Group-Level Statistical Analysis\n{'='*60}")
    print(f"Subjects: {st['num_subjects']} | '{condition_1}' vs '{condition_2}' | "
          f"alpha={st['alpha']} | permutations={st['n_permutations']}\n")

    print("Step 1: Merging FOOOF results across subjects...")
    ped_1, _, ped_2, _ = merge_fooof_results(st['num_subjects'], output_folder,
                                             condition_1, condition_2)

    print("\nStep 2: Performing statistical analysis...")
    epochs_file = os.path.join(output_folder, subject_name,
                               f'{subject_name}_{condition_1}_epochs-epo.fif')
    T_obs, T_obs_plot, clusters, cluster_p_values = perform_cluster_statistics(
        ped_1, ped_2, epochs_file, st['ch_type'], st['alpha'], st['threshold'],
        st['n_permutations'], st['tail'], st['out_type'])

    print("\nStep 3: Saving results...")
    save_statistics_results(T_obs, T_obs_plot, clusters, cluster_p_values,
                            output_folder, condition_1, condition_2)

    return T_obs, T_obs_plot, clusters, cluster_p_values


# ============================================================================
# VISUAL INSPECTION
# ============================================================================

class VisualInspector:
    """Visual inspection / quality control of sensor-space results."""

    EVENT_COLOR = {100: 'g', 200: 'g', 101: 'g', 102: 'g',
                   128: 'r', 208: 'r', 152: 'r', 155: 'r',
                   110: 'g', 120: 'g', 104: 'g', 201: 'g',
                   202: 'g', 203: 'g', 204: 'g', 210: 'g',
                   220: 'g', 255: 'r', 103: 'g', 105: 'g', 205: 'g'}

    def __init__(self, config):
        self.config = config
        self.folder = config['paths']['data_folder']
        self.subject_name = config['subject']['subject_name']
        self.results_folder = os.path.join(config['paths'].get('output_folder', self.folder),
                                           self.subject_name)
        self.condition_1 = config['conditions']['condition_1']['name']
        self.condition_2 = config['conditions']['condition_2']['name']

    def inspect_raw_data(self, raw_data, events=None, title='Raw data'):
        if events is not None:
            raw_data.plot(events=events, title=title, event_color=self.EVENT_COLOR)
        else:
            raw_data.plot(title=title)

    def inspect_psd(self, raw_data, fmax=80, n_jobs=1):
        return raw_data.plot_psd(fmax=fmax, average=True, n_jobs=n_jobs)

    def inspect_ica_components(self, ica, title='ICA components'):
        ica.plot_components(sensors=True, colorbar=True, title=title, outlines='head')

    def inspect_ica_sources(self, ica, raw_data):
        ica.plot_sources(raw_data, show_scrollbars=False)
        print(f"ICA excluded components: {ica.exclude}")

    def inspect_epochs(self, epochs, title=None):
        epochs.plot(title=title or f"Epochs: {self.subject_name}")

    def inspect_evoked(self, evoked, condition_name, save=False):
        fig = evoked.plot(titles=f'Evoked data of {self.subject_name} for condition {condition_name}')
        if save:
            fig.savefig(os.path.join(self.results_folder,
                                     f'Evoked_{self.subject_name}_{condition_name}.png'))
        return fig

    def inspect_time_frequency(self, power, condition_name, baseline=None,
                               mode='logratio', combine='mean'):
        power.plot(combine=combine, title=f'{condition_name}')
        power.plot_joint(title=f'{condition_name}')
        power.plot_topo(baseline=baseline, mode=mode, title=f'{condition_name}')

    def inspect_fooof_fit(self, fm, spectrum, freqs, plt_log=False):
        from fooof.plts.spectra import plot_spectrum
        plot_spectrum(fm.freqs, spectrum.T)

    def inspect_statistics_results(self, T_obs, T_obs_plot, vmin=-5, vmax=5):
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        for a, data, title in ((ax[0], T_obs, 'All T-values'),
                               (ax[1], T_obs_plot, 'Significant T-values')):
            im = a.imshow(data, aspect='auto', origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
            a.set_title(title)
            a.set_xlabel('Time points')
            a.set_ylabel('Frequency bins')
            plt.colorbar(im, ax=a)
        plt.tight_layout()
        return fig

    def load_and_inspect_epochs(self, condition):
        epochs_file = os.path.join(self.results_folder,
                                   f'{self.subject_name}_{condition}_epochs-epo.fif')
        if not os.path.exists(epochs_file):
            print(f"Epochs file not found: {epochs_file}")
            return None
        epochs = mne.read_epochs(epochs_file, preload=True)
        self.inspect_epochs(epochs, title=f'{condition} condition')
        return epochs

    def load_and_inspect_power(self, condition):
        power_file = os.path.join(self.results_folder,
                                  f'{self.subject_name}_power_{condition}-tfr.h5')
        if not os.path.exists(power_file):
            print(f"Power file not found: {power_file}")
            return None
        power = mne.time_frequency.read_tfrs(power_file)[0]
        baseline = self.config['baseline']
        self.inspect_time_frequency(power, condition,
                                    baseline=(baseline['tmin'], baseline['tmax']),
                                    mode=self.config['time_frequency']['mode'])
        return power

    def create_comparison_report(self):
        print(f"\n=== Visual Inspection Report for {self.subject_name} ===\n")
        epochs_1 = self.load_and_inspect_epochs(self.condition_1)
        epochs_2 = self.load_and_inspect_epochs(self.condition_2)
        self.load_and_inspect_power(self.condition_1)
        self.load_and_inspect_power(self.condition_2)

        if epochs_1 is not None and epochs_2 is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            epochs_1.average().plot(axes=axes[0], show=False)
            axes[0].set_title(f'Condition {self.condition_1}')
            epochs_2.average().plot(axes=axes[1], show=False)
            axes[1].set_title(f'Condition {self.condition_2}')
            plt.tight_layout()
            plt.show()

        print("\n=== Inspection Complete ===\n")


# ============================================================================
# PUBLICATION FIGURES
# ============================================================================

def fig_3(folder, excel_with_beh_results):
    """Behavioural results: paired t-test, box/strip plot and Hedges's g."""
    import matplotlib
    import pandas as pd
    import seaborn as sns

    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)

    df = pd.read_excel('{}/{}.xlsx'.format(folder, excel_with_beh_results))
    results = df.to_numpy()
    Accuracy_S = results[:, 0]
    Accuracy_T = results[:, 1]

    t_test = stats.ttest_rel(Accuracy_T, Accuracy_S, nan_policy='propagate',
                             alternative='two-sided')
    print(t_test)

    beh = sns.boxplot(data=df, saturation=0.75, width=0.8, dodge=True,
                      fliersize=5, whis=1.5)
    beh = sns.stripplot(data=df, color='k')
    plt.savefig("Beh_results.png", format='png')

    # Hedges's g with pooled standard deviation
    dof = len(Accuracy_S) + len(Accuracy_T) - 2
    s_pooled = np.sqrt(((len(Accuracy_S) - 1) * np.var(Accuracy_S) +
                        (len(Accuracy_T) - 1) * np.var(Accuracy_T)) / dof)
    hedgess_g = abs(np.mean(Accuracy_S) - np.mean(Accuracy_T)) / s_pooled
    print(f"Hedges's g = {hedgess_g:.3f}")

    return t_test, hedgess_g, beh


def fig_4(num_subjects, condition_1, condition_2, T_obs, T_obs_plot,
          folder, vmin_4b, vmax_4b, vmax_4c, vmin_4c, freq_int, Rect):
    """Sensor-space results: grand-average TFR, T-value map and topomap."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    power_1_list = [mne.time_frequency.read_tfrs('S{}_power_{}-tfr.h5'.format(i + 1, condition_1))[0]
                    for i in range(num_subjects)]
    grand_average_1 = mne.grand_average(power_1_list, interpolate_bads=True, drop_bads=True)

    fig, ax = plt.subplots(2, 2)

    # FIGURE 4A
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    grand_average_1.plot(mode='logratio', axes=ax1, colorbar=True,
                         combine='mean', yscale='linear')
    ax1.set_title('Spatial power averaged across subjects', pad=30)
    if Rect:
        ax1.add_patch(plt.Rectangle((0.008, 3.3), 3.985, 77.5, ls="-", lw=1, ec="b", fc="none"))
    for x in (-7, -6, -4.5, -3, -1.5, 5):
        ax1.axvline(x=x, ymin=0, ymax=1, color="black", linestyle="--", linewidth=0.7)
    plt.yticks([13, 26, 39, 52, 66, 80], ['6', '10', '17', '28', '48', '80'])
    for x, y, s in ((-7.2, 87, 'Cue'), (-7.3, 82, 'Onset'),
                    (-6.2, 87, 'Item'), (-6.05, 82, '1'),
                    (-4.7, 87, 'Item'), (-4.55, 82, '2'),
                    (-3.2, 87, 'Item'), (-3.05, 82, '3'),
                    (-1.65, 87, 'Item'), (-1.55, 82, '4'),
                    (-0.45, 87, 'Retention'), (-0.2, 82, 'onset'),
                    (3.75, 87, 'Probe'), (3.75, 82, 'onset'),
                    (4.6, 87, 'Response'), (4.8, 82, 'onset')):
        ax1.text(x=x, y=y, s=s)

    # FIGURE 4B
    ax2 = plt.subplot2grid((2, 2), (1, 0), rowspan=1)
    ax2.add_patch(plt.Rectangle((130, 10.5), 60, 7, ls="--", lw=0.5, ec="k", fc="none"))
    plt.imshow(T_obs, aspect='auto', origin='lower', cmap='RdBu_r', vmin=vmin_4b, vmax=vmax_4b)
    plt.colorbar(label='T values')
    plt.contour(T_obs_plot, levels=1, colors='g', alpha=0.5,
                linewidths=[0.5], linestyles='solid', origin=None)
    plt.title('Frequency-spatial plot of T values for contrasting condition (temporal - spatial)')
    plt.yticks([0, 5, 10, 15, 20, 25, 29], ['4', '6', '10', '17', '28', '48', '80'])
    ax2.set_xlabel('Channels')
    ax2.set_ylabel('Frequency (Hz)')

    # FIGURE 4C
    os.chdir(folder)
    info = mne.io.read_raw_fif(folder + '/S1_filtered.fif', allow_maxshield=False,
                               preload=False, on_split_missing='raise').info
    what_to_analyze = np.mean(freq_int, axis=0)
    print(np.max(what_to_analyze), np.min(what_to_analyze))

    ax3 = plt.subplot2grid((2, 2), (1, 1), rowspan=1)
    mne.viz.plot_topomap(what_to_analyze, pos=info, ch_type='grad', image_interp='cubic',
                         cmap='Reds', axes=ax3, show=True, vlim=(-vmin_4c, vmax_4c))
    ax3.set_title('Topomap of T-values for significant frequencies')

    cax = fig.add_axes([0.9, 0.1, 0.008, 0.32])
    norm = Normalize(vmin=vmin_4c, vmax=vmax_4c)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=plt.colormaps["RdBu_r"]),
                        cax=cax, ticks=[vmin_4c, vmax_4c])
    cbar.outline.set_visible(True)
    cax.yaxis.set_ticks_position('right')
    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    cbar.set_ticklabels(["-3", "-2", "-1", "0", "1", "2", "3"])
    cbar.set_label('T values', rotation=90)

    plt.subplots_adjust(left=0.1, right=1, bottom=0.1, top=0.8, wspace=None)
    return fig
