"""
Visual Inspection Module for MEG Sensor Space Analysis
========================================================

This module contains all visualization and quality control functions
that can be run independently for manual inspection of the data.

These functions should be used AFTER the main analysis pipeline 
to verify data quality and inspect results.

@author: Nikita Otstavnov, 2023 (refactored 2026)
"""

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from fooof.plts.spectra import plot_spectrum, plot_spectra


class VisualInspector:
    """
    Class to handle all visual inspection operations for MEG data.
    """
    
    def __init__(self, config):
        """
        Initialize the visual inspector with configuration.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing paths and parameters
        """
        self.config = config
        self.folder = config['paths']['data_folder']
        self.subject_name = config['subject']['name']
        self.condition_1 = config['conditions']['condition_1']['name']
        self.condition_2 = config['conditions']['condition_2']['name']
        
    def inspect_raw_data(self, raw_data, events=None, title='Raw data'):
        """
        Plot raw MEG data for visual inspection.
        
        Parameters
        ----------
        raw_data : mne.io.Raw
            Raw MEG data object
        events : array, optional
            Events array for marking on the plot
        title : str
            Title for the plot
        """
        if events is not None:
            event_color = {100: 'g', 200: 'g', 101: 'g', 102: 'g', 
                          128: 'r', 208: 'r', 152: 'r', 155: 'r', 
                          110: 'g', 120: 'g', 104: 'g', 201: 'g', 
                          202: 'g', 203: 'g', 204: 'g', 210: 'g', 
                          220: 'g', 255: 'r', 103: 'g', 105: 'g', 205: 'g'}
            raw_data.plot(events=events, title=title, event_color=event_color)
        else:
            raw_data.plot(title=title)
    
    def inspect_psd(self, raw_data, fmax=80, n_jobs=1):
        """
        Plot power spectral density of raw data.
        
        Parameters
        ----------
        raw_data : mne.io.Raw
            Raw MEG data object
        fmax : float
            Maximum frequency to display
        n_jobs : int
            Number of parallel jobs
        """
        fig = raw_data.plot_psd(fmax=fmax, average=True, n_jobs=n_jobs)
        return fig
    
    def inspect_ica_components(self, ica, title='ICA components'):
        """
        Plot ICA components for visual inspection.
        
        Parameters
        ----------
        ica : mne.preprocessing.ICA
            Fitted ICA object
        title : str
            Title for the plot
        """
        ica.plot_components(sensors=True, colorbar=True, 
                           title=title, outlines='head')
    
    def inspect_ica_sources(self, ica, raw_data):
        """
        Plot ICA sources for component selection.
        
        Parameters
        ----------
        ica : mne.preprocessing.ICA
            Fitted ICA object
        raw_data : mne.io.Raw
            Raw MEG data object
        """
        ica.plot_sources(raw_data, show_scrollbars=False)
        print(f"ICA excluded components: {ica.exclude}")
    
    def inspect_epochs(self, epochs, title=None):
        """
        Plot epochs for visual inspection and rejection.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Epochs object
        title : str, optional
            Title for the plot
        """
        if title is None:
            title = f"Epochs: {self.subject_name}"
        epochs.plot(title=title)
    
    def inspect_evoked(self, evoked, condition_name, save=False):
        """
        Plot evoked response for a condition.
        
        Parameters
        ----------
        evoked : mne.Evoked
            Evoked response object
        condition_name : str
            Name of the condition
        save : bool
            Whether to save the figure
        """
        title = f'Evoked data of {self.subject_name} for condition {condition_name}'
        fig = evoked.plot(titles=title)
        
        if save:
            os.chdir(self.folder)
            fig.savefig(f'Evoked_{self.subject_name}_{condition_name}.png')
        
        return fig
    
    def inspect_time_frequency(self, power, condition_name, 
                               baseline=None, mode='logratio', 
                               combine='mean'):
        """
        Visualize time-frequency representation.
        
        Parameters
        ----------
        power : mne.time_frequency.AverageTFR
            Time-frequency power object
        condition_name : str
            Name of the condition
        baseline : tuple or None
            Baseline period (tmin, tmax)
        mode : str
            Baseline mode
        combine : str
            How to combine channels ('mean', 'median', etc.)
        """
        # Simple mean plot
        power.plot(combine=combine, title=f'{condition_name}')
        
        # Joint plot with topomaps
        power.plot_joint(title=f'{condition_name}')
        
        # Topographic representation
        power.plot_topo(baseline=baseline, mode=mode, 
                       title=f'{condition_name}')
    
    def inspect_fooof_fit(self, fm, spectrum, freqs, plt_log=False):
        """
        Plot FOOOF model fit for a spectrum.
        
        Parameters
        ----------
        fm : FOOOF
            Fitted FOOOF model
        spectrum : array
            Power spectrum data
        freqs : array
            Frequency values
        plt_log : bool
            Whether to use log scale
        """
        plot_spectrum(fm.freqs, spectrum.T)
    
    def inspect_statistics_results(self, T_obs, T_obs_plot, 
                                   vmin=-5, vmax=5):
        """
        Visualize statistical test results.
        
        Parameters
        ----------
        T_obs : array
            Observed T-values
        T_obs_plot : array
            T-values with only significant clusters
        vmin, vmax : float
            Color scale limits
        """
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot all T-values
        im1 = ax[0].imshow(T_obs, aspect='auto', origin='lower', 
                          cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax[0].set_title('All T-values')
        ax[0].set_xlabel('Time points')
        ax[0].set_ylabel('Frequency bins')
        plt.colorbar(im1, ax=ax[0])
        
        # Plot significant T-values only
        im2 = ax[1].imshow(T_obs_plot, aspect='auto', origin='lower', 
                          cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax[1].set_title('Significant T-values')
        ax[1].set_xlabel('Time points')
        ax[1].set_ylabel('Frequency bins')
        plt.colorbar(im2, ax=ax[1])
        
        plt.tight_layout()
        return fig
    
    def load_and_inspect_epochs(self, condition):
        """
        Load and inspect epochs for a given condition.
        
        Parameters
        ----------
        condition : str
            Condition name ('S' or 'T')
        """
        os.chdir(self.folder)
        epochs_file = f'{self.subject_name}_{condition}_epochs-epo.fif'
        
        if os.path.exists(epochs_file):
            epochs = mne.read_epochs(epochs_file, preload=True)
            self.inspect_epochs(epochs, title=f'{condition} condition')
            return epochs
        else:
            print(f"Epochs file not found: {epochs_file}")
            return None
    
    def load_and_inspect_power(self, condition):
        """
        Load and inspect time-frequency power for a given condition.
        
        Parameters
        ----------
        condition : str
            Condition name ('S' or 'T')
        """
        os.chdir(self.folder)
        power_file = f'{self.subject_name}_power_{condition}-tfr.h5'
        
        if os.path.exists(power_file):
            power = mne.time_frequency.read_tfrs(power_file)[0]
            baseline = self.config['baseline']
            mode = self.config['time_frequency']['mode']
            self.inspect_time_frequency(power, condition, 
                                       baseline=(baseline['tmin'], baseline['tmax']),
                                       mode=mode)
            return power
        else:
            print(f"Power file not found: {power_file}")
            return None
    
    def create_comparison_report(self):
        """
        Create a comprehensive visual report comparing both conditions.
        """
        print(f"\n=== Visual Inspection Report for {self.subject_name} ===\n")
        
        # Load data for both conditions
        print("Loading epochs...")
        epochs_1 = self.load_and_inspect_epochs(self.condition_1)
        epochs_2 = self.load_and_inspect_epochs(self.condition_2)
        
        # Load power data
        print("Loading time-frequency data...")
        power_1 = self.load_and_inspect_power(self.condition_1)
        power_2 = self.load_and_inspect_power(self.condition_2)
        
        # Compare evoked responses
        if epochs_1 is not None and epochs_2 is not None:
            print("Creating evoked response comparison...")
            evoked_1 = epochs_1.average()
            evoked_2 = epochs_2.average()
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            evoked_1.plot(axes=axes[0], show=False)
            axes[0].set_title(f'Condition {self.condition_1}')
            evoked_2.plot(axes=axes[1], show=False)
            axes[1].set_title(f'Condition {self.condition_2}')
            plt.tight_layout()
            plt.show()
        
        print("\n=== Inspection Complete ===\n")


def run_visual_inspection_pipeline(config_path='config.yaml'):
    """
    Run a complete visual inspection pipeline.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    """
    import yaml
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create inspector
    inspector = VisualInspector(config)
    
    # Run inspection
    inspector.create_comparison_report()


if __name__ == "__main__":
    # Example usage
    print("Visual Inspection Module")
    print("=" * 50)
    print("\nThis module provides tools for visual inspection of MEG data.")
    print("Import this module in your analysis scripts or run interactively.")
    print("\nExample usage:")
    print("  from visual_inspection import VisualInspector")
    print("  inspector = VisualInspector(config)")
    print("  inspector.inspect_raw_data(raw_data, events)")
