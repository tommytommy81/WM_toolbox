"""
Example Usage: Connectivity Analysis
====================================

This script demonstrates how to configure and run all-to-all source-space
connectivity analysis on MEG data.

Requirements:
- Completed sensor space analysis (CSD files: <subject>_<condition>_csd.h5)
- FreeSurfer anatomical reconstruction (in subjects_dir)
- Coregistration transformation file (<subject>-trans.fif)
- conpy package: pip install conpy

The analysis itself is delegated to the functions in
STWM_functions_for_connectivity.py, all of which receive the single `config`
dictionary, exactly as in STWM_functions_core.py.

@author: Nikita Otstavnov, 2023 (refactored 2026)
"""

import os
import yaml

# ============================================================================
# CONFIGURATION
# ============================================================================

# Create configuration dictionary
config = {
    'paths': {
        'data_folder': '/path/to/your/data',    # Main folder with MEG data
        'output_folder': '/path/to/output',      # Output folder (optional, defaults to data_folder)
        'subjects_dir': '/path/to/freesurfer/subjects'  # FreeSurfer subjects directory
    },

    'subject': {
        'subject_name': 'S1',                    # Subject identifier
        'file_name': 'S1_preprocessed-raw.fif'   # Preprocessed MEG file
    },

    'conditions': {
        'condition_1': {
            'name': 'Spatial',                   # First condition name
        },
        'condition_2': {
            'name': 'Temporal',                  # Second condition name
        }
    },

    # Forward model parameters
    'forward_model': {
        'conductivity': [0.3],                   # BEM conductivity (single shell)
        'ico': 5,                                # BEM surface decimation
        'mindist': 0.0                           # Minimum distance from inner skull (mm)
    },

    # Connectivity-specific parameters
    'connectivity': {
        'spacing': 'oct6',                       # Source space: 'oct6', 'ico4', 'ico5'
        'max_sensor_dist': 0.07,                 # Max distance from sensors (m)
        'min_dist': 0.04,                        # Min distance between connected vertices (m)
        'regularization': 0.05,                  # Regularization for the inverse solution
        'freq_min': 8,                           # Minimum frequency (Hz)
        'freq_max': 12,                          # Maximum frequency (Hz)
        'num_subjects': 3,                       # Number of subjects (group-level steps)
        'average_ref_file': 'S1_filtered.fif',   # Reference recording for fsaverage forward info
        'average_trans_file': 'Av-trans.fif',    # Coregistration for the fsaverage template

        # Single-subject visualization
        'atlas': 'aparc',                        # Brain atlas: 'aparc', 'aparc.a2009s'
        'n_lines': 1000,                         # Number of connections to visualize
        'vmin': None,                            # Lower colorbar limit (None = auto)
        'vmax': None,                            # Upper colorbar limit (None = auto)
        'figure': None,                          # Existing figure handle (None = new figure)
        'size': 800,                             # Brain figure size (pixels)
        'borders': True,                         # Draw parcellation borders
        'hemi': 'both',                          # Hemisphere: 'lh', 'rh', 'both'

        # Group-level cluster-permutation statistics
        'cluster_threshold': 5.0,                # Cluster-forming threshold
        'n_permutations': 1000,                  # Number of permutations
        'alpha': 0.05,                           # Significance level
        'tail': 0,                               # Test tail: 0, 1, or -1
        'max_spread': 0.013,                     # Max spatial spread of a bundle (m)
        'seed': 42,                              # RNG seed
        'summary': 'sum',                        # Parcellation summary
        'brain_mode': 'absmax',                  # make_stc mode for rendering
        'hemi_stat': 'both',                     # Hemisphere for statistics rendering
        'views': ['lateral', 'medial'],          # Views for statistics rendering

        # Statistics visualization
        'regexp': None,                          # Label-selection regular expression
        'weight_by_degree': False,               # Weight parcellation by node degree
        'n_lines_stat': 100,                     # Connections in statistics circle plot
        'vmin_stat': None,                       # Lower colorbar limit for statistics
        'vmax_stat': None,                       # Upper colorbar limit for statistics
        'fontsize_names': 8,                     # Font size for label names
        'fontsize_colorbar': 8                   # Font size for colorbar
    },

    # Processing parameters
    'processing': {
        'n_jobs': 4                              # Number of parallel jobs
    }
}

# ============================================================================
# SAVE CONFIGURATION
# ============================================================================

# Save configuration to YAML file
config_file = 'config_connectivity.yaml'
with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Configuration saved to: {config_file}")
print("\nTo run connectivity analysis for one subject, use:")
print(f"python connectivity_individual_analysis.py --config {config_file} --subject S1")

# ============================================================================
# IMPORTANT NOTES
# ============================================================================

print("\n" + "="*70)
print("IMPORTANT SETUP STEPS:")
print("="*70)
print("""
1. Install conpy package:
   pip install conpy
   (alternative: conda install -c conda-forge conpy)

2. Prerequisites:
   - Run sensor_space_individual_analysis.py first to generate the CSD files:
     <subject>_<condition_1>_csd.h5  and  <subject>_<condition_2>_csd.h5
   - FreeSurfer reconstruction completed
   - Coregistration file (<subject>-trans.fif) in the data folder

3. Connectivity Analysis Workflow:

   Group-level prerequisites (run ONCE for the whole sample):
     src_average(config)          -> fsaverage template source space
     new_morphing(config)         -> morph the template onto every subject
     new_morphed_forward_model()  -> forward model per subject
     pairs_identification(config) -> shared vertices + Average-pairs.npy

   Individual-level analysis (connectivity_individual_analysis.py):
     Step 1: new_morphing            -> subject source space
     Step 2: new_morphed_forward_model -> subject forward model
     Step 3: connectivity_estimation -> DICS connectivity per condition
     Step 4: connectivity_vizualization -> contrast circle plot + brain

   Group-level statistics (run ONCE across subjects):
     connectivity_statistics(config)
     connectivity_statistics_visualization(config)

4. Connectivity Method (DICS):
   - Dynamic Imaging of Coherent Sources
   - Functional connectivity in source space from CSD matrices
   - Captures frequency-specific interactions

5. Key parameters to adjust:
   max_sensor_dist (0.07 m): only sources visible to sensors
   min_dist (0.04 m):        minimum distance between connected vertices
   regularization (0.05):    noise sensitivity of the inverse solution

6. Frequency bands for connectivity:
   Theta (4-8 Hz), Alpha (8-12 Hz), Beta (12-30 Hz), Gamma (30-80 Hz)
""")

# ============================================================================
# EXAMPLE: Group-Level Workflow
# ============================================================================

print("\n" + "="*70)
print("GROUP-LEVEL WORKFLOW (run once for the whole sample):")
print("="*70)

example_group = '''
import yaml
from STWM_functions_for_connectivity import (
    src_average, new_morphing, new_morphed_forward_model,
    pairs_identification, connectivity_statistics,
    connectivity_statistics_visualization)

with open('config_connectivity.yaml') as f:
    config = yaml.safe_load(f)

subjects = ['S1', 'S2', 'S3']

# Step 1: build the fsaverage template source space (once)
src_average(config)

# Step 2: morph + forward model for every subject
for subject in subjects:
    config['subject']['subject_name'] = subject
    config['subject']['file_name']    = f'{subject}_preprocessed-raw.fif'
    new_morphing(config)
    new_morphed_forward_model(config)

# Step 3: identify the common connectivity pairs (once)
pairs_identification(config)

# Step 4: estimate connectivity for every subject
import subprocess
for subject in subjects:
    subprocess.run(
        f'python connectivity_individual_analysis.py '
        f'--config config_connectivity.yaml --subject {subject}',
        shell=True)

# Step 5: group statistics and visualization
connectivity_statistics(config)
connectivity_statistics_visualization(config)
'''

print(example_group)

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

print("\n" + "="*70)
print("TROUBLESHOOTING:")
print("="*70)
print("""
1. "conpy not found":
   pip install conpy  (or conda install -c conda-forge conpy)

2. "CSD files not found":
   Run sensor_space_individual_analysis.py first; keep the CSD files in the
   data folder.

3. "Transformation file not found":
   Run coregistration (mne.gui.coregistration()) and save as
   <subject>-trans.fif in the data folder.

4. "No vertices in sensor range":
   Increase max_sensor_dist (e.g. 0.07 -> 0.10) and check coregistration.

5. "Average-pairs.npy not found":
   Run the group-level pairs_identification(config) step first.

6. "Memory error":
   Use a coarser source space (ico4), reduce n_jobs, or fewer pairs.
""")

print("\n" + "="*70)
print("For more information:")
print("- conpy documentation: https://aaltoimaginglanguage.github.io/conpy/")
print("- MNE connectivity: https://mne.tools/stable/auto_tutorials/connectivity/")
print("="*70)
