"""
Example Usage: Connectivity Analysis
====================================

This script demonstrates how to run connectivity analysis on MEG data.

Requirements:
- Completed sensor space analysis (CSD files)
- FreeSurfer anatomical reconstruction
- Coregistration transformation file
- conpy package: pip install conpy

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
        'data_folder': '/path/to/your/data',  # Main folder with MEG data
        'output_folder': '/path/to/output',    # Output folder (optional)
        'subjects_dir': '/path/to/freesurfer/subjects'  # FreeSurfer subjects directory
    },
    
    'subject': {
        'name': 'S1',                          # Subject identifier
        'file_name': 'S1_preprocessed-raw.fif'  # Preprocessed MEG file
    },
    
    'conditions': {
        'condition_1': {
            'name': 'Spatial',                 # First condition name
        },
        'condition_2': {
            'name': 'Temporal',                # Second condition name
        }
    },
    
    # Connectivity-specific parameters
    'connectivity': {
        'spacing': 'oct6',                     # Source space: 'oct6', 'ico4', 'ico5'
        'max_sensor_dist': 0.07,               # Max distance from sensors (m)
        'min_dist': 0.04,                      # Min distance between connected vertices (m)
        'regularization': 0.05,                # Regularization for inverse solution
        'freq_min': 8,                         # Minimum frequency (Hz)
        'freq_max': 12,                        # Maximum frequency (Hz)
        'atlas': 'aparc',                      # Brain atlas: 'aparc', 'aparc.a2009s'
        'n_lines': 1000                        # Number of connections to visualize
    },
    
    # Forward model parameters
    'forward_model': {
        'conductivity': [0.3],                 # BEM conductivity
        'ico': 5,                              # BEM surface decimation
        'mindist': 0.0                         # Minimum distance from inner skull (mm)
    },
    
    # Processing parameters
    'processing': {
        'n_jobs': 4                            # Number of parallel jobs
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
print("\nTo run connectivity analysis, use:")
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
   
   Alternative installation:
   conda install -c conda-forge conpy

2. Prerequisites:
   - Run sensor space analysis first to generate CSD files:
     - S1_Spatial_csd.h5
     - S1_Temporal_csd.h5
   - FreeSurfer reconstruction completed
   - Coregistration file (S1-trans.fif) in data folder

3. Connectivity Analysis Workflow:
   
   Step 1: Create fsaverage source space (done once)
   - Sets up template brain with vertices
   
   Step 2: Morph to subject anatomy
   - Maps template vertices to subject's brain
   
   Step 3: Create forward model
   - Links source space to MEG sensors
   
   Step 4: Identify connectivity pairs
   - Selects vertex pairs for connectivity estimation
   - Usually done once across all subjects
   
   Step 5: Estimate connectivity
   - Computes DICS beamformer connectivity
   - Separate for each condition

4. Connectivity Method (DICS):
   - Dynamic Imaging of Coherent Sources
   - Estimates functional connectivity in source space
   - Uses cross-spectral density (CSD) matrices
   - Captures frequency-specific interactions

5. Parameters to Adjust:
   
   max_sensor_dist (0.07 m):
   - Only includes sources visible to sensors
   - Smaller = fewer but more reliable sources
   - Larger = more coverage but noisier
   
   min_dist (0.04 m):
   - Minimum distance between connected vertices
   - Prevents spurious local connections
   - Typical range: 0.03-0.05 m
   
   regularization (0.05):
   - Controls noise sensitivity in inverse solution
   - Lower (0.01): more detail, more noise
   - Higher (0.1): smoother, less detail

6. Source Space Choice:
   - oct6: ~4098 vertices, good balance
   - ico4: ~2562 vertices, faster
   - ico5: ~10242 vertices, detailed but slower

7. Frequency Bands for Connectivity:
   Common choices:
   - Theta (4-8 Hz): Memory, attention
   - Alpha (8-12 Hz): Attention, inhibition
   - Beta (12-30 Hz): Motor, cognitive control
   - Gamma (30-80 Hz): Local processing
""")

# ============================================================================
# EXAMPLE: Group-Level Analysis
# ============================================================================

print("\n" + "="*70)
print("GROUP-LEVEL ANALYSIS WORKFLOW:")
print("="*70)

example_group = '''
# Step 1: Identify connectivity pairs (do this ONCE for all subjects)
# This creates a common set of vertex pairs across subjects

import numpy as np
import mne
import conpy

subjects = ['S1', 'S2', 'S3', 'S4', 'S5']
spacing = 'oct6'
subjects_dir = '/path/to/freesurfer/subjects'
folder = '/path/to/data'

# Load fsaverage source space
fsaverage = mne.read_source_spaces(f'Sub_for_con_Avg-{spacing}-src.fif')

# Load all forward models
fwds = []
for subj in subjects:
    fwd = mne.read_forward_solution(f'{subj}-for_con-morphed-fwd.fif')
    fwd_tan = conpy.forward_to_tangential(fwd)
    fwds.append(fwd_tan)

# Identify shared vertices across subjects
vert_inds = conpy.select_shared_vertices(fwds, ref_src=fsaverage,
                                         subjects_dir=subjects_dir)

# Create connectivity pairs
pairs = conpy.all_to_all_connectivity_pairs(fwds[0], min_dist=0.04)

# Save pairs for all subjects to use
np.save('Average-pairs.npy', pairs)

print(f"Created {len(pairs[0])} connectivity pairs")

# Step 2: Compute connectivity for each subject (run for each subject)
for subject in subjects:
    cmd = f'python connectivity_individual_analysis.py --config config_connectivity.yaml --subject {subject}'
    import subprocess
    subprocess.run(cmd, shell=True)

# Step 3: Load and compare connectivity across subjects
connectivity_spatial = []
connectivity_temporal = []

for subject in subjects:
    con_s = np.load(f'{subject}_Spatial_connectivity.npy')
    con_t = np.load(f'{subject}_Temporal_connectivity.npy')
    connectivity_spatial.append(con_s)
    connectivity_temporal.append(con_t)

# Convert to arrays
connectivity_spatial = np.array(connectivity_spatial)   # (n_subjects, n_pairs)
connectivity_temporal = np.array(connectivity_temporal)

# Perform statistics (e.g., paired t-test)
from scipy import stats
t_vals, p_vals = stats.ttest_rel(connectivity_spatial, connectivity_temporal, axis=0)

# Multiple comparisons correction
from mne.stats import fdr_correction
_, p_corrected = fdr_correction(p_vals)

# Find significant connections
sig_connections = p_corrected < 0.05
print(f"Found {sig_connections.sum()} significant connections")
'''

print(example_group)

# ============================================================================
# EXAMPLE: Visualization
# ============================================================================

print("\n" + "="*70)
print("VISUALIZATION EXAMPLE:")
print("="*70)

example_viz = '''
import numpy as np
import mne
from mne.viz import circular_layout, plot_connectivity_circle
import matplotlib.pyplot as plt

# Load connectivity results
subject = 'S1'
con_spatial = np.load(f'{subject}_Spatial_connectivity.npy')
con_temporal = np.load(f'{subject}_Temporal_connectivity.npy')
pairs = np.load('Average-pairs.npy')

# Load source space for label information
fsaverage = mne.read_source_spaces('Sub_for_con_Avg-oct6-src.fif')

# Load atlas labels
labels = mne.read_labels_from_annot('fsaverage', parc='aparc',
                                    subjects_dir='/path/to/freesurfer/subjects')

# Create connectivity matrix from pairs and values
n_vertices = len(fsaverage[0]['vertno']) + len(fsaverage[1]['vertno'])
conn_matrix = np.zeros((n_vertices, n_vertices))
for i, (v1, v2) in enumerate(zip(pairs[0], pairs[1])):
    conn_matrix[v1, v2] = con_spatial[i]
    conn_matrix[v2, v1] = con_spatial[i]  # Make symmetric

# Visualize as circular plot
label_names = [label.name for label in labels]
label_colors = [label.color for label in labels]

node_order = list(range(len(label_names)))
node_angles = circular_layout(label_names, node_order, start_pos=90)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
plot_connectivity_circle(conn_matrix, label_names, n_lines=100,
                        node_angles=node_angles, node_colors=label_colors,
                        title='Spatial Condition Connectivity', ax=ax)
plt.show()

# Or visualize on brain surface
brain = mne.viz.plot_connectom(
    con_spatial[:1000],  # Plot top 1000 connections
    pairs,
    pos=fsaverage[0]['rr'][fsaverage[0]['vertno']],  # Vertex positions
    node_size=5,
    edge_cmap='RdBu_r',
    subjects_dir='/path/to/freesurfer/subjects'
)
'''

print(example_viz)

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

print("\n" + "="*70)
print("TROUBLESHOOTING:")
print("="*70)
print("""
Common Issues:

1. "conpy not found":
   - Install: pip install conpy
   - Or: conda install -c conda-forge conpy

2. "CSD files not found":
   - Run sensor_space_individual_analysis.py first
   - Make sure CSD files are in the data folder

3. "Transformation file not found":
   - Run coregistration: mne.gui.coregistration()
   - Save as: S1-trans.fif in data folder

4. "No vertices in sensor range":
   - Increase max_sensor_dist (e.g., from 0.07 to 0.10)
   - Check coregistration quality

5. "Memory error":
   - Use coarser source space (ico4 instead of ico5)
   - Reduce n_jobs
   - Process fewer connectivity pairs

6. "Forward solution fails":
   - Check FreeSurfer reconstruction completed
   - Verify BEM surfaces exist
   - Check transformation file is correct

7. "Too few connectivity pairs":
   - Decrease min_dist (e.g., from 0.04 to 0.03)
   - Use denser source space (ico5 instead of ico4)
   - Check that subjects have overlapping coverage
""")

print("\n" + "="*70)
print("For more information:")
print("- conpy documentation: https://aaltoimaginglanguage.github.io/conpy/")
print("- MNE connectivity: https://mne.tools/stable/auto_tutorials/connectivity/")
print("="*70)
