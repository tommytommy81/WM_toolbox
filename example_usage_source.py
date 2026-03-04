"""
Example Usage: Source Space Analysis
====================================

This script demonstrates how to run source space analysis on MEG data.

Requirements:
- Preprocessed MEG data (.fif file)
- FreeSurfer anatomical reconstruction (in subjects_dir)
- Coregistration transformation file (*-trans.fif)

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
        'output_folder': '/path/to/output',    # Output folder (optional, defaults to data_folder)
        'subjects_dir': '/path/to/freesurfer/subjects'  # FreeSurfer subjects directory
    },
    
    'subject': {
        'name': 'S1',                          # Subject identifier
        'file_name': 'S1_preprocessed-raw.fif'  # Preprocessed MEG file
    },
    
    'conditions': {
        'condition_1': {
            'name': 'Spatial',                 # First condition name
            'event_id': 26                     # Event ID for condition 1
        },
        'condition_2': {
            'name': 'Temporal',                # Second condition name
            'event_id': 46                     # Event ID for condition 2
        },
        'baseline': {
            'name': 'Baseline',                # Baseline condition
            'event_id': None                   # If separate baseline exists
        }
    },
    
    # Source space parameters
    'source_space': {
        'spacing': 'ico4',                     # Source space decimation: 'ico4', 'ico5', 'oct6'
        'surfaces': 'white',                   # Brain surface: 'white', 'pial'
        'orientation': 'coronal'               # Slice orientation for visualization
    },
    
    # Forward model parameters
    'forward_model': {
        'conductivity': [0.3],                 # BEM conductivity (single shell)
        'ico': 5,                              # BEM surface decimation
        'mindist': 5.0,                        # Minimum distance from inner skull (mm)
        'surfaces': 'white',                   # Surface for source space
        'coord_frame': 'mri'                   # Coordinate frame: 'mri' or 'head'
    },
    
    # Source estimate parameters
    'source_estimate': {
        'freq_min': 8,                         # Minimum frequency (Hz)
        'freq_max': 12,                        # Maximum frequency (Hz)
        'orientation': 'fix',                  # Orientation constraint: 'fix', 'free', 'loose'
        'surf_ori': True,                      # Orient dipoles perpendicular to cortex
        'force_fixed': True,                   # Force fixed orientation
        'reg': 0.05,                          # Regularization parameter
        'depth': 1.0,                         # Depth weighting
        'method': 'dSPM'                      # Inversion method: 'MNE', 'dSPM', 'sLORETA'
    },
    
    # Visualization parameters
    'visualization': {
        'hemi': 'both',                        # Hemisphere: 'lh', 'rh', 'both'
        'surface': 'inflated',                 # Surface type: 'inflated', 'pial', 'white'
        'views': ['dorsal', 'lateral', 'medial', 'ventral']  # View angles
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
config_file = 'config_source.yaml'
with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Configuration saved to: {config_file}")
print("\nTo run source space analysis, use:")
print(f"python source_space_individual_analysis.py --config {config_file} --subject S1")

# ============================================================================
# IMPORTANT NOTES
# ============================================================================

print("\n" + "="*70)
print("IMPORTANT SETUP STEPS:")
print("="*70)
print("""
1. FreeSurfer Reconstruction:
   - Run FreeSurfer reconstruction on your subject's MRI:
     recon-all -s S1 -i /path/to/MRI.nii -all
   - This creates anatomical surfaces needed for source localization

2. Coregistration:
   - Align MEG sensors with MRI anatomy:
     mne.gui.coregistration(subject='S1', subjects_dir=subjects_dir)
   - Save transformation as: S1-trans.fif
   - This file must be in your data folder

3. Data Requirements:
   - Preprocessed MEG data (filtered, ICA applied, bad channels marked)
   - Events/triggers properly coded
   - Baseline condition identified (if applicable)

4. Frequency Bands:
   Common choices:
   - Delta: 1-4 Hz
   - Theta: 4-8 Hz
   - Alpha: 8-12 Hz
   - Beta: 12-30 Hz
   - Gamma: 30-80 Hz

5. Source Space Spacing:
   - ico4: ~2562 sources per hemisphere (faster, less detail)
   - ico5: ~10242 sources per hemisphere (slower, more detail)
   - oct6: ~4098 sources per hemisphere (octahedral grid)

6. Regularization (reg parameter):
   - Controls noise sensitivity
   - Typical range: 0.01 to 0.1
   - Lower values: more sensitive, noisier
   - Higher values: smoother, less detail

7. Coordinate Frames:
   - 'mri': MRI coordinate system (for source space)
   - 'head': MEG coordinate system (for sensors)
   - Transformation links these two systems
""")

# ============================================================================
# EXAMPLE: Run for Multiple Subjects
# ============================================================================

print("\n" + "="*70)
print("BATCH PROCESSING EXAMPLE:")
print("="*70)

example_batch = '''
# Process multiple subjects
subjects = ['S1', 'S2', 'S3', 'S4', 'S5']

for subject in subjects:
    print(f"\\nProcessing {subject}...")
    config['subject']['name'] = subject
    config['subject']['file_name'] = f'{subject}_preprocessed-raw.fif'
    
    # Save subject-specific config
    config_file = f'config_{subject}.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    # Run analysis
    import subprocess
    cmd = f'python source_space_individual_analysis.py --config {config_file}'
    subprocess.run(cmd, shell=True)
'''

print(example_batch)

# ============================================================================
# EXAMPLE: Checking Results
# ============================================================================

print("\n" + "="*70)
print("CHECKING RESULTS:")
print("="*70)

example_check = '''
import mne

# Load and inspect source estimate
stc = mne.read_source_estimate('S1_Spatial_8-12Hz-lh.stc')

print("Source Estimate Information:")
print(f"  Shape: {stc.data.shape}")  # (n_vertices, n_times)
print(f"  Time points: {len(stc.times)}")
print(f"  Time range: {stc.times[0]:.3f} - {stc.times[-1]:.3f} s")
print(f"  Peak activation: {stc.data.max():.2e}")

# Visualize
brain = stc.plot(
    subject='S1',
    subjects_dir='/path/to/freesurfer/subjects',
    hemi='both',
    surface='inflated',
    time_viewer=True
)
'''

print(example_check)

print("\n" + "="*70)
print("For more information, see:")
print("- MNE-Python documentation: https://mne.tools/")
print("- Source estimation tutorial: https://mne.tools/stable/auto_tutorials/inverse/")
print("="*70)
