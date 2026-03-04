# MEG Analysis Pipeline for Spatial-Temporal Working Memory

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MNE](https://img.shields.io/badge/MNE-1.4+-orange.svg)](https://mne.tools/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Overview

This repository contains a comprehensive, modular pipeline for analyzing MEG (Magnetoencephalography) data with a focus on frequency-domain analysis across experimental conditions. The pipeline supports sensor space, source space, and connectivity analyses.

**Key Features:**
- ✅ **Configuration-driven**: All parameters in YAML files
- ✅ **Modern MNE API**: Uses latest MNE-Python functions
- ✅ **Modular design**: Reusable, maintainable components
- ✅ **Batch processing**: Easy to run on multiple subjects
- ✅ **FOOOF integration**: Separate periodic and aperiodic components
- ✅ **Three analysis levels**: Sensor, source, and connectivity

---

## 📁 Repository Structure

```
.
├── README.md                                # This file
├── requirements.txt                         # Python dependencies
├── config.yaml                              # Main configuration file
│
├── Core Analysis Scripts
│   ├── sensor_space_individual_analysis.py      # Sensor-level analysis
│   ├── sensor_space_group_statistics.py         # Sensor-level statistics
│   ├── source_space_individual_analysis.py      # Source localization
│   └── connectivity_individual_analysis.py      # Connectivity estimation
│
├── Example Usage & Guides
│   ├── example_usage_sensor.py                  # Sensor analysis guide
│   ├── example_usage_source.py                  # Source analysis guide
│   └── example_usage_connectivity.py            # Connectivity guide
│
├── Core Functions
│   ├── STWM_functions_core.py                   # Core processing functions
│   └── STWM - functions for connectivity        # Connectivity utilities
│
├── Utilities
│   ├── visual_inspection.py                     # Optional QC visualization
│   ├── activate_env.sh                          # Environment activation
│   └── run_batch_analysis.sh                    # Batch processing script
│
└── Legacy Code (for reference)
    └── WMspatiotemporal/                        # Original analysis scripts
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/SpatialTemporalWorkingMemoryMEG.git
cd SpatialTemporalWorkingMemoryMEG-main

# Create and activate virtual environment
python -m venv meg_env
source meg_env/bin/activate  # On Windows: meg_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Your Analysis

Edit `config.yaml` with your data paths and parameters:

```yaml
paths:
  data_folder: '/path/to/your/data'
  output_folder: '/path/to/output'
  subjects_dir: '/path/to/freesurfer/subjects'  # For source/connectivity analysis

subject:
  name: 'S1'
  file_name: 'S1_preprocessed-raw.fif'

conditions:
  condition_1:
    name: 'Spatial'
    event_id: 26
  condition_2:
    name: 'Temporal'
    event_id: 46
```

### 3. Run Analysis

```bash
# Sensor space analysis
python sensor_space_individual_analysis.py --config config.yaml --subject S1

# Source space analysis (requires FreeSurfer reconstruction)
python source_space_individual_analysis.py --config config.yaml --subject S1

# Connectivity analysis (requires conpy package)
python connectivity_individual_analysis.py --config config.yaml --subject S1
```

---

## 📊 Analysis Workflows

### Sensor Space Analysis

**Purpose:** Frequency-domain analysis in sensor space with FOOOF decomposition

**Requirements:**
- Preprocessed MEG data (.fif file)
- Events/triggers properly coded

**Output:**
- Epochs files (*-epo.fif)
- Time-frequency representations (*-tfr.h5)
- FOOOF decompositions (*.npy)
- Cross-spectral density matrices (*-csd.h5)

**Example:**
```python
# Run the example script
python example_usage_sensor.py

# Or directly
python sensor_space_individual_analysis.py --config config.yaml --subject S1
```

**Key Features:**
- ✅ Modern `compute_tfr()` API (replaces deprecated `tfr_morlet()`)
- ✅ Automatic frequency interpolation for FOOOF (handles log-spaced frequencies)
- ✅ Configurable peak width limits based on frequency resolution
- ✅ Parallel processing with joblib

---

### Source Space Analysis

**Purpose:** Localize neural activity to cortical sources

**Requirements:**
- Preprocessed MEG data
- FreeSurfer anatomical reconstruction
- Coregistration transformation file (*-trans.fif)

**Output:**
- Source space (*-src.fif)
- Forward model (*-fwd.fif)
- Source estimates (*-lh.stc, *-rh.stc)
- Morphed estimates to fsaverage
- Visualization images (*.png)

**Workflow:**
1. Create source space from FreeSurfer surfaces
2. Compute forward model (links sources to sensors)
3. Apply inverse operator to estimate source activity
4. Morph to fsaverage for group comparison
5. Visualize on cortical surface

**Example:**
```python
# Run the example script for detailed guidance
python example_usage_source.py

# Run analysis
python source_space_individual_analysis.py --config config.yaml --subject S1
```

---

### Connectivity Analysis

**Purpose:** Estimate functional connectivity between brain regions

**Requirements:**
- Cross-spectral density matrices (from sensor analysis)
- FreeSurfer reconstruction
- Coregistration file
- conpy package (`pip install conpy`)

**Output:**
- Morphed source space (*_for_con-morph-src.fif)
- Forward model for connectivity (*-for_con-morphed-fwd.fif)
- Connectivity matrices (*_connectivity.npy)

**Workflow:**
1. Setup fsaverage template source space
2. Morph to individual anatomy
3. Create forward model with vertex selection
4. Identify connectivity pairs
5. Estimate DICS beamformer connectivity
6. Compute for each condition

**Example:**
```python
# Run the example script for detailed guidance
python example_usage_connectivity.py

# Run analysis
python connectivity_individual_analysis.py --config config.yaml --subject S1
```

**Note:** Connectivity pair identification is typically done once across all subjects. See `example_usage_connectivity.py` for group-level workflow.

---

## 🔧 Configuration Guide

### Essential Parameters

```yaml
# Time-frequency analysis
time_frequency:
  min_freq_log: 0.60206      # log10(4) - minimum frequency
  max_freq_log: 1.90309      # log10(80) - maximum frequency  
  freq_resolution: 30         # Number of frequencies
  n_cycles: 5                 # Morlet wavelet cycles
  decim: 20                   # Time decimation factor

# Epoching
epoching:
  tmin: -4.0                  # Start time (seconds)
  tmax: 3.5                   # End time (seconds)
  reject_criteria:
    grad: 3000e-13            # Rejection threshold (T/m)
  flat_criteria:
    grad: 1e-13               # Flat detection threshold

# Source estimation
source_estimate:
  freq_min: 8                 # Frequency band minimum
  freq_max: 12                # Frequency band maximum
  reg: 0.05                   # Regularization parameter
  method: 'dSPM'              # Inverse method: MNE, dSPM, sLORETA

# Connectivity
connectivity:
  spacing: 'oct6'             # Source space resolution
  max_sensor_dist: 0.07       # Max distance from sensors (m)
  min_dist: 0.04              # Min pair distance (m)
  regularization: 0.05        # Inverse regularization
```

### Frequency Bands

Common choices for analysis:
- **Delta:** 1-4 Hz (sleep, attention)
- **Theta:** 4-8 Hz (memory, meditation)
- **Alpha:** 8-12 Hz (relaxation, attention)
- **Beta:** 12-30 Hz (motor, active thinking)
- **Gamma:** 30-80 Hz (perception, consciousness)

---

## 📦 Dependencies

### Required Packages

```
mne>=1.4.0              # MEG/EEG analysis
numpy>=1.21.0           # Numerical computing
scipy>=1.7.0            # Scientific computing
matplotlib>=3.4.0       # Visualization
pyyaml>=5.4.0           # Configuration files
fooof>=1.0.0            # Spectral parameterization
joblib>=1.0.0           # Parallel processing
pandas>=1.3.0           # Data manipulation
```

### Optional Packages

```
conpy                   # For connectivity analysis
seaborn>=0.11.0        # Enhanced visualization
jupyter>=1.0.0         # Interactive analysis
```

### Installation

```bash
# All required packages
pip install -r requirements.txt

# Connectivity analysis support
pip install conpy
```

---

## 💡 Usage Examples

### Batch Processing Multiple Subjects

```bash
#!/bin/bash
# run_batch_analysis.sh

SUBJECTS=("S1" "S2" "S3" "S4" "S5")

for subject in "${SUBJECTS[@]}"; do
    echo "Processing $subject..."
    python sensor_space_individual_analysis.py \
        --config config.yaml \
        --subject "$subject"
done
```

### Python Batch Script

```python
import subprocess
from pathlib import Path

subjects = ['S1', 'S2', 'S3', 'S4', 'S5']
config_file = 'config.yaml'

for subject in subjects:
    print(f"\n{'='*60}")
    print(f"Processing {subject}")
    print(f"{'='*60}\n")
    
    cmd = [
        'python', 'sensor_space_individual_analysis.py',
        '--config', config_file,
        '--subject', subject
    ]
    
    subprocess.run(cmd, check=True)
    print(f"✓ {subject} completed")
```

### Interactive Analysis (Jupyter)

See `example_usage_sensor.ipynb` for an interactive notebook example.

---

## 🐛 Troubleshooting

### Common Issues

**1. "reject_criteria imported as string"**
- **Fixed** in latest version with automatic type conversion
- Config values are now properly converted to floats

**2. "tfr_morlet() is a legacy function"**
- **Fixed** - Now uses modern `compute_tfr(method="morlet")`

**3. "joblib not installed. Cannot run in parallel"**
- **Solution:** `pip install joblib`
- Enables parallel processing for faster computation

**4. "FOOOF frequency spacing error"**
- **Fixed** - Automatic interpolation to linear frequency spacing
- Proper peak width limits based on frequency resolution

**5. "conpy not found"**
- **Solution:** `pip install conpy` (for connectivity analysis only)

**6. "Transformation file not found"**
- **Solution:** Run MNE coregistration GUI
  ```python
  import mne
  mne.gui.coregistration(subject='S1', subjects_dir='/path/to/subjects')
  ```
- Save as `S1-trans.fif` in data folder

**7. "FreeSurfer reconstruction missing"**
- **Solution:** Run FreeSurfer reconstruction
  ```bash
  recon-all -s S1 -i /path/to/MRI.nii -all
  ```

### Debug Mode

Enable verbose output for troubleshooting:

```python
# In config.yaml
processing:
  verbose: True
  
# Or set environment variable
export MNE_LOGGING_LEVEL=DEBUG
```

---

## 📚 Documentation & Resources

### MNE-Python Documentation
- [Main Documentation](https://mne.tools/stable/index.html)
- [Time-Frequency Analysis](https://mne.tools/stable/auto_tutorials/time-freq/index.html)
- [Source Estimation](https://mne.tools/stable/auto_tutorials/inverse/index.html)
- [Connectivity Analysis](https://mne.tools/stable/auto_tutorials/connectivity/index.html)

### FOOOF Documentation
- [FOOOF Overview](https://fooof-tools.github.io/fooof/)
- [Tutorial](https://fooof-tools.github.io/fooof/auto_tutorials/index.html)

### Connectivity (conpy)
- [conpy Documentation](https://aaltoimaginglanguage.github.io/conpy/)

### FreeSurfer
- [FreeSurfer Documentation](https://surfer.nmr.mgh.harvard.edu/)
- [Reconstruction Tutorial](https://surfer.nmr.mgh.harvard.edu/fswiki/ReconAllDevTable)

---

## 🔬 Analysis Pipeline Details

### Sensor Space Pipeline

```
Raw MEG Data (preprocessed)
    ↓
Extract Events
    ↓
Create Epochs (with artifact rejection)
    ↓
Compute Time-Frequency Representations
    ├── Power
    └── Inter-Trial Coherence (ITC)
    ↓
Apply FOOOF Decomposition
    ├── Periodic Component (oscillations)
    └── Aperiodic Component (1/f background)
    ↓
Compute Cross-Spectral Density
    ↓
Statistical Analysis (group level)
```

### Source Space Pipeline

```
Preprocessed MEG + MRI
    ↓
Coregistration (align MEG & MRI)
    ↓
Create Source Space (cortical surface)
    ↓
Compute Forward Model (sources → sensors)
    ↓
Load Epochs & Apply Inverse Operator
    ↓
Source Estimates (activity on cortex)
    ↓
Morph to fsaverage (for group analysis)
    ↓
Statistical Analysis
```

### Connectivity Pipeline

```
Source Space + CSD Matrices
    ↓
Setup fsaverage Template
    ↓
Morph to Individual Anatomy
    ↓
Create Forward Model (restricted vertices)
    ↓
Identify Connectivity Pairs
    ↓
Compute DICS Beamformer Connectivity
    ↓
Compare Across Conditions
    ↓
Group-Level Statistics
```

---

## 📖 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{otstavnov2023meg,
  author = {Otstavnov, Nikita},
  title = {MEG Analysis Pipeline for Spatial-Temporal Working Memory},
  year = {2023-2026},
  url = {https://github.com/your-repo/SpatialTemporalWorkingMemoryMEG}
}
```

And the underlying tools:

**MNE-Python:**
```bibtex
@article{gramfort2013meg,
  title={MEG and EEG data analysis with MNE-Python},
  author={Gramfort, Alexandre and Luessi, Martin and Larson, Eric and Engemann, Denis A and Strohmeier, Daniel and Brodbeck, Christian and Goj, Roman and Jas, Mainak and Brooks, Teon and Parkkonen, Lauri and others},
  journal={Frontiers in neuroscience},
  volume={7},
  pages={267},
  year={2013},
  publisher={Frontiers}
}
```

**FOOOF:**
```bibtex
@article{donoghue2020parameterizing,
  title={Parameterizing neural power spectra into periodic and aperiodic components},
  author={Donoghue, Thomas and Haller, Matar and Peterson, Erik J and Varma, Paroma and Sebastian, Priyadarshini and Gao, Richard and Noto, Torben and Lara, Antonio H and Wallis, Jacqueline D and Knight, Robert T and others},
  journal={Nature neuroscience},
  volume={23},
  number={12},
  pages={1655--1665},
  year={2020},
  publisher={Nature Publishing Group}
}
```

---

## 👥 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙋 Support

For questions, issues, or suggestions:

- **Issues:** [GitHub Issues](https://github.com/your-repo/SpatialTemporalWorkingMemoryMEG/issues)
- **Discussions:** [GitHub Discussions](https://github.com/your-repo/SpatialTemporalWorkingMemoryMEG/discussions)
- **Email:** your.email@institution.edu

---

## 📅 Changelog

### Version 2.0 (2026) - Major Refactoring
- ✅ Modern MNE API (`compute_tfr` vs `tfr_morlet`)
- ✅ Configuration-driven analysis
- ✅ Automatic FOOOF frequency interpolation
- ✅ Fixed type conversion issues
- ✅ Added source space analysis
- ✅ Added connectivity analysis
- ✅ Comprehensive documentation
- ✅ Example usage scripts

### Version 1.0 (2023) - Original Implementation
- Initial sensor space analysis pipeline
- Basic FOOOF integration
- Group-level statistics

---

## 🎯 Future Directions

Planned features:
- [ ] Automated quality control metrics
- [ ] Machine learning integration
- [ ] Real-time analysis support
- [ ] Web-based visualization dashboard
- [ ] Docker containerization
- [ ] Cloud computing support (AWS/GCP)

---

**Author:** Nikita Otstavnov (2023-2026)

**Last Updated:** March 4, 2026
