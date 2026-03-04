# MEG Sensor Space Analysis Pipeline (Refactored)

## Overview

This repository contains a refactored and modularized pipeline for analyzing MEG (Magnetoencephalography) data in sensor space, with a focus on frequency-domain comparisons between experimental conditions.

The refactored code separates:
- **Core data processing** (no visualization)
- **Visual inspection** (optional quality control)
- **Configuration** (YAML-based parameter management)

## Repository Structure

```
.
├── config.yaml                              # Main configuration file
├── sensor_space_individual_analysis.py      # Individual subject analysis
├── sensor_space_group_statistics.py         # Group-level statistics
├── STWM_functions_core.py                   # Core processing functions (no plots)
├── visual_inspection.py                     # Visual QC module (optional)
├── Script_for_Figure_generating.py          # Publication figures
│
├── Analysis - sensor space - individual level.py      # LEGACY (original)
├── Analysis - sensor space - group statistics level.py # LEGACY (original)
├── STWM_functions.py                        # LEGACY (original)
│
└── README_refactored.md                     # This file
```

## Key Features

### ✅ Configuration-Driven
- All parameters in one YAML file
- Easy to modify without touching code
- Reproducible analysis settings

### ✅ Modular Architecture
- Core processing separate from visualization
- Reusable functions
- Clean, maintainable code

### ✅ Assumes Preprocessed Data
- No ICA components selection
- No manual artifact rejection
- Focus on time-frequency analysis

### ✅ Command-Line Interface
- Easy to batch process subjects
- Scriptable for HPC environments

## Quick Start

### 1. Configure Your Analysis

Edit `config.yaml` to set your parameters:

```yaml
paths:
  data_folder: '/path/to/your/data'
  output_folder: '/path/to/output'

subject:
  name: 'S1'
  file_name: '1_test_1_tsss_mc_trans.fif'

conditions:
  condition_1:
    name: 'S'        # Spatial condition
    event_id: 155
  condition_2:
    name: 'T'        # Temporal condition
    event_id: 255

time_window:
  tmin: 0.0          # Analysis window
  tmax: 4.0
```

### 2. Run Individual Subject Analysis

```bash
python sensor_space_individual_analysis.py --config config.yaml --subject S1
```

This will:
1. Load preprocessed MEG data
2. Extract events and create epochs
3. Compute time-frequency representations
4. Apply FOOOF decomposition (separate periodic/aperiodic)
5. Compute cross-spectral density
6. Save all intermediate results

**Output files:**
- `S1_S_epochs-epo.fif` / `S1_T_epochs-epo.fif`
- `S1_power_S-tfr.h5` / `S1_power_T-tfr.h5`
- `S1_S_ped_crop.npy` / `S1_T_ped_crop.npy` (FOOOF periodic)
- `S1_S_aper_crop.npy` / `S1_T_aper_crop.npy` (FOOOF aperiodic)
- `S1_S_csd.h5` / `S1_T_csd.h5`

### 3. Run Group-Level Statistics

After processing all subjects, update `config.yaml`:

```yaml
statistics:
  num_subjects: 10
  alpha: 0.05
  threshold: 0.01
  n_permutations: 5000
```

Then run:

```bash
python sensor_space_group_statistics.py --config config.yaml
```

This performs cluster-based permutation testing comparing the two conditions across subjects.

**Output files:**
- `T_obs_S_vs_T.npy` (all T-values)
- `T_obs_significant_S_vs_T.npy` (significant clusters only)
- `cluster_p_values_S_vs_T.npy`
- `statistics_summary_S_vs_T.txt`

### 4. Visual Inspection (Optional)

If you need to visually inspect data quality:

```python
from visual_inspection import VisualInspector
import yaml

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Create inspector
inspector = VisualInspector(config)

# Inspect epochs
inspector.load_and_inspect_epochs('S')
inspector.load_and_inspect_epochs('T')

# Inspect time-frequency
inspector.load_and_inspect_power('S')
inspector.load_and_inspect_power('T')

# Create comparison report
inspector.create_comparison_report()
```

## Configuration File Reference

### Essential Parameters

| Section | Parameter | Description | Example |
|---------|-----------|-------------|---------|
| **paths** | `data_folder` | Input data directory | `/data/meg/` |
| | `output_folder` | Output directory | `/results/` |
| **subject** | `name` | Subject ID | `'S1'` |
| | `file_name` | MEG file name | `'1_test_tsss.fif'` |
| **conditions** | `condition_1.name` | First condition name | `'S'` |
| | `condition_1.event_id` | Trigger code | `155` |
| | `condition_2.name` | Second condition name | `'T'` |
| | `condition_2.event_id` | Trigger code | `255` |
| **time_window** | `tmin` | Analysis start (s) | `0.0` |
| | `tmax` | Analysis end (s) | `4.0` |
| **epoching** | `tmin` | Epoch start (s) | `-8.0` |
| | `tmax` | Epoch end (s) | `8.0` |
| | `reject_criteria.grad` | Rejection threshold | `3000e-13` |
| **time_frequency** | `min_freq_log` | log10(min freq) | `0.60206` (4 Hz) |
| | `max_freq_log` | log10(max freq) | `1.90309` (80 Hz) |
| | `freq_resolution` | Number of freqs | `30` |
| | `n_cycles` | Morlet cycles | `5` |
| **statistics** | `num_subjects` | Group size | `10` |
| | `alpha` | Significance level | `0.05` |
| | `n_permutations` | Permutations | `5000` |

## Workflow

### Individual Level Processing

```
Load Preprocessed Data
        ↓
Extract Events
        ↓
Create Epochs (2 conditions)
        ↓
Time-Frequency Analysis (Morlet)
        ↓
FOOOF Decomposition
  ├── Periodic component
  └── Aperiodic component
        ↓
Cross-Spectral Density
        ↓
Save Results
```

### Group Level Processing

```
Load FOOOF Results (all subjects)
        ↓
Merge Across Subjects
        ↓
Cluster Permutation Test
  ├── Condition 1 vs Condition 2
  └── Spatio-temporal clustering
        ↓
Extract Significant Clusters
        ↓
Save Statistics
```

## Comparison: Old vs New Structure

| Aspect | Original Code | Refactored Code |
|--------|---------------|-----------------|
| **Configuration** | Hard-coded variables | YAML config file |
| **Visualization** | Mixed with processing | Separate module |
| **Preprocessing** | Included (ICA, filtering) | Assumes done |
| **Execution** | Interactive (Jupyter-style) | Command-line scripts |
| **Reusability** | Monolithic scripts | Modular functions |
| **Documentation** | Inline comments | Comprehensive README |

## Core Functions API

### STWM_functions_core.py

Key functions available:

```python
# Data loading
load_preprocessed_data(file_path, meg_type='grad')

# Epoching
create_epochs(raw_data, events, event_id, tmin, tmax, ...)
save_epochs(epochs, file_path)

# Time-frequency
compute_time_frequency(epochs, freqs, n_cycles=5, ...)
save_tfr(tfr, file_path)

# FOOOF
apply_fooof_single_channel(spectrum, freqs)
apply_fooof_multi_channel(spectrum, freqs)

# CSD
compute_csd(epochs, freqs, tmin, tmax, ...)
save_csd(csd, file_path)

# Statistics
cluster_permutation_test(data, adjacency, n_permutations=5000, ...)
extract_significant_clusters(T_obs, clusters, cluster_p_values, alpha=0.05)
```

## Batch Processing Example

Process multiple subjects:

```bash
#!/bin/bash
# process_all_subjects.sh

for subj in S1 S2 S3 S4 S5; do
    echo "Processing $subj..."
    python sensor_space_individual_analysis.py \
        --config config.yaml \
        --subject $subj
done

echo "Running group statistics..."
python sensor_space_group_statistics.py --config config.yaml
```

## Dependencies

Required packages:
- `mne` (MEG/EEG analysis)
- `numpy`
- `scipy`
- `pyyaml` (configuration)
- `fooof` (spectral parameterization)
- `matplotlib` (for visual inspection)

Install:
```bash
pip install mne numpy scipy pyyaml fooof matplotlib
```

## Tips & Best Practices

### 1. **Start with Example Config**
   - Copy `config.yaml` and modify for your data
   - Keep a separate config for each experiment

### 2. **Verify First Subject**
   - Run full pipeline on one subject
   - Use visual inspection to verify quality
   - Then batch process remaining subjects

### 3. **Use Version Control**
   - Commit `config.yaml` with each analysis
   - Track parameter changes for reproducibility

### 4. **HPC Optimization**
   - Increase `n_jobs` for parallel processing
   - Use array jobs for subject-level processing
   - Group statistics on single node

### 5. **Output Organization**
   ```
   results/
   ├── subject_level/
   │   ├── S1/
   │   ├── S2/
   │   └── ...
   └── group_level/
       ├── statistics/
       └── figures/
   ```

## Troubleshooting

### "Config file not found"
- Ensure `config.yaml` is in working directory
- Or specify full path: `--config /path/to/config.yaml`

### "Events not found"
- Check `stim_channel` in config
- Verify event IDs match your paradigm

### "No significant clusters"
- Try adjusting `threshold` in statistics
- Increase `n_permutations`
- Check data quality visually

### Memory issues
- Reduce `freq_resolution`
- Increase `decim` (decimation factor)
- Process fewer channels at once

## Migration from Original Code

If you have existing analysis scripts:

1. **Extract Parameters**: Copy hard-coded values to `config.yaml`
2. **Update Imports**: Change to new function names
3. **Remove Plots**: Delete plotting code (use `visual_inspection.py` instead)
4. **Test**: Run on pilot subject, compare results

## Citation

If you use this code, please cite the original study:

```
[Your citation here]
```

## Support

For questions or issues:
- Check configuration file carefully
- Review error messages
- Inspect intermediate outputs
- Use visual inspection module for QC

## License

[Specify license]

## Authors

- Original: Nikita Otstavnov (2023)
- Refactored: 2026

---

**Note**: Original scripts (`Analysis - sensor space - *.py`) are preserved for reference but are not actively maintained. Use the new refactored scripts for new analyses.
