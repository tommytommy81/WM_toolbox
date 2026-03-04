# Migration Guide: Old → New Pipeline

## Overview

This guide helps you transition from the original analysis scripts to the refactored, modular pipeline.

## Key Changes

### 1. Configuration Management

**OLD (Hard-coded)**
```python
# In Analysis - sensor space - individual level.py
folder           = 'C:/Users/User/Desktop/For testing'
file_name        = '3_test_1_tsss_mc_trans.fif'
subject_name     = 'S3'
min_freqs        = 4
max_freqs        = 80
main_event_1     = 155
main_event_2     = 255
```

**NEW (Config file)**
```yaml
# In config.yaml
paths:
  data_folder: '/path/to/data'
subject:
  name: 'S3'
  file_name: '3_test_1_tsss_mc_trans.fif'
filtering:
  min_freq: 4
  max_freq: 80
conditions:
  condition_1:
    event_id: 155
  condition_2:
    event_id: 255
```

### 2. Visual Inspection

**OLD (Inline)**
```python
# Mixed with analysis
raw_data.plot()
ica.plot_sources(raw_data)
epochs_1.plot()
power.plot_joint()
```

**NEW (Separate module)**
```python
# After analysis, if needed:
from visual_inspection import VisualInspector

inspector = VisualInspector(config)
inspector.inspect_epochs(epochs)
inspector.inspect_time_frequency(power, 'condition_1')
```

### 3. Script Execution

**OLD (Interactive)**
```python
# Run sections manually in IDE
%matplotlib qt
# ... code ...
print(ica.exclude)
ica.exclude = []  # Manual adjustment
```

**NEW (Command-line)**
```bash
# Non-interactive
python sensor_space_individual_analysis.py --config config.yaml --subject S3
```

### 4. Function Organization

**OLD (STWM_functions.py)**
- Mixed processing and plotting
- Functions include `raw_data.plot()` calls
- Hard to reuse without visualization

**NEW (STWM_functions_core.py)**
- Pure data processing
- No plotting code
- Reusable, testable functions

## Step-by-Step Migration

### Step 1: Create Configuration File

Extract all parameters from your old script:

```python
# OLD script parameters
folder           = 'C:/Users/User/Desktop/For testing'
file_name        = '1_test_1_tsss_mc_trans.fif'
subject_name     = 'S1'
condition_1      = 'S'
condition_2      = 'T'
main_event_1     = 155
main_event_2     = 255
tmin_epo         = -8
tmax_epo         = 8
# ... etc
```

Create `config.yaml`:

```yaml
paths:
  data_folder: 'C:/Users/User/Desktop/For testing'
subject:
  name: 'S1'
  file_name: '1_test_1_tsss_mc_trans.fif'
conditions:
  condition_1:
    name: 'S'
    event_id: 155
  condition_2:
    name: 'T'
    event_id: 255
epoching:
  tmin: -8
  tmax: 8
```

### Step 2: Test on One Subject

Run the new pipeline on a subject you've already analyzed:

```bash
python sensor_space_individual_analysis.py --config config.yaml --subject S1
```

Compare outputs with old analysis:
- Check epochs counts
- Compare time-frequency shapes
- Verify FOOOF results

### Step 3: Batch Process Remaining Subjects

Edit `run_batch_analysis.sh`:

```bash
SUBJECTS=("S1" "S2" "S3" "S4" "S5")  # Your subject list
```

Run:
```bash
./run_batch_analysis.sh config.yaml
```

### Step 4: Group Statistics

The new script handles this automatically:

```bash
python sensor_space_group_statistics.py --config config.yaml
```

### Step 5: Visual Inspection (Optional)

If you need quality control:

```python
from visual_inspection import VisualInspector
inspector = VisualInspector(config)
inspector.create_comparison_report()
```

## Equivalence Table

| Old Function | New Function | Location |
|--------------|--------------|----------|
| `perform_initial_analysis()` | `load_preprocessed_data()` | STWM_functions_core.py |
| `ica_apply()` | `apply_ica()` | STWM_functions_core.py |
| `event_renaming()` | `extract_events()` | STWM_functions_core.py |
| `epochs_initialization()` | `create_epochs()` | STWM_functions_core.py |
| `evoked_data()` | *Removed* (plotting only) | visual_inspection.py |
| `time_freq()` | `compute_time_frequency()` | STWM_functions_core.py |
| `fooof_application()` | `apply_fooof_multi_channel()` | STWM_functions_core.py |
| `fooof_merger()` | `merge_fooof_results()` | sensor_space_group_statistics.py |
| `sensor_statistics()` | `perform_cluster_statistics()` | sensor_space_group_statistics.py |
| `csd_calc()` | `compute_csd()` | STWM_functions_core.py |

## Parameter Mapping

| Old Variable | New Config Path | Notes |
|--------------|-----------------|-------|
| `folder` | `paths.data_folder` | |
| `subject_name` | `subject.name` | |
| `file_name` | `subject.file_name` | |
| `condition_1` | `conditions.condition_1.name` | |
| `main_event_1` | `conditions.condition_1.event_id` | |
| `tmin_epo` | `epoching.tmin` | |
| `tmax_epo` | `epoching.tmax` | |
| `reject_criteria` | `epoching.reject_criteria` | |
| `min_freq` | `time_frequency.min_freq_log` | Now log10 scale |
| `max_freq` | `time_frequency.max_freq_log` | Now log10 scale |
| `freq_res` | `time_frequency.freq_resolution` | |
| `n_cycles` | `time_frequency.n_cycles` | |
| `t_interest_min` | `time_window.tmin` | |
| `t_interest_max` | `time_window.tmax` | |
| `num_subjects` | `statistics.num_subjects` | |
| `alpha` | `statistics.alpha` | |
| `threshold` | `statistics.threshold` | |

## Common Issues

### Issue 1: File paths

**Problem**: Windows paths don't work on Mac/Linux

**Solution**: Use forward slashes in config.yaml:
```yaml
paths:
  data_folder: '/path/to/data'  # Works everywhere
```

### Issue 2: Missing YAML module

**Problem**: `ModuleNotFoundError: No module named 'yaml'`

**Solution**:
```bash
pip install pyyaml
```

### Issue 3: Different results

**Problem**: Slightly different numerical results

**Causes**:
- Random seed differences (ICA)
- MNE version differences
- Different baseline correction

**Solution**: Verify major patterns match, small numerical differences are normal

### Issue 4: Memory errors

**Problem**: Out of memory with large datasets

**Solution**: Adjust in config.yaml:
```yaml
time_frequency:
  freq_resolution: 20  # Reduce from 30
  decim: 30            # Increase from 20
```

## Backwards Compatibility

The original scripts are preserved:
- `Analysis - sensor space - individual level.py`
- `Analysis - sensor space - group statistics level.py`
- `STWM_functions.py`

You can continue using them if needed, but they won't be updated.

## Advantages of New Pipeline

✅ **Reproducibility**: Config file tracks all parameters

✅ **Scalability**: Easy batch processing

✅ **Modularity**: Reuse functions in custom analyses

✅ **Maintainability**: Cleaner, documented code

✅ **Flexibility**: Easy to modify parameters

✅ **Automation**: Command-line interface for HPC

## Getting Help

1. Check [README_refactored.md](README_refactored.md)
2. Review [example_usage.py](example_usage.py)
3. Compare old and new outputs
4. Use visual inspection for QC

## Rollback Plan

If you need to revert to old pipeline:

1. Keep old scripts (they're preserved)
2. Save config.yaml for future use
3. Document any issues encountered
4. Consider hybrid approach:
   - Use new pipeline for processing
   - Use old scripts for specific visualizations

## Next Steps After Migration

1. ✅ Verify results match old pipeline
2. ✅ Document any parameter adjustments
3. ✅ Set up batch processing
4. ✅ Create analysis templates
5. ✅ Train collaborators on new workflow

---

**Questions?** Review the documentation or inspect the example scripts.
