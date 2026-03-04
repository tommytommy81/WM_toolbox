# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Analysis

Edit `config.yaml`:

```yaml
paths:
  data_folder: '/path/to/your/meg/data'
  output_folder: '/path/to/output'

subject:
  name: 'S1'
  file_name: 'your_meg_file.fif'

conditions:
  condition_1:
    name: 'S'        # Your first condition
    event_id: 155    # Trigger code
  condition_2:
    name: 'T'        # Your second condition  
    event_id: 255    # Trigger code
```

### 3. Process One Subject

```bash
python sensor_space_individual_analysis.py --config config.yaml
```

### 4. Process All Subjects

Edit subject list in `run_batch_analysis.sh`:

```bash
SUBJECTS=("S1" "S2" "S3" "S4" "S5")
```

Then run:

```bash
./run_batch_analysis.sh config.yaml
```

### 5. View Results

```python
from visual_inspection import VisualInspector
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

inspector = VisualInspector(config)
inspector.create_comparison_report()
```

---

## 📁 File Overview

### You Need to Edit

| File | Purpose |
|------|---------|
| `config.yaml` | **Main configuration** - Edit this for your analysis |
| `run_batch_analysis.sh` | **Batch script** - Edit subject list |

### You Run

| File | Purpose |
|------|---------|
| `sensor_space_individual_analysis.py` | Process individual subjects |
| `sensor_space_group_statistics.py` | Run group statistics |
| `visual_inspection.py` | Visual quality control |

### You Read

| File | Purpose |
|------|---------|
| `README_refactored.md` | **Full documentation** |
| `MIGRATION_GUIDE.md` | Transition from old code |
| `example_usage.py` | Usage examples |

### Support Files

| File | Purpose |
|------|---------|
| `STWM_functions_core.py` | Core processing functions |
| `requirements.txt` | Python dependencies |
| `REFACTORING_SUMMARY.md` | Technical summary |

---

## 🎯 Common Tasks

### Process Single Subject

```bash
python sensor_space_individual_analysis.py \
    --config config.yaml \
    --subject S1
```

### Process Multiple Subjects

```bash
for subj in S1 S2 S3; do
    python sensor_space_individual_analysis.py \
        --config config.yaml \
        --subject $subj
done
```

### Run Group Statistics

```bash
# After all subjects processed
python sensor_space_group_statistics.py --config config.yaml
```

### Visual Quality Check

```python
from visual_inspection import VisualInspector
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

inspector = VisualInspector(config)
inspector.load_and_inspect_epochs('S')
inspector.load_and_inspect_power('S')
```

---

## 📊 Output Files

### Individual Subject

After processing subject S1 with conditions 'S' and 'T':

```
S1_S_epochs-epo.fif          # Epochs for condition S
S1_T_epochs-epo.fif          # Epochs for condition T
S1_power_S-tfr.h5            # Time-frequency power (S)
S1_power_T-tfr.h5            # Time-frequency power (T)
S1_itc_S-tfr.h5              # Inter-trial coherence (S)
S1_itc_T-tfr.h5              # Inter-trial coherence (T)
S1_S_ped_crop.npy            # FOOOF periodic component (S)
S1_T_ped_crop.npy            # FOOOF periodic component (T)
S1_S_aper_crop.npy           # FOOOF aperiodic component (S)
S1_T_aper_crop.npy           # FOOOF aperiodic component (T)
S1_S_csd.h5                  # Cross-spectral density (S)
S1_T_csd.h5                  # Cross-spectral density (T)
```

### Group Level

After running group statistics:

```
list_S_ped_crop.npy                    # Merged periodic (S)
list_T_ped_crop.npy                    # Merged periodic (T)
T_obs_S_vs_T.npy                       # T-values (all)
T_obs_significant_S_vs_T.npy           # T-values (significant only)
cluster_p_values_S_vs_T.npy            # Cluster p-values
statistics_summary_S_vs_T.txt          # Summary report
```

---

## ⚙️ Key Configuration Parameters

### Must Configure

```yaml
paths:
  data_folder: '/your/data/path'      # WHERE IS YOUR DATA?

subject:
  name: 'S1'                           # SUBJECT ID
  file_name: 'file.fif'                # MEG FILE NAME

conditions:
  condition_1:
    name: 'S'                          # CONDITION 1 NAME
    event_id: 155                      # TRIGGER CODE
  condition_2:
    name: 'T'                          # CONDITION 2 NAME
    event_id: 255                      # TRIGGER CODE
```

### Commonly Modified

```yaml
time_window:
  tmin: 0.0        # Analysis window start (seconds)
  tmax: 4.0        # Analysis window end (seconds)

epoching:
  tmin: -8.0       # Epoch start (seconds)
  tmax: 8.0        # Epoch end (seconds)

statistics:
  num_subjects: 10  # Number of subjects
  alpha: 0.05       # Significance level
```

---

## 🔧 Troubleshooting

### "Config file not found"
- Ensure `config.yaml` exists in current directory
- Or specify path: `--config /full/path/to/config.yaml`

### "File not found: *.fif"
- Check `paths.data_folder` in config
- Check `subject.file_name` in config
- Verify file exists at specified location

### "No events found"
- Check `events.stim_channel` in config (default: 'STI101')
- Verify trigger codes in `conditions.*.event_id`

### Memory errors
- Reduce `time_frequency.freq_resolution` (e.g., 20 instead of 30)
- Increase `time_frequency.decim` (e.g., 30 instead of 20)
- Reduce `processing.n_jobs` (e.g., 4 instead of 7)

---

## 📚 Next Steps

1. ✅ Read [README_refactored.md](README_refactored.md) for full documentation
2. ✅ Check [example_usage.py](example_usage.py) for code examples
3. ✅ Review [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) if migrating from old code
4. ✅ Start with one subject, verify results
5. ✅ Batch process all subjects
6. ✅ Run group statistics
7. ✅ Generate publication figures

---

## 🆘 Need Help?

1. Check documentation files
2. Review example_usage.py
3. Inspect error messages carefully
4. Use visual_inspection.py to verify data quality
5. Compare with original scripts if needed

---

## ✨ Key Features

- ✅ Configuration-driven (no code changes needed)
- ✅ Assumes preprocessed data (ICA already done)
- ✅ Command-line interface (easy batch processing)
- ✅ Separate visual inspection (optional QC)
- ✅ Modular design (reusable functions)
- ✅ Comprehensive documentation

---

**Ready to start? Edit `config.yaml` and run your first subject!**

```bash
python sensor_space_individual_analysis.py --config config.yaml
```
