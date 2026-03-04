#!/bin/bash
#
# Batch Processing Script for MEG Sensor Space Analysis
# ======================================================
# 
# This script processes multiple subjects sequentially or in parallel.
#
# Usage:
#   ./run_batch_analysis.sh config.yaml
#
# For parallel processing (if using job scheduler):
#   Edit the PARALLEL variable below
#

set -e  # Exit on error

# Configuration
CONFIG_FILE="${1:-config.yaml}"
PARALLEL=false  # Set to true for HPC parallel processing

# Subject list - EDIT THIS for your subjects
SUBJECTS=("S1" "S2" "S3")

echo "========================================"
echo "MEG Batch Analysis Pipeline"
echo "========================================"
echo "Config: $CONFIG_FILE"
echo "Subjects: ${SUBJECTS[@]}"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Step 1: Process individual subjects
echo "Step 1: Processing individual subjects..."
echo ""

if [ "$PARALLEL" = false ]; then
    # Sequential processing
    for subj in "${SUBJECTS[@]}"; do
        echo "----------------------------------------"
        echo "Processing subject: $subj"
        echo "----------------------------------------"
        
        python sensor_space_individual_analysis.py \
            --config "$CONFIG_FILE" \
            --subject "$subj"
        
        if [ $? -eq 0 ]; then
            echo "✓ Subject $subj completed successfully"
        else
            echo "✗ Error processing subject $subj"
            exit 1
        fi
        echo ""
    done
else
    # Parallel processing (example for SLURM)
    echo "Submitting parallel jobs..."
    for subj in "${SUBJECTS[@]}"; do
        sbatch --job-name="meg_$subj" \
               --output="logs/meg_${subj}_%j.out" \
               --wrap="python sensor_space_individual_analysis.py --config $CONFIG_FILE --subject $subj"
    done
    
    echo "Jobs submitted. Waiting for completion..."
    echo "Check job status with: squeue -u $USER"
    echo ""
    echo "After all jobs complete, run group statistics manually:"
    echo "  python sensor_space_group_statistics.py --config $CONFIG_FILE"
    exit 0
fi

# Step 2: Group-level statistics
echo "========================================"
echo "Step 2: Running group-level statistics"
echo "========================================"
echo ""

python sensor_space_group_statistics.py --config "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Analysis pipeline completed!"
    echo "========================================"
    echo ""
    echo "Results saved to output folder specified in config"
else
    echo ""
    echo "✗ Error in group statistics"
    exit 1
fi

echo ""
echo "Next steps:"
echo "  1. Review statistics summary file"
echo "  2. Run visual inspection (optional)"
echo "  3. Generate publication figures"
