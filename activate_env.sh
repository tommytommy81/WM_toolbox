#!/bin/bash
# Activation script for the MEG analysis environment
# Usage: source activate_env.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate the virtual environment
source "$SCRIPT_DIR/meg_env/bin/activate"

echo "✅ MEG analysis environment activated!"
echo "Python: $(which python)"
echo "MNE-Python version: $(python -c 'import mne; print(mne.__version__)')"
echo ""
echo "To deactivate: deactivate"
