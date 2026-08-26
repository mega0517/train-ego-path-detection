#!/bin/bash
# Launch the TEP-Net GUI Application

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import PyQt5" 2>/dev/null || {
    echo "Installing PyQt5..."
    python3 -m pip install PyQt5
}

# Launch the GUI
cd "$SCRIPT_DIR"
echo "Launching TEP-Net GUI Application..."
python3 gui_app.py
