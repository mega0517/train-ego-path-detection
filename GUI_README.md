# TEP-Net GUI Application

A modern graphical user interface for train ego-path detection inference.

## Features

- **File Selection**: Browse and select image or video files for inference
- **Model Management**: Choose from available trained models in the `weights/` directory
- **Device Selection**: Select compute device (CPU, CUDA, MPS, or specific GPU)
- **Crop Modes**: 
  - **None**: Run inference on the full image
  - **Auto**: Automatic crop detection
  - **Manual**: Specify custom crop coordinates (left, top, right, bottom)
- **Real-time Preview**: View input images and results in real-time
- **Batch Processing**: Process entire videos with frame-by-frame progress tracking
- **Output Management**: Save results to a custom output directory

## Installation

### Prerequisites
- Python 3.8 or higher
- All dependencies from `requirements.txt`

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

The GUI requires PyQt5, which will be automatically installed.

## Usage

### Linux/macOS
```bash
chmod +x launch_gui.sh
./launch_gui.sh
```

Or directly:
```bash
python3 gui_app.py
```

### Windows
```bash
launch_gui.bat
```

Or directly:
```bash
python gui_app.py
```

## GUI Walkthrough

1. **Load Model**: 
   - Models are automatically detected from the `weights/` directory
   - Select your preferred model from the dropdown
   - Model information (method and backbone) is displayed

2. **Select Input**:
   - Click "Browse" under "Input File"
   - Choose an image (JPG, PNG) or video (MP4, AVI)
   - Preview is automatically displayed

3. **Configure Device**:
   - Select your compute device
   - Available options depend on your hardware (CPU, CUDA, MPS)
   - Current device status is shown

4. **Set Crop Mode**:
   - **None**: Uses the entire image
   - **Auto**: Automatically determines optimal crop region (requires model with autocrop)
   - **Manual**: Manually specify crop coordinates (left, top, right, bottom)

5. **Choose Output Directory**:
   - Click "Select Output Directory" to choose where to save results
   - Defaults to `output/` in the project root

6. **Run Inference**:
   - Click "Run Inference" button
   - Progress bar shows processing status
   - Results are displayed in the preview pane
   - Output is automatically saved to the specified directory

## Output

- **Images**: Results are saved as PNG files in the output directory
- **Videos**: Frame-by-frame inference with ego-path visualization
- File naming: `result_<original_filename>`

## Keyboard Shortcuts

- The GUI is fully mouse-driven, but you can:
  - Use Tab to navigate between fields
  - Use Enter to confirm selections

## Troubleshooting

### "No models found"
- Ensure the `weights/` directory exists and contains model directories
- Each model directory must contain `config.yaml` and `best.pt`

### PyQt5 Installation Issues
- Explicitly install: `pip install PyQt5`
- On Ubuntu/Debian: `sudo apt-get install python3-pyqt5`
- On macOS: `brew install pyqt5` (if using Homebrew)

### Inference Errors
- Check that all required model files are present
- Ensure the input file is valid and readable
- Verify GPU memory if using CUDA

### Performance Tips
- Use CPU for small images if GPU memory is limited
- For videos, consider processing shorter clips
- Auto-crop mode requires additional inference iterations

## System Requirements

- **GPU (Recommended)**: NVIDIA GPU with CUDA support for faster inference
- **Memory**: At least 2GB RAM (4GB+ recommended for GPU)
- **Display**: Any monitor capable of 1024x768 resolution

## References

- See [README.md](README.md) for main documentation
- Check [AGENTS.md](AGENTS.md) for development guidelines
- Review `configs/` for model configuration details
