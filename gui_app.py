#!/usr/bin/env python3
"""
TEP-Net Train Ego-Path Detection GUI Application
Provides an interactive interface for model inference with file, model, device, and crop selection.
"""

import os
import sys

import torch
import yaml
from PIL import Image
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QButtonGroup,
    QGroupBox,
    QFormLayout,
    QSpinBox,
    QProgressBar,
    QSplitter,
    QFileIconProvider,
)

from src.utils.interface import Detector
from src.utils.visualization import draw_egopath


class FastFolderIconProvider(QFileIconProvider):
    """Icon provider that skips per-file mime-type resolution.

    The non-native QFileDialog otherwise resolves an icon for every entry on the
    UI thread (mime detection per file), which is the main slowdown in folders
    with many files. Since the folder dialog only shows directories, returning a
    single cached folder icon for everything keeps it fast without looking empty.
    """

    def __init__(self):
        super().__init__()
        self._folder_icon = super().icon(QFileIconProvider.Folder)

    def icon(self, _):  # handles both the IconType and QFileInfo overloads
        return self._folder_icon


class InferenceWorker(QThread):
    """Worker thread for running inference without blocking the UI."""

    finished = pyqtSignal()
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        detector_single=None,
        detector_rnn=None,
        input_path=None,
        input_paths=None,
        is_video=False,
    ):
        super().__init__()
        self.detector_single = detector_single
        self.detector_rnn = detector_rnn
        self.input_path = input_path
        self.input_paths = input_paths
        self.is_video = is_video
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            if self.is_video:
                self.process_video()
            elif self.input_paths is not None:
                self.process_folder()
            else:
                self.process_image()
            self.finished.emit()
        except Exception as e:
            self.error_occurred.emit(f"Inference error: {str(e)}")

    def run_both(self, img):
        """Run both checkpoints and return (single_frame_vis, rnn_vis).

        Each panel comes from its own model: the single-frame visualization from
        the standalone single-frame checkpoint and the RNN visualization from the
        temporal checkpoint. If no standalone single-frame model is loaded but the
        RNN is, the single-frame panel falls back to the RNN's embedded base net
        (via detect_pair) so the comparison is still populated.
        """
        single_vis = None
        rnn_vis = None

        if self.detector_single is not None:
            crop_s = self.detector_single.get_crop_coords()
            single_vis = draw_egopath(
                img, self.detector_single.detect(img), crop_coords=crop_s
            )
            if self.detector_rnn is not None:
                crop_r = self.detector_rnn.get_crop_coords()
                rnn_vis = draw_egopath(
                    img, self.detector_rnn.detect(img), crop_coords=crop_r
                )
        elif self.detector_rnn is not None:
            # No standalone single-frame model: derive it from the RNN's base net.
            crop_r = self.detector_rnn.get_crop_coords()
            single_res, rnn_res = self.detector_rnn.detect_pair(img)
            single_vis = draw_egopath(img, single_res, crop_coords=crop_r)
            if rnn_res is not None:
                rnn_vis = draw_egopath(img, rnn_res, crop_coords=crop_r)

        return single_vis, rnn_vis

    def process_image(self):
        """Process a single image."""
        try:
            img = Image.open(self.input_path)
            single_vis, rnn_vis = self.run_both(img)
            self.result_ready.emit(
                {
                    "type": "image",
                    "original": img,
                    "single_image": single_vis,
                    "rnn_image": rnn_vis,
                }
            )
        except Exception as e:
            self.error_occurred.emit(f"Image processing error: {str(e)}")

    def process_video(self):
        """Process a video file."""
        try:
            import cv2

            cap = cv2.VideoCapture(self.input_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_count = 0
            while True:
                if self._stop_requested:
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                single_vis, rnn_vis = self.run_both(frame)

                self.result_ready.emit(
                    {
                        "type": "video_frame",
                        "frame_num": frame_count,
                        "total_frames": total_frames,
                        "original": frame,
                        "single_image": single_vis,
                        "rnn_image": rnn_vis,
                    }
                )

                frame_count += 1

            cap.release()
        except Exception as e:
            self.error_occurred.emit(f"Video processing error: {str(e)}")

    def process_folder(self):
        """Process a folder of images sequentially."""
        try:
            total_files = len(self.input_paths)
            for idx, image_path in enumerate(self.input_paths):
                if self._stop_requested:
                    break
                img = Image.open(image_path)
                single_vis, rnn_vis = self.run_both(img)

                self.result_ready.emit(
                    {
                        "type": "folder_image",
                        "index": idx,
                        "total": total_files,
                        "input_path": image_path,
                        "original": img,
                        "single_image": single_vis,
                        "rnn_image": rnn_vis,
                    }
                )
        except Exception as e:
            self.error_occurred.emit(f"Folder processing error: {str(e)}")


class TEPNetGUI(QMainWindow):
    """Main GUI application for TEP-Net inference."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TEP-Net Train Ego-Path Detection")
        self.setMinimumSize(1024, 700)

        self.base_path = os.path.dirname(__file__)
        self.settings = QSettings("TEP-Net", "TEPNetGUI")
        # Reused by the folder dialog; held as an attribute so it isn't GC'd.
        self._folder_icon_provider = FastFolderIconProvider()
        self.detector = None
        self.current_image = None
        self.last_single_image = None
        self.last_rnn_image = None
        self.inference_worker = None
        self.custom_model_path = None
        self.selected_model = None
        self.model_buttons = {}
        self.model_dirs = {}
        self.base_weights_path = os.path.join(self.base_path, "egopath", "weights")
        self.rnn_weights_path = os.path.join(self.base_path, "egopathrnn", "weights")
        self.input_files = []
        self.is_folder_mode = False
        self.default_browse_dir = "/home/bhkim/Documents/AIWORKS_2025/bo_dataset_2025/"
        self.last_browse_dir = (
            self.default_browse_dir
            if os.path.isdir(self.default_browse_dir)
            else self.base_path
        )
        self.supported_image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

        self.default_input_path = os.path.join(self.base_path, "data", "egopath.jpg")

        self._initial_sized = False
        self.init_ui()
        self.detect_device()
        self.load_models()
        self.load_default_image()

    def showEvent(self, event):
        super().showEvent(event)
        # Size to the left panel only once it is laid out: its sizeHint is not
        # final until the window is shown (word-wrapped labels, font metrics).
        if not self._initial_sized:
            self._initial_sized = True
            self.set_initial_size()

    def set_initial_size(self):
        """Open tall enough to show the whole left control panel, capped to the screen."""
        screen = QApplication.primaryScreen().availableGeometry()
        # Window-frame (title bar / borders) height, known only after show.
        frame_h = max(0, self.frameGeometry().height() - self.height())
        # Height the left panel needs to show every group without the outer scroll.
        needed_height = self.left_panel.sizeHint().height() + 40  # layout margins
        max_client = screen.height() - frame_h - 40  # keep the frame on screen
        height = min(needed_height, max_client)
        width = min(3200, screen.width() - 80)
        self.resize(max(1024, width), max(700, height))

    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()

        # Left panel: Controls
        left_panel = QWidget()
        self.left_panel = left_panel
        left_layout = QVBoxLayout()
        left_panel.setMinimumWidth(320)
        left_panel.setMaximumWidth(700)
        left_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)

        # Title
        title = QLabel("TEP-Net Inference")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        left_layout.addWidget(title)

        # Control groups
        left_layout.addWidget(self.create_file_group())
        left_layout.addWidget(self.create_model_group())
        left_layout.addWidget(self.create_device_group())
        left_layout.addWidget(self.create_crop_group())
        left_layout.addWidget(self.create_output_group())

        left_layout.addStretch()

        # Run button
        self.run_button = QPushButton("Run Inference")
        self.run_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }"
        )
        self.run_button.clicked.connect(self.run_inference)
        left_layout.addWidget(self.run_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }"
        )
        self.stop_button.clicked.connect(self.stop_inference)
        self.stop_button.setEnabled(False)
        left_layout.addWidget(self.stop_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready")
        left_layout.addWidget(self.status_label)

        left_panel.setLayout(left_layout)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_panel)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(340)

        # Right panel: Image display
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        display_label = QLabel("Preview")
        display_font = QFont()
        display_font.setPointSize(12)
        display_font.setBold(True)
        display_label.setFont(display_font)
        right_layout.addWidget(display_label)

        preview_area = QWidget()
        preview_layout = QHBoxLayout()
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(10)

        original_col, self.original_display_label = self.make_preview_column(
            "원본 (Original)", "Input preview"
        )
        single_col, self.single_display_label = self.make_preview_column(
            "단일 프레임 (Single-frame)", "Result preview"
        )
        rnn_col, self.rnn_display_label = self.make_preview_column(
            "RNN (Temporal)", "Result preview"
        )
        preview_layout.addWidget(original_col)
        preview_layout.addWidget(single_col)
        preview_layout.addWidget(rnn_col)

        preview_area.setLayout(preview_layout)

        scroll = QScrollArea()
        scroll.setWidget(preview_area)
        scroll.setWidgetResizable(True)
        right_layout.addWidget(scroll)

        right_panel.setLayout(right_layout)

        # Combine panels with adjustable splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([700, 900])

        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)

    def make_preview_column(self, title, placeholder):
        """Build a captioned image-preview column; returns (container, image_label)."""
        column = QWidget()
        col_layout = QVBoxLayout()
        col_layout.setContentsMargins(0, 0, 0, 0)
        col_layout.setSpacing(4)

        caption = QLabel(title)
        caption.setAlignment(Qt.AlignCenter)
        caption_font = QFont()
        caption_font.setBold(True)
        caption.setFont(caption_font)
        col_layout.addWidget(caption)

        image_label = QLabel(placeholder)
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet(
            "border: 2px solid #ccc; background-color: #f8f8f8; min-height: 300px;"
        )
        image_label.setMinimumSize(280, 300)
        col_layout.addWidget(image_label, 1)

        column.setLayout(col_layout)
        return column, image_label

    def create_file_group(self):
        """Create file selection group."""
        group = QGroupBox("Input File")
        layout = QVBoxLayout()

        self.file_label = QLineEdit()
        self.file_label.setReadOnly(True)
        self.file_label.setPlaceholderText("No file or folder selected")
        layout.addWidget(self.file_label)

        browse_file_button = QPushButton("Browse File")
        browse_file_button.clicked.connect(self.browse_file)
        layout.addWidget(browse_file_button)

        browse_folder_button = QPushButton("Browse Folder")
        browse_folder_button.clicked.connect(self.browse_folder)
        layout.addWidget(browse_folder_button)

        group.setLayout(layout)
        return group

    def create_model_group(self):
        """Create model selection group."""
        group = QGroupBox("Model Selection")
        layout = QVBoxLayout()

        self.model_button_group = QButtonGroup(self)
        self.model_button_group.setExclusive(True)

        self.model_buttons_layout = QVBoxLayout()
        self.model_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.model_buttons_layout.setSpacing(5)

        self.model_buttons_widget = QWidget()
        self.model_buttons_widget.setLayout(self.model_buttons_layout)

        self.model_scroll = QScrollArea()
        self.model_scroll.setWidgetResizable(True)
        self.model_scroll.setWidget(self.model_buttons_widget)
        self.model_scroll.setMinimumHeight(500)
        layout.addWidget(self.model_scroll)

        self.custom_model_label = QLineEdit()
        self.custom_model_label.setReadOnly(True)
        self.custom_model_label.setPlaceholderText("Optional custom model folder")
        layout.addWidget(self.custom_model_label)

        self.model_browse_button = QPushButton("Browse Custom Model")
        self.model_browse_button.clicked.connect(self.browse_model)
        layout.addWidget(self.model_browse_button)

        self.model_info_label = QLabel()
        self.model_info_label.setWordWrap(True)
        layout.addWidget(self.model_info_label)

        group.setLayout(layout)
        return group

    def create_device_group(self):
        """Create device selection group."""
        group = QGroupBox("Device")
        layout = QFormLayout()

        self.device_combo = QComboBox()
        available_devices = ["cpu"]
        if torch.cuda.is_available():
            available_devices.append("cuda")
            available_devices.extend(
                [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            )
        if torch.backends.mps.is_available():
            available_devices.append("mps")

        self.device_combo.addItems(available_devices)
        if "cuda:0" in available_devices:
            self.device_combo.setCurrentText("cuda:0")
        layout.addRow("Device:", self.device_combo)

        self.device_info_label = QLabel()
        self.device_info_label.setWordWrap(True)
        layout.addRow("Status:", self.device_info_label)

        group.setLayout(layout)
        return group

    def create_crop_group(self):
        """Create crop mode selection group."""
        group = QGroupBox("Crop Settings")
        layout = QVBoxLayout()

        crop_type_layout = QFormLayout()
        self.crop_combo = QComboBox()
        self.crop_combo.addItems(["None", "Auto", "Manual"])
        self.crop_combo.currentIndexChanged.connect(self.on_crop_mode_changed)
        crop_type_layout.addRow("Mode:", self.crop_combo)
        layout.addLayout(crop_type_layout)

        # Manual crop coordinates
        self.crop_widget = QWidget()
        crop_coords_layout = QFormLayout()

        self.crop_left = QSpinBox()
        self.crop_left.setRange(0, 10000)
        crop_coords_layout.addRow("Left (x):", self.crop_left)

        self.crop_top = QSpinBox()
        self.crop_top.setRange(0, 10000)
        crop_coords_layout.addRow("Top (y):", self.crop_top)

        self.crop_right = QSpinBox()
        self.crop_right.setRange(0, 10000)
        self.crop_right.setValue(1369)
        crop_coords_layout.addRow("Right (x):", self.crop_right)

        self.crop_bottom = QSpinBox()
        self.crop_bottom.setRange(0, 10000)
        self.crop_bottom.setValue(1079)
        crop_coords_layout.addRow("Bottom (y):", self.crop_bottom)

        self.crop_widget.setLayout(crop_coords_layout)
        self.crop_widget.setVisible(False)
        layout.addWidget(self.crop_widget)

        group.setLayout(layout)
        return group

    def create_output_group(self):
        """Create output directory selection group."""
        group = QGroupBox("Output")
        layout = QVBoxLayout()

        self.output_label = QLineEdit()
        self.output_label.setReadOnly(True)
        # Restore the last-used output directory, falling back to <base>/output.
        saved_output = self.settings.value(
            "output_dir", os.path.join(self.base_path, "output")
        )
        self.output_label.setText(saved_output)
        layout.addWidget(self.output_label)

        output_button = QPushButton("Select Output Directory")
        output_button.clicked.connect(self.browse_output)
        layout.addWidget(output_button)

        group.setLayout(layout)
        return group

    def detect_device(self):
        """Detect and display device information."""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            self.device_info_label.setText(f"✓ CUDA available: {device_name}")
        elif torch.backends.mps.is_available():
            self.device_info_label.setText("✓ MPS (Metal) available")
        else:
            self.device_info_label.setText("Using CPU")

    def load_models(self):
        """Load base models from egopath/weights and RNN models from egopathrnn/weights."""
        for i in reversed(range(self.model_buttons_layout.count())):
            widget = self.model_buttons_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        for button in self.model_button_group.buttons():
            self.model_button_group.removeButton(button)

        self.model_buttons.clear()
        self.model_dirs.clear()
        self.selected_model = None
        self.model_info_label.setText("")

        def list_dirs(path):
            if not os.path.isdir(path):
                return []
            return sorted(
                d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
            )

        base_models = [d for d in list_dirs(self.base_weights_path) if not d.endswith("RNN")]
        rnn_models = {d for d in list_dirs(self.rnn_weights_path) if d.endswith("RNN")}

        first_selectable = None
        # one row per base model: [model button][RNN button]
        for model_name in base_models:
            base_button = self.make_model_button(
                model_name, path=os.path.join(self.base_weights_path, model_name)
            )
            rnn_name = model_name + "RNN"
            rnn_button = self.make_model_button(
                rnn_name, text="RNN", path=os.path.join(self.rnn_weights_path, rnn_name)
            )
            rnn_button.setMaximumWidth(90)
            if rnn_name in rnn_models:
                rnn_button.setToolTip(f"Run temporal RNN model ({rnn_name})")
            else:
                rnn_button.setEnabled(False)
                rnn_button.setToolTip(
                    f"RNN version not found in egopathrnn/weights/{rnn_name}. Train it first."
                )
            self.add_model_row(base_button, rnn_button)
            if first_selectable is None:
                first_selectable = model_name

        # standalone rows for RNN models without a matching base model
        for rnn_name in sorted(rnn_models):
            if rnn_name[:-3] in base_models:
                continue
            self.add_model_row(
                self.make_model_button(
                    rnn_name, path=os.path.join(self.rnn_weights_path, rnn_name)
                )
            )
            if first_selectable is None:
                first_selectable = rnn_name

        if first_selectable is None:
            self.model_info_label.setText("No models found")
            return
        self.on_model_selected(first_selectable)

    def make_model_button(self, name, text=None, path=None):
        """Create a checkable model-selection button bound to a weights directory."""
        button = QPushButton(text or name)
        button.setCheckable(True)
        button.clicked.connect(lambda checked, m=name: self.on_model_selected(m))
        self.model_button_group.addButton(button)
        self.model_buttons[name] = button
        if path is not None:
            self.model_dirs[name] = path
        return button

    def add_model_row(self, *buttons):
        """Add one or more buttons as a horizontal row to the model list."""
        row = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(5)
        for i, button in enumerate(buttons):
            row_layout.addWidget(button, 1 if i == 0 else 0)
        row.setLayout(row_layout)
        self.model_buttons_layout.addWidget(row)

    def load_default_image(self):
        """Load the default input image from data/egopath.jpg."""
        if os.path.exists(self.default_input_path):
            self.file_label.setText(self.default_input_path)
            self.is_folder_mode = False
            self.input_files = [self.default_input_path]
            self.load_preview(self.default_input_path)
        else:
            self.file_label.setPlaceholderText("No file or folder selected")

    def on_model_selected(self, model_name):
        """Update model info when a model button is selected."""
        self.selected_model = model_name
        self.custom_model_path = None
        self.custom_model_label.clear()

        for name, button in self.model_buttons.items():
            button.setChecked(name == model_name)

        self.show_model_info(self.get_model_path())

        single_path, rnn_path = self.get_model_pair()
        lines = ["", "Comparison panels:"]
        if single_path:
            lines.append(f"• Single-frame: {os.path.basename(single_path)}")
        elif rnn_path:
            lines.append("• Single-frame: RNN base net (fallback)")
        else:
            lines.append("• Single-frame: (missing)")
        lines.append(
            f"• RNN: {os.path.basename(rnn_path)}" if rnn_path else "• RNN: (missing)"
        )
        self.model_info_label.setText(
            self.model_info_label.text() + "\n" + "\n".join(lines)
        )

    def show_model_info(self, model_path):
        """Read a model's config.yaml and display its method/backbone."""
        config_file = (
            os.path.join(model_path, "config.yaml") if model_path else None
        )
        if not config_file or not os.path.exists(config_file):
            self.model_info_label.setText("No config found")
            return
        with open(config_file) as f:
            config = yaml.safe_load(f)
        method = config.get("method", "Unknown")
        backbone = config.get("backbone", "Unknown")
        info = f"Method: {method}\nBackbone: {backbone}"
        if config.get("temporal"):
            info += f"\nTemporal RNN (tracks {config.get('seq_len', '?')} frames)"
        self.model_info_label.setText(info)

    def get_model_path(self):
        """Return the active model path, preferring custom selection."""
        if self.custom_model_path:
            return self.custom_model_path
        if not self.selected_model:
            return None
        return self.model_dirs.get(self.selected_model)

    def get_model_pair(self):
        """Resolve the (single-frame, RNN) checkpoint paths for the current selection.

        Selecting either a base model "X" or its "XRNN" counterpart yields both
        paths when each exists on disk, so the two distinct checkpoints can be
        loaded and compared side by side. Either entry may be None when that
        variant is missing.
        """
        if self.custom_model_path:
            # A custom folder is run on its own; slot it by its temporal flag.
            config_file = os.path.join(self.custom_model_path, "config.yaml")
            is_temporal = False
            if os.path.exists(config_file):
                with open(config_file) as f:
                    is_temporal = bool(yaml.safe_load(f).get("temporal"))
            if is_temporal:
                return None, self.custom_model_path
            return self.custom_model_path, None

        name = self.selected_model
        if not name:
            return None, None
        base_name = name[:-3] if name.endswith("RNN") else name
        rnn_name = base_name + "RNN"
        return self.model_dirs.get(base_name), self.model_dirs.get(rnn_name)

    def browse_model(self):
        """Browse for a custom model directory."""
        model_dir = QFileDialog.getExistingDirectory(
            self, "Select Model Directory", os.path.join(self.base_path, "egopath", "weights")
        )
        if model_dir:
            self.custom_model_path = model_dir
            self.selected_model = None
            for button in self.model_button_group.buttons():
                button.setChecked(False)
            self.custom_model_label.setText(model_dir)
            self.show_model_info(self.get_model_path())

    def on_crop_mode_changed(self):
        """Update crop mode settings."""
        mode = self.crop_combo.currentText()
        self.crop_widget.setVisible(mode == "Manual")

    def browse_file(self):
        """Browse for input file."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Input File",
            "",
            "Image/Video Files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.mp4 *.avi);;All Files (*)",
        )
        if file_path:
            self.file_label.setText(file_path)
            self.is_folder_mode = False
            self.input_files = [file_path]
            self.load_preview(file_path)

    def browse_folder(self):
        """Browse for an input folder containing images."""
        # Use a non-native dialog so its size can be controlled (native file
        # dialogs ignore resize requests).
        dialog = QFileDialog(self, "Select Input Folder", self.last_browse_dir)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        # Avoid per-file mime-type icon resolution, which is the main slowdown
        # of the non-native dialog in directories with many files.
        dialog.setIconProvider(self._folder_icon_provider)
        dialog.resize(1400, 800)
        if dialog.exec_() != QFileDialog.Accepted:
            return
        selected = dialog.selectedFiles()
        folder_path = selected[0] if selected else ""
        if folder_path:
            self.last_browse_dir = folder_path
            image_files = sorted(
                [
                    os.path.join(folder_path, f)
                    for f in os.listdir(folder_path)
                    if os.path.splitext(f)[1].lower() in self.supported_image_extensions
                ]
            )
            if not image_files:
                QMessageBox.warning(
                    self,
                    "Error",
                    "No supported image files found in the selected folder.",
                )
                return

            self.file_label.setText(folder_path)
            self.is_folder_mode = True
            self.input_files = image_files
            self.load_preview(image_files[0])

    def browse_output(self):
        """Browse for output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.output_label.text() or self.base_path
        )
        if dir_path:
            self.output_label.setText(dir_path)
            # Remember it across sessions.
            self.settings.setValue("output_dir", dir_path)

    def reset_result_panels(self):
        """Clear the single-frame and RNN result panels."""
        self.last_single_image = None
        self.last_rnn_image = None
        self.single_display_label.setText("Result preview")
        self.rnn_display_label.setText("Result preview")

    def load_preview(self, file_path):
        """Load and display file preview."""
        try:
            if os.path.isdir(file_path):
                image_files = sorted(
                    [
                        os.path.join(file_path, f)
                        for f in os.listdir(file_path)
                        if os.path.splitext(f)[1].lower() in self.supported_image_extensions
                    ]
                )
                if not image_files:
                    self.original_display_label.clear()
                    self.reset_result_panels()
                    self.status_label.setText("No supported images in folder")
                    return
                file_path = image_files[0]

            ext = os.path.splitext(file_path)[1].lower()
            if ext in self.supported_image_extensions:
                img = Image.open(file_path)
                self.current_image = img
                self.reset_result_panels()
                self.display_image(img, self.original_display_label)
            elif ext in [".mp4", ".avi"]:
                import cv2

                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    self.current_image = img
                    self.reset_result_panels()
                    self.display_image(img, self.original_display_label)
                cap.release()
            else:
                self.original_display_label.clear()
                self.reset_result_panels()
                self.status_label.setText("Unsupported preview file type")
        except Exception as e:
            self.status_label.setText(f"Preview error: {str(e)}")

    def display_image(self, img, label):
        """Display image in the specified label."""
        try:
            pixmap = QPixmap.fromImage(self.pil_to_qimage(img.convert("RGB")))
            if label.width() > 0 and label.height() > 0:
                scaled = pixmap.scaled(
                    label.width(),
                    label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                label.setPixmap(scaled)
                return
            label.setPixmap(pixmap)
        except Exception as e:
            self.status_label.setText(f"Display error: {str(e)}")

    @staticmethod
    def pil_to_qimage(pil_image):
        """Convert PIL image to QImage."""
        rgb_image = pil_image.convert("RGB")
        data = rgb_image.tobytes()
        w, h = rgb_image.size
        bytes_per_line = 3 * w
        # .copy() detaches the QImage from the temporary `data` buffer, which is
        # freed once this function returns (otherwise the pixmap can show garbage).
        return QImage(data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

    def update_preview_images(self):
        """Rescale and refresh current preview images on resize."""
        if self.current_image is not None:
            self.display_image(self.current_image, self.original_display_label)
        if self.last_single_image is not None:
            self.display_image(self.last_single_image, self.single_display_label)
        if self.last_rnn_image is not None:
            self.display_image(self.last_rnn_image, self.rnn_display_label)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_preview_images()

    def get_crop_coords(self):
        """Get crop coordinates based on selected mode."""
        mode = self.crop_combo.currentText()
        if mode == "None":
            return None
        elif mode == "Auto":
            return "auto"
        else:  # Manual
            return (
                self.crop_left.value(),
                self.crop_top.value(),
                self.crop_right.value(),
                self.crop_bottom.value(),
            )

    def run_inference(self):
        """Run inference on the selected file."""
        # Validation
        file_path = self.file_label.text()
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "Error", "Please select a valid input file")
            return

        single_path, rnn_path = self.get_model_pair()
        if not single_path and not rnn_path:
            QMessageBox.warning(self, "Error", "Please select a valid model")
            return

        output_dir = self.output_label.text()
        if not output_dir or not os.path.exists(output_dir):
            QMessageBox.warning(self, "Error", "Please select a valid output directory")
            return

        # Get parameters
        device = self.device_combo.currentText()
        crop_coords = self.get_crop_coords()

        # Create detectors (single-frame and RNN are distinct checkpoints)
        try:
            self.status_label.setText("Initializing model(s)...")
            self.run_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            detector_single = None
            detector_rnn = None
            if single_path and os.path.exists(single_path):
                detector_single = Detector(
                    model_path=single_path,
                    crop_coords=crop_coords,
                    runtime="pytorch",
                    device=device,
                )
            if rnn_path and os.path.exists(rnn_path):
                detector_rnn = Detector(
                    model_path=rnn_path,
                    crop_coords=crop_coords,
                    runtime="pytorch",
                    device=device,
                )

            # Keep a reference so closeEvent etc. behave; prefer the RNN one.
            self.detector = detector_rnn or detector_single

            if detector_single is None and detector_rnn is None:
                raise RuntimeError("No usable model checkpoint found")
            if detector_single is None and detector_rnn is not None:
                # Single-frame panel will fall back to the RNN base net.
                self.single_display_label.setText("Single-frame (RNN base net)")
            if detector_rnn is None:
                self.rnn_display_label.setText("No RNN model")

            # Determine mode: video, folder, or individual image
            current_path = self.file_label.text()
            ext = os.path.splitext(current_path)[1].lower()
            is_video = ext in [".mp4", ".avi"]
            is_folder = self.is_folder_mode or os.path.isdir(current_path)

            if is_folder:
                input_path = None
                input_paths = self.input_files
            else:
                input_path = current_path
                input_paths = None

            # Run inference in worker thread
            self.inference_worker = InferenceWorker(
                detector_single=detector_single,
                detector_rnn=detector_rnn,
                input_path=input_path,
                input_paths=input_paths,
                is_video=is_video,
            )
            self.inference_worker.result_ready.connect(self.on_inference_result)
            self.inference_worker.error_occurred.connect(self.on_inference_error)
            self.inference_worker.finished.connect(self.on_inference_finished)
            self.inference_worker.start()

            self.status_label.setText("Running inference...")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.run_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Failed to initialize model: {str(e)}")

    def show_results(self, data):
        """Display the original/single-frame/RNN trio from a result payload."""
        if data.get("original") is not None:
            self.current_image = data["original"]
        if self.current_image is not None:
            self.display_image(self.current_image, self.original_display_label)
        if data.get("single_image") is not None:
            self.last_single_image = data["single_image"]
            self.display_image(self.last_single_image, self.single_display_label)
        if data.get("rnn_image") is not None:
            self.last_rnn_image = data["rnn_image"]
            self.display_image(self.last_rnn_image, self.rnn_display_label)

    def save_results(self, data, basename):
        """Save the single-frame and RNN visualizations side by side in the output dir."""
        output_dir = self.output_label.text()
        os.makedirs(output_dir, exist_ok=True)
        saved = []
        if data.get("single_image") is not None:
            path = os.path.join(output_dir, f"result_single_{basename}")
            data["single_image"].save(path)
            saved.append(path)
        if data.get("rnn_image") is not None:
            path = os.path.join(output_dir, f"result_rnn_{basename}")
            data["rnn_image"].save(path)
            saved.append(path)
        return saved

    def on_inference_result(self, data):
        """Handle inference result."""
        try:
            if data["type"] == "image":
                self.show_results(data)
                saved = self.save_results(
                    data, os.path.basename(self.file_label.text())
                )
                self.status_label.setText(
                    f"Saved: {', '.join(saved)}" if saved else "Inference complete!"
                )

            elif data["type"] == "video_frame":
                progress = (data["frame_num"] + 1) / data["total_frames"] * 100
                self.progress_bar.setValue(int(progress))
                self.show_results(data)
                self.status_label.setText(
                    f"Processing: {data['frame_num']+1}/{data['total_frames']} frames"
                )

            elif data["type"] == "folder_image":
                progress = (data["index"] + 1) / data["total"] * 100
                self.progress_bar.setValue(int(progress))
                self.show_results(data)
                self.save_results(data, os.path.basename(data["input_path"]))
                self.status_label.setText(
                    f"[{data['index']+1}/{data['total']}] {os.path.basename(data['input_path'])}"
                )

        except Exception as e:
            self.status_label.setText(f"Result handling error: {str(e)}")

    def stop_inference(self):
        """Request the running inference worker to stop."""
        if self.inference_worker is not None:
            self.inference_worker.request_stop()
            self.status_label.setText("Stopping inference...")
            self.stop_button.setEnabled(False)

    def on_inference_error(self, error_msg):
        """Handle inference error."""
        self.status_label.setText("Error occurred")
        QMessageBox.critical(self, "Inference Error", error_msg)
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)

    def on_inference_finished(self):
        """Handle inference completion."""
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Inference complete!")

    def closeEvent(self, event):
        """Stop and wait for the worker thread before closing the window."""
        if self.inference_worker is not None and self.inference_worker.isRunning():
            self.inference_worker.request_stop()
            self.inference_worker.wait(5000)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    gui = TEPNetGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
