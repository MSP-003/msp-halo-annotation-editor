"""
Input panel widget: pickers for annotations, images, output folder, plus
user-tunable parameters for separator and bilateral splitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from haloqc.core.separator import SeparatorParams
from haloqc.core.bilateral import BilateralParams


@dataclass
class InputSelections:
    annotation_paths: list[Path]
    images_folder: Path | None
    # Separate output folders for each stage. Default matches: both point at
    # the same folder the user picked before this refactor. The pipeline
    # writes _TissueSeparated.annotations to separator_output_folder and
    # _Bilateral.annotations to bilateral_output_folder.
    separator_output_folder: Path
    bilateral_output_folder: Path
    # If True, append "_TissueSeparated" / "_Bilateral" to output filenames.
    # If False (default), output filename matches input stem exactly, so HALO
    # batch-import works out of the box.
    append_stage_suffix: bool
    separator_params: SeparatorParams
    bilateral_params: BilateralParams
    qc_channel: str  # "auto", "composite", or a specific channel name

    @property
    def output_folder(self) -> Path:
        """Back-compat: code paths that previously used a single folder get
        the separator folder. Prefer the stage-specific properties in new code.
        """
        return self.separator_output_folder


class _FolderPicker(QWidget):
    """A single-row picker: [mode dropdown] [path] [browse].

    Supports three modes when allow_single_file=True:
    - Folder: one directory
    - File(s): one OR multiple files (Ctrl/Cmd-click to multi-select)

    When files are multi-selected, path() returns the parent folder of the
    first file (useful for downstream "auto output folder" logic) and
    paths() returns the full list of selected files.
    """
    changed = Signal()

    def __init__(
        self,
        allow_single_file: bool = False,
        file_extensions: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._allow_single_file = allow_single_file
        self._file_extensions = file_extensions
        self._selected_files: list[Path] = []  # populated in multi-file mode

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.mode_combo = QComboBox()
        if allow_single_file:
            self.mode_combo.addItems(["Folder", "File(s)"])
        else:
            self.mode_combo.addItems(["Folder"])
            self.mode_combo.setEnabled(False)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("No selection")
        self.path_edit.setReadOnly(True)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._on_browse)

        layout.addWidget(self.mode_combo, 1)
        layout.addWidget(self.path_edit, 4)
        layout.addWidget(self.browse_btn, 0)
        self.setLayout(layout)

    def _on_browse(self) -> None:
        if self.mode_combo.currentText() == "File(s)" and self._allow_single_file:
            # Multi-select file dialog: user can Ctrl/Cmd-click multiple
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Select file(s)", "",
                self._file_extensions or "All files (*)",
            )
            if paths:
                self._selected_files = [Path(p) for p in paths]
                if len(paths) == 1:
                    display = paths[0]
                else:
                    first_name = Path(paths[0]).name
                    display = f"{len(paths)} files selected: {first_name}, ..."
                self.path_edit.setText(display)
                self.changed.emit()
        else:
            folder = QFileDialog.getExistingDirectory(self, "Select folder", "")
            if folder:
                self._selected_files = []
                self.path_edit.setText(folder)
                self.changed.emit()

    def path(self) -> Path | None:
        """Return the selected path. In multi-file mode, returns the parent
        folder of the first selected file (useful for auto-output-folder
        logic that needs a containing directory).
        """
        if self._selected_files:
            return self._selected_files[0].parent
        txt = self.path_edit.text().strip()
        return Path(txt) if txt else None

    def paths(self) -> list[Path]:
        """Return all selected files when in multi-file mode; empty list
        otherwise. Callers that care about the multi-file case should check
        this first."""
        return list(self._selected_files)

    def mode(self) -> Literal["folder", "file"]:
        return "file" if self.mode_combo.currentText() == "File(s)" else "folder"


class InputPanel(QWidget):
    """Main input configuration panel."""
    run_requested = Signal(object)  # emits InputSelections

    def __init__(self, parent=None):
        super().__init__(parent)
        # Wrap in a scroll area so small windows still work
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        from PySide6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidgetResizable(True)
        outer_layout.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(32, 20, 32, 20)
        layout.setSpacing(14)

        # ---- Hero header ----
        header = QLabel(
            '<span style="font-weight: 600;">MSP HALO Annotation Editor</span>'
            '<span style="color: #5d6b82; font-weight: 400;">'
            '&nbsp;&nbsp;|&nbsp;&nbsp;Tissue Separator &amp; Bilateral Splitter</span>'
        )
        header.setStyleSheet(
            "font-size: 22px; letter-spacing: -0.3px; background: transparent;"
        )
        subtitle = QLabel(
            "Post-process HALO-AI annotation exports with automated tissue "
            "separation, bilateral splitting, and interactive QC."
        )
        subtitle.setStyleSheet(
            "font-size: 13px; color: #9aa0ab; background: transparent;"
        )
        subtitle.setObjectName("subtitle")
        layout.addWidget(header)
        layout.addWidget(subtitle)

        # ---- File pickers ----
        files_box = QGroupBox("Input / Output")
        files_form = QFormLayout()
        files_form.setSpacing(10)
        files_form.setContentsMargins(12, 8, 12, 12)
        files_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.annotations_picker = _FolderPicker(
            allow_single_file=True,
            file_extensions="HALO Annotations (*.annotations)",
        )
        self.images_picker = _FolderPicker(allow_single_file=False)
        self.separator_output_picker = _FolderPicker(allow_single_file=False)
        self.bilateral_output_picker = _FolderPicker(allow_single_file=False)

        files_form.addRow("Annotations:", self.annotations_picker)
        files_form.addRow("Images (.ndpis):", self.images_picker)

        # Auto-subfolders checkbox: when on (default), outputs go into
        # TissueSeparated/ and BilateralSplit/ subfolders created alongside
        # the annotation input. The two manual pickers below become disabled.
        self.auto_subfolders_cb = QCheckBox(
            "Save outputs in subfolders alongside input "
            "(auto-create 'TissueSeparated' and 'BilateralSplit' folders)"
        )
        self.auto_subfolders_cb.setChecked(True)
        self.auto_subfolders_cb.setToolTip(
            "When checked, output folders are derived from the annotation input "
            "location. Uncheck to pick output folders manually."
        )
        self.auto_subfolders_cb.toggled.connect(self._on_auto_subfolders_toggled)
        files_form.addRow("", self.auto_subfolders_cb)

        files_form.addRow("Tissue-separated output:", self.separator_output_picker)
        files_form.addRow("Bilateral output:", self.bilateral_output_picker)

        self.append_suffix_cb = QCheckBox(
            "Append stage suffix to filenames (_TissueSeparated / _Bilateral)"
        )
        self.append_suffix_cb.setChecked(False)
        self.append_suffix_cb.setToolTip(
            "Off (default): output filenames exactly match input, so HALO batch "
            "re-import works. On: adds _TissueSeparated or _Bilateral suffix."
        )
        files_form.addRow("", self.append_suffix_cb)

        # Apply initial enabled state (auto mode starts on → pickers disabled)
        self._on_auto_subfolders_toggled(self.auto_subfolders_cb.isChecked())

        files_box.setLayout(files_form)
        layout.addWidget(files_box)

        # ---- Separator + Bilateral side-by-side ----
        params_row = QHBoxLayout()
        params_row.setSpacing(16)

        # Separator column
        sep_box = QGroupBox("Tissue Separator")
        sep_form = QFormLayout()
        sep_form.setSpacing(8)
        sep_form.setContentsMargins(12, 8, 12, 12)
        sep_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.expected_tissues = QSpinBox()
        self.expected_tissues.setRange(1, 64)
        self.expected_tissues.setValue(8)
        sep_form.addRow("Expected tissue pieces:", self.expected_tissues)

        self.grid_forcing = QLineEdit("4x2")
        self.grid_forcing.setPlaceholderText("e.g. 4x2, or blank for auto")
        sep_form.addRow("Grid layout (CxR):", self.grid_forcing)

        self.merge_threshold_um = QDoubleSpinBox()
        self.merge_threshold_um.setRange(0.0, 10000.0)
        self.merge_threshold_um.setValue(400.0)
        self.merge_threshold_um.setSuffix(" um")
        sep_form.addRow("Merge distance threshold:", self.merge_threshold_um)

        self.um_per_pixel = QDoubleSpinBox()
        self.um_per_pixel.setRange(0.001, 100.0)
        self.um_per_pixel.setDecimals(4)
        self.um_per_pixel.setValue(0.5)
        self.um_per_pixel.setSuffix(" um/px")
        sep_form.addRow("Microns per pixel:", self.um_per_pixel)

        self.merge_enabled = QCheckBox("Merge nearby tissue fragments")
        self.merge_enabled.setChecked(True)
        sep_form.addRow("", self.merge_enabled)

        self.force_to_expected = QCheckBox("Force extra merges to reach expected count")
        sep_form.addRow("", self.force_to_expected)

        sep_box.setLayout(sep_form)
        params_row.addWidget(sep_box, 1)

        # Bilateral column (with QC display folded in since it's only one field)
        bil_box = QGroupBox("Bilateral Splitter")
        bil_form = QFormLayout()
        bil_form.setSpacing(8)
        bil_form.setContentsMargins(12, 8, 12, 12)
        bil_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.ipsi_side = QComboBox()
        self.ipsi_side.addItems(["Right", "Left"])
        bil_form.addRow("Ipsilateral side:", self.ipsi_side)

        self.midline_method = QComboBox()
        self.midline_method.addItems([
            "Principal axis (smart)", "Vertical at bbox center (classic)",
        ])
        bil_form.addRow("Midline detection:", self.midline_method)

        # Subtle divider label
        from PySide6.QtWidgets import QFrame
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("color: #32363d; background: #32363d; max-height: 1px;")
        bil_form.addRow("", divider)

        self.qc_channel = QComboBox()
        self.qc_channel.addItems([
            "Composite (all channels)", "Auto (prefer DAPI)",
            "DAPI", "FITC", "TRTC", "Cy5", "Bright Field",
        ])
        bil_form.addRow("QC display channel:", self.qc_channel)

        bil_box.setLayout(bil_form)
        params_row.addWidget(bil_box, 1)

        layout.addLayout(params_row)

        # ---- Run button ----
        run_row = QHBoxLayout()
        run_row.setSpacing(12)
        self.status_label = QLabel("")
        self.status_label.setObjectName("statusError")
        self.status_label.setStyleSheet(
            "color: #e06868; background: transparent; font-weight: 500;"
        )
        self.run_btn = QPushButton("Run Tissue Separation  →")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setMinimumWidth(240)
        self.run_btn.setDefault(True)  # QSS styles :default as primary/accent
        self.run_btn.setStyleSheet(
            "font-size: 14px; font-weight: 600; padding: 8px 20px;"
        )
        self.run_btn.clicked.connect(self._on_run_clicked)
        run_row.addWidget(self.status_label, 1)
        run_row.addWidget(self.run_btn, 0)
        layout.addLayout(run_row)

        layout.addStretch(1)

    def _on_auto_subfolders_toggled(self, checked: bool) -> None:
        """When auto-subfolders is on, disable the manual output pickers so
        it's clear those fields won't be used. Uncheck to re-enable them."""
        self.separator_output_picker.setEnabled(not checked)
        self.bilateral_output_picker.setEnabled(not checked)

    def collect_selections(self) -> InputSelections | None:
        """Validate inputs and return InputSelections, or None if invalid."""
        from haloqc.io.pairing import find_annotations

        ap = self.annotations_picker.path()
        ip = self.images_picker.path()

        if ap is None:
            self.status_label.setText("Pick an annotations file/folder")
            return None

        # Resolve annotation file list, supporting multi-file picker mode.
        multi_files = self.annotations_picker.paths()
        if multi_files:
            paths = multi_files
        else:
            paths = find_annotations(ap)
        if not paths:
            self.status_label.setText("No .annotations files found")
            return None

        # Resolve output folders: either auto-derived from input or user-picked.
        auto = self.auto_subfolders_cb.isChecked()
        if auto:
            # Derive parent folder from the input. For a folder input, `ap`
            # IS the folder. For single/multi file, `_FolderPicker.path()`
            # already returns the parent folder of the first file.
            base = ap if ap.is_dir() else ap.parent
            sep_op = base / "TissueSeparated"
            bil_op = base / "BilateralSplit"
        else:
            sep_op = self.separator_output_picker.path()
            bil_op = self.bilateral_output_picker.path()
            if sep_op is None:
                self.status_label.setText("Pick a tissue-separated output folder")
                return None
            if bil_op is None:
                self.status_label.setText("Pick a bilateral output folder")
                return None

        # Images folder is optional: pipeline still runs without NDPIs
        self.status_label.setText("")

        # Parse ordering method
        grid_forcing_text = self.grid_forcing.text().strip() or None

        sep_params = SeparatorParams(
            expected_tissues=self.expected_tissues.value(),
            order_method="grid",
            grid_forcing=grid_forcing_text,
            merge_split_tissues=self.merge_enabled.isChecked(),
            merge_distance_microns=self.merge_threshold_um.value(),
            microns_per_pixel=self.um_per_pixel.value(),
            allow_force_to_expected=self.force_to_expected.isChecked(),
        )

        bil_params = BilateralParams(
            ipsi_side="right" if self.ipsi_side.currentText() == "Right" else "left",
            midline_method=(
                "principal_axis"
                if self.midline_method.currentIndex() == 0
                else "vertical_bbox"
            ),
        )

        return InputSelections(
            annotation_paths=paths,
            images_folder=ip,
            separator_output_folder=sep_op,
            bilateral_output_folder=bil_op,
            append_stage_suffix=self.append_suffix_cb.isChecked(),
            separator_params=sep_params,
            bilateral_params=bil_params,
            qc_channel=self.qc_channel.currentText(),
        )

    def _on_run_clicked(self):
        sel = self.collect_selections()
        if sel is not None:
            self.run_requested.emit(sel)
