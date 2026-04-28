"""
Bilateral QC view (stage 2).

For the selected sample, shows the slide thumbnail + all tissue outlines,
each with its midline cut line visible. The user can:
- click a tissue to select it and see its ipsi/contra area stats
- drag the selected tissue's cut endpoints to correct the cut
- click Recompute to rerun the bilateral split with the edited cut
- click Save to write the final output

When the user changes a cut, we store it in the per-sample `manual_cuts`
dict and rerun `split_bilateral` for just that sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from haloqc.core.bilateral import (
    BilateralParams,
    MidlineCut,
    detect_midline,
    split_bilateral,
)
from haloqc.core.qc import check_bilateral
from haloqc.pipeline import SampleResult
from haloqc.ui.canvas import AnnotationCanvas
from haloqc.ui.cut_line_item import CutLineItem
from haloqc.ui.sample_list import SampleListWidget
from haloqc.ui.thumbnail_controller import RenderRequest, ThumbnailController


class BilateralQCView(QWidget):
    """Stage 2 QC: review and correct bilateral cuts."""

    save_and_finish = Signal()
    back_to_separator = Signal()
    recompute_sample = Signal(int, dict)  # (sample_idx, manual_cuts)

    def __init__(self, thumb_ctrl: ThumbnailController | None = None, parent=None):
        super().__init__(parent)
        self._samples: list[SampleResult] = []
        self._selected_idx: int | None = None
        self._bilateral_params = BilateralParams()
        # Per-sample manual cuts: { sample_idx: { tissue_name: MidlineCut } }
        self._manual_cuts: dict[int, dict[str, MidlineCut]] = {}
        # Per-sample active cut items (so we can remove them before redraw)
        self._active_cut_items: list[CutLineItem] = []
        self._active_cut_by_tissue: dict[str, CutLineItem] = {}
        self._selected_tissue: str | None = None
        self._thumb_ctrl = thumb_ctrl or ThumbnailController()

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(12)
        title = QLabel("Stage 2 · Bilateral Split QC")
        title.setStyleSheet(
            "font-size: 18px; font-weight: 600; background: transparent; "
            "letter-spacing: -0.2px;"
        )
        toolbar.addWidget(title)
        toolbar.addStretch(1)
        self.back_btn = QPushButton("← Back to Separator")
        self.back_btn.clicked.connect(self.back_to_separator.emit)
        toolbar.addWidget(self.back_btn)
        self.save_btn = QPushButton("Save Final Output")
        self.save_btn.setMinimumHeight(36)
        self.save_btn.setDefault(True)
        self.save_btn.setStyleSheet("font-weight: 600; padding: 6px 16px;")
        self.save_btn.clicked.connect(self.save_and_finish.emit)
        toolbar.addWidget(self.save_btn)
        root.addLayout(toolbar)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left: sample list
        self.sample_list = SampleListWidget()
        self.sample_list.sample_selected.connect(self._on_sample_selected)
        splitter.addWidget(self.sample_list)

        # Center: canvas
        self.canvas = AnnotationCanvas()
        self.canvas.tissue_clicked.connect(self._on_tissue_clicked)
        splitter.addWidget(self.canvas)

        # Right: controls
        right = QWidget()
        right_layout = QVBoxLayout(right)

        # Display controls (same as separator view for consistency)
        display_box = QGroupBox("Display")
        display_layout = QVBoxLayout()
        display_layout.addWidget(QLabel("Channel mode:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems([
            "Composite (all channels)", "Auto (prefer DAPI)",
            "DAPI", "FITC", "TRTC", "Cy5", "Bright Field",
        ])
        self.channel_combo.currentTextChanged.connect(self._refresh_image)
        display_layout.addWidget(self.channel_combo)
        contrast_row = QHBoxLayout()
        contrast_row.addWidget(QLabel("Contrast pct:"))
        self.low_pct_spin = QDoubleSpinBox()
        self.low_pct_spin.setRange(0.0, 49.0); self.low_pct_spin.setValue(1.0)
        self.low_pct_spin.setSuffix(" lo")
        self.low_pct_spin.valueChanged.connect(self._refresh_image)
        self.high_pct_spin = QDoubleSpinBox()
        self.high_pct_spin.setRange(51.0, 100.0); self.high_pct_spin.setValue(99.0)
        self.high_pct_spin.setSuffix(" hi")
        self.high_pct_spin.valueChanged.connect(self._refresh_image)
        contrast_row.addWidget(self.low_pct_spin)
        contrast_row.addWidget(self.high_pct_spin)
        display_layout.addLayout(contrast_row)
        self.fit_btn = QPushButton("Fit slide to view")
        self.fit_btn.clicked.connect(self.canvas.fit_to_scene)
        display_layout.addWidget(self.fit_btn)
        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("Background:"))
        self.bg_combo = QComboBox()
        self.bg_combo.addItems(["Dark", "Light"])
        self.bg_combo.currentTextChanged.connect(self._on_bg_changed)
        bg_row.addWidget(self.bg_combo)
        display_layout.addLayout(bg_row)
        display_box.setLayout(display_layout)
        right_layout.addWidget(display_box)

        # Selected tissue controls
        sel_box = QGroupBox("Selected Tissue")
        sel_layout = QVBoxLayout()
        self.selected_label = QLabel("(click a tissue to select)")
        sel_layout.addWidget(self.selected_label)

        rotate_row = QHBoxLayout()
        rotate_row.addWidget(QLabel("Rotate:"))
        self.rot_left_btn = QPushButton("−5°")
        self.rot_left_btn.clicked.connect(lambda: self._rotate_selected(-5))
        self.rot_right_btn = QPushButton("+5°")
        self.rot_right_btn.clicked.connect(lambda: self._rotate_selected(+5))
        rotate_row.addWidget(self.rot_left_btn)
        rotate_row.addWidget(self.rot_right_btn)
        sel_layout.addLayout(rotate_row)

        translate_row = QHBoxLayout()
        translate_row.addWidget(QLabel("Shift (px):"))
        self.shift_left_btn = QPushButton("◀")
        self.shift_left_btn.clicked.connect(lambda: self._translate_selected(-500, 0))
        self.shift_right_btn = QPushButton("▶")
        self.shift_right_btn.clicked.connect(lambda: self._translate_selected(+500, 0))
        self.shift_up_btn = QPushButton("▲")
        self.shift_up_btn.clicked.connect(lambda: self._translate_selected(0, -500))
        self.shift_down_btn = QPushButton("▼")
        self.shift_down_btn.clicked.connect(lambda: self._translate_selected(0, +500))
        translate_row.addWidget(self.shift_left_btn)
        translate_row.addWidget(self.shift_right_btn)
        translate_row.addWidget(self.shift_up_btn)
        translate_row.addWidget(self.shift_down_btn)
        sel_layout.addLayout(translate_row)

        self.reset_cut_btn = QPushButton("Reset to auto-detected cut")
        self.reset_cut_btn.clicked.connect(self._reset_selected_cut)
        sel_layout.addWidget(self.reset_cut_btn)

        sel_box.setLayout(sel_layout)
        right_layout.addWidget(sel_box)

        # Recompute button
        self.recompute_btn = QPushButton("Recompute this sample")
        self.recompute_btn.clicked.connect(self._do_recompute)
        right_layout.addWidget(self.recompute_btn)

        # QC flags + diagnostics
        flags_box = QGroupBox("QC Flags")
        flags_layout = QVBoxLayout()
        self.flags_list = QListWidget()
        self.flags_list.itemClicked.connect(self._on_flag_clicked)
        flags_layout.addWidget(self.flags_list)
        flags_box.setLayout(flags_layout)
        right_layout.addWidget(flags_box)

        diag_box = QGroupBox("Diagnostics")
        diag_layout = QVBoxLayout()
        self.diag_text = QTextEdit()
        self.diag_text.setReadOnly(True)
        self.diag_text.setMaximumHeight(140)
        diag_layout.addWidget(self.diag_text)
        diag_box.setLayout(diag_layout)
        right_layout.addWidget(diag_box)

        right_layout.addStretch(1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([260, 900, 340])
        root.addWidget(splitter)

    # ------------------------------------------------------------------
    def set_bilateral_params(self, params: BilateralParams) -> None:
        self._bilateral_params = params

    def load_samples(self, samples: list[SampleResult]) -> None:
        self._samples = list(samples)
        self._manual_cuts = {i: {} for i in range(len(samples))}
        labels = [s.pair.annotation_path.name for s in self._samples]
        self.sample_list.populate(labels)
        for i, s in enumerate(self._samples):
            self._update_sample_status(i)
        if self._samples:
            self._select_sample(0)

    def _update_sample_status(self, idx: int) -> None:
        s = self._samples[idx]
        if s.error:
            self.sample_list.set_status(idx, "error", s.error)
        elif s.bilateral_qc is None:
            self.sample_list.set_status(idx, None)
        else:
            sev = s.bilateral_qc.max_severity
            tooltip = "\n".join(
                f"[{f.severity.upper()}] {f.message}" for f in s.bilateral_qc.flags
            )
            self.sample_list.set_status(idx, sev, tooltip)

    # ------------------------------------------------------------------
    def _on_sample_selected(self, row: int) -> None:
        if 0 <= row < len(self._samples):
            self._select_sample(row)

    def _on_bg_changed(self, text: str) -> None:
        self.canvas.set_background_theme("light" if text == "Light" else "dark")

    def _select_sample(self, idx: int) -> None:
        self._selected_idx = idx
        self._selected_tissue = None
        self.selected_label.setText("(click a tissue to select)")
        self._refresh_image()
        self._refresh_overlay()
        self._refresh_cut_lines()
        self._refresh_flags()
        self._refresh_diagnostics()

    def _current(self) -> SampleResult | None:
        if self._selected_idx is None or self._selected_idx >= len(self._samples):
            return None
        return self._samples[self._selected_idx]

    # ------------------------------------------------------------------
    def _refresh_image(self) -> None:
        s = self._current()
        if s is None:
            return
        slide_set = s.pair.slide_set
        scene_size = self._thumb_ctrl.scene_size_for(slide_set)
        if scene_size == (0, 0):
            scene_size = self._scene_size_from_samples(s)

        req = RenderRequest(
            slide_set=slide_set,
            mode=self.channel_combo.currentText(),
            downsample=32.0,
            low_pct=self.low_pct_spin.value(),
            high_pct=self.high_pct_spin.value(),
        )
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            rgb = self._thumb_ctrl.render(req)
        finally:
            QApplication.restoreOverrideCursor()
        self.canvas.set_slide_image(rgb, scene_size)
        self._refresh_overlay()
        self._refresh_cut_lines()
        self.canvas.fit_to_scene()

    def _refresh_overlay(self) -> None:
        s = self._current()
        if s is None or s.separated_file is None:
            self.canvas.set_tissue_layers([])
            return
        self.canvas.set_tissue_layers(
            s.separated_file.layers, highlight=self._selected_tissue,
        )

    def _refresh_cut_lines(self) -> None:
        # Remove old items
        for item in self._active_cut_items:
            item.remove_from_scene()
        self._active_cut_items.clear()
        self._active_cut_by_tissue.clear()

        s = self._current()
        if s is None or s.separated_file is None:
            return

        scene_w, scene_h = self.canvas.scene_size()
        bbox = (0.0, 0.0, scene_w, scene_h)

        manual = self._manual_cuts.get(self._selected_idx, {})

        for layer in s.separated_file.layers:
            if not layer.positive_regions:
                continue
            if layer.name in manual:
                cut = manual[layer.name]
            else:
                cut = detect_midline(layer, method=self._bilateral_params.midline_method)

            # Clip the displayed cut line to the tissue's bounding box for clarity
            all_pts = np.vstack([r.vertices for r in layer.positive_regions])
            mn = all_pts.min(axis=0)
            mx = all_pts.max(axis=0)
            pad = 0.15 * max(mx[0] - mn[0], mx[1] - mn[1])
            tissue_bbox = (mn[0] - pad, mn[1] - pad, mx[0] + pad, mx[1] + pad)

            cut_item = CutLineItem(cut, tissue_bbox, handle_radius=min(
                max(scene_w, scene_h) / 400, 500
            ))
            cut_item.add_to_scene(self.canvas.scene())
            cut_item.changed.connect(
                lambda c, name=layer.name: self._on_cut_edited(name, c)
            )
            self._active_cut_items.append(cut_item)
            self._active_cut_by_tissue[layer.name] = cut_item

    def _refresh_flags(self) -> None:
        self.flags_list.clear()
        s = self._current()
        if s is None or s.bilateral_qc is None:
            return
        for flag in s.bilateral_qc.flags:
            sym = {"info": "●", "warn": "▲", "error": "✕"}[flag.severity]
            color = {"info": "#2a9d5b", "warn": "#d29922", "error": "#cf4141"}[flag.severity]
            prefix = f"{flag.tissue_name}: " if flag.tissue_name else ""
            it = QListWidgetItem(f"{sym}  {prefix}{flag.message}")
            it.setForeground(QBrush(QColor(color)))
            it.setData(Qt.UserRole, flag.tissue_name)
            self.flags_list.addItem(it)

    def _refresh_diagnostics(self) -> None:
        s = self._current()
        if s is None or s.bilateral is None:
            self.diag_text.clear()
            return
        self.diag_text.setPlainText("\n".join(s.bilateral.diagnostics))

    def _on_flag_clicked(self, item: QListWidgetItem) -> None:
        tissue_name = item.data(Qt.UserRole)
        if tissue_name:
            self._on_tissue_clicked(tissue_name)

    def _on_tissue_clicked(self, tissue_name: str) -> None:
        self._selected_tissue = tissue_name
        self.selected_label.setText(f"<b>{tissue_name}</b>")
        # Zoom & highlight
        s = self._current()
        if s is None or s.separated_file is None:
            return
        for layer in s.separated_file.layers:
            if layer.name == tissue_name and layer.positive_regions:
                v = layer.positive_regions[0].vertices
                mn = v.min(axis=0); mx = v.max(axis=0)
                self.canvas.zoom_to_region((mn[0], mn[1], mx[0], mx[1]))
                self.canvas.set_tissue_layers(
                    s.separated_file.layers, highlight=tissue_name,
                )
                break

    # ------------------------------------------------------------------
    # Manual cut editing
    # ------------------------------------------------------------------
    def _on_cut_edited(self, tissue_name: str, cut: MidlineCut) -> None:
        if self._selected_idx is None:
            return
        self._manual_cuts.setdefault(self._selected_idx, {})[tissue_name] = cut

    def _rotate_selected(self, degrees: float) -> None:
        if self._selected_tissue is None:
            return
        item = self._active_cut_by_tissue.get(self._selected_tissue)
        if item is not None:
            item.rotate_about_center(degrees)

    def _translate_selected(self, dx: float, dy: float) -> None:
        if self._selected_tissue is None:
            return
        item = self._active_cut_by_tissue.get(self._selected_tissue)
        if item is not None:
            item.translate(dx, dy)

    def _reset_selected_cut(self) -> None:
        if self._selected_idx is None or self._selected_tissue is None:
            return
        s = self._current()
        if s is None or s.separated_file is None:
            return
        # Remove manual cut, redraw
        self._manual_cuts[self._selected_idx].pop(self._selected_tissue, None)
        self._refresh_cut_lines()

    # ------------------------------------------------------------------
    # Recompute
    # ------------------------------------------------------------------
    def _do_recompute(self) -> None:
        if self._selected_idx is None:
            return
        s = self._samples[self._selected_idx]
        if s.separated_file is None:
            return
        manual = self._manual_cuts.get(self._selected_idx, {})
        result = split_bilateral(
            s.separated_file,
            self._bilateral_params,
            manual_cuts=manual,
        )
        s.bilateral = result
        s.bilateral_qc = check_bilateral(
            result,
            separated_file=s.separated_file,
            sample_name=s.pair.annotation_path.stem,
        )
        self._update_sample_status(self._selected_idx)
        self._refresh_flags()
        self._refresh_diagnostics()

    # ------------------------------------------------------------------
    def current_manual_cuts(self) -> dict[int, dict[str, MidlineCut]]:
        return self._manual_cuts

    def samples(self) -> list[SampleResult]:
        return self._samples

    # ------------------------------------------------------------------
    @staticmethod
    def _scene_size_from_samples(s: SampleResult) -> tuple[int, int]:
        if s.separated_file is None:
            return (1000, 1000)
        max_x = 0; max_y = 0
        for layer in s.separated_file.layers:
            for region in layer.regions:
                v = region.vertices
                max_x = max(max_x, int(v[:, 0].max()) + 100)
                max_y = max(max_y, int(v[:, 1].max()) + 100)
        return (max_x or 1000, max_y or 1000)
