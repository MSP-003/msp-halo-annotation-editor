"""
Separator QC view (stage 1).

For each sample, shows:
- the slide thumbnail (composite or single channel, user-adjustable)
- the tissue separation overlay (one color per tissue, numbered)
- QC flags as a sidebar list
- a "Continue to bilateral" button when the user is satisfied

Flow:
- Left: list of all samples with status badges (info/warn/error)
- Center: large canvas
- Right: display controls + QC flag list for the selected sample
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from haloqc.core.edits import (
    EditHistory,
    delete_tissue,
    merge_tissues,
    move_regions,
    rename_tissue,
    renumber_sequential,
    shift_tissue_numbers,
    split_tissue_by_regions,
)
from haloqc.pipeline import SampleResult
from haloqc.ui.canvas import AnnotationCanvas
from haloqc.ui.edit_dialogs import (
    MoveRegionsDialog,
    PickTargetTissueDialog,
    RenameCollisionDialog,
    ShiftNumbersDialog,
    SplitTissueDialog,
)
from haloqc.ui.sample_list import SampleListWidget
from haloqc.ui.thumbnail_controller import RenderRequest, ThumbnailController


class SeparatorQCView(QWidget):
    """Stage 1 QC: review tissue separation output."""

    continue_to_bilateral = Signal()  # user clicked the advance button
    back_to_input = Signal()
    sample_edited = Signal(int)        # emitted when a sample's separation is edited;
                                        # payload is sample index

    def __init__(self, thumb_ctrl: ThumbnailController | None = None, parent=None):
        super().__init__(parent)
        self._samples: list[SampleResult] = []
        self._selected_idx: int | None = None
        self._thumb_ctrl = thumb_ctrl or ThumbnailController()
        # Per-sample edit history (single-level undo). Keyed by sample index.
        self._edit_history: dict[int, EditHistory] = {}
        # Per-sample edited flag, so the main window knows to invalidate bilateral
        self._sample_dirty: set[int] = set()
        # Currently selected tissue name (for highlight persistence)
        self._selected_tissue: str | None = None
        # Snapshot of InputSelections needed to re-run QC after edits. The
        # MainWindow sets this via set_selections() before loading samples.
        self._selections_snapshot = None

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        # Top toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(12)
        title = QLabel("Stage 1 · Tissue Separation QC")
        title.setStyleSheet(
            "font-size: 18px; font-weight: 600; background: transparent; "
            "letter-spacing: -0.2px;"
        )
        toolbar.addWidget(title)
        toolbar.addStretch(1)
        self.back_btn = QPushButton("← Back to Input")
        self.back_btn.clicked.connect(self.back_to_input.emit)
        toolbar.addWidget(self.back_btn)
        self.continue_btn = QPushButton("Approve & Continue to Bilateral  →")
        self.continue_btn.setMinimumHeight(36)
        self.continue_btn.setDefault(True)
        self.continue_btn.setStyleSheet("font-weight: 600; padding: 6px 16px;")
        self.continue_btn.clicked.connect(self.continue_to_bilateral.emit)
        toolbar.addWidget(self.continue_btn)
        root.addLayout(toolbar)

        # Main splitter: [sample list | canvas | controls+flags]
        splitter = QSplitter(Qt.Horizontal)

        # Left: sample list
        self.sample_list = SampleListWidget()
        self.sample_list.sample_selected.connect(self._on_sample_selected)
        splitter.addWidget(self.sample_list)

        # Center: canvas
        self.canvas = AnnotationCanvas()
        self.canvas.tissue_clicked.connect(self._on_tissue_clicked)
        self.canvas.tissue_right_clicked.connect(self._on_tissue_right_clicked)
        self.canvas.tissue_double_clicked.connect(self._on_tissue_double_clicked)
        splitter.addWidget(self.canvas)

        # Right: controls
        right = QWidget()
        right_layout = QVBoxLayout(right)

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
        self.low_pct_spin.setRange(0.0, 49.0)
        self.low_pct_spin.setValue(1.0)
        self.low_pct_spin.setSuffix(" lo")
        self.low_pct_spin.valueChanged.connect(self._refresh_image)
        self.high_pct_spin = QDoubleSpinBox()
        self.high_pct_spin.setRange(51.0, 100.0)
        self.high_pct_spin.setValue(99.0)
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

        # ---- Edit controls ----
        edit_box = QGroupBox("Edit")
        edit_layout = QVBoxLayout()
        hint = QLabel(
            "<span style='color:#888'>"
            "<b>Right-click</b> a tissue (on canvas or in Tissues list) for "
            "edit options.<br>"
            "<b>Double-click</b> to rename quickly.<br><br>"
            "<b>Merge</b> = combine two whole tissues into one.<br>"
            "<b>Move regions</b> = reassign specific polygons to another tissue."
            "</span>"
        )
        hint.setWordWrap(True)
        edit_layout.addWidget(hint)

        self.renumber_btn = QPushButton("Renumber tissues (reading order)")
        self.renumber_btn.clicked.connect(self._on_renumber_clicked)
        edit_layout.addWidget(self.renumber_btn)

        self.shift_btn = QPushButton("Shift numbering...")
        self.shift_btn.setToolTip(
            "Use this when a physical slice is missing and tissues need to "
            "be renumbered (e.g. slice 1 is missing, shift all up by 1)."
        )
        self.shift_btn.clicked.connect(self._on_shift_clicked)
        edit_layout.addWidget(self.shift_btn)

        self.undo_btn = QPushButton("Undo last edit")
        self.undo_btn.setShortcut(QKeySequence.Undo)
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self._on_undo_clicked)
        edit_layout.addWidget(self.undo_btn)

        edit_box.setLayout(edit_layout)
        right_layout.addWidget(edit_box)

        # Tissue list: every tissue shown by name + region count.
        # Clicking a row selects (highlights) that tissue on the canvas.
        # This makes tissues visible in the list even when they're too
        # small or overlapping to spot on the canvas.
        tissues_box = QGroupBox("Tissues")
        tissues_layout = QVBoxLayout()
        self.tissue_list = QListWidget()
        self.tissue_list.itemClicked.connect(self._on_tissue_list_clicked)
        self.tissue_list.itemDoubleClicked.connect(self._on_tissue_list_double_clicked)
        self.tissue_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tissue_list.customContextMenuRequested.connect(self._on_tissue_list_context)
        self.tissue_list.setMaximumHeight(200)
        tissues_layout.addWidget(self.tissue_list)
        tissues_box.setLayout(tissues_layout)
        right_layout.addWidget(tissues_box)

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
        self.diag_text.setMaximumHeight(180)
        diag_layout.addWidget(self.diag_text)
        diag_box.setLayout(diag_layout)
        right_layout.addWidget(diag_box)

        right_layout.addStretch(1)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([260, 900, 320])
        root.addWidget(splitter)

    # ------------------------------------------------------------------
    # Public: load samples
    # ------------------------------------------------------------------
    def set_selections(self, selections) -> None:
        """Called by MainWindow before load_samples. Captures the run params
        so edits can re-check QC against them."""
        self._selections_snapshot = selections

    def load_samples(self, samples: list[SampleResult]) -> None:
        self._samples = list(samples)
        # Fresh samples from the pipeline: clear any leftover edit state
        self._edit_history = {}
        self._sample_dirty = set()
        labels = [s.pair.annotation_path.name for s in self._samples]
        self.sample_list.populate(labels)
        # Set statuses
        for i, s in enumerate(self._samples):
            if s.error:
                self.sample_list.set_status(i, "error", s.error)
            elif s.separator_qc is None:
                self.sample_list.set_status(i, None)
            else:
                sev = s.separator_qc.max_severity
                tooltip = "\n".join(
                    f"[{f.severity.upper()}] {f.message}"
                    for f in s.separator_qc.flags
                )
                self.sample_list.set_status(i, sev, tooltip)

        if self._samples:
            self._select_sample(0)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def _on_sample_selected(self, row: int) -> None:
        if 0 <= row < len(self._samples):
            self._select_sample(row)

    def _on_bg_changed(self, text: str) -> None:
        self.canvas.set_background_theme("light" if text == "Light" else "dark")

    def _select_sample(self, idx: int) -> None:
        self._selected_idx = idx
        self._selected_tissue = None  # reset selection when switching samples
        self._refresh_image()
        self._refresh_overlay()
        self._refresh_tissue_list()
        self._refresh_flags()
        self._refresh_diagnostics()
        self._update_undo_button()

    def _current(self) -> SampleResult | None:
        if self._selected_idx is None or self._selected_idx >= len(self._samples):
            return None
        return self._samples[self._selected_idx]

    # ------------------------------------------------------------------
    # Refreshes
    # ------------------------------------------------------------------
    def _refresh_image(self) -> None:
        s = self._current()
        if s is None:
            return
        slide_set = s.pair.slide_set
        scene_size = self._thumb_ctrl.scene_size_for(slide_set)
        # Fall back to bbox of annotations if no slide
        if scene_size == (0, 0) and s.separated_file is not None:
            scene_size = _scene_size_from_annotations(s)

        mode = self.channel_combo.currentText()
        req = RenderRequest(
            slide_set=slide_set,
            mode=mode,
            downsample=32.0,
            low_pct=self.low_pct_spin.value(),
            high_pct=self.high_pct_spin.value(),
        )
        # Show busy cursor while reading NDPI (can take several seconds first time)
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            rgb = self._thumb_ctrl.render(req)
        finally:
            QApplication.restoreOverrideCursor()
        self.canvas.set_slide_image(rgb, scene_size)
        # Re-draw overlays on top
        self._refresh_overlay()
        self.canvas.fit_to_scene()

    def _refresh_overlay(self) -> None:
        s = self._current()
        if s is None or s.separated_file is None:
            self.canvas.set_tissue_layers([])
            return
        # Only highlight if the selected tissue still exists (edits may have
        # renamed or deleted it)
        if self._selected_tissue is not None:
            if not any(l.name == self._selected_tissue for l in s.separated_file.layers):
                self._selected_tissue = None
        self.canvas.set_tissue_layers(
            s.separated_file.layers,
            highlight=self._selected_tissue,
        )

    def _refresh_flags(self) -> None:
        self.flags_list.clear()
        s = self._current()
        if s is None or s.separator_qc is None:
            return
        for flag in s.separator_qc.flags:
            sym = {"info": "●", "warn": "▲", "error": "✕"}[flag.severity]
            color = {"info": "#2a9d5b", "warn": "#d29922", "error": "#cf4141"}[flag.severity]
            prefix = f"{flag.tissue_name}: " if flag.tissue_name else ""
            it = QListWidgetItem(f"{sym}  {prefix}{flag.message}")
            it.setForeground(QBrush(QColor(color)))
            it.setData(Qt.UserRole, flag.tissue_name)
            self.flags_list.addItem(it)

    def _refresh_diagnostics(self) -> None:
        """Render diagnostics from the separator run PLUS the current edited
        state of the tissues. The second section reflects edits made after
        the initial separation ran."""
        s = self._current()
        if s is None:
            self.diag_text.clear()
            return

        lines: list[str] = []
        if s.separation is not None:
            lines.append("=== Initial separation ===")
            lines.extend(s.separation.diagnostics)

        if s.separated_file is not None:
            lines.append("")
            lines.append(f"=== Current state ({len(s.separated_file.layers)} tissues) ===")
            import numpy as np
            for layer in s.separated_file.layers:
                pos = layer.positive_regions
                neg = layer.negative_regions
                if pos:
                    c = np.mean([r.vertices.mean(axis=0) for r in pos], axis=0)
                    centroid_str = f"centroid=({int(c[0])}, {int(c[1])}) px"
                else:
                    centroid_str = "NO POSITIVE REGIONS"
                lines.append(
                    f"  {layer.name}: {len(pos)} positive, {len(neg)} negative, {centroid_str}"
                )

        self.diag_text.setPlainText("\n".join(lines))

    def _on_flag_clicked(self, item: QListWidgetItem) -> None:
        tissue_name = item.data(Qt.UserRole)
        if not tissue_name:
            return
        s = self._current()
        if s is None or s.separated_file is None:
            return
        for layer in s.separated_file.layers:
            if layer.name == tissue_name and layer.positive_regions:
                # Zoom to bbox
                verts = layer.positive_regions[0].vertices
                mn = verts.min(axis=0); mx = verts.max(axis=0)
                self.canvas.zoom_to_region((mn[0], mn[1], mx[0], mx[1]))
                self.canvas.set_tissue_layers(s.separated_file.layers, highlight=tissue_name)
                break

    # ------------------------------------------------------------------
    # Editing: context menu, rename, merge, split, move, delete
    # ------------------------------------------------------------------
    def _on_tissue_clicked(self, layer_name: str) -> None:
        """Left-click on a tissue outline: select it (highlight)."""
        self._selected_tissue = layer_name
        self._refresh_overlay()

    def _on_tissue_list_clicked(self, item) -> None:
        """Left-click a row in the Tissues list: select and zoom to it."""
        name = item.data(Qt.UserRole)
        if not name:
            return
        self._selected_tissue = name
        self._refresh_overlay()
        self._zoom_to_tissue(name)

    def _on_tissue_list_double_clicked(self, item) -> None:
        name = item.data(Qt.UserRole)
        if not name:
            return
        self._do_rename(name)

    def _on_tissue_list_context(self, pos) -> None:
        """Right-click a row in the Tissues list: open the edit context menu."""
        item = self.tissue_list.itemAt(pos)
        if item is None:
            return
        name = item.data(Qt.UserRole)
        if not name:
            return
        # Select and zoom first so the user can see what they're about to edit
        self._selected_tissue = name
        self._refresh_overlay()
        self._zoom_to_tissue(name)
        # Translate local widget coords into global coords for the menu
        screen_pos = self.tissue_list.viewport().mapToGlobal(pos)
        self._on_tissue_right_clicked(name, screen_pos)

    def _zoom_to_tissue(self, name: str) -> None:
        s = self._current()
        if s is None or s.separated_file is None:
            return
        for layer in s.separated_file.layers:
            if layer.name == name and layer.positive_regions:
                import numpy as np
                all_pts = np.vstack([r.vertices for r in layer.positive_regions])
                mn = all_pts.min(axis=0); mx = all_pts.max(axis=0)
                self.canvas.zoom_to_region((mn[0], mn[1], mx[0], mx[1]))
                return

    def _refresh_tissue_list(self) -> None:
        """Rebuild the right-sidebar Tissues list from the current sample."""
        self.tissue_list.clear()
        s = self._current()
        if s is None or s.separated_file is None:
            return

        import numpy as np
        from haloqc.core.colors import rgb_for_layer

        for layer in s.separated_file.layers:
            pos = layer.positive_regions
            neg = layer.negative_regions
            # Compute a size proxy for the row label (helps spot tiny/stray tissues)
            if pos:
                all_pts = np.vstack([r.vertices for r in pos])
                bbox_w = int(all_pts[:, 0].max() - all_pts[:, 0].min())
                bbox_h = int(all_pts[:, 1].max() - all_pts[:, 1].min())
                size_str = f"{bbox_w}×{bbox_h}px"
            else:
                size_str = "NO POSITIVE"
            text = (
                f"{layer.name}  —  {len(pos)} pos / {len(neg)} neg  —  {size_str}"
            )
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, layer.name)
            # Color-code the row with the tissue's palette color so the list
            # correlates visually with the canvas overlay
            r, g, b = rgb_for_layer(layer.name)
            item.setForeground(QBrush(QColor(r, g, b)))
            self.tissue_list.addItem(item)

    def _on_tissue_right_clicked(self, layer_name: str, screen_pos) -> None:
        """Show the context menu when the user right-clicks a tissue outline."""
        s = self._current()
        if s is None or s.separated_file is None:
            return

        # Set selection so the user sees which tissue the menu will operate on
        self._selected_tissue = layer_name
        self._refresh_overlay()

        menu = QMenu(self)
        menu.setTitle(layer_name)
        # Header row so the user sees exactly which tissue is about to change
        header = menu.addAction(f"⦿  {layer_name}")
        header.setEnabled(False)
        menu.addSeparator()

        act_rename = menu.addAction("Rename...")
        act_merge = menu.addAction("Merge entire tissue into another...")
        act_split = menu.addAction("Split this tissue into multiple tissues...")
        act_move = menu.addAction("Move some regions to another tissue...")
        menu.addSeparator()
        act_delete = menu.addAction("Delete this tissue")

        chosen = menu.exec(screen_pos)
        if chosen is None:
            return
        if chosen is act_rename:
            self._do_rename(layer_name)
        elif chosen is act_merge:
            self._do_merge(layer_name)
        elif chosen is act_split:
            self._do_split(layer_name)
        elif chosen is act_move:
            self._do_move(layer_name)
        elif chosen is act_delete:
            self._do_delete(layer_name)

    def _on_tissue_double_clicked(self, layer_name: str) -> None:
        """Double-click on a tissue = quick rename."""
        self._selected_tissue = layer_name
        self._refresh_overlay()
        self._do_rename(layer_name)

    # --- individual edit operations ---

    def _do_rename(self, layer_name: str) -> None:
        new_name, ok = QInputDialog.getText(
            self, "Rename tissue",
            f"New name for {layer_name}:",
            text=layer_name,
        )
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name or new_name == layer_name:
            return

        s = self._current()
        if s is None or s.separated_file is None:
            return

        # Check for collision before trying the edit, so we can offer the
        # smart resolution dialog
        collision = any(l.name == new_name for l in s.separated_file.layers)
        if collision:
            resolved = self._resolve_rename_collision(layer_name, new_name)
            if resolved is None:
                return  # user cancelled
            self._apply_edit(resolved, label=f"rename {layer_name} → {new_name}",
                             renumber_after=False)
            return

        self._apply_edit(
            lambda af: rename_tissue(af, layer_name, new_name),
            label=f"rename {layer_name} → {new_name}",
            renumber_after=False,
        )

    def _resolve_rename_collision(self, source_name: str, target_name: str):
        """Show a dialog asking the user how to resolve a rename collision.

        Returns a lambda suitable for _apply_edit, or None if the user cancels.
        """
        dialog = RenameCollisionDialog(source_name, target_name, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        choice = dialog.choice()
        if choice == RenameCollisionDialog.CHOICE_SHIFT:
            # Compute the shift parameters that achieve source_name → target_name
            import re
            m_src = re.match(r"^Tissue_(\d+)$", source_name)
            m_tgt = re.match(r"^Tissue_(\d+)$", target_name)
            if not (m_src and m_tgt):
                return None
            src_n = int(m_src.group(1))
            tgt_n = int(m_tgt.group(1))
            offset = tgt_n - src_n
            if offset > 0:
                # shift everything from src_n up by offset
                return lambda af: shift_tissue_numbers(
                    af, offset, from_number=src_n,
                )
            else:
                # shift everything from 1..src_n DOWN by -offset
                return lambda af: shift_tissue_numbers(
                    af, offset, from_number=1, to_number=src_n,
                )
        elif choice == RenameCollisionDialog.CHOICE_SWAP:
            # Three-step swap using a temp name
            def do_swap(af):
                import re
                existing_numbers = set()
                for l in af.layers:
                    m = re.match(r"^Tissue_(\d+)$", l.name)
                    if m:
                        existing_numbers.add(int(m.group(1)))
                tmp = 99
                while tmp in existing_numbers:
                    tmp += 1
                tmp_name = f"Tissue_{tmp:02d}"
                af2 = rename_tissue(af, target_name, tmp_name)
                af2 = rename_tissue(af2, source_name, target_name)
                af2 = rename_tissue(af2, tmp_name, source_name)
                return af2
            return do_swap
        return None

    def _on_shift_clicked(self) -> None:
        s = self._current()
        if s is None or s.separated_file is None:
            return
        dialog = ShiftNumbersDialog(s.separated_file, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        offset = dialog.offset()
        from_n = dialog.from_number()
        to_n = dialog.to_number()
        self._apply_edit(
            lambda af: shift_tissue_numbers(
                af, offset, from_number=from_n, to_number=to_n,
            ),
            label=f"shift by {offset:+d}",
            renumber_after=False,  # shift itself is the renumbering
        )

    def _do_merge(self, layer_name: str) -> None:
        s = self._current()
        if s is None or s.separated_file is None:
            return
        dialog = PickTargetTissueDialog(
            s.separated_file,
            exclude=layer_name,
            title=f"Merge {layer_name}",
            prompt=f"Merge {layer_name} into which tissue?",
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        target = dialog.picked_tissue()
        if not target:
            return
        self._apply_edit(
            lambda af: merge_tissues(af, layer_name, target),
            label=f"merge {layer_name} → {target}",
            renumber_after=True,
        )

    def _do_split(self, layer_name: str) -> None:
        s = self._current()
        if s is None or s.separated_file is None:
            return
        layer = next(
            (l for l in s.separated_file.layers if l.name == layer_name), None,
        )
        if layer is None:
            return
        dialog = SplitTissueDialog(layer, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        indices = dialog.picked_indices()
        new_name = dialog.new_name()
        self._apply_edit(
            lambda af: split_tissue_by_regions(af, layer_name, indices, new_name),
            label=f"split {layer_name}",
            renumber_after=True,
        )

    def _do_move(self, layer_name: str) -> None:
        s = self._current()
        if s is None or s.separated_file is None:
            return
        layer = next(
            (l for l in s.separated_file.layers if l.name == layer_name), None,
        )
        if layer is None:
            return
        dialog = MoveRegionsDialog(s.separated_file, layer, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        indices = dialog.picked_indices()
        target = dialog.target_tissue()
        if not indices or not target:
            return
        self._apply_edit(
            lambda af: move_regions(af, layer_name, indices, target),
            label=f"move regions from {layer_name} → {target}",
            renumber_after=False,
        )

    def _do_delete(self, layer_name: str) -> None:
        reply = QMessageBox.question(
            self, "Delete tissue",
            f"Remove {layer_name} entirely? This will delete all its regions "
            "from the output file.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._apply_edit(
            lambda af: delete_tissue(af, layer_name),
            label=f"delete {layer_name}",
            renumber_after=True,
        )

    # --- edit plumbing ---

    def _apply_edit(self, fn, *, label: str, renumber_after: bool) -> None:
        """Run an edit function on the current sample's separated_file.

        Snapshots history, applies the edit, optionally re-numbers, refreshes
        the canvas, marks the sample dirty, and emits sample_edited.
        Shows a QMessageBox on validation errors (e.g. "A layer named X already exists").
        """
        if self._selected_idx is None:
            return
        s = self._samples[self._selected_idx]
        if s.separated_file is None:
            return

        history = self._edit_history.setdefault(self._selected_idx, EditHistory())
        history.record(s.separated_file, label)

        try:
            new_af = fn(s.separated_file)
        except ValueError as e:
            QMessageBox.warning(self, "Edit rejected", str(e))
            history.snapshot = None  # don't leave a stale snapshot that wasn't really an edit
            history.label = None
            return

        if renumber_after:
            new_af = renumber_sequential(new_af)

        s.separated_file = new_af
        self._sample_dirty.add(self._selected_idx)

        # Recompute QC flags against the edited state so the flag list and
        # sample-list status badge reflect the new tissue count, areas, etc.
        self._recompute_separator_qc(s)

        self._refresh_overlay()
        self._refresh_tissue_list()
        self._refresh_flags()
        self._refresh_diagnostics()
        self._refresh_sample_status(self._selected_idx)
        self._update_undo_button()
        self.sample_edited.emit(self._selected_idx)

    def _recompute_separator_qc(self, sample: SampleResult) -> None:
        """Re-run the separator QC checks on the edited tissues.

        The separation diagnostics (initial output) stay as-is, but the flag
        list gets updated to reflect the current tissue count, area outliers,
        etc. This matters because after merging two small fragments, for
        example, a "tissue count too low" warning might resolve.
        """
        from haloqc.core.qc import check_separation
        from haloqc.core.separator import SeparationResult, TissueGroup

        if sample.separated_file is None:
            return
        if self._selections_snapshot is None:
            return

        # Rebuild TissueGroups from the edited layers so check_separation can
        # work against them. Each edited layer becomes one TissueGroup.
        groups = [
            TissueGroup(regions=list(l.regions))
            for l in sample.separated_file.layers
        ]
        synthetic_result = SeparationResult(
            groups=groups,
            diagnostics=[],
            nearest_distances_um=[],  # not recomputed; QC check tolerates empty
        )
        sample.separator_qc = check_separation(
            synthetic_result,
            expected_tissues=self._selections_snapshot.separator_params.expected_tissues,
            merge_threshold_um=self._selections_snapshot.separator_params.merge_distance_microns,
            sample_name=sample.pair.annotation_path.stem,
        )

    def _on_undo_clicked(self) -> None:
        if self._selected_idx is None:
            return
        history = self._edit_history.get(self._selected_idx)
        if history is None or not history.can_undo():
            return
        s = self._samples[self._selected_idx]
        restored, _label = history.undo()
        s.separated_file = restored
        # Sample may still be dirty because earlier edits may have happened
        # before the snapshot we just restored. Safer to leave dirty flag set.
        self._recompute_separator_qc(s)
        self._refresh_overlay()
        self._refresh_tissue_list()
        self._refresh_flags()
        self._refresh_diagnostics()
        self._update_undo_button()
        self.sample_edited.emit(self._selected_idx)

    def _on_renumber_clicked(self) -> None:
        self._apply_edit(
            lambda af: renumber_sequential(af),
            label="renumber",
            renumber_after=False,  # fn already does it
        )

    def _update_undo_button(self) -> None:
        if self._selected_idx is None:
            self.undo_btn.setEnabled(False)
            return
        history = self._edit_history.get(self._selected_idx)
        self.undo_btn.setEnabled(bool(history and history.can_undo()))

    def _refresh_sample_status(self, idx: int) -> None:
        """Re-compute the list-item status for one sample (e.g. after an edit)."""
        s = self._samples[idx]
        dirty_marker = " (edited)" if idx in self._sample_dirty else ""
        base_name = s.pair.annotation_path.name
        # Temporarily update the stored label so the symbol prefix uses the
        # dirty indicator
        item = self.sample_list.item(idx)
        if item is not None:
            item.setData(Qt.UserRole + 1, base_name + dirty_marker)
        if s.error:
            self.sample_list.set_status(idx, "error", s.error)
        elif s.separator_qc is None:
            self.sample_list.set_status(idx, None)
        else:
            sev = s.separator_qc.max_severity
            tooltip = "\n".join(
                f"[{f.severity.upper()}] {f.message}"
                for f in s.separator_qc.flags
            )
            if idx in self._sample_dirty:
                tooltip = "(edited — bilateral will re-run on Continue)\n" + tooltip
            self.sample_list.set_status(idx, sev, tooltip)

    # ------------------------------------------------------------------
    # Public: expose dirty samples so MainWindow can invalidate bilateral
    # ------------------------------------------------------------------
    def dirty_samples(self) -> set[int]:
        """Return indices of samples whose separated_file has been edited
        since the last bilateral run. MainWindow uses this to force re-run."""
        return set(self._sample_dirty)

    def clear_dirty(self) -> None:
        """Called by MainWindow after bilateral has been re-run on dirty samples."""
        self._sample_dirty.clear()
        # Refresh all sample list entries so the (edited) marker goes away
        for i in range(len(self._samples)):
            self._refresh_sample_status(i)


def _scene_size_from_annotations(s: SampleResult) -> tuple[int, int]:
    """Fallback: estimate scene size from max vertex coords."""
    max_x = 0
    max_y = 0
    if s.separated_file is None:
        return (1000, 1000)
    for layer in s.separated_file.layers:
        for region in layer.regions:
            v = region.vertices
            max_x = max(max_x, int(v[:, 0].max()) + 100)
            max_y = max(max_y, int(v[:, 1].max()) + 100)
    return (max_x or 1000, max_y or 1000)
