"""
Modal dialogs for manual tissue correction operations.

Each dialog is small and purpose-built. The calling view passes in the
current AnnotationFile state and the source layer name; the dialog
returns user choices via its public properties after `exec()`.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from haloqc.io.annotations import AnnotationFile, AnnotationLayer, Region


def _region_summary(region: Region, idx: int) -> str:
    """Human-readable one-liner for a region in a list dialog."""
    v = region.vertices
    c = v.mean(axis=0)
    kind = "HOLE (negative)" if region.is_negative else "positive"
    return (
        f"Region #{idx}: {kind}, {len(v)} vertices, "
        f"centroid ({int(c[0])}, {int(c[1])})"
    )


class PickTargetTissueDialog(QDialog):
    """Modal: let the user pick one tissue layer from a list.

    Used for 'Merge into...' and the target half of 'Move region to...'.
    """

    def __init__(
        self,
        af: AnnotationFile,
        exclude: str,
        title: str = "Pick target tissue",
        prompt: str = "Pick the target tissue:",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._picked_name: str | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(prompt))

        self._list = QListWidget()
        for layer in af.layers:
            if layer.name == exclude:
                continue
            item = QListWidgetItem(layer.name)
            item.setData(Qt.UserRole, layer.name)
            self._list.addItem(item)
        self._list.setCurrentRow(0)

        # Auto-fit list height so all tissue rows are visible without
        # scrolling. Each row is ~28px with the theme padding; add header
        # chrome. Cap at 80% of the primary screen so we never exceed display.
        n_items = self._list.count()
        row_px = 30
        desired_list_h = max(160, n_items * row_px + 16)
        screen = (
            self.screen() if parent is None else parent.screen()
        )
        max_dialog_h = int(screen.availableGeometry().height() * 0.80) if screen else 900
        # reserve ~120px for prompt label + buttons
        max_list_h = max(200, max_dialog_h - 120)
        self._list.setMinimumHeight(min(desired_list_h, max_list_h))

        layout.addWidget(self._list)

        # Give the dialog a reasonable minimum width so tissue names aren't
        # truncated
        self.setMinimumWidth(380)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        item = self._list.currentItem()
        if item is None:
            return
        self._picked_name = item.data(Qt.UserRole)
        self.accept()

    def picked_tissue(self) -> str | None:
        return self._picked_name


class SplitTissueDialog(QDialog):
    """Modal: show the regions of one layer as a checklist so the user can
    select which regions should be split out into a new layer.
    """

    def __init__(self, layer: AnnotationLayer, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Split {layer.name}")
        self._picked_indices: list[int] = []
        self._new_name: str | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"<b>{layer.name}</b> contains {len(layer.regions)} regions.\n"
            "Check the regions to split out into a new tissue:"
        ))

        self._checks: list[QCheckBox] = []
        list_widget = QListWidget()
        for i, region in enumerate(layer.regions):
            cb = QCheckBox(_region_summary(region, i))
            list_item = QListWidgetItem()
            list_widget.addItem(list_item)
            list_widget.setItemWidget(list_item, cb)
            self._checks.append(cb)
        list_widget.setMinimumHeight(250)
        layout.addWidget(list_widget)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("New tissue name (blank = auto):"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. Tissue_07 or Cortex_Fragment")
        name_row.addWidget(self._name_edit)
        layout.addLayout(name_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        self._picked_indices = [i for i, cb in enumerate(self._checks) if cb.isChecked()]
        if not self._picked_indices:
            QMessageBox.warning(self, "Nothing selected", "Check at least one region to split out.")
            return
        name = self._name_edit.text().strip()
        self._new_name = name or None
        self.accept()

    def picked_indices(self) -> list[int]:
        return self._picked_indices

    def new_name(self) -> str | None:
        return self._new_name


class MoveRegionsDialog(QDialog):
    """Modal: pick which regions of the source to move, and which tissue to move them to."""

    def __init__(
        self,
        af: AnnotationFile,
        source_layer: AnnotationLayer,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Move regions from {source_layer.name}")
        self._picked_indices: list[int] = []
        self._target_name: str | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"Select regions of <b>{source_layer.name}</b> to move:"
        ))

        self._checks: list[QCheckBox] = []
        list_widget = QListWidget()
        for i, region in enumerate(source_layer.regions):
            cb = QCheckBox(_region_summary(region, i))
            item = QListWidgetItem()
            list_widget.addItem(item)
            list_widget.setItemWidget(item, cb)
            self._checks.append(cb)
        list_widget.setMinimumHeight(200)
        layout.addWidget(list_widget)

        layout.addWidget(QLabel("Target tissue:"))
        self._target_list = QListWidget()
        for layer in af.layers:
            if layer.name == source_layer.name:
                continue
            it = QListWidgetItem(layer.name)
            it.setData(Qt.UserRole, layer.name)
            self._target_list.addItem(it)
        if self._target_list.count() > 0:
            self._target_list.setCurrentRow(0)
        # Auto-fit target list height (same as PickTargetTissueDialog)
        n_tgt = self._target_list.count()
        self._target_list.setMinimumHeight(max(120, min(n_tgt * 30 + 16, 400)))
        layout.addWidget(self._target_list)

        self.setMinimumWidth(480)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        self._picked_indices = [i for i, cb in enumerate(self._checks) if cb.isChecked()]
        if not self._picked_indices:
            QMessageBox.warning(self, "Nothing selected", "Check at least one region to move.")
            return
        tgt_item = self._target_list.currentItem()
        if tgt_item is None:
            QMessageBox.warning(self, "No target", "Pick a target tissue.")
            return
        self._target_name = tgt_item.data(Qt.UserRole)
        self.accept()

    def picked_indices(self) -> list[int]:
        return self._picked_indices

    def target_tissue(self) -> str | None:
        return self._target_name


class ShiftNumbersDialog(QDialog):
    """Modal: shift tissue numbers up or down by an offset, optionally
    restricted to a number range. Useful for missing-slice scenarios.

    Example: seven tissues currently labeled Tissue_01..07 need to be
    Tissue_02..08 because physical slice 1 is missing.
    """

    def __init__(self, af: AnnotationFile, parent=None, suggested_offset: int = 1,
                 suggested_from: int = 1):
        super().__init__(parent)
        self.setWindowTitle("Shift tissue numbers")
        self._offset: int = 0
        self._from_number: int = 1
        self._to_number: int | None = None

        import re
        tissue_nums = sorted(
            int(m.group(1))
            for l in af.layers
            if (m := re.match(r"^Tissue_(\d+)$", l.name))
        )

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "<b>Shift tissue numbering</b><br>"
            "Use this when physical slices are missing and tissues should be "
            "renumbered to match their grid position."
        ))

        if tissue_nums:
            layout.addWidget(QLabel(
                f"<span style='color:#888'>Currently: Tissue_"
                f"{tissue_nums[0]:02d}..Tissue_{tissue_nums[-1]:02d}</span>"
            ))

        # Offset row
        offset_row = QHBoxLayout()
        offset_row.addWidget(QLabel("Shift by:"))
        from PySide6.QtWidgets import QSpinBox
        self._offset_spin = QSpinBox()
        self._offset_spin.setRange(-50, 50)
        self._offset_spin.setValue(suggested_offset)
        self._offset_spin.setSuffix("  (positive = up, negative = down)")
        offset_row.addWidget(self._offset_spin)
        layout.addLayout(offset_row)

        # Range row (default: apply to all)
        self._range_cb = QCheckBox(
            "Only shift tissues in a number range (otherwise shift all)"
        )
        self._range_cb.setChecked(suggested_from > 1)
        layout.addWidget(self._range_cb)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("From:"))
        self._from_spin = QSpinBox()
        self._from_spin.setRange(1, 100)
        self._from_spin.setValue(suggested_from)
        self._from_spin.setPrefix("Tissue_")
        range_row.addWidget(self._from_spin)
        range_row.addWidget(QLabel("To:"))
        self._to_spin = QSpinBox()
        self._to_spin.setRange(1, 100)
        self._to_spin.setValue(tissue_nums[-1] if tissue_nums else 1)
        self._to_spin.setPrefix("Tissue_")
        range_row.addWidget(self._to_spin)
        layout.addLayout(range_row)

        self._from_spin.setEnabled(self._range_cb.isChecked())
        self._to_spin.setEnabled(self._range_cb.isChecked())
        self._range_cb.toggled.connect(self._from_spin.setEnabled)
        self._range_cb.toggled.connect(self._to_spin.setEnabled)

        preview_label = QLabel()
        preview_label.setStyleSheet("color: #4a90d9; padding: 4px 0;")
        layout.addWidget(preview_label)
        self._preview = preview_label
        self._offset_spin.valueChanged.connect(self._update_preview)
        self._from_spin.valueChanged.connect(self._update_preview)
        self._to_spin.valueChanged.connect(self._update_preview)
        self._range_cb.toggled.connect(self._update_preview)
        self._tissue_nums = tissue_nums
        self._update_preview()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_preview(self) -> None:
        offset = self._offset_spin.value()
        if offset == 0 or not self._tissue_nums:
            self._preview.setText("")
            return
        from_n = self._from_spin.value() if self._range_cb.isChecked() else 1
        to_n = self._to_spin.value() if self._range_cb.isChecked() else 999
        affected = [n for n in self._tissue_nums if from_n <= n <= to_n]
        if not affected:
            self._preview.setText("(No tissues in that range.)")
            return
        lo, hi = min(affected), max(affected)
        self._preview.setText(
            f"Preview: Tissue_{lo:02d}..Tissue_{hi:02d} → "
            f"Tissue_{lo+offset:02d}..Tissue_{hi+offset:02d}"
            f"  ({len(affected)} tissue{'s' if len(affected)!=1 else ''})"
        )

    def _on_accept(self) -> None:
        self._offset = self._offset_spin.value()
        if self._offset == 0:
            QMessageBox.warning(self, "No shift", "Pick a non-zero offset.")
            return
        if self._range_cb.isChecked():
            self._from_number = self._from_spin.value()
            self._to_number = self._to_spin.value()
            if self._from_number > self._to_number:
                QMessageBox.warning(
                    self, "Invalid range",
                    "'From' must not be greater than 'To'.",
                )
                return
        else:
            self._from_number = 1
            self._to_number = None
        self.accept()

    def offset(self) -> int:
        return self._offset

    def from_number(self) -> int:
        return self._from_number

    def to_number(self) -> int | None:
        return self._to_number


class RenameCollisionDialog(QDialog):
    """Shown when the user tries to rename Tissue_NN to Tissue_MM but MM
    already exists. Offers three options: shift everything up, swap names,
    or cancel.
    """

    CHOICE_SHIFT = "shift"
    CHOICE_SWAP = "swap"
    CHOICE_CANCEL = "cancel"

    def __init__(self, source_name: str, target_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Name already in use")
        self._choice = self.CHOICE_CANCEL

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"<b>{target_name} already exists.</b><br><br>"
            f"You're trying to rename <b>{source_name}</b> → "
            f"<b>{target_name}</b>.<br>"
            f"What would you like to do?"
        ))

        # Determine what the shift would do
        import re
        m_src = re.match(r"^Tissue_(\d+)$", source_name)
        m_tgt = re.match(r"^Tissue_(\d+)$", target_name)
        shift_available = bool(m_src and m_tgt)

        if shift_available:
            src_n = int(m_src.group(1))
            tgt_n = int(m_tgt.group(1))
            offset = tgt_n - src_n
            if offset > 0:
                shift_label = (
                    f"Shift {source_name} and all higher-numbered tissues up "
                    f"by {offset}\n"
                    f"(so {source_name} becomes {target_name}, and everything "
                    f"after shifts too)"
                )
            else:
                shift_label = (
                    f"Shift {source_name} and all lower-numbered tissues down "
                    f"by {-offset}\n"
                    f"(so {source_name} becomes {target_name}, and everything "
                    f"before shifts too)"
                )
            self._shift_btn = QPushButton(shift_label)
            self._shift_btn.clicked.connect(self._pick_shift)
            self._shift_btn.setMinimumHeight(50)
            layout.addWidget(self._shift_btn)

        self._swap_btn = QPushButton(
            f"Swap names: {source_name} ⇄ {target_name}"
        )
        self._swap_btn.clicked.connect(self._pick_swap)
        self._swap_btn.setMinimumHeight(40)
        layout.addWidget(self._swap_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self._cancel_btn)

    def _pick_shift(self) -> None:
        self._choice = self.CHOICE_SHIFT
        self.accept()

    def _pick_swap(self) -> None:
        self._choice = self.CHOICE_SWAP
        self.accept()

    def choice(self) -> str:
        return self._choice

