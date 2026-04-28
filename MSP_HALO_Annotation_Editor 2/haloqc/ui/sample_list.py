"""
Sample list widget showing all samples in the batch with QC status badges.
Used on the left side of both separator and bilateral QC views.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import QListWidget, QListWidgetItem


_SEVERITY_COLORS = {
    "info": "#2a9d5b",
    "warn": "#d29922",
    "error": "#cf4141",
    None: "#555",
}


class SampleListWidget(QListWidget):
    """Left-side list of samples, each showing a status color + name."""
    sample_selected = Signal(int)  # row index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.setMinimumWidth(260)
        self.setUniformItemSizes(True)
        self.currentRowChanged.connect(self._on_row_changed)

    def populate(self, sample_labels: list[str]) -> None:
        self.clear()
        for label in sample_labels:
            it = QListWidgetItem(label)
            it.setForeground(QBrush(QColor("#eee")))
            # Store the original label so set_status can rebuild the display text
            it.setData(Qt.UserRole + 1, label)
            self.addItem(it)
        if self.count() > 0:
            self.setCurrentRow(0)

    def set_status(self, row: int, severity: str | None, detail: str = "") -> None:
        item = self.item(row)
        if item is None:
            return
        color = _SEVERITY_COLORS.get(severity, _SEVERITY_COLORS[None])
        symbol = {"info": "●", "warn": "▲", "error": "✕"}.get(severity, "○")
        # Pull the original label from UserRole+1
        base_name = item.data(Qt.UserRole + 1) or item.text()
        item.setText(f"{symbol}  {base_name}")
        item.setForeground(QBrush(QColor(color)))
        if detail:
            item.setToolTip(detail)

    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            self.sample_selected.emit(row)
