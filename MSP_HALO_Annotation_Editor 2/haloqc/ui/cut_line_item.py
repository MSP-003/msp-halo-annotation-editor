"""
Interactive midline cut line for the bilateral QC view.

Renders as a line with two draggable endpoint handles and a central
translation handle. Emits a signal whenever the geometry changes so the
parent view can recompute the bilateral split for that tissue.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
)

from haloqc.core.bilateral import MidlineCut


class _EndpointHandle(QGraphicsEllipseItem):
    """Draggable circle endpoint. Notifies parent on move."""

    def __init__(self, radius: float, parent_line: "CutLineItem", index: int):
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        self._parent_line = parent_line
        self._index = index
        self.setBrush(QBrush(QColor("#ffcc00")))
        pen = QPen(QColor("#ffffff"))
        pen.setWidthF(radius * 0.25)
        self.setPen(pen)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setZValue(101)
        self.setCursor(Qt.CrossCursor)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self._parent_line._on_handle_moved()
        return super().itemChange(change, value)


class CutLineItem(QObject):
    """
    A cut line composed of a line segment plus two draggable endpoints.

    Managed as a group of QGraphicsItems added to a scene, but exposes a
    unified API and signal. The scene must be set via `add_to_scene`.
    """

    changed = Signal(object)  # emits the updated MidlineCut

    def __init__(
        self,
        cut: MidlineCut,
        scene_bbox: tuple[float, float, float, float],
        *,
        color: str = "#ffcc00",
        handle_radius: float = 250.0,
    ):
        super().__init__()
        self._cut = cut
        self._scene_bbox = scene_bbox
        self._color = color
        self._handle_radius = handle_radius

        # Current endpoints (in scene coords)
        p1, p2 = cut.endpoints_through_bbox(scene_bbox)
        self._p1 = np.asarray(p1, dtype=float)
        self._p2 = np.asarray(p2, dtype=float)

        # The visible line
        self._line = QGraphicsLineItem()
        pen = QPen(QColor(color))
        pen.setWidthF(handle_radius * 0.3)
        pen.setStyle(Qt.DashLine)
        pen.setCosmetic(False)
        self._line.setPen(pen)
        self._line.setZValue(100)

        # Two endpoint handles
        self._handles = [
            _EndpointHandle(handle_radius, self, 0),
            _EndpointHandle(handle_radius, self, 1),
        ]
        self._suspend_signal = False
        self._update_geometry_from_endpoints()

    # ------------------------------------------------------------------
    # Scene attachment
    # ------------------------------------------------------------------
    def add_to_scene(self, scene) -> None:
        scene.addItem(self._line)
        for h in self._handles:
            scene.addItem(h)
        self._suspend_signal = True
        try:
            self._handles[0].setPos(self._p1[0], self._p1[1])
            self._handles[1].setPos(self._p2[0], self._p2[1])
        finally:
            self._suspend_signal = False

    def remove_from_scene(self) -> None:
        for h in self._handles:
            if h.scene() is not None:
                h.scene().removeItem(h)
        if self._line.scene() is not None:
            self._line.scene().removeItem(self._line)

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    def _on_handle_moved(self) -> None:
        """Called when either endpoint handle is dragged."""
        if self._suspend_signal:
            return
        p1 = self._handles[0].pos()
        p2 = self._handles[1].pos()
        self._p1 = np.array([p1.x(), p1.y()])
        self._p2 = np.array([p2.x(), p2.y()])
        self._update_geometry_from_endpoints(update_handles=False)
        self.changed.emit(self.current_cut())

    def _update_geometry_from_endpoints(self, *, update_handles: bool = True) -> None:
        self._line.setLine(
            self._p1[0], self._p1[1], self._p2[0], self._p2[1]
        )
        if update_handles:
            self._suspend_signal = True
            try:
                self._handles[0].setPos(self._p1[0], self._p1[1])
                self._handles[1].setPos(self._p2[0], self._p2[1])
            finally:
                self._suspend_signal = False

    def current_cut(self) -> MidlineCut:
        """Return the current MidlineCut derived from endpoint positions."""
        return MidlineCut.from_endpoints(self._p1, self._p2)

    def set_cut(self, cut: MidlineCut) -> None:
        """Replace the cut (used when user picks a tissue programmatically)."""
        self._cut = cut
        p1, p2 = cut.endpoints_through_bbox(self._scene_bbox)
        self._p1 = np.asarray(p1, dtype=float)
        self._p2 = np.asarray(p2, dtype=float)
        self._update_geometry_from_endpoints(update_handles=True)

    def translate(self, dx: float, dy: float) -> None:
        """Translate the cut line by (dx, dy) in scene coords."""
        self._p1 = self._p1 + np.array([dx, dy])
        self._p2 = self._p2 + np.array([dx, dy])
        self._update_geometry_from_endpoints()
        self.changed.emit(self.current_cut())

    def rotate_about_center(self, degrees: float) -> None:
        """Rotate both endpoints about the cut's midpoint."""
        theta = np.radians(degrees)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        center = (self._p1 + self._p2) / 2
        for attr in ("_p1", "_p2"):
            p = getattr(self, attr) - center
            rotated = np.array([
                cos_t * p[0] - sin_t * p[1],
                sin_t * p[0] + cos_t * p[1],
            ])
            setattr(self, attr, rotated + center)
        self._update_geometry_from_endpoints()
        self.changed.emit(self.current_cut())
