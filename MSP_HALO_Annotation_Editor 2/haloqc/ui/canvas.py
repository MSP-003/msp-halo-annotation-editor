"""
QGraphicsView-based canvas for showing an NDPI thumbnail plus annotation
polygon overlays. Coordinates in the scene are pixels at NDPI level 0
(full resolution), which is the same coordinate system the .annotations
files use, so overlays don't need transformation.

The image is drawn into the scene at its full level-0 size via a
QPixmap with a scale transform that maps thumbnail pixels -> level-0
pixels. This lets us use the same scene coordinate system for everything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from haloqc.io.annotations import AnnotationLayer, Region


# Colors for tissue labeling overlays (cycled)
from haloqc.core.colors import TISSUE_PALETTE_HEX, rgb_for_layer

# Kept for back-compat imports; canonical source is haloqc.core.colors
TISSUE_PALETTE = TISSUE_PALETTE_HEX


def color_for_layer(name: str) -> QColor:
    """Return the QColor used for a layer's outline and fill.

    Delegates to the shared palette in haloqc.core.colors so that the visual
    we show on the canvas matches what HALO will render when the output file
    is re-imported. Plain tissue layers get the palette color; ipsi/contra
    pairs share a hue and differ in lightness.
    """
    r, g, b = rgb_for_layer(name)
    return QColor(r, g, b)


def _numpy_to_qimage(arr: np.ndarray) -> QImage:
    """Convert an RGB uint8 numpy array (H, W, 3) into a QImage.

    The QImage must be copied out of the numpy buffer because PySide doesn't
    keep the array alive on its own.
    """
    if arr.ndim == 2:
        # Grayscale -> stack to RGB
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    h, w, _ = arr.shape
    # Use RGB888 format; make contiguous and copy
    arr = np.ascontiguousarray(arr)
    qimg = QImage(arr.data, w, h, arr.strides[0], QImage.Format_RGB888).copy()
    return qimg


@dataclass
class TissueOverlayStyle:
    line_width: float = 8.0            # in scene (level-0) pixels
    positive_alpha: int = 255          # stroke opacity
    negative_alpha: int = 200          # for hole outlines
    fill_alpha: int = 30               # interior fill alpha
    # Label font size is a fraction of the tissue's minimum bbox dimension.
    # This keeps labels visually similar-sized regardless of scanner resolution
    # or slide layout. 0.15 means the label height is ~15% of the tissue's
    # smaller dimension — big enough to read from a fit-to-view, small enough
    # not to obscure the tissue when zoomed.
    label_font_fraction: float = 0.15
    label_font_px_min: int = 200       # floor for very small tissues
    label_font_px_max: int = 2500      # ceiling for very large tissues


class AnnotationCanvas(QGraphicsView):
    """
    Pan/zoom canvas that shows a slide thumbnail plus annotation overlays.

    Use:
        canvas.set_slide_image(arr_rgb, scene_size_px=(W, H))
        canvas.set_tissue_layers([layer1, layer2, ...])   # one per tissue
    """

    tissue_clicked = Signal(str)                # layer.name for the clicked tissue
    tissue_right_clicked = Signal(str, object)  # layer_name, QPoint (screen coords)
    tissue_double_clicked = Signal(str)          # layer.name
    scene_clicked = Signal(QPointF)              # emitted on left-click in empty scene space

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        self._image_item: QGraphicsPixmapItem | None = None
        self._tissue_items: list[QGraphicsPathItem] = []
        self._label_items: list[QGraphicsSimpleTextItem] = []
        self._scene_size = (1000.0, 1000.0)  # level-0 width, height
        self._style = TissueOverlayStyle()
        self._background_theme = "dark"
        self.set_background_theme("dark")

    def set_background_theme(self, theme: str) -> None:
        """Set the canvas background color. theme = 'dark' or 'light'.
        Matches the app theme bg_app value for a seamless look."""
        self._background_theme = theme
        if theme == "light":
            # match theme.LIGHT.bg_app
            self.setBackgroundBrush(QBrush(QColor("#f5f5f7")))
        else:
            # match theme.DARK.bg_app
            self.setBackgroundBrush(QBrush(QColor("#1b1d21")))

    def background_theme(self) -> str:
        return self._background_theme

    # ------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------
    def set_slide_image(
        self,
        rgb_array: np.ndarray | None,
        scene_size_px: tuple[int, int],
    ) -> None:
        """Set the background slide image.

        `rgb_array` is an (H, W, 3) uint8 thumbnail. `scene_size_px` is the
        full-resolution (level-0) size that this thumbnail represents. The
        thumbnail is scaled up to scene coordinates so it aligns with
        annotation coordinates.
        """
        if self._image_item is not None:
            self._scene.removeItem(self._image_item)
            self._image_item = None

        W0, H0 = scene_size_px
        self._scene_size = (float(W0), float(H0))

        # Extend the scene rect beyond the slide extent so the user can scroll
        # past the slide edges when zoomed in. Cut-line handles near the
        # bottom/right of the last row of tissues otherwise become unreachable
        # because Qt won't scroll past the scene rect.
        pad_x = (W0 or 1000) * 0.15
        pad_y = (H0 or 1000) * 0.15

        if rgb_array is None:
            # Empty placeholder scene
            self._scene.setSceneRect(
                -pad_x, -pad_y,
                (W0 or 1000) + 2 * pad_x,
                (H0 or 1000) + 2 * pad_y,
            )
            return

        qimg = _numpy_to_qimage(rgb_array)
        pix = QPixmap.fromImage(qimg)
        item = QGraphicsPixmapItem(pix)
        th, tw = rgb_array.shape[:2]
        # Scale the thumbnail up to level-0 pixel size
        if tw > 0 and th > 0 and W0 > 0 and H0 > 0:
            sx = W0 / tw
            sy = H0 / th
            t = QTransform()
            t.scale(sx, sy)
            item.setTransform(t)
        item.setZValue(-10)
        self._scene.addItem(item)
        self._image_item = item
        self._scene.setSceneRect(
            -pad_x, -pad_y,
            W0 + 2 * pad_x,
            H0 + 2 * pad_y,
        )

    # ------------------------------------------------------------------
    # Tissue overlays
    # ------------------------------------------------------------------
    def set_tissue_layers(
        self,
        layers: list[AnnotationLayer],
        *,
        highlight: str | None = None,
        show_labels: bool = True,
    ) -> None:
        """Replace all tissue overlays with the given layers."""
        for it in self._tissue_items:
            self._scene.removeItem(it)
        for it in self._label_items:
            self._scene.removeItem(it)
        self._tissue_items.clear()
        self._label_items.clear()

        for i, layer in enumerate(layers):
            color = color_for_layer(layer.name)
            path = self._build_tissue_path(layer)
            is_highlight = (highlight == layer.name)

            pen = QPen(color)
            # Selected tissue gets a much thicker border so it's unmistakable
            pen.setWidthF(self._style.line_width * (2.5 if is_highlight else 1.0))
            pen.setCosmetic(False)
            pen.setJoinStyle(Qt.RoundJoin)

            fill_color = QColor(color)
            fill_color.setAlpha(
                self._style.fill_alpha * (5 if is_highlight else 1)
            )

            item = _ClickablePathItem(path, layer.name)
            item.setPen(pen)
            item.setBrush(QBrush(fill_color))
            # Selected tissue renders above others so its highlighted fill
            # isn't obscured by neighbors
            item.setZValue(15 if is_highlight else 5)
            item.clicked.connect(self.tissue_clicked)
            item.right_clicked.connect(self.tissue_right_clicked)
            item.double_clicked.connect(self.tissue_double_clicked)
            self._scene.addItem(item)
            self._tissue_items.append(item)

            if show_labels and layer.positive_regions:
                label = QGraphicsSimpleTextItem(layer.name)
                # Scale font to tissue size: use a fraction of the smaller
                # bbox dimension so squat and tall tissues both get readable
                # labels without dwarfing the tissue itself.
                all_pts = np.vstack([r.vertices for r in layer.positive_regions])
                bbox_w = float(all_pts[:, 0].max() - all_pts[:, 0].min())
                bbox_h = float(all_pts[:, 1].max() - all_pts[:, 1].min())
                target_px = min(bbox_w, bbox_h) * self._style.label_font_fraction
                target_px = max(self._style.label_font_px_min, target_px)
                target_px = min(self._style.label_font_px_max, target_px)
                font = label.font()
                font.setPixelSize(int(target_px))
                font.setBold(True)
                label.setFont(font)
                label.setBrush(QBrush(color))
                # Outline the text so it stays legible on any background color
                # (white, black, or fluorescence signal). We fake an outline by
                # using a slightly darker/lighter pen.
                from PySide6.QtGui import QPen as _QPen
                outline = QColor(0, 0, 0) if color.lightnessF() > 0.5 else QColor(255, 255, 255)
                pen_label = _QPen(outline)
                pen_label.setWidthF(max(2.0, target_px * 0.04))
                label.setPen(pen_label)
                # Position at centroid of first positive region
                c = layer.positive_regions[0].vertices.mean(axis=0)
                rect = label.boundingRect()
                label.setPos(c[0] - rect.width() / 2, c[1] - rect.height() / 2)
                label.setZValue(20)
                self._scene.addItem(label)
                self._label_items.append(label)

    def _build_tissue_path(self, layer: AnnotationLayer) -> QPainterPath:
        """One QPainterPath per layer combining all regions via even-odd rule."""
        path = QPainterPath()
        path.setFillRule(Qt.OddEvenFill)
        # Add positives first, then negatives - even-odd fill makes negatives holes
        for reg in layer.positive_regions + layer.negative_regions:
            sub = self._region_to_subpath(reg)
            if sub is not None:
                path.addPath(sub)
        return path

    def _region_to_subpath(self, region: Region) -> QPainterPath | None:
        verts = region.vertices
        if len(verts) < 3:
            return None
        poly = QPolygonF([QPointF(float(v[0]), float(v[1])) for v in verts])
        p = QPainterPath()
        p.addPolygon(poly)
        p.closeSubpath()
        return p

    # ------------------------------------------------------------------
    # Extra overlays (arbitrary items)
    # ------------------------------------------------------------------
    def add_overlay_item(self, item: QGraphicsItem) -> None:
        self._scene.addItem(item)

    def remove_overlay_item(self, item: QGraphicsItem) -> None:
        if item.scene() is self._scene:
            self._scene.removeItem(item)

    def scene_size(self) -> tuple[float, float]:
        return self._scene_size

    # ------------------------------------------------------------------
    # Zoom / pan
    # ------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def fit_to_scene(self) -> None:
        """Fit the view to the SLIDE extent (0, 0, W, H), not the padded scene
        rect. The scene rect includes 15% padding around the slide so users can
        scroll past the edges when zoomed in, but "fit to scene" should show
        the actual slide filling the viewport, not leave a big empty border.
        """
        from PySide6.QtCore import QRectF
        W, H = self._scene_size
        if W <= 0 or H <= 0:
            # Fall back to scene rect if we somehow don't have a slide extent
            rect = self._scene.sceneRect()
            if rect.isEmpty():
                return
            self.fitInView(rect, Qt.KeepAspectRatio)
            return
        # Small buffer around slide (5%) so outlines near the edge don't
        # get clipped by the viewport border
        buf_x = W * 0.05
        buf_y = H * 0.05
        self.fitInView(
            QRectF(-buf_x, -buf_y, W + 2 * buf_x, H + 2 * buf_y),
            Qt.KeepAspectRatio,
        )

    def zoom_to_region(self, bbox: tuple[float, float, float, float], pad: float = 0.3) -> None:
        """Zoom/pan to fit the given scene-space bbox, with 30% padding so
        edit handles (cut-line endpoints, etc.) that extend past the bbox
        remain visible."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1; h = y2 - y1
        rect = QRectF(x1 - pad * w, y1 - pad * h, w * (1 + 2 * pad), h * (1 + 2 * pad))
        self.fitInView(rect, Qt.KeepAspectRatio)

    def mousePressEvent(self, event) -> None:
        # Pass through to scene items first; if nothing claims it, emit scene_clicked
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton and not event.isAccepted():
            scene_pt = self.mapToScene(event.pos())
            self.scene_clicked.emit(scene_pt)


class _ClickablePathItemProxy(QObject):
    """Workaround: QGraphicsPathItem can't inherit from QObject, so we need
    a sibling QObject to host the signals."""
    clicked = Signal(str)
    right_clicked = Signal(str, object)  # layer_name, QPoint (screen coords)
    double_clicked = Signal(str)


class _ClickablePathItem(QGraphicsPathItem):
    """QGraphicsPathItem that emits signals on click / right-click / double-click."""

    def __init__(self, path: QPainterPath, layer_name: str):
        super().__init__(path)
        self._layer_name = layer_name
        self._proxy = _ClickablePathItemProxy()
        self.clicked = self._proxy.clicked
        self.right_clicked = self._proxy.right_clicked
        self.double_clicked = self._proxy.double_clicked
        self.setAcceptHoverEvents(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._proxy.clicked.emit(self._layer_name)
            event.accept()
        elif event.button() == Qt.RightButton:
            # screenPos() returns a QPoint in global screen coords, ready for
            # QMenu.exec(). No conversion needed.
            self._proxy.right_clicked.emit(
                self._layer_name, event.screenPos(),
            )
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._proxy.double_clicked.emit(self._layer_name)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)
