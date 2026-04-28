"""
Bilateral midline splitter.

Takes tissue-separated annotations (one layer per tissue piece) and splits
each tissue along its anatomical midline into ipsilateral and contralateral
sub-layers. Used for ipsi-vs-contra percent-area analysis in HALO.

Port of MSP_HALO_BILATERAL_MIDLINE_CUTTER_v70.m with improvements:

1. Midline detection defaults to "principal-axis" mode: PCA on the tissue's
   positive polygon vertices, then a cut perpendicular to the major axis
   through the centroid. This handles tilted coronal sections correctly.
   The old "vertical-at-bbox-center" mode is available as a fallback.

2. The cut is represented as a (point, normal) pair, not a vertical x-value.
   This lets the UI rotate and translate the cut line interactively for
   manual correction, and then recompute just that tissue's output.

3. Same polygon-traversal logic as v70 for the actual cutting, so we match
   its behavior on simple crossings. Polygons with 4+ crossings produce
   the same (slightly malformed) output as v70 did — Mark confirmed that
   works fine in practice and we're deferring shapely-based clipping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from haloqc.core.geometry import principal_axis, split_polygon_by_line
from haloqc.io.annotations import AnnotationFile, AnnotationLayer, Region

Side = Literal["right", "left"]
MidlineMethod = Literal["principal_axis", "vertical_bbox"]


@dataclass
class MidlineCut:
    """Geometric definition of a cut line for one tissue.

    The cut is an infinite line passing through `point` perpendicular to
    `normal`. Equivalently: all points p on the line satisfy
    `dot(p - point, normal) == 0`. The "positive side" is where the dot
    product is > 0.
    """
    point: np.ndarray  # (2,) point on the line
    normal: np.ndarray  # (2,) unit vector normal to the line

    @classmethod
    def from_endpoints(cls, p1: np.ndarray, p2: np.ndarray) -> "MidlineCut":
        """Construct from two endpoints (e.g., user dragged endpoints in UI)."""
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        direction = p2 - p1
        norm = np.hypot(direction[0], direction[1])
        if norm < 1e-9:
            raise ValueError("Cut line endpoints must be distinct")
        direction = direction / norm
        normal = np.array([-direction[1], direction[0]])
        point = (p1 + p2) / 2
        return cls(point=point, normal=normal)

    def endpoints_through_bbox(
        self, bbox: tuple[float, float, float, float], margin: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Two endpoints of the cut clipped to a bounding box (for rendering)."""
        x1, y1, x2, y2 = bbox
        x1 -= margin; y1 -= margin; x2 += margin; y2 += margin
        # Direction along the line is perpendicular to the normal
        direction = np.array([-self.normal[1], self.normal[0]])
        # Walk from point in both directions until we hit a bbox edge
        # Parameterize as point + t*direction; find t values where we leave bbox
        ts = []
        if abs(direction[0]) > 1e-9:
            ts.append((x1 - self.point[0]) / direction[0])
            ts.append((x2 - self.point[0]) / direction[0])
        if abs(direction[1]) > 1e-9:
            ts.append((y1 - self.point[1]) / direction[1])
            ts.append((y2 - self.point[1]) / direction[1])
        # Keep only the pair that gives valid endpoints inside the bbox
        valid: list[tuple[float, np.ndarray]] = []
        for t in ts:
            p = self.point + t * direction
            if x1 - 1e-6 <= p[0] <= x2 + 1e-6 and y1 - 1e-6 <= p[1] <= y2 + 1e-6:
                valid.append((t, p))
        if len(valid) < 2:
            # Fall back to a long segment in each direction
            L = max(x2 - x1, y2 - y1)
            return self.point - L * direction, self.point + L * direction
        valid.sort(key=lambda x: x[0])
        return valid[0][1], valid[-1][1]


def detect_midline(
    layer: AnnotationLayer,
    method: MidlineMethod = "principal_axis",
) -> MidlineCut:
    """Compute the midline cut for a tissue layer.

    For `principal_axis`: pools all vertices of all positive polygons in the
    layer, runs PCA, and returns a cut through the centroid perpendicular to
    the long axis. This is the anatomical midline for a roughly-symmetric
    coronal section regardless of how it's tilted on the slide.

    For `vertical_bbox`: returns a vertical cut at the x-midpoint of the
    layer's bounding box. Matches the MATLAB v70 default.
    """
    positives = layer.positive_regions
    if not positives:
        raise ValueError(f"Layer {layer.name} has no positive regions to split")

    # Pool vertices for robust statistics
    all_pts = np.vstack([r.vertices for r in positives])

    if method == "vertical_bbox":
        min_x = all_pts[:, 0].min()
        max_x = all_pts[:, 0].max()
        mid_x = (min_x + max_x) / 2
        mean_y = all_pts[:, 1].mean()
        return MidlineCut(
            point=np.array([mid_x, mean_y]),
            normal=np.array([1.0, 0.0]),  # vertical line => horizontal normal
        )

    # Principal-axis method
    centroid, axis_major, axis_minor = principal_axis(all_pts)
    # Midline is perpendicular to the major axis, so its normal IS the major axis
    return MidlineCut(point=centroid, normal=axis_major)


def apply_cut_to_layer(
    layer: AnnotationLayer,
    cut: MidlineCut,
    ipsi_side: Side = "right",
) -> tuple[AnnotationLayer, AnnotationLayer]:
    """Split a tissue layer into (ipsi_layer, contra_layer).

    We define: positive side (dot(p - point, normal) > 0) = direction of
    the cut's normal vector.

    When `ipsi_side="right"`, we take the +x-ish side as ipsi if the normal
    is close to pointing right (|nx| > |ny|). For a principal-axis cut of a
    roughly-symmetric coronal slice, the normal lies along the anatomical
    left-right axis, so this correctly assigns sides. For a vertical_bbox
    cut, normal is always +x, so right-side = positive side.
    """
    # Decide which dot-product sign corresponds to "ipsilateral"
    nx = cut.normal[0]
    if ipsi_side == "right":
        ipsi_positive = nx >= 0  # +x side is ipsi iff normal points right
    else:
        ipsi_positive = nx < 0

    ipsi_regions: list[Region] = []
    contra_regions: list[Region] = []

    for region in layer.regions:
        # Close the polygon for splitting if not already
        verts = region.vertices
        pos_poly, neg_poly = split_polygon_by_line(verts, cut.point, cut.normal)
        # pos_poly = positive-side portion, neg_poly = negative-side portion
        if ipsi_positive:
            ipsi_verts, contra_verts = pos_poly, neg_poly
        else:
            ipsi_verts, contra_verts = neg_poly, pos_poly

        if ipsi_verts is not None and len(ipsi_verts) >= 3:
            ipsi_regions.append(Region(
                vertices=ipsi_verts,
                is_negative=region.is_negative,
                region_type=region.region_type,
                has_endcaps=region.has_endcaps,
            ))
        if contra_verts is not None and len(contra_verts) >= 3:
            contra_regions.append(Region(
                vertices=contra_verts,
                is_negative=region.is_negative,
                region_type=region.region_type,
                has_endcaps=region.has_endcaps,
            ))

    from haloqc.core.colors import halo_line_color_for_layer

    ipsi_name = f"{layer.name} - Ipsi"
    contra_name = f"{layer.name} - Contra"

    ipsi_layer = AnnotationLayer(
        name=ipsi_name,
        regions=ipsi_regions,
        visible=layer.visible,
        line_color=halo_line_color_for_layer(ipsi_name),
    )
    contra_layer = AnnotationLayer(
        name=contra_name,
        regions=contra_regions,
        visible=layer.visible,
        line_color=halo_line_color_for_layer(contra_name),
    )
    return ipsi_layer, contra_layer


@dataclass
class BilateralParams:
    ipsi_side: Side = "right"
    midline_method: MidlineMethod = "principal_axis"


@dataclass
class BilateralResult:
    output_file: AnnotationFile
    cuts: dict[str, MidlineCut]  # layer name -> cut used
    diagnostics: list[str]


def split_bilateral(
    separated: AnnotationFile,
    params: BilateralParams,
    *,
    manual_cuts: dict[str, MidlineCut] | None = None,
) -> BilateralResult:
    """Run bilateral splitting on a tissue-separated annotation file.

    If `manual_cuts` provides a cut for a given layer name, that cut is used
    instead of the auto-detected one. This is how the UI passes user edits
    back into the pipeline.
    """
    manual_cuts = manual_cuts or {}
    out_layers: list[AnnotationLayer] = []
    cuts_used: dict[str, MidlineCut] = {}
    diagnostics: list[str] = []

    for layer in separated.layers:
        if not layer.positive_regions:
            diagnostics.append(f"Skipped {layer.name}: no positive regions")
            continue
        if layer.name in manual_cuts:
            cut = manual_cuts[layer.name]
            diagnostics.append(f"{layer.name}: using manual cut")
        else:
            cut = detect_midline(layer, method=params.midline_method)
        ipsi, contra = apply_cut_to_layer(layer, cut, ipsi_side=params.ipsi_side)
        cuts_used[layer.name] = cut

        ipsi_count = len(ipsi.regions)
        contra_count = len(contra.regions)
        diagnostics.append(
            f"{layer.name}: {ipsi_count} ipsi regions, {contra_count} contra regions"
        )

        if ipsi_count > 0:
            out_layers.append(ipsi)
        if contra_count > 0:
            out_layers.append(contra)

    return BilateralResult(
        output_file=AnnotationFile(layers=out_layers),
        cuts=cuts_used,
        diagnostics=diagnostics,
    )
