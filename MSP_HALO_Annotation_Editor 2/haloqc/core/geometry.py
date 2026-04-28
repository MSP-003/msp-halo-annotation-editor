"""
Polygon geometry utilities.

Kept deliberately dependency-light: numpy only, no shapely. All functions
operate on (N, 2) numpy arrays of vertices. Polygons are assumed to be closed
in the topological sense but not necessarily repeat the first vertex at the
end (we handle both).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Point-in-polygon
# ---------------------------------------------------------------------------

def point_in_polygon(point: np.ndarray, poly: np.ndarray) -> bool:
    """Ray casting point-in-polygon test.

    Matches MATLAB's `inpolygon` for the `in` return (points on the boundary
    are also considered inside here, consistent with `in | on`).
    """
    x, y = float(point[0]), float(point[1])
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        # Boundary case: if the point lies on this edge, treat as inside.
        if _point_on_segment((x, y), (xi, yi), (xj, yj)):
            return True
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _point_on_segment(p, a, b, eps: float = 1e-9) -> bool:
    """True if point p lies on segment ab within tolerance."""
    px, py = p
    ax, ay = a
    bx, by = b
    # Cross product check for collinearity
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) > eps * max(1.0, abs(bx - ax) + abs(by - ay)):
        return False
    # Bounding box check
    if min(ax, bx) - eps <= px <= max(ax, bx) + eps and \
       min(ay, by) - eps <= py <= max(ay, by) + eps:
        return True
    return False


# ---------------------------------------------------------------------------
# Polygon-polygon minimum distance
# ---------------------------------------------------------------------------

def polygon_bbox(poly: np.ndarray) -> tuple[float, float, float, float]:
    """(min_x, min_y, max_x, max_y) of a polygon."""
    mn = poly.min(axis=0)
    mx = poly.max(axis=0)
    return (float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1]))


def bbox_min_distance(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Minimum distance between two axis-aligned bounding boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(0.0, max(bx1 - ax2, ax1 - bx2))
    dy = max(0.0, max(by1 - ay2, ay1 - by2))
    return float(np.hypot(dx, dy))


def polygon_min_distance(
    poly_a: np.ndarray,
    poly_b: np.ndarray,
    *,
    early_stop_threshold: float | None = None,
) -> float:
    """Minimum distance between two polygons.

    Returns 0 if they overlap (one polygon contains a vertex of the other).
    Matches the MATLAB script's polyMinDistance behavior.

    Optimizations over the MATLAB version:
      - Bounding-box precheck: if bboxes are separated by more than
        `early_stop_threshold`, returns that separation distance as a lower
        bound (caller can use this to short-circuit merge decisions).
      - Vectorized segment-segment distance computation.

    Parameters
    ----------
    early_stop_threshold : optional
        If the bbox-to-bbox distance already exceeds this, skip the expensive
        edge computation and just return the bbox distance. Callers that only
        care about whether distance <= threshold can pass the threshold here.
    """
    # Bbox precheck
    bbox_a = polygon_bbox(poly_a)
    bbox_b = polygon_bbox(poly_b)
    bbox_d = bbox_min_distance(bbox_a, bbox_b)
    if early_stop_threshold is not None and bbox_d > early_stop_threshold:
        return bbox_d

    # Containment test only if bboxes overlap (otherwise polygons can't overlap)
    if bbox_d == 0.0:
        for p in poly_a:
            if point_in_polygon(p, poly_b):
                return 0.0
        for p in poly_b:
            if point_in_polygon(p, poly_a):
                return 0.0

    # Full edge-to-edge computation
    edges_a = _edges(poly_a)
    edges_b = _edges(poly_b)
    return _min_segment_distance_batch(edges_a, edges_b)


def _edges(poly: np.ndarray) -> np.ndarray:
    n = len(poly)
    p1 = poly
    p2 = np.roll(poly, -1, axis=0)
    return np.stack([p1, p2], axis=1)  # (N, 2, 2)


def _min_segment_distance_batch(edges_a: np.ndarray, edges_b: np.ndarray) -> float:
    """Minimum distance between any segment in A and any segment in B."""
    a1 = edges_a[:, 0, :]  # (Na, 2)
    a2 = edges_a[:, 1, :]
    b1 = edges_b[:, 0, :]  # (Nb, 2)
    b2 = edges_b[:, 1, :]

    # 4 point-to-segment evaluations per (a, b) pair:
    #   dist(a1, b_seg), dist(a2, b_seg), dist(b1, a_seg), dist(b2, a_seg)
    d1 = _point_to_segment_batch(a1, b1, b2)  # (Na, Nb)
    d2 = _point_to_segment_batch(a2, b1, b2)
    d3 = _point_to_segment_batch(b1, a1, a2).T  # transpose to (Na, Nb)
    d4 = _point_to_segment_batch(b2, a1, a2).T
    stacked = np.stack([d1, d2, d3, d4], axis=0)
    return float(stacked.min())


def _point_to_segment_batch(pts: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Distance from each point in `pts` to each segment (s1[i], s2[i]).

    pts: (P, 2), s1: (S, 2), s2: (S, 2) -> returns (P, S).
    """
    # Broadcast shapes
    p = pts[:, None, :]    # (P, 1, 2)
    a = s1[None, :, :]     # (1, S, 2)
    b = s2[None, :, :]
    ab = b - a             # (1, S, 2)
    ap = p - a             # (P, S, 2)
    ab_sq = np.sum(ab * ab, axis=-1)  # (1, S)
    # Avoid divide-by-zero for degenerate segments
    ab_sq = np.where(ab_sq < 1e-30, 1e-30, ab_sq)
    t = np.sum(ap * ab, axis=-1) / ab_sq  # (P, S)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[..., None] * ab          # (P, S, 2)
    diff = p - proj
    return np.sqrt(np.sum(diff * diff, axis=-1))  # (P, S)


# ---------------------------------------------------------------------------
# Union-Find (for merging clusters)
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[rj] = ri

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            out.setdefault(r, []).append(i)
        return out


# ---------------------------------------------------------------------------
# Principal axis (for smart midline detection)
# ---------------------------------------------------------------------------

def principal_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute principal axis of a point set via PCA.

    Returns
    -------
    centroid : (2,) array
        Mean of the points.
    axis_major : (2,) unit vector
        Direction of maximum variance (tissue long axis).
    axis_minor : (2,) unit vector
        Perpendicular direction (anatomical midline direction for a symmetric
        coronal section).
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    # 2x2 covariance
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
    axis_major = eigvecs[:, 1]
    axis_minor = eigvecs[:, 0]
    # Make major axis point "up" (positive y) for consistent orientation
    if axis_major[1] < 0:
        axis_major = -axis_major
    # Make minor axis point "right" (positive x)
    if axis_minor[0] < 0:
        axis_minor = -axis_minor
    return centroid, axis_major, axis_minor


# ---------------------------------------------------------------------------
# Line-polygon intersection (for bilateral cutting)
# ---------------------------------------------------------------------------

def line_polygon_crossings(
    poly: np.ndarray,
    line_point: np.ndarray,
    line_normal: np.ndarray,
) -> list[tuple[int, np.ndarray]]:
    """Find points where polygon edges cross an infinite line.

    The line is defined by a point `line_point` on it and a normal vector
    `line_normal`. The signed distance from any point p to the line is
    `dot(p - line_point, line_normal)`.

    Returns a list of (edge_index, crossing_point) tuples sorted by edge_index.
    """
    n = len(poly)
    # Signed distances for each vertex
    d = (poly - line_point) @ line_normal  # (n,)
    crossings: list[tuple[int, np.ndarray]] = []
    for i in range(n):
        j = (i + 1) % n
        di, dj = d[i], d[j]
        # Proper sign change (not touching)
        if (di < 0 and dj > 0) or (di > 0 and dj < 0):
            t = di / (di - dj)
            cross = poly[i] + t * (poly[j] - poly[i])
            crossings.append((i, cross))
    return crossings


def split_polygon_by_line(
    poly: np.ndarray,
    line_point: np.ndarray,
    line_normal: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Split a polygon by an infinite line into (positive_side, negative_side).

    Positive side = where dot(p - line_point, line_normal) > 0.

    This implementation walks the polygon once, emitting vertices to whichever
    side they belong and inserting crossing points where edges cross the line.
    It assumes the polygon crosses the line 0 or 2 times (convex-ish behavior).
    For C-shaped polygons with 4+ crossings, the output will be malformed but
    self-consistent with the MATLAB v70 script's behavior.

    Returns None for a side if no vertices fall on it.
    """
    n = len(poly)
    # Ensure not closed (last != first), match MATLAB convention
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
        n -= 1

    d = (poly - line_point) @ line_normal  # signed distances

    pos_verts: list[np.ndarray] = []
    neg_verts: list[np.ndarray] = []

    for i in range(n):
        j = (i + 1) % n
        pi = poly[i]
        pj = poly[j]
        di = d[i]
        dj = d[j]

        # Emit current vertex to appropriate side (>= puts boundary on pos side)
        if di >= 0:
            pos_verts.append(pi)
        if di <= 0:
            neg_verts.append(pi)

        # If edge crosses the line (strict sign change), emit crossing point
        if (di > 0 and dj < 0) or (di < 0 and dj > 0):
            t = di / (di - dj)
            cross = pi + t * (pj - pi)
            pos_verts.append(cross)
            neg_verts.append(cross)

    pos_arr = np.asarray(pos_verts) if len(pos_verts) >= 3 else None
    neg_arr = np.asarray(neg_verts) if len(neg_verts) >= 3 else None
    return pos_arr, neg_arr
