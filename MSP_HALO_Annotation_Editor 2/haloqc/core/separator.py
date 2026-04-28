"""
Tissue separation algorithm.

Port of HALO_Tissue_Separator_v12.m. Takes an AnnotationFile with regions from
one or more source layers and splits them into one AnnotationLayer per tissue
piece.

Algorithm:
1. Flatten all regions from all input layers.
2. Split into positives (tissue outlines) and negatives (holes: ventricles,
   cortex gaps).
3. Seed one group per positive region.
4. Attach each negative to the positive whose polygon contains its centroid
   (or the nearest positive if no positive contains it).
5. Compute polygon-to-polygon minimum distance between every pair of groups
   (using positive regions only). Merge pairs whose distance is below the
   user's micron threshold via union-find.
6. Optionally force-merge closest pairs until the group count matches the
   expected tissue count.
7. Order groups by reading order (top-down, left-right for 4x2 grids) or a
   user-specified ordering method.
8. Emit one AnnotationLayer per group named Tissue_01, Tissue_02, ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from haloqc.core.geometry import (
    UnionFind,
    point_in_polygon,
    polygon_min_distance,
)
from haloqc.io.annotations import (
    AnnotationFile,
    AnnotationLayer,
    Region,
    flatten_regions,
)

OrderMethod = Literal["grid", "horizontal", "vertical"]


@dataclass
class SeparatorParams:
    """All user-tunable parameters for tissue separation."""
    expected_tissues: int | None = 8
    order_method: OrderMethod = "grid"
    grid_tolerance: float = 0.2
    grid_forcing: str | None = "4x2"        # "4x2", "2x4", or None for auto
    merge_split_tissues: bool = True
    merge_distance_microns: float = 400.0
    microns_per_pixel: float = 0.5
    allow_force_to_expected: bool = False


@dataclass
class TissueGroup:
    """Internal representation of one tissue piece during processing."""
    regions: list[Region] = field(default_factory=list)

    @property
    def positive_regions(self) -> list[Region]:
        return [r for r in self.regions if not r.is_negative]

    @property
    def negative_regions(self) -> list[Region]:
        return [r for r in self.regions if r.is_negative]

    @property
    def tissue_centroid(self) -> np.ndarray:
        """Mean of positive-region centroids. Used for ordering."""
        pos = self.positive_regions
        if not pos:
            return np.array([0.0, 0.0])
        return np.mean([r.centroid for r in pos], axis=0)

    @property
    def total_positive_area(self) -> float:
        return sum(r.area() for r in self.positive_regions)


@dataclass
class SeparationResult:
    """Output of running tissue separation."""
    groups: list[TissueGroup]
    diagnostics: list[str] = field(default_factory=list)
    # Nearest inter-tissue distances in microns, sorted ascending
    nearest_distances_um: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def separate_tissues(
    annot_file: AnnotationFile,
    params: SeparatorParams,
) -> SeparationResult:
    """Run the full tissue separation pipeline."""
    diagnostics: list[str] = []

    all_regions = flatten_regions(annot_file)
    if not all_regions:
        raise ValueError("No regions found in input annotations")

    positives = [r for r in all_regions if not r.is_negative]
    negatives = [r for r in all_regions if r.is_negative]
    diagnostics.append(
        f"Found {len(positives)} positive regions and {len(negatives)} negative regions"
    )

    if not positives:
        raise ValueError("No positive (tissue) regions in input")

    # Initial groups: one per positive
    groups = [TissueGroup(regions=[p]) for p in positives]

    # Attach negatives
    groups = _attach_negatives(groups, negatives)

    # Distance-based merging
    px_thresh = params.merge_distance_microns / params.microns_per_pixel
    diagnostics.append(
        f"Merge threshold: {params.merge_distance_microns:.1f} um "
        f"({px_thresh:.1f} px at {params.microns_per_pixel:.4f} um/px)"
    )

    if params.merge_split_tissues:
        groups = _merge_groups_by_distance(groups, px_thresh)

    # Forced merge to hit expected count
    if params.expected_tissues is not None:
        n_now = len(groups)
        if n_now > params.expected_tissues:
            diagnostics.append(
                f"After merging by threshold: {n_now} groups. Target is {params.expected_tissues}."
            )
            if params.allow_force_to_expected:
                groups = _force_merge_to_target(groups, params.expected_tissues)
                diagnostics.append(
                    f"After forced merges toward target: {len(groups)} groups."
                )
        elif n_now < params.expected_tissues:
            diagnostics.append(
                f"Warning: {n_now} groups found, below expected {params.expected_tissues}. "
                "Check positivity detection."
            )

    # Ordering
    groups = _order_groups(groups, params, diagnostics)

    # Distance diagnostics (in microns)
    nearest_um = _list_nearest_distances(groups, params.microns_per_pixel)

    for i, g in enumerate(groups, start=1):
        pos_count = len(g.positive_regions)
        neg_count = len(g.negative_regions)
        c = g.tissue_centroid
        diagnostics.append(
            f"Tissue {i:02d}: {pos_count} positive, {neg_count} negative, "
            f"centroid=({c[0]:.0f}, {c[1]:.0f}) px"
        )

    diagnostics.append(f"Total tissues identified: {len(groups)}")

    return SeparationResult(
        groups=groups,
        diagnostics=diagnostics,
        nearest_distances_um=nearest_um,
    )


def separation_to_annotation_file(
    result: SeparationResult,
    *,
    base_visible: str = "True",
    base_line_color: str = "1772542",  # kept for backwards compat; no longer used
    source_path=None,
) -> AnnotationFile:
    """Convert a SeparationResult into an AnnotationFile ready for writing.

    Each group becomes one AnnotationLayer named Tissue_01, Tissue_02, etc.
    Each layer gets a distinct LineColor so that when the output is imported
    back into HALO, each tissue outline shows in its own color.

    Regions within each layer preserve the positive/negative order from the
    input, since ordering can matter to Halo for rendering z-stacking.
    """
    from haloqc.core.colors import halo_line_color_for_layer

    layers: list[AnnotationLayer] = []
    for i, group in enumerate(result.groups, start=1):
        name = f"Tissue_{i:02d}"
        layers.append(
            AnnotationLayer(
                name=name,
                regions=list(group.regions),
                visible=base_visible,
                line_color=halo_line_color_for_layer(name),
            )
        )
    return AnnotationFile(layers=layers, source_path=source_path)


# ---------------------------------------------------------------------------
# Step 1: attach negatives to positives
# ---------------------------------------------------------------------------

def _attach_negatives(
    groups: list[TissueGroup],
    negatives: list[Region],
) -> list[TissueGroup]:
    """Attach each negative region to its containing positive, or nearest."""
    if not negatives:
        return groups

    # Build flat list of (group_index, positive_region) for searching
    flat: list[tuple[int, Region]] = []
    for gi, g in enumerate(groups):
        for r in g.regions:
            if not r.is_negative:
                flat.append((gi, r))

    pos_centroids = np.array([r.centroid for _, r in flat])

    for neg in negatives:
        c = neg.centroid
        # Containment test first
        assigned = False
        for gi, pos in flat:
            if point_in_polygon(c, pos.vertices):
                groups[gi].regions.append(neg)
                assigned = True
                break
        if not assigned:
            # Nearest-centroid fallback
            d = np.linalg.norm(pos_centroids - c, axis=1)
            idx = int(np.argmin(d))
            gi = flat[idx][0]
            groups[gi].regions.append(neg)

    return groups


# ---------------------------------------------------------------------------
# Step 2: merge groups by pairwise polygon distance
# ---------------------------------------------------------------------------

def _pairwise_group_distances(
    groups: list[TissueGroup],
    *,
    early_stop_threshold: float | None = None,
) -> np.ndarray:
    """(n, n) symmetric matrix of minimum polygon distances between groups.

    Distance is measured between positive regions only (negatives don't define
    tissue boundaries). Diagonal is inf.

    If `early_stop_threshold` is given, pairs whose distance exceeds it may be
    reported as a bbox-based lower bound rather than the exact distance. This
    is safe for merging decisions, which only care about distance <= threshold.
    """
    n = len(groups)
    D = np.full((n, n), np.inf)
    for i in range(n):
        pos_i = groups[i].positive_regions
        for j in range(i + 1, n):
            pos_j = groups[j].positive_regions
            if not pos_i or not pos_j:
                continue
            d = np.inf
            for ra in pos_i:
                for rb in pos_j:
                    d = min(d, polygon_min_distance(
                        ra.vertices, rb.vertices,
                        early_stop_threshold=early_stop_threshold,
                    ))
                    if d == 0:
                        break
                if d == 0:
                    break
            D[i, j] = d
            D[j, i] = d
    return D


def _merge_groups_by_distance(
    groups: list[TissueGroup],
    px_thresh: float,
) -> list[TissueGroup]:
    """Merge any two groups whose minimum polygon distance is below threshold."""
    n = len(groups)
    if n <= 1:
        return groups

    D = _pairwise_group_distances(groups, early_stop_threshold=px_thresh)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= px_thresh:
                uf.union(i, j)

    # Consolidate
    new_groups: list[TissueGroup] = []
    for root_members in uf.groups().values():
        merged = TissueGroup(regions=[])
        for idx in root_members:
            merged.regions.extend(groups[idx].regions)
        new_groups.append(merged)
    return new_groups


def _force_merge_to_target(
    groups: list[TissueGroup],
    target_count: int,
) -> list[TissueGroup]:
    """Greedily merge closest pairs until at target count."""
    groups = list(groups)
    while len(groups) > target_count:
        D = _pairwise_group_distances(groups)
        if np.all(np.isinf(D)):
            break
        i, j = np.unravel_index(np.argmin(D), D.shape)
        i, j = int(i), int(j)
        if i > j:
            i, j = j, i
        # Merge j into i
        groups[i].regions.extend(groups[j].regions)
        groups.pop(j)
    return groups


# ---------------------------------------------------------------------------
# Step 3: ordering
# ---------------------------------------------------------------------------

def _order_groups(
    groups: list[TissueGroup],
    params: SeparatorParams,
    diagnostics: list[str],
) -> list[TissueGroup]:
    """Order groups for consistent Tissue_01..N labeling."""
    n = len(groups)
    if n == 0:
        return groups

    centers = np.array([g.tissue_centroid for g in groups])

    # Special case: expected 8 pieces, got 8, use 4x2 reading order
    if params.expected_tissues == 8 and n == 8:
        diagnostics.append("Ordering used: fixed reading order 4x2")
        return _order_reading_two_rows(groups, centers, n_cols=4)

    # Forced grid (e.g. "4x2") with matching count
    if params.order_method == "grid" and params.grid_forcing:
        parsed = _parse_cxr(params.grid_forcing)
        if parsed is not None:
            n_cols, n_rows = parsed
            if n_cols * n_rows == n:
                diagnostics.append(f"Ordering used: grid {n_cols}x{n_rows} (forced)")
                return _order_grid_fixed(groups, centers, n_cols, n_rows)

    # Fallback to auto grid / horizontal / vertical
    if params.order_method == "horizontal":
        idx = np.argsort(centers[:, 0])
        diagnostics.append("Ordering used: horizontal")
    elif params.order_method == "vertical":
        idx = np.argsort(centers[:, 1])
        diagnostics.append("Ordering used: vertical")
    else:
        idx = _detect_grid_arrangement(centers, params.grid_tolerance, diagnostics)

    return [groups[int(i)] for i in idx]


def _order_reading_two_rows(
    groups: list[TissueGroup],
    centers: np.ndarray,
    n_cols: int,
) -> list[TissueGroup]:
    """Top row left-to-right, then bottom row left-to-right."""
    n = len(groups)
    y_order = np.argsort(centers[:, 1])
    top_idx = y_order[:min(n_cols, n)]
    bot_idx = np.array([i for i in y_order if i not in set(top_idx.tolist())])
    top_sorted = top_idx[np.argsort(centers[top_idx, 0])]
    bot_sorted = bot_idx[np.argsort(centers[bot_idx, 0])] if len(bot_idx) else bot_idx
    order = np.concatenate([top_sorted, bot_sorted])
    return [groups[int(i)] for i in order]


def _parse_cxr(txt: str) -> tuple[int, int] | None:
    import re
    m = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", txt.lower())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def _order_grid_fixed(
    groups: list[TissueGroup],
    centers: np.ndarray,
    n_cols: int,
    n_rows: int,
) -> list[TissueGroup]:
    """Sort into n_rows rows by Y quantiles, then sort each row by X."""
    y = centers[:, 1]
    y_order = np.argsort(y)
    sorted_centers = centers[y_order]
    if n_rows > 1:
        # Use quantile edges to split into rows
        edges = np.quantile(sorted_centers[:, 1], np.arange(1, n_rows) / n_rows)
        row_idx = np.ones(len(groups), dtype=int)
        for r, edge in enumerate(edges, start=1):
            row_idx[sorted_centers[:, 1] > edge] = r + 1
    else:
        row_idx = np.ones(len(groups), dtype=int)

    new_order: list[int] = []
    for r in range(1, n_rows + 1):
        in_row = np.where(row_idx == r)[0]
        xs_in_row = sorted_centers[in_row, 0]
        sorted_within = in_row[np.argsort(xs_in_row)]
        new_order.extend(y_order[sorted_within].tolist())
    return [groups[i] for i in new_order]


def _detect_grid_arrangement(
    centroids: np.ndarray,
    tolerance: float,
    diagnostics: list[str],
) -> np.ndarray:
    """Auto-detect rows by Y proximity, sort within each row by X."""
    n = len(centroids)
    if n == 1:
        return np.array([0])
    y_positions = centroids[:, 1]
    y_order = np.argsort(y_positions)
    sorted_y = y_positions[y_order]
    y_range = y_positions.max() - y_positions.min()
    y_tol = tolerance * y_range if y_range > 0 else 100.0

    rows: list[list[int]] = [[int(y_order[0])]]
    current_y = sorted_y[0]
    for i in range(1, n):
        if abs(sorted_y[i] - current_y) < y_tol:
            rows[-1].append(int(y_order[i]))
        else:
            rows.append([int(y_order[i])])
            current_y = sorted_y[i]

    result: list[int] = []
    for row in rows:
        xs = centroids[row, 0]
        row_sorted = [row[i] for i in np.argsort(xs)]
        result.extend(row_sorted)
    diagnostics.append(f"Detected grid arrangement: {len(rows)} rows")
    for r, row in enumerate(rows, start=1):
        diagnostics.append(f"  Row {r}: {len(row)} tissues")
    return np.array(result)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _list_nearest_distances(
    groups: list[TissueGroup],
    microns_per_pixel: float,
) -> list[float]:
    """Nearest-neighbor distance per group, in microns, sorted ascending.

    For each group, reports the distance to its closest neighbor. Distances
    beyond 2000 microns (4000 px at 0.5 um/px) are clamped: at that range the
    tissues are clearly unrelated and an exact number isn't useful for QC,
    while computing it exactly on 2000+ vertex polygons is slow.
    """
    n = len(groups)
    if n < 2:
        return []
    # 4 mm lower-bound gives room for anything that could plausibly be related
    early_stop_px = 4000.0 / max(microns_per_pixel, 1e-6) * microns_per_pixel
    early_stop_px = 4000.0  # pixels; about 2 mm at 0.5 um/px
    D = _pairwise_group_distances(groups, early_stop_threshold=early_stop_px)
    nearest_px = D.min(axis=1)
    out_um = [float(d * microns_per_pixel) for d in nearest_px if np.isfinite(d)]
    out_um.sort()
    return out_um
