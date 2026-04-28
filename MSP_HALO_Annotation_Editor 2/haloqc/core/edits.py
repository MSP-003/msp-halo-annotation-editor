"""
Manual edit operations on tissue separator output.

Every operation here takes an AnnotationFile and returns a new one (or
mutates in place with a snapshot). The UI wraps these in an undo stack.

Operations available:
  - rename_tissue: change a layer's display name
  - merge_tissues: combine all regions from one layer into another, delete the source
  - delete_tissue: remove a layer entirely (for noise/dust)
  - split_tissue_by_regions: promote selected regions from a layer to a new layer
  - move_regions: take specific regions out of one layer and append to another
  - shift_tissue_numbers: offset a range of Tissue_NN numbers by +offset or -offset
  - renumber_sequential: rename layers Tissue_01..N in reading order

All operations preserve the region-order invariant (positives first, then
negatives) because HALO's z-stacking can depend on it.

Tissue color is re-derived from the final layer name via
haloqc.core.colors.halo_line_color_for_layer, so colors stay consistent with
the canonical palette after any edit.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

from haloqc.core.colors import halo_line_color_for_layer
from haloqc.io.annotations import AnnotationFile, AnnotationLayer, Region


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clone(af: AnnotationFile) -> AnnotationFile:
    """Deep copy an AnnotationFile for the undo stack."""
    return copy.deepcopy(af)


def _find_layer(af: AnnotationFile, name: str) -> AnnotationLayer:
    for layer in af.layers:
        if layer.name == name:
            return layer
    raise KeyError(f"No layer named {name!r}")


def _find_layer_index(af: AnnotationFile, name: str) -> int:
    for i, layer in enumerate(af.layers):
        if layer.name == name:
            return i
    raise KeyError(f"No layer named {name!r}")


def _sort_regions_positives_first(regions: list[Region]) -> list[Region]:
    positives = [r for r in regions if not r.is_negative]
    negatives = [r for r in regions if r.is_negative]
    return positives + negatives


# ---------------------------------------------------------------------------
# Public: rename
# ---------------------------------------------------------------------------

def rename_tissue(af: AnnotationFile, old_name: str, new_name: str) -> AnnotationFile:
    """Return a new AnnotationFile with one layer renamed. Color is recomputed
    from the new name so `Tissue_03` → `Cortex_L` produces a name-stable color.
    """
    if not new_name or not new_name.strip():
        raise ValueError("New name cannot be empty")
    new_name = new_name.strip()
    if old_name == new_name:
        return af
    af2 = _clone(af)
    if any(l.name == new_name for l in af2.layers):
        raise ValueError(f"A layer named {new_name!r} already exists")
    layer = _find_layer(af2, old_name)
    layer.name = new_name
    layer.line_color = halo_line_color_for_layer(new_name)
    return af2


# ---------------------------------------------------------------------------
# Public: merge
# ---------------------------------------------------------------------------

def merge_tissues(
    af: AnnotationFile,
    source_name: str,
    target_name: str,
) -> AnnotationFile:
    """Merge all regions of `source_name` into `target_name`, then delete source.
    Target retains its name and color.
    """
    if source_name == target_name:
        raise ValueError("Cannot merge a layer into itself")
    af2 = _clone(af)
    src = _find_layer(af2, source_name)
    tgt = _find_layer(af2, target_name)
    combined = tgt.regions + src.regions
    tgt.regions = _sort_regions_positives_first(combined)
    af2.layers = [l for l in af2.layers if l.name != source_name]
    return af2


# ---------------------------------------------------------------------------
# Public: delete
# ---------------------------------------------------------------------------

def delete_tissue(af: AnnotationFile, name: str) -> AnnotationFile:
    """Remove a layer entirely."""
    af2 = _clone(af)
    _find_layer(af2, name)  # raises if missing
    af2.layers = [l for l in af2.layers if l.name != name]
    return af2


# ---------------------------------------------------------------------------
# Public: split
# ---------------------------------------------------------------------------

def split_tissue_by_regions(
    af: AnnotationFile,
    source_name: str,
    region_indices: Iterable[int],
    new_name: str | None = None,
) -> AnnotationFile:
    """Move the specified regions (by index within the source layer) into a
    new layer. `region_indices` refer to the source layer's regions list.

    If `new_name` is None, we auto-pick `Tissue_NN` with the lowest unused N.

    The new layer is inserted immediately after the source so visual ordering
    stays local. Renumber the whole file afterward if the user prefers clean
    sequential names — that's a separate explicit step.
    """
    idx_set = set(int(i) for i in region_indices)
    if not idx_set:
        raise ValueError("No regions specified to split out")

    af2 = _clone(af)
    src = _find_layer(af2, source_name)
    if any(i < 0 or i >= len(src.regions) for i in idx_set):
        raise ValueError("Region index out of range")

    moved = [src.regions[i] for i in sorted(idx_set)]
    kept = [r for i, r in enumerate(src.regions) if i not in idx_set]

    if not any(not r.is_negative for r in moved):
        raise ValueError(
            "Can't split out only negative regions - a new tissue needs at least "
            "one positive polygon"
        )
    if not any(not r.is_negative for r in kept):
        raise ValueError(
            "Splitting out these regions would leave the source with no positive "
            "polygon. Delete the source instead, or pick fewer regions to move."
        )

    src.regions = _sort_regions_positives_first(kept)

    if new_name is None:
        new_name = _next_auto_tissue_name(af2)
    elif any(l.name == new_name for l in af2.layers):
        raise ValueError(f"A layer named {new_name!r} already exists")

    new_layer = AnnotationLayer(
        name=new_name,
        regions=_sort_regions_positives_first(moved),
        visible=src.visible,
        line_color=halo_line_color_for_layer(new_name),
    )
    src_idx = _find_layer_index(af2, source_name)
    af2.layers.insert(src_idx + 1, new_layer)
    return af2


def _next_auto_tissue_name(af: AnnotationFile) -> str:
    """Return the lowest unused Tissue_NN name."""
    import re
    used = set()
    for l in af.layers:
        m = re.match(r"^Tissue_(\d+)$", l.name)
        if m:
            used.add(int(m.group(1)))
    n = 1
    while n in used:
        n += 1
    return f"Tissue_{n:02d}"


# ---------------------------------------------------------------------------
# Public: move regions
# ---------------------------------------------------------------------------

def move_regions(
    af: AnnotationFile,
    source_name: str,
    region_indices: Iterable[int],
    target_name: str,
) -> AnnotationFile:
    """Move specific regions from one layer to another.

    Same index-based semantics as split. Raises if the move would leave the
    source with no positive regions (user should delete the source instead).
    """
    if source_name == target_name:
        raise ValueError("Source and target are the same layer")
    idx_set = set(int(i) for i in region_indices)
    if not idx_set:
        raise ValueError("No regions specified")

    af2 = _clone(af)
    src = _find_layer(af2, source_name)
    tgt = _find_layer(af2, target_name)

    if any(i < 0 or i >= len(src.regions) for i in idx_set):
        raise ValueError("Region index out of range")

    moved = [src.regions[i] for i in sorted(idx_set)]
    kept = [r for i, r in enumerate(src.regions) if i not in idx_set]

    if not any(not r.is_negative for r in kept):
        raise ValueError(
            "Moving these regions would leave the source layer with no positive "
            "polygon. Delete the source instead, or merge it into the target."
        )

    src.regions = _sort_regions_positives_first(kept)
    tgt.regions = _sort_regions_positives_first(tgt.regions + moved)
    return af2


# ---------------------------------------------------------------------------
# Public: shift tissue numbers
# ---------------------------------------------------------------------------

def shift_tissue_numbers(
    af: AnnotationFile,
    offset: int,
    *,
    from_number: int = 1,
    to_number: int | None = None,
    custom_name_prefix: str = "Tissue_",
) -> AnnotationFile:
    """Offset a range of Tissue_NN names by `offset`.

    Use case: slice #1 of the grid is missing, so separator labeled the seven
    present tissues as Tissue_01..07, but anatomically they're slices 2..8.
    shift_tissue_numbers(af, +1) renames every Tissue_NN -> Tissue_(NN+offset).

    Parameters
    ----------
    af : AnnotationFile
    offset : int
        Amount to add (positive) or subtract (negative) from each number.
    from_number : int, default 1
        Only shift layers whose number is >= from_number.
    to_number : int or None, default None
        Only shift layers whose number is <= to_number. None = no upper bound.
    custom_name_prefix : str, default "Tissue_"
        Only rename layers matching this prefix pattern. Custom-named layers
        (Cortex_L etc.) are left untouched.

    Raises
    ------
    ValueError
        If the shift would collide with an existing tissue name outside the
        shifted range, or if it would produce a non-positive number.
    """
    import re

    if offset == 0:
        return af

    af2 = _clone(af)
    pattern = re.compile(rf"^{re.escape(custom_name_prefix)}(\d+)$")

    # Collect (layer, old_number, new_number) for every layer we'd touch
    to_shift: list[tuple[AnnotationLayer, int, int]] = []
    for layer in af2.layers:
        m = pattern.match(layer.name)
        if not m:
            continue
        num = int(m.group(1))
        if num < from_number:
            continue
        if to_number is not None and num > to_number:
            continue
        new_num = num + offset
        if new_num < 1:
            raise ValueError(
                f"Shifting {layer.name} by {offset:+d} would produce a "
                f"non-positive number ({new_num})."
            )
        to_shift.append((layer, num, new_num))

    if not to_shift:
        return af2

    # Detect collisions: a new name must not already exist on a layer that's
    # NOT itself being shifted to a different name in this pass.
    shifted_old_names = {l.name for l, _, _ in to_shift}
    existing_names = {l.name for l in af2.layers}
    for layer, _, new_num in to_shift:
        new_name = f"{custom_name_prefix}{new_num:02d}"
        if new_name in existing_names and new_name not in shifted_old_names:
            raise ValueError(
                f"Shifting would collide with existing {new_name!r}. Rename "
                f"or shift the colliding tissue first."
            )

    # Apply renames in an order that avoids self-collision:
    # shifting up -> process highest number first so Tissue_07 becomes 08
    # before Tissue_06 becomes 07
    # shifting down -> process lowest first
    to_shift.sort(key=lambda t: t[1], reverse=(offset > 0))

    for layer, _, new_num in to_shift:
        new_name = f"{custom_name_prefix}{new_num:02d}"
        layer.name = new_name
        layer.line_color = halo_line_color_for_layer(new_name)

    return af2


# ---------------------------------------------------------------------------
# Public: renumber sequential (reading order)
# ---------------------------------------------------------------------------

def renumber_sequential(
    af: AnnotationFile,
    *,
    custom_name_prefix: str = "Tissue_",
) -> AnnotationFile:
    """Rename every layer to `Tissue_01`, `Tissue_02`, etc. in reading order
    (top-to-bottom by centroid Y, then left-to-right by X within each row).

    Custom user-renamed layers (e.g. `Cortex_L`) are preserved — only layers
    whose name matches the `Tissue_NN` pattern get renumbered. Recomputes
    line_color from the new name.

    This is what the "Renumber" button calls. Also automatically called after
    every destructive edit (merge/delete/split) when the UI has auto-renumber
    enabled.
    """
    import re

    af2 = _clone(af)

    # Split layers into ones we should renumber and ones we shouldn't
    pattern = re.compile(rf"^{re.escape(custom_name_prefix)}\d+$")
    numbered = [(i, l) for i, l in enumerate(af2.layers) if pattern.match(l.name)]
    custom = [(i, l) for i, l in enumerate(af2.layers) if not pattern.match(l.name)]

    if not numbered:
        return af2

    # Compute reading order on the numbered layers only
    def centroid_of(layer: AnnotationLayer) -> tuple[float, float]:
        pos = [r for r in layer.regions if not r.is_negative]
        if not pos:
            return (0.0, 0.0)
        xs = []
        ys = []
        for r in pos:
            c = r.vertices.mean(axis=0)
            xs.append(float(c[0]))
            ys.append(float(c[1]))
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    centroids = [(i, l, centroid_of(l)) for i, l in numbered]

    if centroids:
        ys = [c[2][1] for c in centroids]
        y_range = max(ys) - min(ys) if len(ys) > 1 else 0.0
        y_tol = 0.2 * y_range if y_range > 0 else 1.0

        # Bucket into rows by Y
        sorted_by_y = sorted(centroids, key=lambda t: t[2][1])
        rows: list[list] = [[sorted_by_y[0]]]
        for entry in sorted_by_y[1:]:
            cur_y = entry[2][1]
            row_y = rows[-1][-1][2][1]
            if abs(cur_y - row_y) < y_tol:
                rows[-1].append(entry)
            else:
                rows.append([entry])
        # Within each row, sort by X
        for row in rows:
            row.sort(key=lambda t: t[2][0])
        ordered = [entry for row in rows for entry in row]
    else:
        ordered = []

    # Assign new names
    for new_i, (orig_i, layer, _c) in enumerate(ordered, start=1):
        new_name = f"{custom_name_prefix}{new_i:02d}"
        layer.name = new_name
        layer.line_color = halo_line_color_for_layer(new_name)

    # Reorder the file's layer list so visual order matches naming order.
    # Custom-named layers keep their original position by stuffing them back in.
    # Simplest: put renumbered first in their new order, then custom-named
    # layers in their original index order.
    renum_layers = [entry[1] for entry in ordered]
    custom_layers = [l for _i, l in custom]
    af2.layers = renum_layers + custom_layers
    return af2


# ---------------------------------------------------------------------------
# Undo stack
# ---------------------------------------------------------------------------

@dataclass
class EditHistory:
    """Simple single-level undo: we snapshot before each edit and can revert
    to the most recent snapshot. For v2 this is single-level only (one Ctrl+Z
    reverts the last edit); multi-level would be straightforward to add later
    by growing this into a list with a pointer.
    """

    snapshot: AnnotationFile | None = None
    label: str | None = None  # human-readable description of the last edit

    def record(self, af: AnnotationFile, label: str) -> None:
        self.snapshot = _clone(af)
        self.label = label

    def can_undo(self) -> bool:
        return self.snapshot is not None

    def undo(self) -> tuple[AnnotationFile, str]:
        if not self.can_undo():
            raise RuntimeError("Nothing to undo")
        af = self.snapshot
        label = self.label or "edit"
        self.snapshot = None
        self.label = None
        return af, label
