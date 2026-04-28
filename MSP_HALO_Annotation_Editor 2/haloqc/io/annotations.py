"""
HALO .annotations file I/O.

HALO annotation files are pseudo-XML with a specific structure:

    <Annotations>
      <Annotation Name="..." Visible="..." LineColor="...">
        <Regions>
          <Region Type="Polygon" HasEndcaps="0" NegativeROA="0">
            <Vertices>
              <V X="123" Y="456" />
              ...
            </Vertices>
          </Region>
          ...
        </Regions>
      </Annotation>
      ...
    </Annotations>

NegativeROA="1" marks a region as a hole (e.g., ventricle) within a positive
tissue region. Coordinates are in pixels at the full-resolution level of the
associated NDPI slide.

We use regex parsing (not a full XML parser) to exactly match the behavior of
the MATLAB v12 / v70 scripts, which HALO is known to accept. The MATLAB
regex only matches integer coordinates; we allow decimals here because some
HALO exports use them, but we round to int on write for byte-level parity
with the existing MATLAB output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Region:
    """A single polygon region within an annotation layer."""
    vertices: np.ndarray           # (N, 2) float array of (x, y) coordinates
    is_negative: bool = False      # NegativeROA="1" = hole
    region_type: str = "Polygon"
    has_endcaps: str = "0"

    @property
    def centroid(self) -> np.ndarray:
        """Mean of vertices (matches MATLAB's `mean(V, 1)`)."""
        return self.vertices.mean(axis=0)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y)"""
        mn = self.vertices.min(axis=0)
        mx = self.vertices.max(axis=0)
        return (mn[0], mn[1], mx[0], mx[1])

    def area(self) -> float:
        """Signed polygon area via shoelace. Absolute value returned."""
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


@dataclass
class AnnotationLayer:
    """A named layer containing one or more regions."""
    name: str
    regions: list[Region] = field(default_factory=list)
    visible: str = "True"
    line_color: str = "1772542"  # HALO stores color as int32; default matches TISSUE layer color

    @property
    def positive_regions(self) -> list[Region]:
        return [r for r in self.regions if not r.is_negative]

    @property
    def negative_regions(self) -> list[Region]:
        return [r for r in self.regions if r.is_negative]


@dataclass
class AnnotationFile:
    """Top-level container for an entire .annotations file."""
    layers: list[AnnotationLayer] = field(default_factory=list)
    source_path: Path | None = None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Same patterns the MATLAB scripts use, slightly relaxed to allow decimal
# coordinates. Using non-greedy matching to handle adjacent regions correctly.
_ANNOT_RE = re.compile(
    r'<Annotation\s+Name="([^"]*)"\s+Visible="([^"]*)"\s+LineColor="([^"]*)">([\s\S]*?)</Annotation>'
)
_REGION_RE = re.compile(
    r'<Region\s+Type="([^"]*)"\s+HasEndcaps="([^"]*)"\s+NegativeROA="([^"]*)">([\s\S]*?)</Region>'
)
_VERTEX_RE = re.compile(
    r'<V\s+X="([0-9.\-]+)"\s+Y="([0-9.\-]+)"\s*/>'
)


def parse_annotations(path: str | Path) -> AnnotationFile:
    """Load a HALO .annotations file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    layers: list[AnnotationLayer] = []
    for m in _ANNOT_RE.finditer(text):
        name, visible, line_color, body = m.groups()
        regions = []
        for rm in _REGION_RE.finditer(body):
            region_type, has_endcaps, neg_attr, region_body = rm.groups()
            is_negative = neg_attr.strip().lower() in ("1", "true")
            vertices = []
            for vm in _VERTEX_RE.finditer(region_body):
                vertices.append((float(vm.group(1)), float(vm.group(2))))
            if len(vertices) < 3:
                # HALO needs at least a triangle; skip degenerate regions.
                continue
            regions.append(
                Region(
                    vertices=np.asarray(vertices, dtype=float),
                    is_negative=is_negative,
                    region_type=region_type,
                    has_endcaps=has_endcaps,
                )
            )
        layers.append(
            AnnotationLayer(
                name=name,
                regions=regions,
                visible=visible,
                line_color=line_color,
            )
        )

    return AnnotationFile(layers=layers, source_path=path)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_annotations(
    annot_file: AnnotationFile,
    path: str | Path,
    *,
    round_vertices: bool = True,
) -> None:
    """Write an AnnotationFile to disk in HALO-compatible XML.

    Formatting matches the MATLAB scripts' output exactly: 2-space base indent,
    no XML declaration, integer-rounded vertices by default.
    """
    path = Path(path)
    lines: list[str] = ["<Annotations>"]

    for layer in annot_file.layers:
        lines.append(
            f'  <Annotation Name="{layer.name}" Visible="{layer.visible}" '
            f'LineColor="{layer.line_color}">'
        )
        lines.append("    <Regions>")
        for region in layer.regions:
            neg_str = "1" if region.is_negative else "0"
            lines.append(
                f'      <Region Type="{region.region_type}" '
                f'HasEndcaps="{region.has_endcaps}" NegativeROA="{neg_str}">'
            )
            lines.append("        <Vertices>")
            for x, y in region.vertices:
                if round_vertices:
                    xs = str(int(round(x)))
                    ys = str(int(round(y)))
                else:
                    xs = f"{x:g}"
                    ys = f"{y:g}"
                lines.append(f'          <V X="{xs}" Y="{ys}" />')
            lines.append("        </Vertices>")
            lines.append("      </Region>")
        lines.append("    </Regions>")
        lines.append("  </Annotation>")

    lines.append("</Annotations>")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Convenience: flatten all regions regardless of source layer
# ---------------------------------------------------------------------------

def flatten_regions(annot_file: AnnotationFile) -> list[Region]:
    """Return all regions from all layers as one list.

    Matches what the MATLAB tissue separator does in its first pass: it treats
    all incoming layers as a single pool of regions and regroups them by
    geometry. This is the right model when the input is one TISSUE layer, as
    in the standard Halo classifier workflow.
    """
    out: list[Region] = []
    for layer in annot_file.layers:
        out.extend(layer.regions)
    return out
