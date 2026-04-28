"""
Shared color palette and HALO LineColor encoding.

HALO stores annotation colors as a packed 24-bit integer in the LineColor
attribute of each <Annotation> element. The encoding is:
    value = (B << 16) | (G << 8) | R

This is HALO's native format - confirmed by decoding 1772542 = 0x1B0BFE
which unpacks to R=0xFE G=0x0B B=0x1B, a bright red, matching what HALO
renders by default. Note this is the low-byte-is-R convention (reverse of
standard 0xRRGGBB hex notation).

We keep the palette here so that:
- The canvas visualizes a tissue with the same color HALO will show
- Bilateral ipsi/contra get lighter/darker versions of the base tissue color
  both in our QC view and when re-imported to HALO
"""

from __future__ import annotations

import colorsys
import re


# Same 10 colors the canvas cycles through for distinct tissue outlines.
# Chosen to be maximally distinguishable with high-saturation, mid-lightness
# hues so they read clearly on both dark and light backgrounds.
TISSUE_PALETTE_HEX: list[str] = [
    "#00ff9f", "#ff6b9d", "#ffc857", "#4ecdc4", "#c780ff",
    "#ff8354", "#7fc8f8", "#c3ea6a", "#ff5757", "#b388eb",
]


def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """'#RRGGBB' -> (r, g, b) ints in 0-255."""
    s = hex_str.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def rgb_to_halo_int(r: int, g: int, b: int) -> int:
    """HALO packs color as B*65536 + G*256 + R (low-byte-is-red)."""
    return (b << 16) | (g << 8) | r


def halo_int_to_rgb(value: int) -> tuple[int, int, int]:
    """Inverse of rgb_to_halo_int."""
    r = value & 0xFF
    g = (value >> 8) & 0xFF
    b = (value >> 16) & 0xFF
    return (r, g, b)


def _adjust_lightness(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """Return a lighter or darker version of `rgb`.

    factor > 1: interpolate toward white. factor=1.3 means 30% of the way to
                pure white in RGB space.
    factor < 1: interpolate toward black. factor=0.65 means 35% of the way to
                black in RGB space (1.0 - 0.65 = 0.35 mix toward black).
    factor == 1: no change.

    We do this in RGB rather than HSL because HSL-lightness scaling produces
    oversaturated colors for already-bright base hues (e.g. HSL-darkening a
    pink at lightness 0.7 pulls it through a highly saturated red-pink that
    can be mistaken for a different tissue's primary color). Linear RGB
    interpolation toward white/black preserves the original color's chroma
    ratio and produces a visually cleaner tint/shade.
    """
    r, g, b = rgb
    if factor >= 1.0:
        # Toward white
        t = min(factor - 1.0, 1.0)
        return (
            int(round(r + (255 - r) * t)),
            int(round(g + (255 - g) * t)),
            int(round(b + (255 - b) * t)),
        )
    # Toward black
    t = 1.0 - factor  # e.g. factor=0.65 -> t=0.35 mix toward black
    return (
        int(round(r * (1 - t))),
        int(round(g * (1 - t))),
        int(round(b * (1 - t))),
    )


def _parse_bilateral_name(name: str) -> tuple[str, str | None]:
    """Parse layer name into (base, side). See color_for_layer for examples."""
    if name.endswith(" - Ipsi"):
        return name[:-len(" - Ipsi")], "ipsi"
    if name.endswith(" - Contra"):
        return name[:-len(" - Contra")], "contra"
    return name, None


def _tissue_number_from_base(base: str) -> int:
    """Extract integer from 'Tissue_03' -> 2 (0-indexed for palette)."""
    m = re.search(r"(\d+)", base)
    if m:
        return int(m.group(1)) - 1
    return abs(hash(base)) % len(TISSUE_PALETTE_HEX)


# Lightness adjustment factors for bilateral pairs.
# Tuned so ipsi and contra stay inside the 0.3-0.85 lightness band, where
# both HALO and our QC canvas render them clearly on any background.
IPSI_LIGHTNESS_FACTOR = 1.3
CONTRA_LIGHTNESS_FACTOR = 0.65


def rgb_for_layer(name: str) -> tuple[int, int, int]:
    """Return the (r, g, b) color tuple for a layer name, applying the same
    palette and ipsi/contra pairing logic used everywhere in haloqc.
    """
    base, side = _parse_bilateral_name(name)
    idx = _tissue_number_from_base(base)
    base_rgb = hex_to_rgb(TISSUE_PALETTE_HEX[idx % len(TISSUE_PALETTE_HEX)])
    if side == "ipsi":
        return _adjust_lightness(base_rgb, IPSI_LIGHTNESS_FACTOR)
    if side == "contra":
        return _adjust_lightness(base_rgb, CONTRA_LIGHTNESS_FACTOR)
    return base_rgb


def halo_line_color_for_layer(name: str) -> str:
    """Return the LineColor integer (as a decimal string) HALO expects.

    This is the key function that makes tissues show as distinct colors when
    re-imported to HALO instead of defaulting to red.
    """
    r, g, b = rgb_for_layer(name)
    return str(rgb_to_halo_int(r, g, b))
