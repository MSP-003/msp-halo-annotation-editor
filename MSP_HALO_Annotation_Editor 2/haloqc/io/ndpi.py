"""
NDPI slide reader with multi-channel compositing and contrast adjustment.

Uses openslide-python to read the multi-resolution pyramid, blends channels
with per-channel color tints and auto-contrast. Everything is cached per
(path, downsample) so switching between composite and single-channel views
doesn't re-read from disk.

openslide is an optional import: if it isn't installed, we fall back to
placeholder gray thumbnails so the separator/bilateral logic still works for
headless batch runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    openslide = None
    OPENSLIDE_AVAILABLE = False


# Channel color tints for fluorescence pseudo-coloring.
# Keys are normalized channel name tokens (case-insensitive, prefix match).
CHANNEL_COLORS: dict[str, tuple[float, float, float]] = {
    "dapi": (0.2, 0.4, 1.0),      # blue
    "fitc": (0.0, 1.0, 0.2),      # green
    "trtc": (1.0, 0.2, 0.0),      # red
    "tritc": (1.0, 0.2, 0.0),     # red (alternate spelling)
    "cy5": (1.0, 0.0, 1.0),       # magenta
    "cy3": (1.0, 0.5, 0.0),       # orange
    "cy7": (0.8, 0.0, 0.4),       # deep pink
    "bright": (1.0, 1.0, 1.0),    # white (pass-through)
    "brightfield": (1.0, 1.0, 1.0),
}

# Default auto-select priority when user picks "Auto"
CHANNEL_PRIORITY = ["dapi", "fitc", "trtc", "tritc", "cy5", "cy3", "bright"]


def normalize_channel_name(name: str) -> str:
    """Lowercase, strip whitespace and leading token. 'Dapi 378' -> 'dapi'."""
    tokens = name.lower().replace("-", " ").split()
    for tok in tokens:
        for key in CHANNEL_COLORS:
            if tok.startswith(key):
                return key
    # Fallback: return the first word
    return tokens[0] if tokens else name.lower()


def channel_color_for(name: str) -> tuple[float, float, float]:
    """Return the RGB tint for a channel name. White for unknown channels."""
    key = normalize_channel_name(name)
    return CHANNEL_COLORS.get(key, (1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# NDPIS manifest parsing
# ---------------------------------------------------------------------------

@dataclass
class SlideSet:
    """A group of per-channel .ndpi files that belong to one sample.

    Parsed from a .ndpis manifest file. Channel names are the keys from the
    [NanoZoomer...] section (e.g., "Fitc 474", "Dapi 378").
    """
    ndpis_path: Path
    channels: dict[str, Path] = field(default_factory=dict)  # channel name -> .ndpi path

    @property
    def stem(self) -> str:
        """Base name used for pairing to annotation files."""
        return self.ndpis_path.stem

    def pick_channel(self, preference: str = "auto") -> tuple[str, Path] | None:
        """Return (channel_name, path) for the preferred channel.

        If `preference` is "auto", walks CHANNEL_PRIORITY and returns the first
        channel that exists. Otherwise, looks for a normalized match.
        """
        if not self.channels:
            return None
        if preference != "auto":
            pref_norm = normalize_channel_name(preference)
            for name, path in self.channels.items():
                if normalize_channel_name(name) == pref_norm:
                    return name, path
            return None
        # Auto: walk priority list
        normalized = {normalize_channel_name(n): (n, p) for n, p in self.channels.items()}
        for key in CHANNEL_PRIORITY:
            if key in normalized:
                return normalized[key]
        # Fall back to first available channel
        first = next(iter(self.channels.items()))
        return first


def parse_ndpis(path: str | Path) -> SlideSet:
    """Parse a .ndpis manifest into a SlideSet."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    channels: dict[str, Path] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("[") or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key.lower().startswith("image"):
            continue
        # The value is the .ndpi filename; derive the channel name from the
        # filename suffix after the sample stem. Format observed:
        #   "S01 - 2025-10-24 10.25.34-Fitc 474.ndpi"
        ndpi_path = path.parent / value
        # Extract the channel descriptor: the part after the last "-" before ".ndpi"
        base = Path(value).stem  # "S01 - 2025-10-24 10.25.34-Fitc 474"
        if "-" in base:
            channel_name = base.rsplit("-", 1)[1].strip()
        else:
            channel_name = f"Channel {len(channels)}"
        channels[channel_name] = ndpi_path

    return SlideSet(ndpis_path=path, channels=channels)


# ---------------------------------------------------------------------------
# NDPI image reading and cache
# ---------------------------------------------------------------------------

@dataclass
class Thumbnail:
    """A single cached grayscale thumbnail at a known downsample level."""
    data: np.ndarray           # (H, W) uint8
    downsample: float          # pixels-per-thumbnail-pixel at level 0
    level: int                 # openslide pyramid level used
    channel_name: str
    slide_dims: tuple[int, int]  # (width, height) at level 0


class NdpiReader:
    """Reads NDPI files via openslide, caches thumbnails by (path, level).

    Non-reentrant: intended for use from the UI thread or a single worker.
    """

    def __init__(self, max_cache_items: int = 64) -> None:
        self._cache: dict[tuple[str, int], Thumbnail] = {}
        self._max_cache = max_cache_items

    def available(self) -> bool:
        return OPENSLIDE_AVAILABLE

    def pick_level_for_downsample(self, slide_path: Path, target_downsample: float) -> int:
        """Pick the openslide pyramid level closest to `target_downsample`."""
        if not OPENSLIDE_AVAILABLE:
            return 0
        with openslide.OpenSlide(str(slide_path)) as sl:
            best = sl.get_best_level_for_downsample(target_downsample)
            return int(best)

    def read_thumbnail(
        self,
        slide_path: str | Path,
        channel_name: str,
        downsample: float = 32.0,
    ) -> Thumbnail:
        """Return a grayscale thumbnail of the slide at ~downsample resolution.

        If openslide isn't available, returns a blank placeholder.
        """
        slide_path = Path(slide_path)

        if not OPENSLIDE_AVAILABLE or not slide_path.exists():
            # Placeholder: 500x500 mid-gray
            return Thumbnail(
                data=np.full((500, 500), 128, dtype=np.uint8),
                downsample=downsample,
                level=0,
                channel_name=channel_name,
                slide_dims=(0, 0),
            )

        with openslide.OpenSlide(str(slide_path)) as sl:
            level = sl.get_best_level_for_downsample(downsample)
            actual_ds = sl.level_downsamples[level]
            w, h = sl.level_dimensions[level]
            region = sl.read_region((0, 0), level, (w, h))  # PIL RGBA
            region = region.convert("L")  # to grayscale
            arr = np.asarray(region, dtype=np.uint8)
            slide_dims = sl.level_dimensions[0]

        return Thumbnail(
            data=arr,
            downsample=float(actual_ds),
            level=int(level),
            channel_name=channel_name,
            slide_dims=(int(slide_dims[0]), int(slide_dims[1])),
        )

    def read_slide_dimensions(self, slide_path: str | Path) -> tuple[int, int]:
        """Return (width, height) of the slide at level 0."""
        if not OPENSLIDE_AVAILABLE or not Path(slide_path).exists():
            return (0, 0)
        with openslide.OpenSlide(str(slide_path)) as sl:
            w, h = sl.level_dimensions[0]
            return (int(w), int(h))


# ---------------------------------------------------------------------------
# Contrast adjustment and compositing
# ---------------------------------------------------------------------------

def auto_contrast_stretch(
    gray: np.ndarray,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> np.ndarray:
    """Return a float image in [0, 1] stretched to the given percentile range."""
    arr = gray.astype(np.float32)
    lo = np.percentile(arr, low_pct)
    hi = np.percentile(arr, high_pct)
    if hi - lo < 1e-6:
        # Flat image: preserve the original intensity (normalized to [0, 1])
        # so that constant-gray placeholders don't end up pitch black.
        return np.clip(arr / 255.0, 0.0, 1.0)
    stretched = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return stretched


def composite_channels(
    thumbnails: list[Thumbnail],
    *,
    enabled: dict[str, bool] | None = None,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> np.ndarray:
    """Blend multiple channel thumbnails into a single RGB image.

    All thumbnails must have the same shape. Each channel is contrast-stretched
    with the given percentiles, multiplied by its pseudocolor tint, and summed.
    The result is clipped to [0, 255] uint8.

    `enabled`: optional map of channel_name -> bool to turn channels on/off.
    """
    if not thumbnails:
        return np.zeros((500, 500, 3), dtype=np.uint8)

    shape = thumbnails[0].data.shape
    for t in thumbnails:
        if t.data.shape != shape:
            raise ValueError("All thumbnails must have the same shape for compositing")

    enabled = enabled or {}
    rgb = np.zeros((shape[0], shape[1], 3), dtype=np.float32)

    for t in thumbnails:
        if enabled.get(t.channel_name, True) is False:
            continue
        stretched = auto_contrast_stretch(t.data, low_pct, high_pct)
        r, g, b = channel_color_for(t.channel_name)
        rgb[..., 0] += stretched * r
        rgb[..., 1] += stretched * g
        rgb[..., 2] += stretched * b

    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def single_channel_image(
    thumbnail: Thumbnail,
    *,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
    apply_color: bool = True,
) -> np.ndarray:
    """Return a colorized single-channel RGB image."""
    stretched = auto_contrast_stretch(thumbnail.data, low_pct, high_pct)
    if not apply_color:
        gray = (stretched * 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)
    r, g, b = channel_color_for(thumbnail.channel_name)
    rgb = np.stack([stretched * r, stretched * g, stretched * b], axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
