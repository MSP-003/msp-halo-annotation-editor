"""
Thumbnail controller.

Thin layer on top of NdpiReader that caches rendered RGB thumbnails per
(slide_path, requested_mode, channel_selection, downsample). Used by the QC
views so switching tissues doesn't re-render, and switching display modes
doesn't re-read the slide.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from haloqc.io.ndpi import (
    NdpiReader,
    SlideSet,
    Thumbnail,
    composite_channels,
    single_channel_image,
)


@dataclass
class RenderRequest:
    slide_set: SlideSet | None
    mode: str  # "composite", "auto", or a specific channel display name
    downsample: float = 32.0
    channel_enabled: dict[str, bool] | None = None
    low_pct: float = 1.0
    high_pct: float = 99.0


class ThumbnailController:
    """Caches per-channel raw thumbnails and per-request composite renders."""

    def __init__(self) -> None:
        self._reader = NdpiReader()
        # Per-channel raw thumbnails: key = (slide_path, channel_name, downsample)
        self._raw_cache: dict[tuple[str, str, float], Thumbnail] = {}

    # ------------------------------------------------------------------
    def scene_size_for(self, slide_set: SlideSet | None) -> tuple[int, int]:
        """Return level-0 dimensions of the slide, or (0, 0) if no slide."""
        if slide_set is None or not slide_set.channels:
            return (0, 0)
        # Use the first channel; all channels of one sample share dimensions
        name, path = next(iter(slide_set.channels.items()))
        return self._reader.read_slide_dimensions(path)

    # ------------------------------------------------------------------
    def render(self, req: RenderRequest) -> np.ndarray | None:
        """Return an (H, W, 3) uint8 array for the requested view.

        Returns None if the slide set is missing or none of the channel files
        on disk can be read. The canvas interprets None as "no background",
        which looks better than compositing placeholder gray over empty space.
        """
        if req.slide_set is None or not req.slide_set.channels:
            return None
        # Bail early if none of the referenced NDPI files exist on disk.
        if not any(p.exists() for p in req.slide_set.channels.values()):
            return None

        mode = req.mode
        channels = req.slide_set.channels

        if mode.startswith("Composite") or mode == "composite":
            thumbs = self._load_channels(
                req.slide_set, list(channels.keys()), req.downsample
            )
            if not thumbs:
                return None
            thumbs = self._match_shapes(thumbs)
            return composite_channels(
                thumbs,
                enabled=req.channel_enabled,
                low_pct=req.low_pct,
                high_pct=req.high_pct,
            )

        # Single-channel modes
        target_key = None
        if mode.startswith("Auto"):
            picked = req.slide_set.pick_channel("auto")
            if picked is None:
                return None
            target_key, _ = picked
        else:
            # mode is either a full channel name or a short token like "DAPI"
            from haloqc.io.ndpi import normalize_channel_name
            target_norm = normalize_channel_name(mode)
            for name in channels:
                if normalize_channel_name(name) == target_norm:
                    target_key = name
                    break
            if target_key is None:
                # Try substring match
                for name in channels:
                    if mode.lower() in name.lower():
                        target_key = name
                        break
            if target_key is None:
                # Fall back to auto
                picked = req.slide_set.pick_channel("auto")
                if picked is None:
                    return None
                target_key, _ = picked

        thumbs = self._load_channels(req.slide_set, [target_key], req.downsample)
        if not thumbs:
            return None
        return single_channel_image(
            thumbs[0], low_pct=req.low_pct, high_pct=req.high_pct,
        )

    # ------------------------------------------------------------------
    def _load_channels(
        self,
        slide_set: SlideSet,
        channel_names: list[str],
        downsample: float,
    ) -> list[Thumbnail]:
        out: list[Thumbnail] = []
        for name in channel_names:
            path = slide_set.channels.get(name)
            if path is None:
                continue
            key = (str(path), name, downsample)
            thumb = self._raw_cache.get(key)
            if thumb is None:
                thumb = self._reader.read_thumbnail(path, name, downsample)
                self._raw_cache[key] = thumb
            out.append(thumb)
        return out

    @staticmethod
    def _match_shapes(thumbs: list[Thumbnail]) -> list[Thumbnail]:
        """Crop all thumbs to the minimum common shape (defensive).

        Normally all channels of one slide share dimensions, but fluorescence
        scanners occasionally produce thumbnails off by a row/column at a
        given level depending on rounding. Cropping to the common shape
        avoids a shape-mismatch crash in compositing.
        """
        if not thumbs:
            return thumbs
        min_h = min(t.data.shape[0] for t in thumbs)
        min_w = min(t.data.shape[1] for t in thumbs)
        cropped: list[Thumbnail] = []
        for t in thumbs:
            if t.data.shape[0] == min_h and t.data.shape[1] == min_w:
                cropped.append(t)
            else:
                cropped.append(Thumbnail(
                    data=t.data[:min_h, :min_w],
                    downsample=t.downsample,
                    level=t.level,
                    channel_name=t.channel_name,
                    slide_dims=t.slide_dims,
                ))
        return cropped
