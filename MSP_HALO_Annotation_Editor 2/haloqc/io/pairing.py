"""
File pairing: match HALO .annotations files to their corresponding NDPI image sets.

HALO annotation files tend to have underscores in their names
(S01_-_2025-10-24_10_25_34.annotations) while the slide scanner outputs use
spaces and dots (S01 - 2025-10-24 10.25.34.ndpis). We normalize both sides
to a common form for comparison.

When multiple annotations might map to one image (e.g., a re-analyzed set),
we take longest-prefix match with warning. When no image is found, the
annotation is paired with None and the UI shows an "image not found" state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from haloqc.io.ndpi import SlideSet, parse_ndpis


@dataclass
class FilePair:
    annotation_path: Path
    slide_set: SlideSet | None
    notes: list[str]  # warnings or diagnostics about this pair


def normalize_stem(stem: str) -> str:
    """Collapse whitespace, underscores, dots, and hyphens into a single '_'."""
    # Keep alphanumerics, collapse everything else to underscore
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", stem)
    normalized = normalized.strip("_").lower()
    return normalized


def find_annotations(folder_or_file: Path) -> list[Path]:
    """Return a list of .annotations files from a path (folder or single file)."""
    folder_or_file = Path(folder_or_file)
    if folder_or_file.is_file():
        return [folder_or_file]
    return sorted(folder_or_file.glob("*.annotations"))


def find_slide_sets(folder: Path) -> list[SlideSet]:
    """Return all SlideSets from a folder of .ndpis manifests."""
    folder = Path(folder)
    sets = []
    for path in sorted(folder.glob("*.ndpis")):
        try:
            sets.append(parse_ndpis(path))
        except Exception as e:
            # Skip manifests we can't parse; the UI can surface this
            continue
    return sets


def pair_files(
    annotation_paths: list[Path],
    slide_sets: list[SlideSet],
) -> list[FilePair]:
    """Pair each annotation file with its best-matching SlideSet.

    Returns one FilePair per annotation path, always in the same order as the
    input. SlideSet is None if no plausible match was found.
    """
    # Build a lookup table: normalized stem -> SlideSet
    by_stem: dict[str, SlideSet] = {}
    duplicate_stems: set[str] = set()
    for s in slide_sets:
        key = normalize_stem(s.stem)
        if key in by_stem:
            duplicate_stems.add(key)
        else:
            by_stem[key] = s

    pairs: list[FilePair] = []
    for ap in annotation_paths:
        notes: list[str] = []
        stem_norm = normalize_stem(ap.stem)
        # Many HALO workflows append "_TissueSeparated" to the stem; strip that
        # so we can still match the original slide.
        cleaned = re.sub(r"_tissueseparated(_bilateral)?$", "", stem_norm)

        match = by_stem.get(cleaned)
        if match is None:
            match = by_stem.get(stem_norm)

        if match is None:
            # Fuzzy fallback: longest common prefix
            best_key = None
            best_len = 0
            for key in by_stem:
                # compare stems from the left
                n = 0
                for a, b in zip(cleaned, key):
                    if a == b:
                        n += 1
                    else:
                        break
                # require at least 8-char common prefix to reduce false positives
                if n >= 8 and n > best_len:
                    best_len = n
                    best_key = key
            if best_key:
                match = by_stem[best_key]
                notes.append(
                    f"Fuzzy match: annotation stem {cleaned!r} matched "
                    f"image stem {best_key!r} via {best_len}-char common prefix"
                )

        if match is None:
            notes.append("No matching .ndpis image found for this annotation")
        elif normalize_stem(match.stem) in duplicate_stems:
            notes.append(
                f"Multiple .ndpis files share the stem {match.stem!r} - using first"
            )

        pairs.append(FilePair(annotation_path=ap, slide_set=match, notes=notes))

    return pairs
