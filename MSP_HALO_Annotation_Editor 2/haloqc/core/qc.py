"""
Quality control checks and report generation.

After running the tissue separator and/or bilateral splitter, we compute
a set of flags on each sample and produce a human-readable log. Flags
escalate in severity: INFO (everything looks fine), WARN (something to
eyeball), ERROR (almost certainly needs manual correction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from haloqc.core.separator import SeparationResult
from haloqc.core.bilateral import BilateralResult, MidlineCut
from haloqc.io.annotations import AnnotationFile

Severity = Literal["info", "warn", "error"]


@dataclass
class QCFlag:
    severity: Severity
    message: str
    tissue_name: str | None = None  # None means it's a sample-level flag


@dataclass
class SampleQC:
    """All QC results for a single sample (one annotation file)."""
    sample_name: str
    flags: list[QCFlag] = field(default_factory=list)

    @property
    def max_severity(self) -> Severity:
        if any(f.severity == "error" for f in self.flags):
            return "error"
        if any(f.severity == "warn" for f in self.flags):
            return "warn"
        return "info"

    @property
    def n_errors(self) -> int:
        return sum(1 for f in self.flags if f.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for f in self.flags if f.severity == "warn")


# ---------------------------------------------------------------------------
# Separator QC
# ---------------------------------------------------------------------------

def check_separation(
    result: SeparationResult,
    expected_tissues: int | None,
    merge_threshold_um: float,
    sample_name: str,
) -> SampleQC:
    """Run QC checks on a tissue separation result."""
    qc = SampleQC(sample_name=sample_name)

    # Check 1: tissue count matches expected
    n_found = len(result.groups)
    if expected_tissues is not None and n_found != expected_tissues:
        sev: Severity = "error" if abs(n_found - expected_tissues) > 1 else "warn"
        qc.flags.append(QCFlag(
            severity=sev,
            message=f"Found {n_found} tissues, expected {expected_tissues}",
        ))

    # Check 2: area outliers (> 2 sigma from cohort mean)
    if n_found >= 3:
        areas = np.array([g.total_positive_area for g in result.groups])
        mean_a = areas.mean()
        std_a = areas.std()
        if std_a > 0:
            z_scores = (areas - mean_a) / std_a
            for i, z in enumerate(z_scores, start=1):
                if abs(z) > 2.0:
                    pct = (areas[i-1] / mean_a - 1.0) * 100
                    direction = "larger" if pct > 0 else "smaller"
                    qc.flags.append(QCFlag(
                        severity="warn",
                        tissue_name=f"Tissue_{i:02d}",
                        message=(
                            f"Area is {abs(pct):.0f}% {direction} than cohort mean "
                            f"(z-score {z:+.2f}) - possible incorrect grouping"
                        ),
                    ))

    # Check 3: ambiguous merges (pairs close to the threshold)
    # We consider nearest-neighbor distances within 1.5x the merge threshold
    # but still outside it as ambiguous.
    if result.nearest_distances_um:
        ambiguous = [
            d for d in result.nearest_distances_um
            if merge_threshold_um < d <= 1.5 * merge_threshold_um
        ]
        if ambiguous:
            qc.flags.append(QCFlag(
                severity="warn",
                message=(
                    f"{len(ambiguous)} inter-tissue distance(s) landed within "
                    f"50% of the merge threshold ({merge_threshold_um:.0f} um). "
                    "Consider verifying these pieces aren't split by the threshold."
                ),
            ))

    if not qc.flags:
        qc.flags.append(QCFlag(severity="info", message="Separation looks clean"))

    return qc


# ---------------------------------------------------------------------------
# Bilateral QC
# ---------------------------------------------------------------------------

def check_bilateral(
    result: BilateralResult,
    separated_file: AnnotationFile,
    sample_name: str,
    *,
    max_area_ratio: float = 2.0,
) -> SampleQC:
    """Run QC checks on a bilateral split result.

    Flags tissues where the ipsi/contra area ratio exceeds `max_area_ratio`,
    which usually indicates the cut line is placed incorrectly.
    """
    qc = SampleQC(sample_name=sample_name)

    # Index output layers by base name (strip " - Ipsi" / " - Contra")
    by_base: dict[str, dict[str, list]] = {}
    for layer in result.output_file.layers:
        if layer.name.endswith(" - Ipsi"):
            base = layer.name[:-len(" - Ipsi")]
            side = "ipsi"
        elif layer.name.endswith(" - Contra"):
            base = layer.name[:-len(" - Contra")]
            side = "contra"
        else:
            continue
        by_base.setdefault(base, {"ipsi": [], "contra": []})[side].append(layer)

    for input_layer in separated_file.layers:
        base = input_layer.name
        sides = by_base.get(base)
        if sides is None:
            qc.flags.append(QCFlag(
                severity="error",
                tissue_name=base,
                message="No ipsi/contra layers produced for this tissue",
            ))
            continue

        ipsi_area = sum(
            _poly_area(r.vertices)
            for layer in sides["ipsi"] for r in layer.positive_regions
        )
        contra_area = sum(
            _poly_area(r.vertices)
            for layer in sides["contra"] for r in layer.positive_regions
        )

        if ipsi_area == 0 or contra_area == 0:
            qc.flags.append(QCFlag(
                severity="error",
                tissue_name=base,
                message="One side has zero area - cut may be outside tissue",
            ))
            continue

        ratio = max(ipsi_area, contra_area) / min(ipsi_area, contra_area)
        if ratio > max_area_ratio:
            larger = "ipsi" if ipsi_area > contra_area else "contra"
            qc.flags.append(QCFlag(
                severity="warn",
                tissue_name=base,
                message=(
                    f"Asymmetric split: {larger} side is {ratio:.1f}x larger "
                    "- cut line may be misplaced"
                ),
            ))

    if not any(f.severity in ("warn", "error") for f in qc.flags):
        qc.flags.append(QCFlag(
            severity="info",
            message="Bilateral splits look symmetric",
        ))

    return qc


def _poly_area(verts: np.ndarray) -> float:
    x, y = verts[:, 0], verts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_qc_log(qc: SampleQC, path: Path, diagnostics: list[str] | None = None) -> None:
    """Write a single-sample plain-text QC log."""
    lines = [
        f"HALO QC Log",
        f"Sample: {qc.sample_name}",
        f"Overall: {qc.max_severity.upper()}  ({qc.n_errors} errors, {qc.n_warnings} warnings)",
        "",
    ]
    if diagnostics:
        lines.append("--- Processing Diagnostics ---")
        lines.extend(diagnostics)
        lines.append("")

    lines.append("--- QC Flags ---")
    for flag in qc.flags:
        prefix = f"[{flag.severity.upper():5s}]"
        if flag.tissue_name:
            lines.append(f"{prefix} {flag.tissue_name}: {flag.message}")
        else:
            lines.append(f"{prefix} {flag.message}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_batch_report(results: list[SampleQC], path: Path) -> None:
    """Write a minimal HTML summary of a batch QC run."""
    rows = []
    for r in results:
        color = {"info": "#d4edda", "warn": "#fff3cd", "error": "#f8d7da"}[r.max_severity]
        flag_html = "<br>".join(
            f"<span style='color:#666'>[{f.severity.upper()}]</span> "
            f"{'<b>' + f.tissue_name + ':</b> ' if f.tissue_name else ''}{f.message}"
            for f in r.flags
        )
        rows.append(
            f"<tr style='background:{color}'>"
            f"<td>{r.sample_name}</td>"
            f"<td style='text-align:center'>{r.max_severity.upper()}</td>"
            f"<td style='text-align:center'>{r.n_errors}</td>"
            f"<td style='text-align:center'>{r.n_warnings}</td>"
            f"<td>{flag_html}</td>"
            "</tr>"
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>HaloQC Batch Report</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', sans-serif; margin: 2em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; vertical-align: top; }}
  th {{ background: #f4f4f4; }}
</style></head>
<body>
<h1>HaloQC Batch Report</h1>
<p>{len(results)} samples processed. {sum(1 for r in results if r.max_severity == 'error')} with errors, {sum(1 for r in results if r.max_severity == 'warn')} with warnings.</p>
<table>
  <thead><tr><th>Sample</th><th>Status</th><th>Errors</th><th>Warnings</th><th>Details</th></tr></thead>
  <tbody>
    {''.join(rows)}
  </tbody>
</table>
</body></html>
"""
    path.write_text(html, encoding="utf-8")
