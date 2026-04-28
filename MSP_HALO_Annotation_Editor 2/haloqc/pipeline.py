"""
End-to-end processing pipeline.

Orchestrates the sequence:
    parse annotations -> separate tissues -> bilateral split -> QC -> write output

Callable from both the CLI (headless batch) and the UI (after the user has
approved the separation stage but before the bilateral stage).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from haloqc.core.bilateral import (
    BilateralParams,
    BilateralResult,
    MidlineCut,
    split_bilateral,
)
from haloqc.core.qc import (
    SampleQC,
    check_bilateral,
    check_separation,
    write_batch_report,
    write_qc_log,
)
from haloqc.core.separator import (
    SeparationResult,
    SeparatorParams,
    separate_tissues,
    separation_to_annotation_file,
)
from haloqc.io.annotations import AnnotationFile, parse_annotations, write_annotations
from haloqc.io.pairing import FilePair


@dataclass
class SampleResult:
    """All artifacts produced for one sample by the pipeline."""
    pair: FilePair
    separated_file: AnnotationFile | None = None
    separation: SeparationResult | None = None
    bilateral: BilateralResult | None = None
    separator_qc: SampleQC | None = None
    bilateral_qc: SampleQC | None = None
    separated_output_path: Path | None = None
    bilateral_output_path: Path | None = None
    qc_log_path: Path | None = None
    error: str | None = None


def run_separation_for_pair(
    pair: FilePair,
    separator_params: SeparatorParams,
    output_folder: Path,
    *,
    append_suffix: bool = True,
) -> SampleResult:
    """Run tissue separation for one pair. Writes an .annotations file to
    `output_folder`. If `append_suffix=True`, filename is `<stem>_TissueSeparated.annotations`.
    If False, filename is `<stem>.annotations` (matches input stem exactly,
    enables HALO batch re-import).
    """
    result = SampleResult(pair=pair)
    try:
        annot_file = parse_annotations(pair.annotation_path)
    except Exception as e:
        result.error = f"Failed to parse annotations: {e}"
        return result

    # Preserve base visibility/color from the first input layer
    base_visible = annot_file.layers[0].visible if annot_file.layers else "True"
    base_color = annot_file.layers[0].line_color if annot_file.layers else "1772542"

    try:
        separation = separate_tissues(annot_file, separator_params)
    except Exception as e:
        result.error = f"Separation failed: {e}"
        return result

    result.separation = separation
    separated_file = separation_to_annotation_file(
        separation,
        base_visible=base_visible,
        base_line_color=base_color,
        source_path=pair.annotation_path,
    )
    result.separated_file = separated_file

    # Write output
    suffix = "_TissueSeparated" if append_suffix else ""
    out_path = output_folder / f"{pair.annotation_path.stem}{suffix}.annotations"
    output_folder.mkdir(parents=True, exist_ok=True)
    write_annotations(separated_file, out_path)
    result.separated_output_path = out_path

    # Run QC
    result.separator_qc = check_separation(
        separation,
        expected_tissues=separator_params.expected_tissues,
        merge_threshold_um=separator_params.merge_distance_microns,
        sample_name=pair.annotation_path.stem,
    )

    return result


def run_bilateral_for_sample(
    sample: SampleResult,
    bilateral_params: BilateralParams,
    output_folder: Path,
    *,
    manual_cuts: dict[str, MidlineCut] | None = None,
    append_suffix: bool = True,
) -> SampleResult:
    """Run bilateral splitting on an already-separated sample. If
    `append_suffix=True`, output filename is `<stem>_Bilateral.annotations`.
    If False, filename is `<stem>.annotations` (matches input stem exactly)."""
    if sample.separated_file is None:
        sample.error = "Cannot run bilateral: no separated file available"
        return sample

    try:
        bilateral = split_bilateral(
            sample.separated_file,
            bilateral_params,
            manual_cuts=manual_cuts,
        )
    except Exception as e:
        sample.error = f"Bilateral failed: {e}"
        return sample

    sample.bilateral = bilateral

    suffix = "_Bilateral" if append_suffix else ""
    out_path = output_folder / f"{sample.pair.annotation_path.stem}{suffix}.annotations"
    output_folder.mkdir(parents=True, exist_ok=True)
    write_annotations(bilateral.output_file, out_path)
    sample.bilateral_output_path = out_path

    sample.bilateral_qc = check_bilateral(
        bilateral,
        separated_file=sample.separated_file,
        sample_name=sample.pair.annotation_path.stem,
    )

    return sample


def finalize_sample(
    sample: SampleResult,
    output_folder: Path,
) -> SampleResult:
    """Write the per-sample QC log combining separator + bilateral flags."""
    flags = []
    if sample.separator_qc:
        for f in sample.separator_qc.flags:
            # Re-tag section
            flags.append(f)
    if sample.bilateral_qc:
        flags.extend(sample.bilateral_qc.flags)

    from haloqc.core.qc import SampleQC
    combined = SampleQC(
        sample_name=sample.pair.annotation_path.stem,
        flags=flags,
    )

    diagnostics: list[str] = []
    if sample.separation:
        diagnostics.append("=== Separation ===")
        diagnostics.extend(sample.separation.diagnostics)
    if sample.bilateral:
        diagnostics.append("=== Bilateral ===")
        diagnostics.extend(sample.bilateral.diagnostics)

    log_path = output_folder / f"{sample.pair.annotation_path.stem}_qc_log.txt"
    write_qc_log(combined, log_path, diagnostics=diagnostics)
    sample.qc_log_path = log_path
    return sample


def write_batch_summary(samples: list[SampleResult], output_folder: Path) -> Path:
    """Write a batch-level HTML summary report."""
    qcs: list[SampleQC] = []
    from haloqc.core.qc import SampleQC
    for s in samples:
        sep_flags = s.separator_qc.flags if s.separator_qc else []
        bil_flags = s.bilateral_qc.flags if s.bilateral_qc else []
        combined = SampleQC(
            sample_name=s.pair.annotation_path.stem,
            flags=sep_flags + bil_flags,
        )
        if s.error:
            from haloqc.core.qc import QCFlag
            combined.flags.insert(0, QCFlag(severity="error", message=s.error))
        qcs.append(combined)

    report_path = output_folder / "batch_qc_report.html"
    write_batch_report(qcs, report_path)
    return report_path
