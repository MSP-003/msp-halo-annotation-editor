"""
Main application window.

Uses a QStackedWidget to switch between three stages:
  1. Input panel   — pick files, set parameters, run separation
  2. Separator QC  — review separation, approve and continue
  3. Bilateral QC  — review cuts, correct as needed, save

The pipeline runs synchronously in the UI thread here. For typical batch
sizes (<50 samples) this is fine on a modern machine. If batches get
large, we'd move heavy work into a QThread with a progress dialog; v1
keeps it simple.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QStackedWidget,
    QStatusBar,
    QWidget,
)

from haloqc.core.bilateral import BilateralParams, split_bilateral
from haloqc.core.qc import check_bilateral
from haloqc.io.pairing import find_slide_sets, pair_files
from haloqc.pipeline import (
    SampleResult,
    finalize_sample,
    run_bilateral_for_sample,
    run_separation_for_pair,
    write_batch_summary,
)
from haloqc.io.annotations import write_annotations
from haloqc.ui.bilateral_view import BilateralQCView
from haloqc.ui.input_panel import InputPanel, InputSelections
from haloqc.ui.separator_view import SeparatorQCView
from haloqc.ui.thumbnail_controller import ThumbnailController


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MSP HALO Annotation Editor  |  Tissue Separator & Bilateral Splitter")
        self.resize(1500, 950)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._thumb_ctrl = ThumbnailController()

        self.input_panel = InputPanel()
        self.input_panel.run_requested.connect(self._on_run_separation)
        self._stack.addWidget(self.input_panel)

        self.separator_view = SeparatorQCView(self._thumb_ctrl)
        self.separator_view.back_to_input.connect(
            lambda: self._stack.setCurrentWidget(self.input_panel)
        )
        self.separator_view.continue_to_bilateral.connect(self._on_continue_to_bilateral)
        self._stack.addWidget(self.separator_view)

        self.bilateral_view = BilateralQCView(self._thumb_ctrl)
        self.bilateral_view.back_to_separator.connect(
            lambda: self._stack.setCurrentWidget(self.separator_view)
        )
        self.bilateral_view.save_and_finish.connect(self._on_save_and_finish)
        self._stack.addWidget(self.bilateral_view)

        self.setStatusBar(QStatusBar())

        # State
        self._selections: InputSelections | None = None
        self._samples: list[SampleResult] = []

        self._stack.setCurrentWidget(self.input_panel)

    # ------------------------------------------------------------------
    def _on_run_separation(self, selections: InputSelections) -> None:
        self._selections = selections

        # Discover slide sets from images folder (if provided)
        slide_sets = []
        if selections.images_folder and selections.images_folder.exists():
            slide_sets = find_slide_sets(selections.images_folder)

        pairs = pair_files(selections.annotation_paths, slide_sets)

        # Pre-flight: warn if any planned output path would overwrite an
        # input annotation file. This catches the "output folder = input
        # folder with no suffix" footgun.
        if not self._confirm_no_overwrite(pairs, selections):
            return

        n = len(pairs)
        progress = QProgressDialog(
            "Running tissue separation...", "Cancel", 0, n, self,
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        samples: list[SampleResult] = []
        for i, p in enumerate(pairs):
            if progress.wasCanceled():
                break
            progress.setLabelText(
                f"Separating tissues: {p.annotation_path.name} ({i+1}/{n})"
            )
            progress.setValue(i)
            # Pump UI
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()

            sample = run_separation_for_pair(
                p, selections.separator_params,
                selections.separator_output_folder,
                append_suffix=selections.append_stage_suffix,
            )
            samples.append(sample)

        progress.setValue(n)
        self._samples = samples
        self.separator_view.set_selections(selections)
        self.separator_view.load_samples(samples)
        self._stack.setCurrentWidget(self.separator_view)
        self.statusBar().showMessage(
            f"Tissue separation complete: {len(samples)} samples", 5000,
        )

    def _confirm_no_overwrite(self, pairs, selections: InputSelections) -> bool:
        """Show a warning if any planned output path matches an input path.

        Returns True if user proceeds, False if they cancel.
        """
        planned_outputs: list[tuple[Path, str]] = []  # (path, description)
        sep_suffix = "_TissueSeparated" if selections.append_stage_suffix else ""
        bil_suffix = "_Bilateral" if selections.append_stage_suffix else ""
        for p in pairs:
            stem = p.annotation_path.stem
            sep_out = selections.separator_output_folder / f"{stem}{sep_suffix}.annotations"
            bil_out = selections.bilateral_output_folder / f"{stem}{bil_suffix}.annotations"
            planned_outputs.append((sep_out, "tissue-separated"))
            planned_outputs.append((bil_out, "bilateral"))

        input_paths = {p.annotation_path.resolve() for p in pairs}
        collisions: list[str] = []
        for out_path, kind in planned_outputs:
            try:
                if out_path.resolve() in input_paths:
                    collisions.append(f"  • {out_path.name} (input file would be overwritten by {kind} output)")
            except OSError:
                pass

        if not collisions:
            return True

        msg = (
            "The following output paths would overwrite input annotation files:\n\n"
            + "\n".join(collisions)
            + "\n\nYou probably want to either:\n"
            "  • Pick a different output folder, or\n"
            "  • Enable \"Append stage suffix to filenames\" in the input panel\n\n"
            "Proceed anyway (overwrites will happen)?"
        )
        reply = QMessageBox.warning(
            self, "Output will overwrite input",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    # ------------------------------------------------------------------
    def _on_continue_to_bilateral(self) -> None:
        if not self._samples or self._selections is None:
            return

        dirty = self.separator_view.dirty_samples()

        # Rewrite the `_TissueSeparated.annotations` file to disk for any
        # sample that was edited, so the bilateral stage reads the corrected
        # separation. Also invalidate the in-memory bilateral result on those
        # samples so the "already_run" check below forces a re-run.
        for idx in dirty:
            s = self._samples[idx]
            if s.separated_file is None or s.separated_output_path is None:
                continue
            write_annotations(s.separated_file, s.separated_output_path)
            # Drop stale bilateral so the re-run covers this sample
            s.bilateral = None
            s.bilateral_qc = None

        # If we've already run bilateral on these samples AND no samples are
        # dirty, just switch views. This preserves any manual cuts the user
        # has made. Edits always force a re-run for the edited samples.
        already_run = (
            not dirty
            and all(s.bilateral is not None for s in self._samples if not s.error)
        )
        if already_run:
            self._stack.setCurrentWidget(self.bilateral_view)
            self.statusBar().showMessage(
                "Returned to bilateral QC (manual cuts preserved)", 3000,
            )
            return

        # Figure out which samples need a bilateral run. If any are dirty,
        # we only re-run those; otherwise we run all that haven't run yet.
        if dirty:
            to_run = [i for i in range(len(self._samples)) if i in dirty]
            progress_label = (
                f"Re-running bilateral for {len(to_run)} edited sample"
                f"{'s' if len(to_run) != 1 else ''}..."
            )
        else:
            to_run = [
                i for i, s in enumerate(self._samples)
                if s.bilateral is None and not s.error
            ]
            progress_label = "Running bilateral split..."

        n = len(to_run)
        if n > 0:
            progress = QProgressDialog(progress_label, "Cancel", 0, n, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)

            for count, i in enumerate(to_run):
                if progress.wasCanceled():
                    break
                sample = self._samples[i]
                progress.setLabelText(
                    f"Bilateral: {sample.pair.annotation_path.name} ({count + 1}/{n})"
                )
                progress.setValue(count)
                from PySide6.QtWidgets import QApplication
                QApplication.processEvents()

                run_bilateral_for_sample(
                    sample,
                    self._selections.bilateral_params,
                    self._selections.bilateral_output_folder,
                    append_suffix=self._selections.append_stage_suffix,
                )
            progress.setValue(n)

        # Mark dirty samples as clean now that bilateral has been re-run
        if dirty:
            self.separator_view.clear_dirty()

        self.bilateral_view.set_bilateral_params(self._selections.bilateral_params)
        self.bilateral_view.load_samples(self._samples)
        self._stack.setCurrentWidget(self.bilateral_view)
        self.statusBar().showMessage(
            f"Bilateral split complete: {len(self._samples)} samples", 5000,
        )

    # ------------------------------------------------------------------
    def _on_save_and_finish(self) -> None:
        if self._selections is None:
            return
        # Persist any pending manual-cut recomputations to disk, plus QC logs
        sep_folder = self._selections.separator_output_folder
        bil_folder = self._selections.bilateral_output_folder
        append_suffix = self._selections.append_stage_suffix
        manual_cuts_by_sample = self.bilateral_view.current_manual_cuts()
        samples = self.bilateral_view.samples()

        bil_suffix = "_Bilateral" if append_suffix else ""

        for i, sample in enumerate(samples):
            # If the user edited cuts and has a live bilateral result, write it
            if sample.bilateral is not None:
                out_path = bil_folder / (
                    f"{sample.pair.annotation_path.stem}{bil_suffix}.annotations"
                )
                bil_folder.mkdir(parents=True, exist_ok=True)
                write_annotations(sample.bilateral.output_file, out_path)
                sample.bilateral_output_path = out_path
            # QC log goes with the separator output (arbitrary choice; they're
            # per-sample logs describing both stages)
            finalize_sample(sample, sep_folder)

        report = write_batch_summary(samples, sep_folder)

        folders_msg = (
            f"Tissue-separated: {sep_folder}\n"
            f"Bilateral: {bil_folder}"
            if sep_folder != bil_folder
            else f"Output folder: {sep_folder}"
        )
        QMessageBox.information(
            self,
            "Saved",
            f"Saved {len(samples)} samples.\n\n{folders_msg}\n\nBatch report: {report.name}",
        )
