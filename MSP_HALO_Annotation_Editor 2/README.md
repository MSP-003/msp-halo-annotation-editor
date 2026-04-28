# MSP HALO Annotation Editor

**Tissue Separator & Bilateral Splitter for HALO-AI annotation exports.**

A Python desktop application that post-processes `.annotations` files exported from HALO / HALO-AI. It automates two correction passes that are otherwise tedious or impossible in HALO itself:

1. **Tissue Separation** — splits HALO-AI's single whole-tissue layer into one layer per tissue section, with grid-aware ordering, merge-by-distance thresholding, and an interactive QC canvas.
2. **Bilateral Splitting** — computes per-tissue midlines using PCA (principal axis) or vertical-bbox modes, splits each tissue into ipsilateral and contralateral halves with HALO-compatible color coding.

Both stages support full manual override: rename, merge, split, move-regions, delete, shift-numbering, drag-to-adjust cut lines, single-level undo. Output filenames match input by default, so HALO batch re-import works without renaming.

---

## Quick start

### Requirements
- Python 3.10 or newer
- (Optional) OpenSlide for NDPI thumbnail rendering in the QC canvas

### Install

```bash
python -m pip install -e .
```

### Run

```bash
python -m haloqc
```

That's it — the GUI opens on the input page.

---

## OpenSlide (optional but recommended)

The separator and bilateral splitter work on annotation geometry alone, so OpenSlide is technically optional. If installed, the QC canvas renders the actual slide image behind your tissue outlines for visual confirmation.

**macOS:** `brew install openslide`

**Windows:** Download from <https://openslide.org/download/>, extract somewhere stable (e.g. `D:\openslide`), and either add the `bin` folder to PATH or place it adjacent to the Python install.

**Linux:** `sudo apt install openslide-tools libopenslide0`

---

## Workflow

1. **Pick inputs** — one `.annotations` file, multiple files (Ctrl/Cmd-click), or a whole folder
2. **Output folders** auto-create alongside the input by default (`TissueSeparated/` and `BilateralSplit/` subfolders), or pick manually
3. **Click Run Tissue Separation** — the separator runs, you land on the Stage 1 QC view
4. **Review and correct** tissue groupings — right-click any tissue for rename, merge, split, move-regions, or delete options. Use Shift Numbering for missing-slice scenarios. Ctrl+Z to undo.
5. **Continue to Bilateral** — bilateral splitter runs, you land on Stage 2 QC view
6. **Review and adjust cuts** — drag the midline endpoints, rotate, or nudge to perfect each per-tissue split
7. **Save Final Output** — writes everything to your output folders

Outputs are drop-in compatible with HALO batch re-import.

---

## Default parameters

| Section | Parameter | Default |
|---|---|---|
| Separator | Expected tissue pieces | 8 |
| Separator | Grid layout (CxR) | 4x2 |
| Separator | Merge distance threshold | 400 µm |
| Separator | Microns per pixel | 0.5 |
| Bilateral | Ipsilateral side | Right |
| Bilateral | Midline detection | Principal axis (PCA) |
| Display | Default channel | Composite (all channels) |

All adjustable in the input panel.

---

## Project layout

```
haloqc/
├── io/           — parse .annotations, .ndpis, file pairing
├── core/         — separator, bilateral, geometry, edits, QC, colors
├── ui/           — PySide6 widgets, theme, dialogs
├── main.py       — entry point
└── pipeline.py   — orchestrates the end-to-end flow
```

---

## Known limitations

- Manual per-channel contrast sliders are not exposed (auto-percentile only)
- The bilateral splitter uses MATLAB-compatible polygon traversal; complex C-shaped tissues crossing the midline at 4+ points may produce slightly malformed boundaries

---

## Author

Mark St. Pierre — Neurology Discovery, Translational Models
