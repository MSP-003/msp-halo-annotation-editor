"""
HaloQC entry point.

Usage:
    python -m haloqc          # launch GUI
    haloqc                    # (after `pip install -e .`)
"""

from __future__ import annotations

import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from haloqc.ui.main_window import MainWindow
from haloqc.ui.theme import apply_theme


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("MSP HALO Annotation Editor")
    app.setOrganizationName("msp")

    # Apply the modern theme BEFORE creating the main window so initial
    # widget construction picks up the stylesheet.
    apply_theme(app, theme="dark")

    # Set a decent default font. QSS handles most sizing but the base font
    # controls system-drawn elements like native menus.
    default_font = QFont("Inter", 10)
    if not default_font.exactMatch():
        # Fall back to Segoe UI / system default
        default_font = QFont("Segoe UI", 10)
    app.setFont(default_font)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
