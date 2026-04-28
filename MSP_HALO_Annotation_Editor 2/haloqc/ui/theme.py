"""
Theme and styling for haloqc.

A central place for the color palette, typography, spacing tokens, and the
Qt Style Sheet (QSS) that applies them app-wide. Dark mode is the default
(imaging workflow benefits from dark chrome); light mode is available via
set_theme("light").

Design references: modern macOS, Linear, Notion. The goal is clean surfaces,
generous padding, subtle borders instead of hard lines, rounded corners, and
a single muted accent color for interactive elements.

Usage:
    from haloqc.ui.theme import apply_theme
    apply_theme(app, "dark")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Palette:
    # Background layers, lightest to deepest surface
    bg_app: str          # app window background
    bg_panel: str        # card/group-box background (one step lighter than bg_app)
    bg_input: str        # text input / combo / spin-box background
    bg_hover: str        # hover state for lists/buttons
    bg_pressed: str      # pressed / active state
    bg_selected: str     # selected row in lists

    # Borders & dividers
    border_subtle: str
    border_strong: str

    # Text
    text_primary: str
    text_secondary: str
    text_disabled: str
    text_on_accent: str

    # Accent (single muted color used for buttons, selections, focus ring)
    accent: str
    accent_hover: str
    accent_pressed: str

    # Semantic (for QC flag badges)
    info: str
    warn: str
    error: str
    success: str


# Dark palette - slate + soft blue accent.
# These hex values were picked to read well at typical monitor gamma and to
# avoid pure black/pure white (which feel harsh).
DARK = Palette(
    bg_app="#1b1d21",
    bg_panel="#23262c",
    bg_input="#2a2d33",
    bg_hover="#2f333a",
    bg_pressed="#383c44",
    bg_selected="#3d4655",
    border_subtle="#32363d",
    border_strong="#434851",
    text_primary="#e4e6eb",
    text_secondary="#9aa0ab",
    text_disabled="#5a6068",
    text_on_accent="#ffffff",
    accent="#5b8def",
    accent_hover="#6e9cf3",
    accent_pressed="#4a7cde",
    info="#5b8def",
    warn="#e0a84a",
    error="#e06868",
    success="#5fb87f",
)

# Light palette - warm off-white with the same accent
LIGHT = Palette(
    bg_app="#f5f5f7",
    bg_panel="#ffffff",
    bg_input="#ffffff",
    bg_hover="#ededf0",
    bg_pressed="#e0e0e5",
    bg_selected="#d5e2fb",
    border_subtle="#e3e3e8",
    border_strong="#c8c8cf",
    text_primary="#1a1a1f",
    text_secondary="#5d6068",
    text_disabled="#a8abb2",
    text_on_accent="#ffffff",
    accent="#3a7afe",
    accent_hover="#5189fe",
    accent_pressed="#2966e8",
    info="#3a7afe",
    warn="#c9851e",
    error="#c94747",
    success="#3d955e",
)


# Type scale (px) and spacing tokens
FONT_FAMILY = (
    '"Inter", "SF Pro Text", -apple-system, BlinkMacSystemFont, '
    '"Segoe UI", "Helvetica Neue", Arial, sans-serif'
)
SIZE_XS = 11
SIZE_SM = 12
SIZE_BASE = 13
SIZE_LG = 15
SIZE_XL = 18
SIZE_XXL = 22

RADIUS_SM = 4
RADIUS_MD = 6
RADIUS_LG = 10

SPACE_1 = 4
SPACE_2 = 8
SPACE_3 = 12
SPACE_4 = 16
SPACE_5 = 24


def build_stylesheet(p: Palette) -> str:
    """Return a Qt Style Sheet string for the given palette.

    QSS selectors target widget classes directly (e.g. QPushButton, not a
    custom objectName) so the theme applies without any per-widget wiring.
    Pseudo-states handle hover/pressed/disabled consistently.
    """
    return f"""
    /* ---------- Base ---------- */
    QWidget {{
        background-color: {p.bg_app};
        color: {p.text_primary};
        font-family: {FONT_FAMILY};
        font-size: {SIZE_BASE}px;
    }}

    QMainWindow, QDialog {{
        background-color: {p.bg_app};
    }}

    QStatusBar {{
        background-color: {p.bg_app};
        color: {p.text_secondary};
        border-top: 1px solid {p.border_subtle};
        padding: 2px 8px;
    }}

    /* ---------- Group box = card ---------- */
    QGroupBox {{
        background-color: {p.bg_panel};
        border: 1px solid {p.border_subtle};
        border-radius: {RADIUS_LG}px;
        padding: {SPACE_4}px {SPACE_3}px {SPACE_3}px {SPACE_3}px;
        margin-top: {SPACE_4}px;
        font-size: {SIZE_LG}px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: {SPACE_3}px;
        padding: 0 {SPACE_2}px;
        color: {p.text_primary};
        background-color: {p.bg_panel};
    }}

    /* ---------- Labels ---------- */
    QLabel {{
        background: transparent;
        color: {p.text_primary};
    }}
    QLabel:disabled {{
        color: {p.text_disabled};
    }}

    /* ---------- Buttons ---------- */
    QPushButton {{
        background-color: {p.bg_input};
        color: {p.text_primary};
        border: 1px solid {p.border_strong};
        border-radius: {RADIUS_MD}px;
        padding: 6px 14px;
        font-size: {SIZE_BASE}px;
        font-weight: 500;
        min-height: 22px;
    }}
    QPushButton:hover {{
        background-color: {p.bg_hover};
        border-color: {p.accent};
    }}
    QPushButton:pressed {{
        background-color: {p.bg_pressed};
    }}
    QPushButton:disabled {{
        background-color: {p.bg_input};
        color: {p.text_disabled};
        border-color: {p.border_subtle};
    }}
    QPushButton:default {{
        background-color: {p.accent};
        color: {p.text_on_accent};
        border-color: {p.accent};
    }}
    QPushButton:default:hover {{
        background-color: {p.accent_hover};
        border-color: {p.accent_hover};
    }}
    QPushButton:default:pressed {{
        background-color: {p.accent_pressed};
    }}

    /* ---------- Line edits / text boxes ---------- */
    QLineEdit, QTextEdit, QPlainTextEdit {{
        background-color: {p.bg_input};
        color: {p.text_primary};
        border: 1px solid {p.border_subtle};
        border-radius: {RADIUS_MD}px;
        padding: 5px 8px;
        selection-background-color: {p.accent};
        selection-color: {p.text_on_accent};
    }}
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
        border: 1px solid {p.accent};
    }}
    QLineEdit:disabled {{
        color: {p.text_disabled};
        background-color: {p.bg_app};
    }}
    QLineEdit:read-only {{
        background-color: {p.bg_app};
        color: {p.text_secondary};
    }}

    /* ---------- Combo / Spin ---------- */
    QComboBox, QSpinBox, QDoubleSpinBox {{
        background-color: {p.bg_input};
        color: {p.text_primary};
        border: 1px solid {p.border_subtle};
        border-radius: {RADIUS_MD}px;
        padding: 4px 8px;
        min-height: 22px;
    }}
    QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
        border-color: {p.border_strong};
    }}
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {p.accent};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    QComboBox::down-arrow {{
        /* Use a simple triangle via CSS since Qt doesn't ship a nice SVG arrow */
        width: 8px;
        height: 8px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {p.bg_panel};
        color: {p.text_primary};
        border: 1px solid {p.border_strong};
        border-radius: {RADIUS_MD}px;
        selection-background-color: {p.accent};
        selection-color: {p.text_on_accent};
        padding: 4px;
        outline: none;
    }}

    /* ---------- Checkbox / radio ---------- */
    QCheckBox, QRadioButton {{
        background: transparent;
        color: {p.text_primary};
        spacing: {SPACE_2}px;
    }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {p.border_strong};
        background-color: {p.bg_input};
    }}
    QCheckBox::indicator {{
        border-radius: 3px;
    }}
    QRadioButton::indicator {{
        border-radius: 8px;
    }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
        border-color: {p.accent};
    }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background-color: {p.accent};
        border-color: {p.accent};
    }}

    /* ---------- List / tree ---------- */
    QListWidget, QTreeWidget, QTableWidget {{
        background-color: {p.bg_panel};
        color: {p.text_primary};
        border: 1px solid {p.border_subtle};
        border-radius: {RADIUS_MD}px;
        padding: 4px;
        outline: none;
        alternate-background-color: {p.bg_panel};
    }}
    QListWidget::item, QTreeWidget::item, QTableWidget::item {{
        padding: 5px 8px;
        border-radius: {RADIUS_SM}px;
        margin: 1px 0;
    }}
    QListWidget::item:hover, QTreeWidget::item:hover {{
        background-color: {p.bg_hover};
    }}
    QListWidget::item:selected, QTreeWidget::item:selected {{
        background-color: {p.bg_selected};
        color: {p.text_primary};
    }}

    /* ---------- Scrollbars ---------- */
    QScrollBar:vertical {{
        background: transparent;
        width: 12px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {p.border_strong};
        min-height: 24px;
        border-radius: 6px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {p.text_secondary};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: none;
        height: 0;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 12px;
        margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: {p.border_strong};
        min-width: 24px;
        border-radius: 6px;
        margin: 2px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {p.text_secondary};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
        background: none;
        width: 0;
    }}

    /* ---------- Splitter ---------- */
    QSplitter::handle {{
        background-color: {p.border_subtle};
    }}
    QSplitter::handle:horizontal {{
        width: 1px;
    }}
    QSplitter::handle:vertical {{
        height: 1px;
    }}
    QSplitter::handle:hover {{
        background-color: {p.accent};
    }}

    /* ---------- Menu ---------- */
    QMenu {{
        background-color: {p.bg_panel};
        color: {p.text_primary};
        border: 1px solid {p.border_strong};
        border-radius: {RADIUS_MD}px;
        padding: 4px;
    }}
    QMenu::item {{
        padding: 6px 18px;
        border-radius: {RADIUS_SM}px;
        margin: 1px;
    }}
    QMenu::item:selected {{
        background-color: {p.accent};
        color: {p.text_on_accent};
    }}
    QMenu::item:disabled {{
        color: {p.text_disabled};
    }}
    QMenu::separator {{
        height: 1px;
        background-color: {p.border_subtle};
        margin: 4px 10px;
    }}

    /* ---------- Tooltips ---------- */
    QToolTip {{
        background-color: {p.bg_pressed};
        color: {p.text_primary};
        border: 1px solid {p.border_strong};
        border-radius: {RADIUS_SM}px;
        padding: 6px 10px;
    }}

    /* ---------- Progress dialog ---------- */
    QProgressBar {{
        background-color: {p.bg_input};
        border: 1px solid {p.border_subtle};
        border-radius: {RADIUS_MD}px;
        text-align: center;
        color: {p.text_primary};
        height: 20px;
    }}
    QProgressBar::chunk {{
        background-color: {p.accent};
        border-radius: {RADIUS_MD}px;
    }}
    """


def apply_theme(app, theme: Literal["dark", "light"] = "dark") -> Palette:
    """Apply the given theme to the QApplication instance. Returns the
    Palette object so callers can pull colors for custom-painted widgets
    (e.g. the canvas background)."""
    palette = DARK if theme == "dark" else LIGHT
    app.setStyleSheet(build_stylesheet(palette))
    return palette
