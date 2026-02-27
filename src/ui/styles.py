BG_APP = "#ffffff"
BG_CARD = "#ffffff"
BG_INPUT = "#ffffff"
BORDER = "#e4e4e7"
BORDER_FOCUS = "#a1a1aa"
TEXT = "#18181b"
TEXT_MUTED = "#71717a"
TEXT_HINT = "#a1a1aa"

RED = "#b91c1c"
RED_SOFT = "#fef2f2"
BLUE = "#1d4ed8"
BLUE_SOFT = "#eff6ff"
NEUTRAL_CARD = "#a16207"
NEUTRAL_SOFT = "#fffbeb"
ASSASSIN_CARD = "#27272a"
ASSASSIN_SOFT = "#27272a"

CARD_FACE = "#fafafa"
CARD_FACE_BORDER = "#d4d4d8"

BTN_PRIMARY_BG = "#2563eb"
BTN_PRIMARY_FG = "#ffffff"
BTN_SECONDARY_BG = "#ffffff"
BTN_SECONDARY_FG = "#18181b"
BTN_RADIUS = "8px"
INPUT_RADIUS = "8px"
CARD_RADIUS = "12px"
PANEL_RADIUS = "12px"

FONT_FAMILY = "Helvetica Neue, Helvetica, Arial, sans-serif"
FONT_SIZE = "13px"
FONT_SIZE_SM = "12px"
FONT_SIZE_LG = "15px"


def global_stylesheet() -> str:
    return f"""
        QWidget {{
            background-color: {BG_APP};
            color: {TEXT};
            font-family: {FONT_FAMILY};
            font-size: {FONT_SIZE};
        }}
        QMainWindow {{
            background-color: {BG_APP};
        }}
        QPushButton {{
            background-color: {BTN_SECONDARY_BG};
            color: {BTN_SECONDARY_FG};
            border: 1px solid {BORDER};
            border-radius: {BTN_RADIUS};
            padding: 10px 18px;
            font-weight: 500;
            min-height: 20px;
        }}
        QPushButton:hover {{
            background-color: #f4f4f5;
            border-color: {BORDER_FOCUS};
        }}
        QPushButton:pressed {{
            background-color: #e4e4e7;
        }}
        QPushButton:disabled {{
            background-color: #fafafa;
            color: {TEXT_MUTED};
            border-color: {BORDER};
        }}
        QPushButton#primary {{
            background-color: {BTN_PRIMARY_BG};
            color: {BTN_PRIMARY_FG};
            border: none;
        }}
        QPushButton#primary:hover {{
            background-color: #1d4ed8;
        }}
        QPushButton#primary:pressed {{
            background-color: #1e40af;
        }}
        QPushButton#primary:disabled {{
            background-color: #d4d4d8;
            color: #a1a1aa;
        }}
        QLineEdit {{
            background-color: {BG_INPUT};
            border: 1px solid {BORDER};
            border-radius: {INPUT_RADIUS};
            padding: 8px 12px;
            min-height: 20px;
            color: {TEXT};
            selection-background-color: {BLUE};
            selection-color: white;
        }}
        QLineEdit:focus {{
            border-color: {BORDER_FOCUS};
        }}
        QLineEdit::placeholder {{
            color: {TEXT_HINT};
        }}
        QSpinBox {{
            background-color: {BG_INPUT};
            border: 1px solid {BORDER};
            border-radius: {INPUT_RADIUS};
            padding: 8px 12px 8px 14px;
            min-height: 20px;
            color: {TEXT};
            selection-background-color: {BLUE};
            selection-color: white;
        }}
        QSpinBox:focus {{
            border-color: {BORDER_FOCUS};
        }}
        QSpinBox::up-button, QSpinBox::down-button {{
            background-color: #fafafa;
            border: none;
            border-left: 1px solid {BORDER};
            width: 22px;
        }}
        QSpinBox::up-button {{
            subcontrol-origin: border;
            subcontrol-position: top right;
            border-top-right-radius: 7px;
            height: 20px;
        }}
        QSpinBox::down-button {{
            subcontrol-origin: border;
            subcontrol-position: bottom right;
            border-bottom-right-radius: 7px;
            height: 20px;
        }}
        QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
            background-color: #f4f4f5;
        }}
        QSpinBox::up-button:pressed, QSpinBox::down-button:pressed {{
            background-color: #e4e4e7;
        }}
        QSpinBox::up-arrow {{
            width: 0;
            height: 0;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-bottom: 6px solid {TEXT_MUTED};
        }}
        QSpinBox::down-arrow {{
            width: 0;
            height: 0;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid {TEXT_MUTED};
        }}
        QCheckBox {{
            spacing: 8px;
        }}
        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border-radius: 4px;
            border: 2px solid {BORDER};
            background-color: white;
        }}
        QCheckBox::indicator:checked {{
            background-color: #2563eb;
            border-color: #2563eb;
        }}
        QLabel {{
            color: {TEXT};
        }}
        QFrame#cardPanel {{
            background-color: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: {PANEL_RADIUS};
        }}
    """
