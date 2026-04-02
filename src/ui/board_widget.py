from PyQt6.QtWidgets import QPushButton, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal

from constants import BOARD_SIZE, GRID_COLS
from dataset_loader import ASSASSIN, BLUE, NEUTRAL, RED
from ui.card_panel import CardPanel
from ui.styles import (
    ASSASSIN_CARD,
    CARD_FACE,
    CARD_FACE_BORDER,
    CARD_RADIUS,
    NEUTRAL_CARD,
    TEXT,
)
from ui.styles import RED as RED_HEX, BLUE as BLUE_HEX


def _card_style(bg: str, fg: str, border: str) -> str:
    return (
        f"background-color: {bg}; color: {fg}; border: 1px solid {border};"
        f" border-radius: {CARD_RADIUS}; font-weight: 500; padding: 4px;"
    )


STYLE_UNREVEALED = _card_style(CARD_FACE, TEXT, CARD_FACE_BORDER)
STYLE_REVEALED_SPY = _card_style("#d4d4d8", "#a1a1aa", "#e4e4e7")
KEY_STYLES = {
    RED: _card_style(RED_HEX, "white", "#991b1b"),
    BLUE: _card_style(BLUE_HEX, "white", "#1e40af"),
    NEUTRAL: _card_style(NEUTRAL_CARD, "white", "#92400e"),
    ASSASSIN: _card_style(ASSASSIN_CARD, "white", "#18181b"),
}


class BoardWidget(CardPanel):
    card_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent, margins=(0, 0, 0, 0), spacing=8)
        layout = self.layout()
        self._buttons = []
        self._spymaster_view = False
        self._words = []
        self._key = []
        self._revealed = []
        for i in range(BOARD_SIZE):
            btn = QPushButton("")
            btn.setMinimumSize(100, 56)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setEnabled(False)
            layout.addWidget(btn, i // GRID_COLS, i % GRID_COLS)
            self._buttons.append(btn)
            btn.clicked.connect(lambda checked=False, idx=i: self.card_clicked.emit(idx))

    def set_spymaster_view(self, on):
        self._spymaster_view = on
        self._refresh_styles()

    def load_board(self, words, key, revealed):
        self._words = list(words)
        self._key = list(key)
        self._revealed = list(revealed)
        for i, btn in enumerate(self._buttons):
            btn.setText(words[i] if i < len(words) else "")
        self._refresh_styles()

    def set_revealed(self, index):
        if 0 <= index < len(self._revealed):
            self._revealed[index] = True
            self._refresh_styles()

    def _style_for(self, index):
        if index >= len(self._key) or index >= len(self._revealed):
            return STYLE_UNREVEALED
        key_type = self._key[index]
        if self._spymaster_view:
            if self._revealed[index]:
                return STYLE_REVEALED_SPY
            return KEY_STYLES.get(key_type, STYLE_UNREVEALED)
        if not self._revealed[index]:
            return STYLE_UNREVEALED
        return KEY_STYLES.get(key_type, STYLE_UNREVEALED)

    def _refresh_styles(self):
        for i, btn in enumerate(self._buttons):
            btn.setStyleSheet(self._style_for(i))

    def set_operative_guessing(self, enabled):
        for i, btn in enumerate(self._buttons):
            btn.setEnabled(
                enabled and i < len(self._revealed) and not self._revealed[i]
            )
