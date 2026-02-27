from PyQt6.QtWidgets import QLabel

from constants import PHASE_SPYMASTER, RED, BLUE
from ui.card_panel import CardPanel
from ui.styles import BLUE as BLUE_HEX, RED as RED_HEX

LABEL_HEADING = "font-size: 15px; font-weight: 600;"
LABEL_COUNT = "font-weight: 600; font-size: 14px;"
TEAM_COLOR = {RED: RED_HEX, BLUE: BLUE_HEX}
TEAM_LABEL = {RED: "Red", BLUE: "Blue"}


class StatusPanel(CardPanel):
    def __init__(self, parent=None):
        super().__init__(parent, margins=(20, 16, 20, 16), spacing=10)
        layout = self.layout()

        self._team_label = QLabel("—")
        self._team_label.setStyleSheet(LABEL_HEADING)
        layout.addWidget(QLabel("Current team:"), 0, 0)
        layout.addWidget(self._team_label, 0, 1)

        self._role_label = QLabel("—")
        layout.addWidget(QLabel("Current role:"), 1, 0)
        layout.addWidget(self._role_label, 1, 1)

        self._red_remaining = QLabel("0")
        self._red_remaining.setStyleSheet(f"color: {RED_HEX}; {LABEL_COUNT}")
        layout.addWidget(QLabel("Red remaining:"), 2, 0)
        layout.addWidget(self._red_remaining, 2, 1)

        self._blue_remaining = QLabel("0")
        self._blue_remaining.setStyleSheet(f"color: {BLUE_HEX}; {LABEL_COUNT}")
        layout.addWidget(QLabel("Blue remaining:"), 3, 0)
        layout.addWidget(self._blue_remaining, 3, 1)

    def update_state(self, current_team, phase, red_remaining, blue_remaining):
        team = current_team if current_team in (RED, BLUE) else RED
        self._team_label.setText(TEAM_LABEL[team])
        self._team_label.setStyleSheet(f"color: {TEAM_COLOR[team]}; {LABEL_HEADING}")

        self._role_label.setText("Spymaster" if phase == PHASE_SPYMASTER else "Operative")
        self._red_remaining.setText(str(red_remaining))
        self._blue_remaining.setText(str(blue_remaining))
