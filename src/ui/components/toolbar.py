from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QPushButton, QWidget


class Toolbar(QWidget):
    start_clicked = pyqtSignal()
    end_game_clicked = pyqtSignal()
    spymaster_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        self._start_btn = QPushButton("Start Game")
        self._start_btn.setObjectName("primary")
        self._start_btn.setMinimumHeight(40)
        self._start_btn.clicked.connect(self.start_clicked.emit)
        layout.addWidget(self._start_btn)

        self._end_btn = QPushButton("End Game")
        self._end_btn.setMinimumHeight(40)
        self._end_btn.setEnabled(False)
        self._end_btn.clicked.connect(self.end_game_clicked.emit)
        layout.addWidget(self._end_btn)

        layout.addStretch()

        self._spymaster_check = QCheckBox("Spymaster view (show card colors)")
        self._spymaster_check.toggled.connect(self.spymaster_toggled.emit)
        layout.addWidget(self._spymaster_check)

    def set_spymaster_checked(self, on: bool) -> None:
        """Sync the checkbox without emitting toggled (caller updates the board)."""
        self._spymaster_check.blockSignals(True)
        self._spymaster_check.setChecked(on)
        self._spymaster_check.blockSignals(False)

    def set_start_enabled(self, enabled: bool) -> None:
        self._start_btn.setEnabled(enabled)
        self._end_btn.setEnabled(not enabled)
