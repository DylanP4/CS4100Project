from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ui.board_widget import BoardWidget
from ui.clue_panel import ClueGuessStack
from ui.status_panel import StatusPanel


class GameView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        self._status = StatusPanel()
        layout.addWidget(self._status)

        self._board = BoardWidget()
        layout.addWidget(self._board, 1)

        self._clue_stack = ClueGuessStack()
        layout.addWidget(self._clue_stack)

    @property
    def status(self) -> StatusPanel:
        return self._status

    @property
    def board(self) -> BoardWidget:
        return self._board

    @property
    def clue_stack(self) -> ClueGuessStack:
        return self._clue_stack
