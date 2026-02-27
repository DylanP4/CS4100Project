from PyQt6.QtWidgets import QMainWindow, QMessageBox, QVBoxLayout, QWidget

from constants import PHASE_SPYMASTER, RED
from game_engine import GameEngine
from ui.components import GameView, Toolbar
from ui.styles import global_stylesheet


def _team_label(team: str) -> str:
    return (team or "").capitalize()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Codenames")
        self.setMinimumSize(700, 620)
        self.resize(800, 680)
        self.setStyleSheet(global_stylesheet())

        self._engine = GameEngine()
        self._engine.set_on_state_change(self._refresh_ui)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(20)
        layout.setContentsMargins(24, 24, 24, 24)

        self._toolbar = Toolbar()
        self._toolbar.start_clicked.connect(self._on_start_game)
        layout.addWidget(self._toolbar)

        self._game_view = GameView()
        self._toolbar.spymaster_toggled.connect(self._game_view.board.set_spymaster_view)
        self._game_view.board.card_clicked.connect(self._on_card_clicked)
        self._game_view.clue_stack.submit_clue.connect(self._on_submit_clue)
        self._game_view.clue_stack.end_turn.connect(self._on_end_turn)
        self._game_view.clue_stack.ai_suggest_requested.connect(self._on_ai_suggest)
        layout.addWidget(self._game_view)

        self._refresh_ui()

    def _on_start_game(self):
        self._engine.start_game()
        self._game_view.clue_stack.clear_clue_inputs()
        self._refresh_ui()

    def _show_message(self, title: str, text: str):
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setIcon(QMessageBox.Icon.NoIcon)
        box.setText(text)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    def _on_ai_suggest(self):
        self._show_message("AI Suggest", "AI Spymaster will be added later (Q-learning).")

    def _on_submit_clue(self, word, number):
        ok, msg = self._engine.submit_clue(word, number)
        if not ok and msg:
            self._show_message("Invalid clue", msg)
            return
        self._refresh_ui()

    def _on_end_turn(self):
        self._engine.end_turn_early()
        self._refresh_ui()

    def _on_card_clicked(self, index):
        ok, outcome, msg = self._engine.guess(index)
        if not ok and msg:
            self._show_message("Invalid move", msg)
            self._refresh_ui()
            return
        if outcome == "assassin":
            QMessageBox.critical(
                self,
                "Game Over",
                "Assassin! " + _team_label(self._engine.loser) + " team loses.",
            )
        elif self._engine.winner is not None and outcome != "assassin":
            QMessageBox.information(
                self,
                "Game Over",
                _team_label(self._engine.winner) + " team wins!",
            )
        self._refresh_ui()

    def _refresh_ui(self):
        engine = self._engine
        view = self._game_view
        status = view.status
        board_widget = view.board
        clue_stack = view.clue_stack

        if engine.board is None:
            status.update_state(RED, PHASE_SPYMASTER, 0, 0)
            board_widget.load_board([], [], [])
            board_widget.set_operative_guessing(False)
            clue_stack.show_spymaster()
            self._toolbar.set_start_enabled(True)
            return

        board = engine.board
        board_widget.load_board(board.words(), board.all_key_assignments(), board.revealed_mask())
        status.update_state(
            engine.current_team,
            engine.phase,
            engine.red_remaining,
            engine.blue_remaining,
        )

        if engine.game_over:
            board_widget.set_operative_guessing(False)
            clue_stack.show_spymaster()
            self._toolbar.set_start_enabled(True)
            return

        if engine.phase == PHASE_SPYMASTER:
            clue_stack.show_spymaster()
            board_widget.set_operative_guessing(False)
        else:
            clue_stack.show_operative()
            clue_stack.update_guess_panel(engine.clue_word, engine.guesses_left)
            board_widget.set_operative_guessing(engine.guesses_left > 0)

        self._toolbar.set_start_enabled(False)
