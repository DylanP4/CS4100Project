from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMainWindow, QMessageBox, QVBoxLayout, QWidget

from ai.agent import SpymasterAgent
from ai.embeddings import load_model
from constants import PHASE_OPERATIVE, PHASE_SPYMASTER, RED
from game_engine import GameEngine
from ui.components import GameView, Toolbar
from ui.styles import global_stylesheet


def _team_label(team: str) -> str:
    return (team or "").capitalize()


class _ModelLoader(QThread):
    """Loads the GloVe model in a background thread so the UI stays responsive."""
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def run(self):
        try:
            load_model()
            self.finished.emit()
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self, training_mode: bool = False):
        super().__init__()
        self._training_mode = training_mode
        title = "Codenames — TRAINING MODE" if training_mode else "Codenames"
        self.setWindowTitle(title)
        self.setMinimumSize(700, 620)
        self.resize(800, 680)
        self.setStyleSheet(global_stylesheet())

        self._engine = GameEngine()
        self._engine.set_on_state_change(self._refresh_ui)

        self._ai = SpymasterAgent()
        self._ai_ready = False  # True once GloVe has finished loading

        # Tracks whether the AI suggested the clue for the current operative turn.
        # Reset to False at the start of every new spymaster phase.
        self._ai_active_turn: bool = False
        # The team the AI is playing for this turn (needed after current_team switches).
        self._ai_team: str = RED
        # Outcomes collected during the operative phase (e.g. ["correct", "neutral"]).
        self._turn_outcomes: list[str] = []

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(20)
        layout.setContentsMargins(24, 24, 24, 24)

        self._toolbar = Toolbar()
        self._toolbar.start_clicked.connect(self._on_start_game)
        self._toolbar.end_game_clicked.connect(self._on_end_game)
        layout.addWidget(self._toolbar)

        self._game_view = GameView()
        self._toolbar.spymaster_toggled.connect(self._game_view.board.set_spymaster_view)
        self._game_view.board.card_clicked.connect(self._on_card_clicked)
        self._game_view.clue_stack.submit_clue.connect(self._on_submit_clue)
        self._game_view.clue_stack.end_turn.connect(self._on_end_turn)
        self._game_view.clue_stack.ai_suggest_requested.connect(self._on_ai_suggest)
        layout.addWidget(self._game_view)

        self._refresh_ui()

        # Start loading GloVe in the background immediately so it's ready
        # by the time the user clicks AI Suggest.
        self._loader = _ModelLoader()
        self._loader.finished.connect(self._on_model_ready)
        self._loader.failed.connect(self._on_model_failed)
        self._loader.start()

    # ── Game flow ──────────────────────────────────────────────────────────────

    def _on_start_game(self):
        self._engine.start_game()
        self._game_view.clue_stack.clear_clue_inputs()
        self._ai_active_turn = False
        self._turn_outcomes = []
        self._refresh_ui()

    def _on_end_game(self):
        if self._ai_active_turn:
            self._record_ai_outcome(done=True)
            self._ai_active_turn = False
        self._engine.reset()
        self._game_view.clue_stack.clear_clue_inputs()
        self._turn_outcomes = []
        self._refresh_ui()

    def _on_submit_clue(self, word, number):
        ok, msg = self._engine.submit_clue(word, number)
        if not ok and msg:
            self._show_message("Invalid clue", msg)
            return
        self._turn_outcomes = []
        # In training mode, record every human clue as a training transition.
        if self._training_mode and not self._ai_active_turn:
            team_words = self._unrevealed_words_for(self._engine.current_team)
            all_board_words = self._unrevealed_board_words()
            recorded = self._ai.record_human_clue(team_words, all_board_words, word, number)
            if recorded:
                self._ai_active_turn = True
                self._ai_team = self._engine.current_team
        self._refresh_ui()

    def _on_end_turn(self):
        # Record outcome before end_turn_early() switches current_team.
        if self._ai_active_turn:
            self._record_ai_outcome(done=False)
            self._ai_active_turn = False
        self._engine.end_turn_early()
        self._refresh_ui()

    def _on_card_clicked(self, index):
        prev_phase = self._engine.phase
        ok, outcome, msg = self._engine.guess(index)
        if not ok and msg:
            self._show_message("Invalid move", msg)
            self._refresh_ui()
            return

        if self._ai_active_turn:
            self._turn_outcomes.append(outcome)

        if outcome == "assassin":
            if self._ai_active_turn:
                self._record_ai_outcome(done=True)
                self._ai_active_turn = False
            QMessageBox.critical(
                self,
                "Game Over",
                "Assassin! " + _team_label(self._engine.loser) + " team loses.",
            )
        elif self._engine.winner is not None:
            if self._ai_active_turn:
                self._record_ai_outcome(done=True)
                self._ai_active_turn = False
            QMessageBox.information(
                self,
                "Game Over",
                _team_label(self._engine.winner) + " team wins!",
            )
        elif prev_phase == PHASE_OPERATIVE and self._engine.phase == PHASE_SPYMASTER:
            # Turn ended naturally (neutral/opponent hit, or out of guesses).
            if self._ai_active_turn:
                self._record_ai_outcome(done=False)
                self._ai_active_turn = False

        self._refresh_ui()

    # ── AI ─────────────────────────────────────────────────────────────────────

    def _on_model_ready(self):
        self._ai_ready = True
        print("[AI] Word vectors ready.")

    def _on_model_failed(self, error: str):
        self._show_message("AI Error", f"Failed to load word vectors:\n{error}")

    def _on_ai_suggest(self):
        if not self._ai_ready:
            self._show_message("AI Suggest", "Still loading word vectors, please wait a moment.")
            return
        if self._engine.board is None or self._engine.game_over:
            return
        if self._engine.phase != PHASE_SPYMASTER:
            return

        team_words = self._unrevealed_words_for(self._engine.current_team)
        all_board_words = self._unrevealed_board_words()

        suggestion = self._ai.suggest(team_words, all_board_words)
        if suggestion is None:
            self._show_message("AI Suggest", "No good clue found for this board.")
            return

        self._ai_active_turn = True
        self._ai_team = self._engine.current_team
        self._turn_outcomes = []

        self._game_view.clue_stack.set_clue(
            suggestion["clue"],
            suggestion["number"],
            suggestion["covered"],
        )

    def _record_ai_outcome(self, done: bool):
        """Store transition and trigger a training step."""
        if done:
            next_team_words = []
            next_board_words = []
        else:
            next_team_words = self._unrevealed_words_for(self._ai_team)
            next_board_words = self._unrevealed_board_words()

        self._ai.record_outcome(
            self._turn_outcomes,
            next_team_words,
            next_board_words,
            done,
        )

    # ── Board helpers ──────────────────────────────────────────────────────────

    def _unrevealed_words_for(self, team: str) -> list[str]:
        """Unrevealed words belonging to the given team."""
        board = self._engine.board
        return [
            board.word_at(i)
            for i in range(board.SIZE)
            if board.key_at(i) == team and not board.is_revealed(i)
        ]

    def _unrevealed_board_words(self) -> list[str]:
        """All unrevealed words currently on the board."""
        board = self._engine.board
        return [
            board.word_at(i)
            for i in range(board.SIZE)
            if not board.is_revealed(i)
        ]

    # ── UI helpers ─────────────────────────────────────────────────────────────

    def _show_message(self, title: str, text: str):
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setIcon(QMessageBox.Icon.NoIcon)
        box.setText(text)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

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
