from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import pyqtSignal

from ui.card_panel import CardPanel
from ui.styles import TEXT_MUTED

BTN_HEIGHT = 40
CLUE_NUM_MIN, CLUE_NUM_MAX = 1, 9


class CluePanel(CardPanel):
    submit_clue = pyqtSignal(str, int, object)
    ai_suggest_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(320)
        layout = self.layout()
        layout.setColumnStretch(1, 1)

        layout.addWidget(QLabel("Clue word:"), 0, 0)
        self._clue_edit = QLineEdit()
        self._clue_edit.setPlaceholderText("One word only")
        self._clue_edit.setMaxLength(50)
        self._clue_edit.setMinimumHeight(BTN_HEIGHT)
        layout.addWidget(self._clue_edit, 0, 1)

        layout.addWidget(QLabel("Number:"), 1, 0)
        self._number_spin = QSpinBox()
        self._number_spin.setRange(CLUE_NUM_MIN, CLUE_NUM_MAX)
        self._number_spin.setValue(1)
        self._number_spin.setMinimumHeight(BTN_HEIGHT)
        layout.addWidget(self._number_spin, 1, 1)

        layout.addWidget(QLabel("Target words:"), 2, 0)
        self._intended_label = QLabel("")
        self._intended_label.setWordWrap(True)
        self._intended_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        self._intended_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._intended_label.setMinimumWidth(0)
        layout.addWidget(self._intended_label, 2, 1)

        self._training_frame = QFrame()
        training_outer = QVBoxLayout(self._training_frame)
        training_outer.setContentsMargins(0, 0, 0, 0)
        training_outer.addWidget(
            QLabel("Training: check team words this clue is meant to target (optional):")
        )
        self._training_checks_host = QWidget()
        self._training_checks_layout = QVBoxLayout(self._training_checks_host)
        self._training_checks_layout.setContentsMargins(0, 0, 0, 0)
        training_outer.addWidget(self._training_checks_host)
        layout.addWidget(self._training_frame, 3, 0, 1, 2)
        self._training_frame.setVisible(False)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        submit_btn = QPushButton("Submit Clue")
        submit_btn.setObjectName("primary")
        submit_btn.setMinimumHeight(BTN_HEIGHT)
        submit_btn.clicked.connect(self._on_submit)
        btn_row.addWidget(submit_btn)
        ai_btn = QPushButton("AI Suggest")
        ai_btn.setMinimumHeight(BTN_HEIGHT)
        ai_btn.clicked.connect(self.ai_suggest_requested.emit)
        btn_row.addWidget(ai_btn)
        layout.addLayout(btn_row, 4, 0, 1, 2)

    def _on_submit(self):
        targets = self._collect_training_targets()
        self.submit_clue.emit(
            self._clue_edit.text().strip(),
            self._number_spin.value(),
            targets,
        )

    def _collect_training_targets(self) -> list[str]:
        if not self._training_frame.isVisible():
            return []
        out = []
        for i in range(self._training_checks_layout.count()):
            item = self._training_checks_layout.itemAt(i)
            w = item.widget() if item else None
            if isinstance(w, QCheckBox) and w.isChecked():
                out.append(w.text())
        return out

    def set_training_target_picker(self, active: bool, team_words: list[str]):
        while self._training_checks_layout.count():
            item = self._training_checks_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        show = active and len(team_words) > 0
        self._training_frame.setVisible(show)
        if not show:
            return
        for w in team_words:
            self._training_checks_layout.addWidget(QCheckBox(w))

    def clear_inputs(self):
        self._clue_edit.clear()
        self._number_spin.setValue(1)
        self._set_intended("")
        self._uncheck_training_targets()

    def _uncheck_training_targets(self):
        for i in range(self._training_checks_layout.count()):
            item = self._training_checks_layout.itemAt(i)
            w = item.widget() if item else None
            if isinstance(w, QCheckBox):
                w.setChecked(False)

    def set_no_suggestion(self):
        self._clue_edit.clear()
        self._number_spin.setValue(1)
        self._set_intended("No good clue found")

    def set_clue(self, word, number, intended_words=None):
        self._clue_edit.setText(word.strip())
        num = max(CLUE_NUM_MIN, min(CLUE_NUM_MAX, number))
        self._number_spin.setValue(num)
        self._set_intended(", ".join(intended_words) if intended_words else "")

    def set_intended_words(self, words):
        self._set_intended(", ".join(words) if words else "")

    def _set_intended(self, text):
        self._intended_label.setText(text)


class GuessPanel(CardPanel):
    end_turn = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = self.layout()

        self._clue_display = QLabel("—")
        self._clue_display.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(QLabel("Current clue:"), 0, 0)
        layout.addWidget(self._clue_display, 0, 1)

        self._guesses_label = QLabel("0")
        layout.addWidget(QLabel("Guesses remaining:"), 1, 0)
        layout.addWidget(self._guesses_label, 1, 1)

        end_btn = QPushButton("End Turn")
        end_btn.setMinimumHeight(BTN_HEIGHT)
        end_btn.clicked.connect(self.end_turn.emit)
        layout.addWidget(end_btn, 2, 0, 1, 2)

    def update_display(self, clue_word, guesses_left):
        self._clue_display.setText(clue_word or "—")
        self._guesses_label.setText(str(guesses_left))


class ClueGuessStack(QWidget):
    submit_clue = pyqtSignal(str, int, object)
    end_turn = pyqtSignal()
    ai_suggest_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stack = QStackedWidget(self)
        self._clue_panel = CluePanel()
        self._clue_panel.submit_clue.connect(self.submit_clue.emit)
        self._clue_panel.ai_suggest_requested.connect(self.ai_suggest_requested.emit)
        self._guess_panel = GuessPanel()
        self._guess_panel.end_turn.connect(self.end_turn.emit)
        self._stack.addWidget(self._clue_panel)
        self._stack.addWidget(self._guess_panel)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._stack)

    def show_spymaster(self):
        self._stack.setCurrentWidget(self._clue_panel)

    def show_operative(self):
        self._stack.setCurrentWidget(self._guess_panel)

    def update_guess_panel(self, clue_word, guesses_left):
        self._guess_panel.update_display(clue_word, guesses_left)

    def clear_clue_inputs(self):
        self._clue_panel.clear_inputs()

    def set_clue(self, word, number, intended_words=None):
        self._clue_panel.set_clue(word, number, intended_words)

    def set_no_suggestion(self):
        self._clue_panel.set_no_suggestion()

    def set_intended_words(self, words):
        self._clue_panel.set_intended_words(words)

    def configure_training_spymaster(self, training: bool, team_words: list[str]):
        self._clue_panel.set_training_target_picker(training, team_words)
