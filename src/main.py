import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    training_mode = "training" in sys.argv
    app = QApplication(sys.argv)
    app.setApplicationName("Codenames")
    win = MainWindow(training_mode=training_mode)
    win.show()
    if training_mode:
        print("[AI] Training mode active — all spymaster clues will be recorded.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
