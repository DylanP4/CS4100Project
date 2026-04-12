import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    use_ai_agent_file = "ai-training" in sys.argv or "llm-training" in sys.argv
    training_mode = "training" in sys.argv or use_ai_agent_file
    app = QApplication(sys.argv)
    app.setApplicationName("Codenames")
    win = MainWindow(training_mode=training_mode, use_ai_agent_file=use_ai_agent_file)
    win.show()
    if training_mode:
        ckpt = "data/ai_agent.pkl" if use_ai_agent_file else "data/agent.pkl"
        print("[AI] Training mode")
        print("    • Use AI Suggest or type a clue, then Submit — play guesses as operative.")
        print(f"    • Each finished clue turn runs one Q-learning update and saves {ckpt}.")
        print("    • Optional: check team-word boxes before submit to tag intended targets.")
        print("    • State uses team / opponent / neutral / assassin words (see SpymasterBoardView).")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
