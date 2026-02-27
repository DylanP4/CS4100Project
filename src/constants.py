from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
WORDS_PATH = DATA_DIR / "words.txt"

BOARD_SIZE = 25
GRID_COLS = 5
GRID_ROWS = 5

RED = "red"
BLUE = "blue"
NEUTRAL = "neutral"
ASSASSIN = "assassin"

STARTING_TEAM_COUNT = 9
OTHER_TEAM_COUNT = 8
NEUTRAL_COUNT = 7
ASSASSIN_COUNT = 1

PHASE_SPYMASTER = "spymaster"
PHASE_OPERATIVE = "operative"
