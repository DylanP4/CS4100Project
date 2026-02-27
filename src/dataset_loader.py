import random

from constants import (
    ASSASSIN,
    ASSASSIN_COUNT,
    BOARD_SIZE,
    BLUE,
    NEUTRAL,
    NEUTRAL_COUNT,
    OTHER_TEAM_COUNT,
    RED,
    STARTING_TEAM_COUNT,
    WORDS_PATH,
)


def load_word_list(path=None):
    path = path or WORDS_PATH
    if not path.exists():
        raise FileNotFoundError("Word list not found: " + str(path))
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().upper()
            if w:
                words.append(w)
    if len(words) < BOARD_SIZE:
        raise ValueError(
            "Word list must have at least " + str(BOARD_SIZE) + " words, got " + str(len(words))
        )
    return words


def sample_board_words(word_list, count=BOARD_SIZE):
    n = min(count, len(word_list))
    return random.sample(word_list, n)


def generate_key_assignment(board_size=BOARD_SIZE, starting_team=RED):
    if board_size != BOARD_SIZE:
        raise ValueError("Only " + str(BOARD_SIZE) + "-card boards are supported.")
    other_team = BLUE if starting_team == RED else RED
    assignment = []
    for _ in range(STARTING_TEAM_COUNT):
        assignment.append(starting_team)
    for _ in range(OTHER_TEAM_COUNT):
        assignment.append(other_team)
    for _ in range(NEUTRAL_COUNT):
        assignment.append(NEUTRAL)
    for _ in range(ASSASSIN_COUNT):
        assignment.append(ASSASSIN)
    random.shuffle(assignment)
    return assignment


def create_board_and_key(word_list_path=None, starting_team=None):
    words = load_word_list(word_list_path)
    board_words = sample_board_words(words, BOARD_SIZE)
    if starting_team is None:
        starting_team = random.choice([RED, BLUE])
    key = generate_key_assignment(BOARD_SIZE, starting_team=starting_team)
    return board_words, key
