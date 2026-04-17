from __future__ import annotations

from constants import ASSASSIN, BLUE, NEUTRAL, RED
from ai.agent import SpymasterBoardView


def spymaster_board_view(board, team: str) -> SpymasterBoardView:
    if board is None:
        return SpymasterBoardView([], [], [], [])
    other = BLUE if team == RED else RED
    team_w: list[str] = []
    opponent_w: list[str] = []
    neutral_w: list[str] = []
    assassin_w: list[str] = []
    for i in range(board.SIZE):
        if board.is_revealed(i):
            continue
        w = board.word_at(i)
        k = board.key_at(i)
        if k == team:
            team_w.append(w)
        elif k == ASSASSIN:
            assassin_w.append(w)
        elif k == NEUTRAL:
            neutral_w.append(w)
        elif k == other:
            opponent_w.append(w)
    return SpymasterBoardView(team_w, opponent_w, neutral_w, assassin_w)


def unrevealed_words(board) -> list[str]:
    if board is None:
        return []
    return [board.word_at(i) for i in range(board.SIZE) if not board.is_revealed(i)]


def word_index(board, word_upper: str) -> int | None:
    wu = word_upper.strip().upper()
    for i in range(board.SIZE):
        if board.is_revealed(i):
            continue
        if board.word_at(i).upper() == wu:
            return i
    return None
