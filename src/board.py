from typing import List

from constants import ASSASSIN, BOARD_SIZE
from dataset_loader import BLUE, NEUTRAL, RED


class Board:
    SIZE = BOARD_SIZE

    def __init__(self, words, key_assignment):
        if len(words) != self.SIZE or len(key_assignment) != self.SIZE:
            raise ValueError("Words and key_assignment must have length " + str(self.SIZE))
        self._words = list(words)
        self._key = list(key_assignment)
        self._revealed = [False] * self.SIZE

    def word_at(self, index):
        return self._words[index]

    def key_at(self, index):
        return self._key[index]

    def is_revealed(self, index):
        return self._revealed[index]

    def reveal(self, index):
        if index < 0 or index >= self.SIZE:
            raise IndexError(index)
        self._revealed[index] = True
        return self._key[index]

    def words(self):
        return list(self._words)

    def key_assignment(self, index):
        return self._key[index]

    def all_key_assignments(self):
        return list(self._key)

    def revealed_mask(self):
        return list(self._revealed)

    def remaining_count(self, team):
        return sum(
            1 for i in range(self.SIZE) if self._key[i] == team and not self._revealed[i]
        )

    def assassin_revealed(self):
        return any(
            self._key[i] == ASSASSIN and self._revealed[i] for i in range(self.SIZE)
        )

    def team_wins(self, team):
        return self.remaining_count(team) == 0
