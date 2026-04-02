from board import Board
from constants import PHASE_OPERATIVE, PHASE_SPYMASTER
from dataset_loader import ASSASSIN, BLUE, NEUTRAL, RED, create_board_and_key


class GameEngine:
    def __init__(self):
        self._board = None
        self._starting_team = RED
        self._current_team = RED
        self._phase = PHASE_SPYMASTER
        self._clue_word = None
        self._clue_number = 0
        self._guesses_left = 0
        self._game_over = False
        self._winner = None
        self._loser = None
        self._on_state_change = None

    def set_on_state_change(self, callback):
        self._on_state_change = callback

    def _emit(self):
        if self._on_state_change is not None:
            self._on_state_change()

    def start_game(self):
        words, key = create_board_and_key(starting_team=None)
        self._board = Board(words, key)
        self._starting_team = RED if key.count(RED) == 9 else BLUE
        self._current_team = self._starting_team
        self._phase = PHASE_SPYMASTER
        self._clue_word = None
        self._clue_number = 0
        self._guesses_left = 0
        self._game_over = False
        self._winner = None
        self._loser = None
        self._emit()

    @property
    def board(self):
        return self._board

    @property
    def current_team(self):
        return self._current_team

    @property
    def phase(self):
        return self._phase

    @property
    def clue_word(self):
        return self._clue_word

    @property
    def clue_number(self):
        return self._clue_number

    @property
    def guesses_left(self):
        return self._guesses_left

    @property
    def game_over(self):
        return self._game_over

    @property
    def winner(self):
        return self._winner

    @property
    def loser(self):
        return self._loser

    @property
    def starting_team(self):
        return self._starting_team

    @property
    def board_words(self):
        return self._board.words() if self._board else []

    @property
    def red_remaining(self):
        return self._board.remaining_count(RED) if self._board else 0

    @property
    def blue_remaining(self):
        return self._board.remaining_count(BLUE) if self._board else 0

    def submit_clue(self, word, number):
        if self._board is None or self._game_over:
            return False, "Game not active."
        if self._phase != PHASE_SPYMASTER:
            return False, "Not the clue phase."
        w = (word or "").strip().upper()
        if not w:
            return False, "Clue cannot be empty."
        if " " in w:
            return False, "Clue must be a single word."
        board_upper = [x.upper() for x in self._board.words()]
        if w in board_upper:
            return False, "Clue cannot be a word on the board."
        try:
            n = int(number)
        except (TypeError, ValueError):
            return False, "Number must be an integer."
        if n < 1:
            return False, "Number must be at least 1."
        self._clue_word = w
        self._clue_number = n
        self._guesses_left = n + 1
        self._phase = PHASE_OPERATIVE
        self._emit()
        return True, ""

    def guess(self, index):
        if self._board is None or self._game_over:
            return False, "", "Game not active."
        if self._phase != PHASE_OPERATIVE:
            return False, "", "Not the guessing phase."
        if index < 0 or index >= Board.SIZE:
            return False, "", "Invalid card."
        if self._board.is_revealed(index):
            return False, "", "Card already revealed."
        if self._guesses_left <= 0:
            return False, "", "No guesses left."

        card_type = self._board.reveal(index)
        self._guesses_left -= 1

        if card_type == ASSASSIN:
            self._game_over = True
            self._loser = self._current_team
            self._winner = BLUE if self._current_team == RED else RED
            self._emit()
            return True, "assassin", "Assassin! Your team loses."

        if card_type == self._current_team:
            if self._board.team_wins(self._current_team):
                self._game_over = True
                self._winner = self._current_team
            self._emit()
            return True, "correct", "Correct!"

        if card_type == NEUTRAL:
            self._end_turn()
            return True, "neutral", "Neutral. Turn ends."

        self._end_turn()
        return True, "opponent", "Opponent's word. Turn ends."

    def end_turn_early(self):
        if self._board is None or self._game_over:
            return False, "Game not active."
        if self._phase != PHASE_OPERATIVE:
            return False, "Not the guessing phase."
        self._end_turn()
        return True, ""

    def reset(self):
        """Abandon the current game and return to the pre-game state."""
        self._board = None
        self._game_over = False
        self._winner = None
        self._loser = None
        self._emit()

    def _end_turn(self):
        self._current_team = BLUE if self._current_team == RED else RED
        self._phase = PHASE_SPYMASTER
        self._clue_word = None
        self._clue_number = 0
        self._guesses_left = 0
        self._emit()
