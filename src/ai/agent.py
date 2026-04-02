"""
SpymasterAgent — ties embeddings and Q-learning together.

This is the only class main_window.py needs to interact with.

Typical call sequence per AI turn:
    suggestion = agent.suggest(team_words, all_board_words)
    # → {"clue": "ocean", "number": 2, "covered": ["SEA", "WAVE"]}

After the human operative finishes guessing:
    agent.record_outcome(outcome, next_team_words, next_all_board_words, done)
    # outcome: list of guess results, e.g. ["correct", "neutral"]
"""

from pathlib import Path

import numpy as np

from ai.embeddings import candidate_clues, get_embedding, load_model
from ai.Q_learning import (
    REWARD_ASSASSIN,
    REWARD_CORRECT,
    REWARD_NEUTRAL,
    REWARD_OPPONENT,
    QLearningAgent,
)

# Where the trained model is saved between sessions.
DEFAULT_SAVE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "agent.pkl"

# Outcome strings that come from game_engine.guess()
_OUTCOME_REWARD = {
    "correct": REWARD_CORRECT,
    "neutral": REWARD_NEUTRAL,
    "opponent": REWARD_OPPONENT,
    "assassin": REWARD_ASSASSIN,
}


class SpymasterAgent:
    """
    High-level AI spymaster.

    Responsibilities
    ----------------
    1. Convert board words → embedding vectors (via embeddings.py).
    2. Generate candidate clues (via candidate_clues()).
    3. Pick the best clue via Q-learning (via QLearningAgent.select_action()).
    4. After the human operative guesses, record the outcome and trigger a
       training step so the agent learns from the interaction.
    """

    def __init__(self, save_path: Path = DEFAULT_SAVE_PATH):
        self._agent = QLearningAgent()
        self._save_path = save_path

        # Persisted state across calls — needed to store the transition after
        # the human finishes guessing.
        self._last_state: np.ndarray | None = None
        self._last_action_vec: np.ndarray | None = None
        self._last_covered: list[str] = []

        if save_path.exists():
            self._agent.load(save_path)

    # ── Public API ─────────────────────────────────────────────────────────────

    def suggest(
        self,
        team_words: list[str],
        all_board_words: list[str],
    ) -> dict | None:
        """
        Generate the AI's clue for this turn.

        Parameters
        ----------
        team_words      : unrevealed words belonging to the AI's team.
        all_board_words : all words currently on the board (revealed + unrevealed).

        Returns
        -------
        {"clue": str, "number": int, "covered": list[str]}
        or None if no candidates could be generated.
        """
        model = load_model()

        # Build vectors for state representation.
        team_vecs, avoid_vecs = self._split_vecs(team_words, all_board_words, model)
        state = self._agent.build_state(team_vecs, avoid_vecs)

        # Get candidate clues and attach their embedding vectors.
        raw_candidates = candidate_clues(team_words, all_board_words)
        if not raw_candidates:
            return None

        candidates = self._attach_vecs(raw_candidates, model)

        # Ask the Q-agent to pick one.
        result = self._agent.select_action(state, candidates)
        if result is None:
            return None

        clue_word, number, action_vec = result

        # Find the covered words for the chosen clue.
        covered = next(
            (c for w, _, c, _ in candidates if w == clue_word),
            [],
        )

        # Persist state/action so record_outcome() can complete the transition.
        self._last_state = state
        self._last_action_vec = action_vec
        self._last_covered = covered

        return {"clue": clue_word, "number": number, "covered": covered}

    def record_outcome(
        self,
        outcomes: list[str],
        next_team_words: list[str],
        next_all_board_words: list[str],
        done: bool,
    ):
        """
        Call this after the human operative finishes guessing to give the agent
        feedback and trigger a training step.

        Parameters
        ----------
        outcomes            : ordered list of guess results for this turn,
                              e.g. ["correct", "correct", "neutral"].
                              Each string must be one of: correct, neutral,
                              opponent, assassin.
        next_team_words     : unrevealed team words after the turn.
        next_all_board_words: all board words after the turn.
        done                : True if the game ended this turn.
        """
        if self._last_state is None or self._last_action_vec is None:
            return  # suggest() was never called this turn

        reward = sum(_OUTCOME_REWARD.get(o, 0.0) for o in outcomes)

        model = load_model()
        next_team_vecs, next_avoid_vecs = self._split_vecs(
            next_team_words, next_all_board_words, model
        )
        next_state = self._agent.build_state(next_team_vecs, next_avoid_vecs)

        # Compute max Q over next state's candidates for the Bellman target.
        if done:
            next_best_q = 0.0
        else:
            next_raw = candidate_clues(next_team_words, next_all_board_words)
            next_candidates = self._attach_vecs(next_raw, model)
            next_best_q = self._agent.best_q(next_state, next_candidates)

        self._agent.store_transition(
            self._last_state,
            self._last_action_vec,
            reward,
            next_best_q,
            done,
        )

        loss = self._agent.train_step()
        if loss is not None:
            print(f"[AI] train loss={loss:.4f}  ε={self._agent.epsilon:.3f}")

        self._agent.decay_epsilon()
        self._agent.save(self._save_path)

        # Clear persisted state.
        self._last_state = None
        self._last_action_vec = None
        self._last_covered = []

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _split_vecs(
        team_words: list[str],
        all_board_words: list[str],
        model,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Split board words into team vectors and avoid vectors."""
        team_upper = {w.upper() for w in team_words}
        team_vecs = [
            model[w.lower()]
            for w in team_words
            if w.lower() in model
        ]
        avoid_vecs = [
            model[w.lower()]
            for w in all_board_words
            if w.upper() not in team_upper and w.lower() in model
        ]
        return team_vecs, avoid_vecs

    @staticmethod
    def _attach_vecs(
        raw_candidates: list[tuple[str, float, list[str]]],
        model,
    ) -> list[tuple[str, float, list[str], np.ndarray]]:
        """Attach the clue word's embedding vector to each candidate tuple."""
        result = []
        for clue_word, score, covered in raw_candidates:
            key = clue_word.lower()
            if key in model:
                result.append((clue_word, score, covered, model[key]))
        return result
