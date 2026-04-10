"""
SpymasterAgent — ties embeddings and Q-learning together.

Typical call sequence per AI turn:
    suggestion = agent.suggest(view, capture_for_learning=True)
    # → {"clue": str, "number": int, "covered": list[str]}
    # Use capture_for_learning=False in play mode so guesses are not trained on.

After the human operative finishes guessing:
    agent.record_outcome(outcomes, next_view, done)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai.embeddings import candidate_clues, load_model
from ai.Q_learning import (
    BATCH_SIZE,
    BUFFER_CAPACITY,
    REWARD_ASSASSIN,
    REWARD_CORRECT,
    REWARD_NEUTRAL,
    REWARD_OPPONENT,
    QLearningAgent,
)

DEFAULT_SAVE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "agent.pkl"
# Optional second Q-learning checkpoint (e.g. alternate data source); never overwrites agent.pkl.
AI_AGENT_SAVE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "ai_agent.pkl"


@dataclass(frozen=True)
class SpymasterBoardView:
    """Unrevealed words on the board from the clue-giver's team perspective."""

    team_words: list[str]
    opponent_words: list[str]
    neutral_words: list[str]
    assassin_words: list[str]


EMPTY_BOARD_VIEW = SpymasterBoardView([], [], [], [])

_OUTCOME_REWARD = {
    "correct": REWARD_CORRECT,
    "neutral": REWARD_NEUTRAL,
    "opponent": REWARD_OPPONENT,
    "assassin": REWARD_ASSASSIN,
}

# Extra reward when operative tags for_clue "current" and picks a spymaster-listed intended word.
INTENT_MATCH_BONUS = 0.08
INTENT_MATCH_BONUS_CAP = 0.32


class SpymasterAgent:
    def __init__(self, save_path: Path = DEFAULT_SAVE_PATH):
        self._agent = QLearningAgent()
        self._save_path = save_path

        self._last_state: np.ndarray | None = None
        self._last_action_vec: np.ndarray | None = None
        self._last_covered: list[str] = []

        if save_path.exists():
            self._agent.load(save_path)

    def suggest(
        self,
        view: SpymasterBoardView,
        *,
        capture_for_learning: bool = True,
    ) -> dict | None:
        model = load_model()

        state = self._state_from_view(view, model, attribution_context=None)

        raw_candidates = candidate_clues(
            view.team_words,
            view.opponent_words,
            view.neutral_words,
            view.assassin_words,
        )
        if not raw_candidates:
            return None

        candidates = self._attach_vecs(raw_candidates, model)

        result = self._agent.select_action(state, candidates)
        if result is None:
            return None

        clue_word, number, action_vec, covered = result

        if capture_for_learning:
            self._last_state = state
            self._last_action_vec = action_vec
            self._last_covered = covered

        return {"clue": clue_word, "number": number, "covered": covered}

    def record_human_clue(
        self,
        view: SpymasterBoardView,
        clue_word: str,
        number: int,
        intended_team_words: list[str] | None = None,
        *,
        attribution_context: np.ndarray | None = None,
        log_training: bool = False,
    ) -> bool:
        model = load_model()
        key = clue_word.lower()
        if key not in model:
            print(f"[AI] '{clue_word}' not in vocabulary — skipping this turn.")
            return False

        team_upper = {t.upper() for t in view.team_words}
        intended = None
        if intended_team_words:
            intended = [w for w in intended_team_words if w.upper() in team_upper]
            if not intended:
                intended = None

        state = self._state_from_view(view, model, attribution_context=attribution_context)
        clue_vec = model[key]
        num_for_action = max(1, min(9, len(intended))) if intended else max(1, min(9, number))
        action_vec = self._agent.build_action_vec(clue_vec, num_for_action)

        self._last_state = state
        self._last_action_vec = action_vec
        self._last_covered = list(intended) if intended else []
        if log_training:
            if intended:
                print(
                    f'[AI] training | human clue recorded: "{clue_word}" '
                    f"n={num_for_action} intended=[{', '.join(intended)}]"
                )
            else:
                print(
                    f'[AI] training | human clue recorded: "{clue_word}" '
                    f"number={number} (no target checkboxes — using number field)"
                )
        return True

    def record_outcome(
        self,
        outcomes: list[str],
        next_view: SpymasterBoardView,
        done: bool,
        *,
        next_attribution_context: np.ndarray | None = None,
        operative_guesses: list[tuple[str, str, str]] | None = None,
        log_training: bool = False,
    ):
        if self._last_state is None or self._last_action_vec is None:
            return

        reward = sum(_OUTCOME_REWARD.get(o, 0.0) for o in outcomes)

        if operative_guesses and self._last_covered:
            intended = {w.upper() for w in self._last_covered}
            extra = 0.0
            for word, _outcome, fc in operative_guesses:
                if fc == "current" and word.strip().upper() in intended:
                    extra += INTENT_MATCH_BONUS
            reward += min(extra, INTENT_MATCH_BONUS_CAP)

        model = load_model()
        next_state = self._state_from_view(
            next_view, model, attribution_context=next_attribution_context
        )

        if done:
            next_best_q = 0.0
        else:
            next_raw = candidate_clues(
                next_view.team_words,
                next_view.opponent_words,
                next_view.neutral_words,
                next_view.assassin_words,
            )
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
        self._agent.decay_epsilon()

        buf_n = len(self._agent.buffer)
        out_str = ",".join(outcomes) if outcomes else "(none)"
        loss_str = f"{loss:.4f}" if loss is not None else f"n/a (buffer < {BATCH_SIZE})"
        log_suffix = (
            f"reward={reward:.2f} guesses=[{out_str}] turn_done={done} "
            f"train_loss={loss_str} replay={buf_n}/{BUFFER_CAPACITY} "
            f"ε={self._agent.epsilon:.4f} grad_steps={self._agent._steps}"
        )
        self._agent.save(
            self._save_path,
            log_suffix=log_suffix if log_training else "",
        )

        self._last_state = None
        self._last_action_vec = None
        self._last_covered = []

    @staticmethod
    def _embed_words(words: list[str], model) -> list[np.ndarray]:
        return [model[w.lower()] for w in words if w.lower() in model]

    def _state_from_view(
        self,
        view: SpymasterBoardView,
        model,
        *,
        attribution_context: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._agent.build_state(
            self._embed_words(view.team_words, model),
            self._embed_words(view.opponent_words, model),
            self._embed_words(view.neutral_words, model),
            self._embed_words(view.assassin_words, model),
            attribution_vec=attribution_context,
        )

    @staticmethod
    def _attach_vecs(
        raw_candidates: list[tuple[str, float, list[str]]],
        model,
    ) -> list[tuple[str, float, list[str], np.ndarray]]:
        result = []
        for clue_word, score, covered in raw_candidates:
            key = clue_word.lower()
            if key in model:
                result.append((clue_word, score, covered, model[key]))
        return result
