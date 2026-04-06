"""
Word embedding utilities for the Codenames AI spymaster.

Loads a pre-trained gensim model (default: GloVe 100d) and exposes:
  - get_embedding(word)       -> np.ndarray | None
  - candidate_clues(...)      -> ranked list of (clue, score, covered_words)

The model is downloaded once (~130 MB) to ~/gensim-data and cached locally.
"""

from collections import defaultdict

import numpy as np

from ai.clue_lexicon import is_allowed_clue_lexeme
from gensim import downloader as api
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine as cosine_distance

# Swap to "glove-wiki-gigaword-300" or "word2vec-google-news-300" for higher accuracy.
DEFAULT_MODEL = "glove-wiki-gigaword-100"

_model: KeyedVectors | None = None


def load_model(model_name: str = DEFAULT_MODEL) -> KeyedVectors:
    """Load (and cache) the gensim KeyedVectors model."""
    global _model
    if _model is None:
        print(f"[AI] Loading word vectors '{model_name}' (first run downloads ~130 MB)...")
        _model = api.load(model_name)
        print("[AI] Word vectors ready.")
    return _model


def get_embedding(word: str) -> np.ndarray | None:
    """Return the embedding vector for a word, or None if not in vocabulary."""
    model = load_model()
    key = word.lower()
    return model[key] if key in model else None


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - cosine_distance(a, b))


EXTRA_TARGET_BREADTH_BIAS = 0.4

# Pool of team words linked to a clue (by similarity); Q-learning picks how many to use.
MAX_STRONG_TEAM_LINKS = 12
# How many distinct clue words to expand (each becomes up to MAX_TARGET_COUNT actions).
MAX_CLUE_ROOTS = 22
# Only k=1..3 targets per clue are offered to Q-learning (tight Codenames play).
MAX_TARGET_COUNT = 3

# Extra similarity to assassin / opponent / neutral hurts more than a flat "avoid" pool.
ASSASSIN_PENALTY_WEIGHT = 5.0
OPPONENT_PENALTY_WEIGHT = 1.7
NEUTRAL_PENALTY_WEIGHT = 1.0


def _weighted_excess_sim_penalty(
    clue_vec: np.ndarray,
    word_vecs: list[np.ndarray],
    floor: float,
    weight: float,
) -> float:
    if not word_vecs or weight <= 0:
        return 0.0
    return weight * sum(
        max(0.0, _cosine_sim(clue_vec, v) - floor) for v in word_vecs
    )


def candidate_clues(
    team_words: list[str],
    opponent_words: list[str],
    neutral_words: list[str],
    assassin_words: list[str],
    top_n: int = 50,
    sim_threshold: float = 0.3,
    avoid_penalty_floor: float = 0.2,
) -> list[tuple[str, float, list[str]]]:
    """
    Return discrete clue actions (clue word + target count k) for Q-learning.

    `top_n` scales how many actions are returned (cap ~max(top_n*6, 150)).

    opponent_words / neutral_words / assassin_words must be the unrevealed non-team
    cards for this clue-giver's team so penalties match real risk (assassin weighted
    highest, then opponent, then neutral).

    For each clue word, up to three actions: k=1, 2, 3 team targets (prefix of the
    similarity-sorted pool). Q-learning picks the clue and k. Cosine similarity only
    builds the pool and heuristic scores, not k beyond that cap.
    """
    model = load_model()

    board_upper = {
        w.upper()
        for w in (team_words + opponent_words + neutral_words + assassin_words)
    }
    team_lower = [w.lower() for w in team_words]

    team_vecs = [(w, model[w]) for w in team_lower if w in model]
    team_display = {w.lower(): w for w in team_words}
    opponent_vecs = [model[w.lower()] for w in opponent_words if w.lower() in model]
    neutral_vecs = [model[w.lower()] for w in neutral_words if w.lower() in model]
    assassin_vecs = [model[w.lower()] for w in assassin_words if w.lower() in model]

    if not team_vecs:
        return []

    mean_team_vec = np.mean([v for _, v in team_vecs], axis=0)
    raw_candidates = model.most_similar(positive=[mean_team_vec], topn=top_n * 6)

    by_clue: dict[str, list[tuple[float, list[str]]]] = defaultdict(list)
    clue_roots = 0
    for clue_word, _ in raw_candidates:
        if clue_roots >= MAX_CLUE_ROOTS:
            break
        if clue_word.upper() in board_upper:
            continue
        if not clue_word.isalpha():
            continue
        if not is_allowed_clue_lexeme(clue_word):
            continue

        clue_vec = model[clue_word]

        team_sims = [(_w, _cosine_sim(clue_vec, v)) for _w, v in team_vecs]
        strong = [(_w, s) for _w, s in team_sims if s >= sim_threshold]
        if not strong:
            continue
        strong.sort(key=lambda x: x[1], reverse=True)
        pool = strong[:MAX_STRONG_TEAM_LINKS]
        clue_roots += 1

        danger = (
            _weighted_excess_sim_penalty(
                clue_vec, assassin_vecs, avoid_penalty_floor, ASSASSIN_PENALTY_WEIGHT
            )
            + _weighted_excess_sim_penalty(
                clue_vec, opponent_vecs, avoid_penalty_floor, OPPONENT_PENALTY_WEIGHT
            )
            + _weighted_excess_sim_penalty(
                clue_vec, neutral_vecs, avoid_penalty_floor, NEUTRAL_PENALTY_WEIGHT
            )
        )

        max_k = min(len(pool), MAX_TARGET_COUNT)
        for k in range(1, max_k + 1):
            picked = pool[:k]
            covered = [team_display.get(w, w) for w, _ in picked]
            team_score = sum(s for _, s in picked)
            base = team_score - danger
            breadth_div = 1.0 + EXTRA_TARGET_BREADTH_BIAS * max(0, k - 1)
            score = base / breadth_div
            by_clue[clue_word].append((score, covered))

    ranked_roots = sorted(
        by_clue.keys(),
        key=lambda cw: max(s for s, _ in by_clue[cw]),
        reverse=True,
    )
    out: list[tuple[str, float, list[str]]] = []
    for cw in ranked_roots:
        for sc, cov in sorted(by_clue[cw], key=lambda x: len(x[1])):
            out.append((cw, sc, cov))

    cap = min(len(out), max(top_n * 6, 150))
    return out[:cap]
