"""
Word embedding utilities for the Codenames AI spymaster.

Loads a pre-trained gensim model (default: GloVe 100d) and exposes:
  - get_embedding(word)       -> np.ndarray | None
  - candidate_clues(...)      -> ranked list of (clue, score, covered_words)

The model is downloaded once (~130 MB) to ~/gensim-data and cached locally.
"""

import random

import numpy as np
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

# When scoring a clue, only the strongest links (by similarity) can be selected as targets.
MAX_STRONG_TEAM_LINKS = 12

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


def _sample_target_word_count(n_available: int) -> int:
    """
    How many team words to tie to this clue candidate.

    Biased toward 1–2, sometimes 3, rarely 4+ (when that many strong links exist).
    """
    if n_available <= 0:
        return 0
    if n_available == 1:
        return 1
    r = random.random()
    if n_available == 2:
        return 1 if r < 0.42 else 2
    if n_available == 3:
        if r < 0.30:
            return 1
        if r < 0.72:
            return 2
        return 3
    # 4+ links in pool
    if r < 0.30:
        return 1
    if r < 0.72:
        return 2
    if r < 0.93:
        return 3
    hi = min(n_available, 6)
    return random.randint(4, hi)


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
    Return the top_n candidate one-word clues for the current spymaster turn.

    opponent_words / neutral_words / assassin_words must be the unrevealed non-team
    cards for this clue-giver's team so penalties match real risk (assassin weighted
    highest, then opponent, then neutral).

    Ranking uses breadth bias and random target count (see _sample_target_word_count).
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

    results = []
    for clue_word, _ in raw_candidates:
        if clue_word.upper() in board_upper:
            continue
        if not clue_word.isalpha():
            continue

        clue_vec = model[clue_word]

        team_sims = [(_w, _cosine_sim(clue_vec, v)) for _w, v in team_vecs]
        strong = [(_w, s) for _w, s in team_sims if s >= sim_threshold]
        if not strong:
            continue
        strong.sort(key=lambda x: x[1], reverse=True)
        pool = strong[:MAX_STRONG_TEAM_LINKS]
        k = _sample_target_word_count(len(pool))
        picked = pool[:k]
        covered = [team_display.get(w, w) for w, _ in picked]
        team_score = sum(s for _, s in picked)
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
        base = team_score - danger
        n = len(covered)
        breadth_div = 1.0 + EXTRA_TARGET_BREADTH_BIAS * max(0, n - 1)
        score = base / breadth_div

        results.append((clue_word, score, covered))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]
