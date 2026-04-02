"""
Word embedding utilities for the Codenames AI spymaster.

Loads a pre-trained gensim model (default: GloVe 100d) and exposes:
  - get_embedding(word)       -> np.ndarray | None
  - candidate_clues(...)      -> ranked list of (clue, score, covered_words)

The model is downloaded once (~130 MB) to ~/gensim-data and cached locally.
"""

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


def candidate_clues(
    team_words: list[str],
    all_board_words: list[str],
    top_n: int = 50,
    sim_threshold: float = 0.3,
    avoid_penalty_floor: float = 0.2,
) -> list[tuple[str, float, list[str]]]:
    """
    Return the top_n candidate one-word clues for the current spymaster turn.

    Parameters
    ----------
    team_words      : unrevealed words belonging to the AI's team.
    all_board_words : all 25 words still on the board (revealed + unrevealed).
    top_n           : number of candidates to return.
    sim_threshold   : minimum cosine similarity for a team word to count as "covered".
    avoid_penalty_floor : similarity above this to a non-team word incurs a penalty.

    Returns
    -------
    List of (clue_word, score, covered_team_words) sorted by score descending.
    score = sum(sim to covered team words) - sum(excess sim to non-team board words)
    """
    model = load_model()

    board_upper = {w.upper() for w in all_board_words}
    team_lower = [w.lower() for w in team_words]
    avoid_lower = [
        w.lower()
        for w in all_board_words
        if w.upper() not in {t.upper() for t in team_words}
    ]

    team_vecs = [(w, model[w]) for w in team_lower if w in model]
    avoid_vecs = [(w, model[w]) for w in avoid_lower if w in model]

    if not team_vecs:
        return []

    # Seed candidates from words most similar to the mean team embedding.
    mean_team_vec = np.mean([v for _, v in team_vecs], axis=0)
    raw_candidates = model.most_similar(positive=[mean_team_vec], topn=top_n * 6)

    results = []
    for clue_word, _ in raw_candidates:
        # Must not be a board word (case-insensitive) and must be purely alphabetic.
        if clue_word.upper() in board_upper:
            continue
        if not clue_word.isalpha():
            continue

        clue_vec = model[clue_word]

        team_sims = [(_w, _cosine_sim(clue_vec, v)) for _w, v in team_vecs]
        covered = [_w for _w, s in team_sims if s >= sim_threshold]

        if not covered:
            continue

        team_score = sum(s for _, s in team_sims if s >= sim_threshold)
        avoid_penalty = sum(
            max(0.0, _cosine_sim(clue_vec, v) - avoid_penalty_floor)
            for _, v in avoid_vecs
        )

        results.append((clue_word, team_score - avoid_penalty, covered))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]
