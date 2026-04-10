"""
Map operative guess → which clue word they claim to be playing (from team history).

Used to build a 100d mean clue-embedding vector concatenated onto the Q state so the
spymaster net can condition on how the team has been linking guesses to past clues.
"""

from __future__ import annotations

import numpy as np

from ai.Q_learning import EMBED_DIM


def _guess_parts(g) -> tuple[str, str, str]:
    if len(g) == 2:
        return str(g[0]), str(g[1]), "current"
    return str(g[0]), str(g[1]), str(g[2])


def operative_attribution_vector(rounds: list[dict], model) -> np.ndarray:
    """
    Mean GloVe vector of clue words that operatives associated with each recorded guess.

    rounds: list of {"clue": str, "number": int, "guesses": [(word, outcome, for_clue), ...]}
    for_clue: "current" = that round's clue; "1","2",... = 1-based index into this list;
    "none" / "arbitrary" / "pass" → skip (no clue embedding).
    """
    vecs: list[np.ndarray] = []
    for ri, r in enumerate(rounds):
        for g in r.get("guesses") or []:
            _w, _o, fc = _guess_parts(g)
            cw = _clue_word_for_tag(fc, rounds, ri)
            if cw:
                key = cw.lower()
                if key in model:
                    vecs.append(model[key])
    if not vecs:
        return np.zeros(EMBED_DIM, dtype=np.float32)
    return np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)


def _clue_word_for_tag(fc: str, rounds: list[dict], round_index: int) -> str | None:
    s = (fc or "").strip().lower()
    if s in ("pass", "none", "arbitrary", "unlinked", "random", ""):
        return None
    if s in ("current", "this", "now"):
        return rounds[round_index].get("clue") if rounds else None
    try:
        k = int(s)
    except ValueError:
        return None
    if 1 <= k <= len(rounds):
        return rounds[k - 1].get("clue")
    return None
