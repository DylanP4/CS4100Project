"""
Filter for AI-generated Codenames clues.

Word list is loaded from data/disallowed_clue_words.txt (one word per line).
"""

from pathlib import Path

MIN_CLUE_WORD_LENGTH = 3

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DEFAULT_LEXICON = _DATA_DIR / "disallowed_clue_words.txt"


def _load_disallowed_words(path: Path = _DEFAULT_LEXICON) -> frozenset[str]:
    if not path.is_file():
        print(f"[AI] Warning: clue blocklist not found at {path} — filter disabled.")
        return frozenset()
    words: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words.add(line.lower())
    return frozenset(words)


DISALLOWED_CLUE_WORDS = _load_disallowed_words()


def is_allowed_clue_lexeme(clue_word: str) -> bool:
    w = clue_word.lower()
    if len(w) < MIN_CLUE_WORD_LENGTH:
        return False
    return w not in DISALLOWED_CLUE_WORDS
