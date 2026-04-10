import json
import re


class SchemaError(ValueError):
    pass


def extract_json_object(text: str) -> dict:
    raw = text.strip()
    raw = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.IGNORECASE | re.DOTALL)
    raw = re.sub(r"\s*```\s*$", "", raw.strip())
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        raise SchemaError("no JSON object found in model output")
    snippet = raw[start : end + 1]
    try:
        out = json.loads(snippet)
    except json.JSONDecodeError as e:
        raise SchemaError(f"invalid JSON: {e}") from e
    if not isinstance(out, dict):
        raise SchemaError("JSON root must be an object")
    return out


def parse_clue(
    obj: dict,
    board_words_upper: set[str],
    team_words_upper: set[str] | None = None,
) -> tuple[str, int, list[str] | None]:
    clue = obj.get("clue") if "clue" in obj else obj.get("word")
    if clue is None:
        raise SchemaError('missing "clue" (one word, not on the board)')
    w = str(clue).strip().upper()
    if not w or " " in w:
        raise SchemaError("clue must be a single word")
    if w in board_words_upper:
        raise SchemaError("clue cannot match any word on the board")
    if "number" not in obj:
        raise SchemaError('missing integer "number" (count of related words)')
    try:
        n = int(obj["number"])
    except (TypeError, ValueError) as e:
        raise SchemaError('"number" must be an integer') from e
    if n < 1:
        raise SchemaError('"number" must be at least 1')

    intended: list[str] | None = None
    raw = obj.get("intended")
    if raw is None:
        raw = obj.get("intended_targets")
    if raw is None:
        raw = obj.get("targets")
    if raw is not None and team_words_upper is not None:
        if not isinstance(raw, list):
            raise SchemaError('"intended" must be a JSON array of your team words')
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            u = str(item).strip().upper()
            if not u or u in seen:
                continue
            if u not in team_words_upper:
                raise SchemaError(f'"intended" word {u!r} must be one of your team words on the board')
            seen.add(u)
            out.append(u)
        intended = out if out else None

    return w, n, intended


def _normalize_for_clue(raw, past_clue_round_count: int) -> str:
    if raw is None:
        return "current"
    if isinstance(raw, bool):
        return "current"
    if isinstance(raw, (int, float)):
        k = int(raw)
        if k < 1 or k > past_clue_round_count:
            raise SchemaError(
                f'"for_clue" round index must be 1..{past_clue_round_count} '
                f"(past clue rounds before this one), or use \"current\""
            )
        return str(k)
    s = str(raw).strip().lower()
    if s in ("current", "this", "now", ""):
        return "current"
    if s in ("none", "arbitrary", "unlinked", "random", "free"):
        return "none" if s == "none" else "arbitrary"
    if s.isdigit():
        k = int(s)
        if k < 1 or k > past_clue_round_count:
            raise SchemaError(
                f'"for_clue" round index must be 1..{past_clue_round_count}, got {k}'
            )
        return str(k)
    raise SchemaError(
        '"for_clue": "current" | <past round 1..N> | "none" | "arbitrary"'
    )


def parse_operative_turn(
    obj: dict,
    unrevealed_upper: set[str],
    must_guess: bool,
    *,
    past_clue_round_count: int = 0,
) -> tuple[str | None, bool, str]:
    """
    Returns (word_upper, is_pass, for_clue_tag).
    for_clue_tag is "pass" on pass; else "current", "1".."N", "none", or "arbitrary".
    """
    wants_pass = obj.get("pass") is True or str(obj.get("action", "")).lower() == "pass"
    if wants_pass:
        if must_guess:
            raise SchemaError("you must guess at least once before passing")
        return None, True, "pass"
    g = obj.get("guess") if "guess" in obj else obj.get("word")
    if g is None:
        raise SchemaError('use {"guess":"WORD","for_clue":"current"} or {"pass":true}')
    w = str(g).strip().upper()
    if w not in unrevealed_upper:
        raise SchemaError(f"guess must be exactly one unrevealed board word; got {w!r}")
    fc = _normalize_for_clue(obj.get("for_clue"), past_clue_round_count)
    return w, False, fc
