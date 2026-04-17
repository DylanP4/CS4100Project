import os
from pathlib import Path


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(root / ".env")


_load_dotenv()


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Primary path for this repo: Groq (`llm_selfplay`, `ai.llm_groq`). Gemini env vars are optional.
GROQ_API_KEY = (os.environ.get("GROQ_API_KEY") or "").strip()
GEMINI_API_KEY = (
    os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
).strip()

GROQ_MODEL = (os.environ.get("GROQ_MODEL") or "llama-3.1-8b-instant").strip()
GEMINI_MODEL = (os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash").strip()

# Minimum seconds between Groq HTTP calls (end of one response → start of next). 0 = off.
# If set > 0, it wins over GROQ_PACE_RPM.
GROQ_MIN_INTERVAL_SECONDS = max(0.0, _env_float("GROQ_MIN_INTERVAL_SECONDS", 0.0))
# Default ~25 RPM (~2.5s between calls) to reduce 429s on free tier. Set GROQ_PACE_RPM=0 to disable pacing.
GROQ_PACE_RPM = max(0.0, _env_float("GROQ_PACE_RPM", 25.0))


def groq_effective_min_interval_s() -> float:
    if GROQ_MIN_INTERVAL_SECONDS > 0:
        return GROQ_MIN_INTERVAL_SECONDS
    if GROQ_PACE_RPM > 0:
        return (60.0 / GROQ_PACE_RPM) * 1.05
    return 0.0


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
