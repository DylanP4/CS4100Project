"""Groq-only convenience API (Gemini remains in llm_providers for optional use)."""

from ai.llm_config import GROQ_API_KEY, GROQ_MODEL
from ai.llm_providers import LLMProviderError, chat_completion


def groq_configured() -> bool:
    return bool(GROQ_API_KEY)


def require_groq() -> None:
    if not GROQ_API_KEY:
        raise LLMProviderError("Set GROQ_API_KEY in the environment or project .env")


def groq_complete(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float = 0.25,
    timeout_s: float = 120.0,
) -> str:
    require_groq()
    messages: list[dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": user})
    return groq_chat_messages(
        messages,
        model=model,
        temperature=temperature,
        timeout_s=timeout_s,
    )


def groq_chat_messages(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.25,
    timeout_s: float = 120.0,
) -> str:
    """OpenAI-style message list (system / user / assistant). Used to avoid resending long context."""
    require_groq()
    return chat_completion(
        "groq",
        messages,
        model=model or GROQ_MODEL,
        temperature=temperature,
        timeout_s=timeout_s,
    )
