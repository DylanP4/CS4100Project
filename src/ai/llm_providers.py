"""
OpenAI-compatible Groq chat + Google Gemini generateContent (free API keys via env).

Requires: GROQ_API_KEY and/or GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment
or in a project-root .env file (optional python-dotenv).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import httpx

from ai.llm_config import (
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    groq_effective_min_interval_s,
)

ProviderName = Literal["groq", "gemini"]

_groq_last_request_end_s = 0.0


@dataclass
class LLMProviderError(Exception):
    message: str
    status_code: int | None = None
    body: str | None = None

    def __str__(self) -> str:
        if self.status_code is not None:
            return f"{self.message} (HTTP {self.status_code})"
        return self.message


def _gemini_contents(rest: list[dict[str, str]]) -> list[dict]:
    """Build Gemini `contents`; merge adjacent same-role turns (API expects user/model alternation)."""
    if not rest:
        raise LLMProviderError("Gemini needs at least one user or assistant message.")
    contents: list[dict] = []
    for m in rest:
        role = "user" if m["role"] == "user" else "model"
        text = m["content"]
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"][0]["text"] += "\n\n" + text
        else:
            contents.append({"role": role, "parts": [{"text": text}]})
    return contents


def _split_messages(
    messages: list[dict[str, str]],
) -> tuple[str | None, list[dict[str, str]]]:
    system_parts: list[str] = []
    rest: list[dict[str, str]] = []
    for m in messages:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if role == "system" and content:
            system_parts.append(content)
        elif role in ("user", "assistant") and content:
            rest.append({"role": role, "content": content})
    system = "\n\n".join(system_parts) if system_parts else None
    return system, rest


def chat_completion(
    provider: ProviderName,
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    timeout_s: float = 120.0,
) -> str:
    """
    Run a chat-style completion. `messages` uses OpenAI roles: system, user, assistant.
    """
    if provider == "groq":
        return _groq_chat(messages, model=model or GROQ_MODEL, temperature=temperature, timeout_s=timeout_s)
    return _gemini_chat(messages, model=model or GEMINI_MODEL, temperature=temperature, timeout_s=timeout_s)


def _groq_chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float,
    timeout_s: float,
) -> str:
    global _groq_last_request_end_s
    if not GROQ_API_KEY:
        raise LLMProviderError("Missing GROQ_API_KEY (set in environment or .env)")

    interval = groq_effective_min_interval_s()
    if interval > 0:
        wait = interval - (time.perf_counter() - _groq_last_request_end_s)
        if wait > 0:
            time.sleep(wait)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{GROQ_BASE_URL}/chat/completions"
    max_attempts = 6
    last_body = ""
    with httpx.Client(timeout=timeout_s) as client:
        for attempt in range(max_attempts):
            try:
                r = client.post(url, json=payload, headers=headers)
            except httpx.RequestError as e:
                # Transient TLS / TCP issues (e.g. "Connection reset by peer") should not abort training.
                if attempt + 1 >= max_attempts:
                    raise LLMProviderError(f"Groq network error after {max_attempts} attempts: {e}") from e
                sleep_s = min(60.0, 1.5**attempt)
                time.sleep(sleep_s)
                if interval > 0:
                    gap = interval - (time.perf_counter() - _groq_last_request_end_s)
                    if gap > 0:
                        time.sleep(gap)
                continue

            _groq_last_request_end_s = time.perf_counter()
            last_body = r.text[:2000]

            if r.status_code == 200:
                data = r.json()
                try:
                    return (data["choices"][0]["message"]["content"] or "").strip()
                except (KeyError, IndexError, TypeError) as e:
                    raise LLMProviderError(f"Unexpected Groq response shape: {e}") from e

            if r.status_code == 429 and attempt + 1 < max_attempts:
                ra = r.headers.get("retry-after")
                try:
                    sleep_s = float(ra) if ra is not None and str(ra).strip() else 0.0
                except ValueError:
                    sleep_s = 0.0
                if sleep_s <= 0:
                    sleep_s = min(60.0, 2.0**attempt)
                time.sleep(sleep_s)
                if interval > 0:
                    gap = interval - (time.perf_counter() - _groq_last_request_end_s)
                    if gap > 0:
                        time.sleep(gap)
                continue

            raise LLMProviderError(
                "Groq request failed",
                status_code=r.status_code,
                body=last_body,
            )


def _gemini_chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float,
    timeout_s: float,
) -> str:
    if not GEMINI_API_KEY:
        raise LLMProviderError(
            "Missing GEMINI_API_KEY or GOOGLE_API_KEY (set in environment or .env)"
        )

    system, rest = _split_messages(messages)
    contents = _gemini_contents(rest)

    body: dict = {
        "contents": contents,
        "generationConfig": {"temperature": temperature},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}

    url = f"{GEMINI_BASE_URL}/models/{model}:generateContent"
    params = {"key": GEMINI_API_KEY}
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, params=params, json=body)

    if r.status_code != 200:
        raise LLMProviderError(
            "Gemini request failed",
            status_code=r.status_code,
            body=r.text[:2000],
        )
    data = r.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        return "".join(texts).strip()
    except (KeyError, IndexError, TypeError) as e:
        err = data.get("error", {})
        if err:
            raise LLMProviderError(
                f"Gemini error: {err.get('message', err)}",
                status_code=r.status_code,
            ) from e
        raise LLMProviderError(f"Unexpected Gemini response shape: {e}") from e


def _main() -> None:
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Smoke-test Groq / Gemini API keys")
    p.add_argument("provider", choices=["groq", "gemini"])
    args = p.parse_args()
    try:
        out = chat_completion(
            args.provider,
            [
                {"role": "system", "content": "Reply with one short sentence."},
                {"role": "user", "content": 'Say exactly: "ok".'},
            ],
            temperature=0.0,
        )
    except LLMProviderError as e:
        print(e, file=sys.stderr)
        if e.body:
            print(e.body, file=sys.stderr)
        sys.exit(1)
    print(out)


if __name__ == "__main__":
    _main()
