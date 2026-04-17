"""
Headless Codenames self-play: Groq spymaster + Groq operative, JSON I/O, engine rules.

Trains the existing Q-learning spymaster into data/ai_agent.pkl (or --checkpoint).

  PYTHONPATH=src python -m llm_selfplay --games 1 -v
"""

from __future__ import annotations

import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import argparse
import os
import random
import time

from constants import BLUE, MAX_CLUE_TARGETS, PHASE_OPERATIVE, PHASE_SPYMASTER, RED
from game_engine import GameEngine
from ai.agent import AI_AGENT_SAVE_PATH, EMPTY_BOARD_VIEW, SpymasterAgent
from ai.embeddings import load_model
from ai.operative_attribution import operative_attribution_vector
from ai.llm_board import spymaster_board_view, unrevealed_words, word_index
from ai.llm_groq import groq_chat_messages, groq_complete, groq_configured, require_groq
from ai.llm_providers import LLMProviderError
from ai.llm_schemas import SchemaError, extract_json_object, parse_clue, parse_operative_turn

SPYMASTER_SYSTEM = f"""Codenames spymaster. One-word clue not on the board (no substring-of-board-word tricks).
Return ONE JSON object only — no markdown, no extra text, no explanations.
Use ONLY these keys: clue, number, intended. Do NOT include any other keys.
Schema: {{"clue":"<word>","number":<1..{MAX_CLUE_TARGETS}>,"intended":["W",...]?}}
Targeting policy:
- "number" is how many team words you intend (1, 2, or {MAX_CLUE_TARGETS} only). Aim for number=1 about 80–90% of the time.
- Use number=2 only when you are confident the guesser will successfully guess both intended words.
- Use number={MAX_CLUE_TARGETS} only when you are almost certain the guesser will get all {MAX_CLUE_TARGETS}.
- At most {MAX_CLUE_TARGETS} words in "intended"; never list more than {MAX_CLUE_TARGETS} team words.
If you include "intended", list exactly "number" words, each copied EXACTLY from the TEAM list in the user message.
Clue must NOT match any board word (team/opp/neutral/assassin) and must be one word.
Never invent, transform, or generalize intended words (e.g., DOCTOR, FIRE, RED, IRON, BEAR, LEMON, IVORY).
If unsure, OMIT "intended" entirely.
Use clear, everyday clues."""

OPERATIVE_SYSTEM = """Codenames operative — no card colors.
Return ONE JSON object only — no markdown, no extra text, no explanations.
Use ONLY these keys: guess, for_clue, pass. Do NOT include any other keys.
Guess: {"guess":"WORD","for_clue":"current"|1|2|...|"none"|"arbitrary"} — for_clue required.
Pass: {"pass":true} only when allowed after at least one guess.
Guess MUST be copied EXACTLY from the UNREVEALED list in the user message.
for_clue MUST be exactly: "current", "none", "arbitrary", or a number 1..N (no extra text).

Rules: one unrevealed board word per guess; max (n+1) guesses for clue number n; don't use the extra guess without a strong reason. n ≈ how many words the clue targets; pick the best obvious fit (what spymaster likely meant), not a stretch if a clearer match exists."""

FAILURE_LOG_ENV_VAR = "CODENAMES_LLM_FAILURE_LOG"
_DEFAULT_FAILURE_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "llm_failures.log"
FAILURE_LOG_PATH = Path(
    (os.environ.get(FAILURE_LOG_ENV_VAR) or "").strip() or _DEFAULT_FAILURE_LOG_PATH
)


def _log_llm_failure(role: str, err: str | None, raw: str | None) -> None:
    if not raw and not err:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"[{ts}] role={role}",
        f"error={err or '(none)'}",
        "raw=",
        raw or "",
        "-" * 40,
    ]
    try:
        FAILURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FAILURE_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass


def _guess_parts(g) -> tuple[str, str, str]:
    if len(g) == 2:
        return g[0], g[1], "current"
    return g[0], g[1], g[2]


def _team_words_confirmed_from_history(rounds: list[dict]) -> list[str]:
    out: list[str] = []
    for r in rounds:
        for g in r.get("guesses") or []:
            w, o, _fc = _guess_parts(g)
            if o == "correct" and w.upper() not in {x.upper() for x in out}:
                out.append(w)
    return out


def _format_operative_history(team: str, prior_rounds: list[dict]) -> str:
    if not prior_rounds:
        return "No prior rounds."
    lines = [f"History ({team.upper()}):"]
    for i, r in enumerate(prior_rounds, 1):
        clue = r["clue"]
        n = r["number"]
        guesses = r.get("guesses") or []
        if not guesses:
            lines.append(f" R{i}: {clue!r} n={n} → (no guesses)")
        else:
            bits = []
            for g in guesses:
                w, o, fc = _guess_parts(g)
                bits.append(f"{w}→{o}(fc={fc})")
            lines.append(f" R{i}: {clue!r} n={n} → " + "; ".join(bits))
    conf = _team_words_confirmed_from_history(prior_rounds)
    lines.append("Confirmed team words: " + (", ".join(conf) if conf else "(none yet)"))
    return "\n".join(lines)


def run_spymaster_turn(
    engine: GameEngine,
    agent: SpymasterAgent,
    *,
    attribution_context,
    verbose: bool,
    max_attempts: int = 6,
) -> tuple[bool, bool]:
    board = engine.board
    team = engine.current_team
    view = spymaster_board_view(board, team)
    board_set = {w.upper() for w in board.words()}
    team_set = {w.upper() for w in view.team_words}

    def user_msg(err: str | None) -> str:
        lines = [
            f"{team.upper()} | team: {', '.join(view.team_words)} | opp: {', '.join(view.opponent_words)} "
            f"| neutral: {', '.join(view.neutral_words)} | assassin: {', '.join(view.assassin_words)}",
            'JSON: {"clue":"...","number":N,"intended":[...]?}',
        ]
        if err:
            lines.append(f"Fix: {err}")
            lines.append("Respond again with ONLY the corrected JSON object. No extra text.")
        return "\n".join(lines)

    clue: str | None = None
    number: int | None = None
    intended_list: list[str] | None = None
    last_err: str | None = None
    last_raw: str | None = None
    for attempt in range(max_attempts):
        temp = 0.25 if attempt == 0 else 0.08
        try:
            raw = groq_complete(SPYMASTER_SYSTEM, user_msg(last_err), temperature=temp)
            last_raw = raw
            obj = extract_json_object(raw)
            clue, number, intended_list = parse_clue(obj, board_set, team_set)
            number = min(MAX_CLUE_TARGETS, max(1, number))
            if intended_list:
                k = len(intended_list)
                if k == 0:
                    intended_list = None
                else:
                    # Match declared count to explicit targets; cap at MAX_CLUE_TARGETS.
                    number = min(MAX_CLUE_TARGETS, max(number, k))
                    if number > k:
                        number = k
            break
        except (SchemaError, ValueError) as e:
            last_err = str(e)
        except LLMProviderError:
            raise
    if clue is None or number is None:
        print(f"[llm_selfplay] spymaster gave up: {last_err}", file=sys.stderr)
        if last_raw:
            print("[llm_selfplay] spymaster last raw output:", file=sys.stderr)
            print(last_raw, file=sys.stderr)
        _log_llm_failure("spymaster", last_err, last_raw)
        return False, False

    ok, msg = engine.submit_clue(clue, number)
    if not ok:
        print(f"[llm_selfplay] engine rejected clue: {msg}", file=sys.stderr)
        return False, False

    recorded = agent.record_human_clue(
        view,
        clue,
        number,
        intended_team_words=intended_list,
        attribution_context=attribution_context,
        log_training=verbose,
    )
    if not recorded and verbose:
        print(f"[llm_selfplay] Q-learning skipped (clue not in embedding vocab): {clue!r}")
    if verbose:
        print(f"[llm_selfplay] clue {clue!r} n={number} team={team} recorded={recorded}")
    return True, recorded


def run_operative_turn(
    engine: GameEngine,
    *,
    team: str,
    prior_rounds: list[dict],
    verbose: bool,
    max_attempts: int = 8,
) -> tuple[list[str], list[tuple[str, str]], str, int]:
    """
    Returns (outcomes_for_reward, guess_log, clue_word, clue_number).
    guess_log entries are (WORD_UPPER, outcome, for_clue_tag) per successful guess() call.
    """
    outcomes: list[str] = []
    guess_log: list[tuple[str, str]] = []
    board = engine.board
    clue = engine.clue_word
    num = engine.clue_number
    if clue is None or num is None:
        return [], [], "", 0
    must_guess = True
    history_block = _format_operative_history(team, prior_rounds)
    max_steps = max(30, (num + 2) * 8)
    operative_thread: list[dict[str, str]] | None = None

    step_i = 0
    while (
        engine.phase == PHASE_OPERATIVE
        and not engine.game_over
        and engine.guesses_left > 0
    ):
        step_i += 1
        if step_i > max_steps:
            print(
                "[llm_selfplay] operative step limit hit; ending turn",
                file=sys.stderr,
            )
            if engine.phase == PHASE_OPERATIVE and not engine.game_over:
                engine.end_turn_early()
            break
        unrevealed = unrevealed_words(board)
        unrevealed_set = {w.upper() for w in unrevealed}

        def operative_user_full(err: str | None) -> str:
            so_far = ""
            if guess_log:
                lines_sf = []
                for g in guess_log:
                    wg, og, tg = _guess_parts(g)
                    lines_sf.append(f"{wg}→{og}({tg})")
                so_far = "This turn: " + "; ".join(lines_sf) + "\n"
            parts = [
                f"{team.upper()} operative.",
                history_block,
                "",
                so_far + f"Clue {clue!r} n={num} | left={engine.guesses_left} | must_guess={must_guess}",
                f"Unrevealed ({len(unrevealed)}): {', '.join(unrevealed)}",
                'JSON: guess+for_clue or pass.',
            ]
            if err:
                parts.append(f"Error: {err}")
            return "\n".join(parts)

        def operative_user_delta(err: str | None) -> str:
            bits = []
            if guess_log:
                bits.append(
                    "Turn: "
                    + "; ".join(f"{wg}→{og}({tg})" for wg, og, tg in (_guess_parts(g) for g in guess_log))
                )
            bits.append(
                f"Clue {clue!r} n={num} | left={engine.guesses_left} | must_guess={must_guess} | "
                f"unrevealed ({len(unrevealed)}): {', '.join(unrevealed)}"
            )
            bits.append("JSON: guess+for_clue or pass.")
            if err:
                bits.append(f"Error: {err}")
            return "\n".join(bits)

        last_err: str | None = None
        last_raw: str | None = None
        word: str | None = None
        is_pass = False
        for_tag = "current"
        commit_messages: list[dict[str, str]] | None = None
        for _ in range(max_attempts):
            temp = 0.25 if last_err is None else 0.08
            if operative_thread is None:
                to_send = [
                    {"role": "system", "content": OPERATIVE_SYSTEM},
                    {"role": "user", "content": operative_user_full(last_err)},
                ]
            else:
                to_send = operative_thread + [
                    {"role": "user", "content": operative_user_delta(last_err)},
                ]
            try:
                raw = groq_chat_messages(to_send, temperature=temp)
                last_raw = raw
                obj = extract_json_object(raw)
                word, is_pass, for_tag = parse_operative_turn(
                    obj,
                    unrevealed_set,
                    must_guess,
                    past_clue_round_count=len(prior_rounds),
                )
                commit_messages = to_send + [{"role": "assistant", "content": raw}]
                break
            except SchemaError as e:
                last_err = str(e)
            except LLMProviderError:
                raise
        else:
            print(f"[llm_selfplay] operative stuck: {last_err}; forcing pass/end", file=sys.stderr)
            if last_raw:
                print("[llm_selfplay] operative last raw output:", file=sys.stderr)
                print(last_raw, file=sys.stderr)
            _log_llm_failure("operative", last_err, last_raw)
            if must_guess and unrevealed:
                idx = random.randrange(len(unrevealed))
                word = unrevealed[idx].upper()
                is_pass = False
                for_tag = "current"
            else:
                engine.end_turn_early()
                break

        if is_pass:
            if commit_messages is not None:
                operative_thread = commit_messages
            engine.end_turn_early()
            if verbose:
                print("[llm_selfplay] operative passed")
            break

        assert word is not None
        idx = word_index(board, word)
        if idx is None:
            last_err = f"internal: lost index for {word!r}"
            continue

        ok, outcome, _msg = engine.guess(idx)
        if not ok:
            last_err = _msg
            continue

        if commit_messages is not None:
            operative_thread = commit_messages

        outcomes.append(outcome)
        guess_log.append((word, outcome, for_tag))
        if verbose:
            print(f"[llm_selfplay] guess {word!r} -> {outcome} for_clue={for_tag!r}")

        if engine.game_over:
            break
        if outcome in ("neutral", "opponent"):
            break

    return outcomes, guess_log, clue, num


def play_one_game(
    agent: SpymasterAgent,
    *,
    verbose: bool,
    seed: int | None = None,
) -> str | None:
    if seed is not None:
        random.seed(seed)
    engine = GameEngine()
    engine.start_game()
    ai_team_for_outcome: str | None = None
    last_recorded = False
    operative_memory: dict[str, list[dict]] = {RED: [], BLUE: []}

    model = load_model()
    while not engine.game_over:
        if engine.phase == PHASE_SPYMASTER:
            team_sm = engine.current_team
            attr_pre = operative_attribution_vector(operative_memory[team_sm], model)
            ok, last_recorded = run_spymaster_turn(
                engine,
                agent,
                attribution_context=attr_pre,
                verbose=verbose,
            )
            if not ok:
                return "aborted_spymaster"
            ai_team_for_outcome = engine.current_team
        elif engine.phase == PHASE_OPERATIVE:
            team = engine.current_team
            prior = operative_memory[team]
            outcomes, guess_log, clue_word, clue_num = run_operative_turn(
                engine,
                team=team,
                prior_rounds=prior,
                verbose=verbose,
            )
            if clue_word:
                operative_memory[team].append(
                    {
                        "clue": clue_word,
                        "number": clue_num,
                        "guesses": list(guess_log),
                    }
                )
            if ai_team_for_outcome is None:
                continue
            done = engine.game_over
            next_view = (
                EMPTY_BOARD_VIEW
                if done
                else spymaster_board_view(engine.board, ai_team_for_outcome)
            )
            if last_recorded:
                next_attr = operative_attribution_vector(
                    operative_memory[ai_team_for_outcome], model
                )
                agent.record_outcome(
                    outcomes,
                    next_view,
                    done,
                    next_attribution_context=next_attr,
                    operative_guesses=list(guess_log),
                    log_training=verbose,
                )
            elif verbose and outcomes:
                print("[llm_selfplay] skipped record_outcome (no recorded clue state)")
        else:
            break

    return engine.winner


def main() -> None:
    global FAILURE_LOG_PATH
    p = argparse.ArgumentParser(description="Groq LLM self-play + Q-learning (ai_agent.pkl)")
    p.add_argument("--games", type=int, default=1, help="full games to play")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(AI_AGENT_SAVE_PATH),
        help="path for SpymasterAgent pickle",
    )
    p.add_argument(
        "--failure-log",
        type=str,
        default=str(FAILURE_LOG_PATH),
        help=f"path for LLM failure log (default via {FAILURE_LOG_ENV_VAR})",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--reset-replay",
        action="store_true",
        help="After load: wipe replay buffer + reset Q-net Adam, save checkpoint, then play (keeps weights).",
    )
    args = p.parse_args()

    if not groq_configured():
        print("Set GROQ_API_KEY (see .env.example)", file=sys.stderr)
        sys.exit(1)
    require_groq()

    FAILURE_LOG_PATH = Path(args.failure_log)

    ckpt = Path(args.checkpoint)
    if args.verbose:
        print("[llm_selfplay] loading embeddings (first run may download GloVe)...")
    t0 = time.perf_counter()
    load_model()
    if args.verbose:
        print(f"[llm_selfplay] embeddings ready in {time.perf_counter() - t0:.1f}s")

    agent = SpymasterAgent(save_path=ckpt)
    if args.reset_replay:
        agent.reset_replay_checkpoint()
        if args.verbose:
            print("[llm_selfplay] replay wiped and checkpoint saved (--reset-replay)")

    completed = 0
    aborts = 0
    attempts = 0
    max_attempts = max(args.games * 3, args.games + 2)
    while completed < args.games:
        attempts += 1
        if attempts > max_attempts:
            print("[llm_selfplay] too many aborted games; stopping early", file=sys.stderr)
            break
        if args.verbose:
            print(f"\n[llm_selfplay] === game {completed + 1}/{args.games} ===")
        seed = args.seed + completed if args.seed is not None else None
        try:
            winner = play_one_game(agent, verbose=args.verbose, seed=seed)
        except LLMProviderError as e:
            print(e, file=sys.stderr)
            if e.body:
                print(e.body, file=sys.stderr)
            sys.exit(2)
        if args.verbose:
            print(f"[llm_selfplay] winner={winner}")
        if winner == "aborted_spymaster":
            aborts += 1
            print("[llm_selfplay] restarting with a fresh game after spymaster failure")
            continue
        completed += 1

    if aborts:
        print(f"[llm_selfplay] aborted games: {aborts}", file=sys.stderr)
        print(f"[llm_selfplay] failure log: {FAILURE_LOG_PATH}", file=sys.stderr)
    print(f"[llm_selfplay] done; checkpoint {ckpt}")


if __name__ == "__main__":
    main()
