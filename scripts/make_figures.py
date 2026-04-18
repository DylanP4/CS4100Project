"""
Generate simple graphs/tables for the writeup WITHOUT re-running training.

Inputs:
  - Q-learning checkpoint pickle (data/ai_agent*.pkl)
  - LLM failure log (data/llm_failures*.log)

Outputs (default: figures/):
  - CSV summaries (always)
  - PNG plots (if matplotlib is installed)

Usage (PowerShell, from repo root):
  python scripts/make_figures.py --checkpoint data\\ai_agent_experiment.pkl --failure-log data\\llm_failures_experiment.log
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


_FAIL_RE = re.compile(
    r"^\[(?P<ts>[\d\-: ]+)\]\s+role=(?P<role>\w+)\s*$", re.IGNORECASE
)
_ERR_RE = re.compile(r"^error=(?P<err>.*)\s*$", re.IGNORECASE)


def _read_failure_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries: list[dict] = []
    cur: dict | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _FAIL_RE.match(line.strip())
        if m:
            if cur:
                entries.append(cur)
            cur = {"ts": m.group("ts").strip(), "role": m.group("role").strip()}
            continue
        m = _ERR_RE.match(line.strip())
        if m and cur is not None:
            cur["error"] = m.group("err").strip()
            continue
    if cur:
        entries.append(cur)
    # Parse timestamps where possible.
    for e in entries:
        try:
            e["dt"] = datetime.strptime(e["ts"], "%Y-%m-%d %H:%M:%S")
        except Exception:
            e["dt"] = None
    return entries


def _load_checkpoint(path: Path) -> dict:
    with path.open("rb") as f:
        data = pickle.load(f)
    return data


def _bucket_reward(r: float) -> str:
    # Rewards can include small bonuses (e.g., +0.08) so bucket by ranges.
    if r <= -1.5:
        return "assassin"
    if r <= -0.5:
        return "opponent"
    if -0.2 <= r <= 0.2:
        return "neutral_or_zero"
    return "positive"


def _maybe_plot(outdir: Path, checkpoint_name: str, rewards: np.ndarray, dones: np.ndarray, failures: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    outdir.mkdir(parents=True, exist_ok=True)

    # Reward histogram.
    plt.figure(figsize=(7, 4))
    plt.hist(rewards, bins=40, color="#2b6cb0", alpha=0.85)
    plt.title(f"Reward Distribution ({checkpoint_name})")
    plt.xlabel("reward")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outdir / "reward_hist.png", dpi=180)
    plt.close()

    # Reward buckets.
    buckets = Counter(_bucket_reward(float(r)) for r in rewards)
    labels = ["positive", "neutral_or_zero", "opponent", "assassin"]
    vals = [buckets.get(k, 0) for k in labels]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, vals, color=["#38a169", "#718096", "#dd6b20", "#e53e3e"])
    plt.title(f"Reward Buckets ({checkpoint_name})")
    plt.ylabel("count")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "reward_buckets.png", dpi=180)
    plt.close()

    # Done fraction.
    done_rate = float(np.mean(dones)) if len(dones) else 0.0
    plt.figure(figsize=(5, 4))
    plt.bar(["non-terminal", "terminal"], [1.0 - done_rate, done_rate], color=["#4a5568", "#2f855a"])
    plt.ylim(0, 1)
    plt.title(f"Terminal Transition Rate ({checkpoint_name})")
    plt.ylabel("fraction")
    plt.tight_layout()
    plt.savefig(outdir / "terminal_rate.png", dpi=180)
    plt.close()

    # Failure log charts (if present).
    if failures:
        # Cumulative failures over time.
        dts = [e["dt"] for e in failures if e.get("dt") is not None]
        if dts:
            dts = sorted(dts)
            xs = dts
            ys = list(range(1, len(dts) + 1))
            plt.figure(figsize=(7, 4))
            plt.plot(xs, ys, color="#805ad5")
            plt.title("Cumulative LLM Schema/Rule Failures")
            plt.xlabel("time")
            plt.ylabel("cumulative count")
            plt.tight_layout()
            plt.savefig(outdir / "failures_cumulative.png", dpi=180)
            plt.close()

        # Top error messages.
        errs = [str(e.get("error") or "").strip() for e in failures if e.get("error")]
        top = Counter(errs).most_common(8)
        if top:
            labels = [t[0][:55] + ("…" if len(t[0]) > 55 else "") for t in top]
            vals = [t[1] for t in top]
            plt.figure(figsize=(10, 4.5))
            plt.bar(range(len(vals)), vals, color="#c53030")
            plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
            plt.title("Top Failure Reasons (truncated)")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(outdir / "failure_top_reasons.png", dpi=180)
            plt.close()


def main() -> int:
    p = argparse.ArgumentParser(description="Generate graphs/tables from existing checkpoint + logs.")
    p.add_argument("--checkpoint", type=str, default="", help="Path to agent .pkl (default: auto-detect)")
    p.add_argument("--failure-log", type=str, default="", help="Path to failure log (default: auto-detect)")
    p.add_argument("--outdir", type=str, default="figures", help="Output directory for figures/CSVs")
    args = p.parse_args()

    repo = Path(__file__).resolve().parent.parent
    outdir = (repo / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt_candidates = [
        repo / "data" / "ai_agent_experiment.pkl",
        repo / "data" / "ai_agent.pkl",
        repo / "data" / "agent.pkl",
    ]
    ckpt_path = Path(args.checkpoint) if args.checkpoint else next((p for p in ckpt_candidates if p.exists()), None)
    if ckpt_path is None or not ckpt_path.exists():
        print("No checkpoint found. Pass --checkpoint path/to.pkl")
        return 2

    log_candidates = [
        repo / "data" / "llm_failures_experiment.log",
        repo / "data" / "llm_failures.log",
    ]
    log_path = Path(args.failure_log) if args.failure_log else next((p for p in log_candidates if p.exists()), None)

    ckpt = _load_checkpoint(ckpt_path)
    steps = int(ckpt.get("steps") or 0)
    epsilon = float(ckpt.get("epsilon") or 0.0)
    replay = ckpt.get("replay") or {}
    transitions = replay.get("transitions") or []

    rewards = np.array([float(t[2]) for t in transitions], dtype=np.float32) if transitions else np.array([], dtype=np.float32)
    dones = np.array([float(t[4]) for t in transitions], dtype=np.float32) if transitions else np.array([], dtype=np.float32)
    buckets = Counter(_bucket_reward(float(r)) for r in rewards) if len(rewards) else Counter()

    failures = _read_failure_log(log_path) if log_path and log_path.exists() else []
    failures_by_role = Counter((e.get("role") or "unknown") for e in failures)
    failures_by_day: dict[str, int] = defaultdict(int)
    failures_by_err = Counter()
    for e in failures:
        if e.get("dt") is not None:
            failures_by_day[e["dt"].strftime("%Y-%m-%d")] += 1
        if e.get("error"):
            failures_by_err[str(e["error"])] += 1

    summary = {
        "checkpoint": str(ckpt_path),
        "steps": steps,
        "epsilon": epsilon,
        "replay_transitions": int(len(transitions)),
        "reward_mean": float(np.mean(rewards)) if len(rewards) else None,
        "reward_std": float(np.std(rewards)) if len(rewards) else None,
        "reward_buckets": dict(buckets),
        "terminal_rate": float(np.mean(dones)) if len(dones) else None,
        "failure_log": str(log_path) if log_path else None,
        "failures_total": int(len(failures)),
        "failures_by_role": dict(failures_by_role),
        "failures_by_day": dict(sorted(failures_by_day.items())),
        "top_failure_reasons": failures_by_err.most_common(10),
    }

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / "summary.txt").write_text(
        "\n".join(
            [
                f"checkpoint: {ckpt_path}",
                f"steps: {steps}",
                f"epsilon: {epsilon:.4f}",
                f"replay transitions: {len(transitions)}",
                f"reward mean/std: {summary['reward_mean']} / {summary['reward_std']}",
                f"terminal rate: {summary['terminal_rate']}",
                f"failure log: {log_path if log_path else '(none)'}",
                f"failures total: {len(failures)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # CSV: rewards (for Excel / Google Sheets).
    with (outdir / "rewards.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["reward", "bucket", "done"])
        for r, d in zip(rewards.tolist(), dones.tolist()):
            w.writerow([r, _bucket_reward(float(r)), int(d)])

    # CSV: failure daily counts.
    with (outdir / "failures_by_day.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "count"])
        for day, cnt in sorted(failures_by_day.items()):
            w.writerow([day, cnt])

    # CSV: top failure reasons.
    with (outdir / "failure_reasons.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["error", "count"])
        for err, cnt in failures_by_err.most_common():
            w.writerow([err, cnt])

    _maybe_plot(outdir, ckpt_path.name, rewards, dones, failures)
    print(f"Wrote figures/CSVs to {outdir}")
    print("If you want PNG plots, install matplotlib: pip install matplotlib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

