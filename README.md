# CS4100 Project — Codenames AI

A Python desktop implementation of **Codenames** (classic edition) with a PyQt6 UI and a Q-learning AI Spymaster that trains through LLM self-play.

---

## What This Project Does

The game ships with four run modes:

| Mode | Command flag | What happens |
|------|-------------|-------------|
| **Human vs Human** | _(none)_ | Standard 4-player Codenames, no AI |
| **Play with AI** | `use-ai-agent` | Trained AI suggests clues; no weights updated |
| **Interactive Training** | `training` | You play Operative; AI learns from your guesses |
| **LLM Self-Play** | `llm_selfplay` script | Groq LLM plays both roles; AI trains headlessly |

---

## Requirements

- Python 3.10+
- PyQt6
- A free [Groq API key](https://console.groq.com/keys) (only needed for LLM self-play)

---

## Setup

**1. Clone and enter the project:**
```bash
git clone <repo-url>
cd CS4100Project
```

**2. Create and activate a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

> On first run, `gensim` will automatically download the GloVe 100-dimension word vectors (~130 MB) to `~/gensim-data`. This happens once and is cached.

**4. Configure your API key (LLM self-play only):**
```bash
cp .env.example .env
# Open .env and set: GROQ_API_KEY=<your key>
```

---

## Running the App

All commands are run from the **project root** (`CS4100Project/`).

### Human vs Human
```bash
python -m src.main
```

### Play with the trained AI Spymaster
```bash
python -m src.main use-ai-agent
```
Loads weights from `data/ai_agent.pkl`. During each Spymaster phase, click **AI Suggest** to get a clue recommendation. No training occurs.

### Interactive training mode
```bash
python -m src.main training
```
- **Spymaster phase**: Click **AI Suggest** or type your own clue word and number.
- Before clicking Submit, optionally check the boxes next to the team words you intended to hint at — this gives the Q-learning agent a cleaner training signal.
- **Operative phase**: Play normally by clicking cards to guess.
- After each completed clue turn, the agent runs one Q-learning update and saves to `data/agent.pkl`.

### LLM Self-Play (headless, recommended for bulk training)
```bash
python -m src.llm_selfplay --games 10 -v
```
A Groq LLM acts as both Spymaster and Operative. The Q-learning agent observes every turn, trains continuously, and saves to `data/ai_agent.pkl` after each turn.

| Flag | Description |
|------|-------------|
| `--games N` | Number of complete games to play (default: 1) |
| `--checkpoint PATH` | Override the save path for the agent pickle |
| `--failure-log PATH` | Override the LLM failure log path |
| `--seed N` | Set a random seed for reproducibility |
| `-v` / `--verbose` | Print per-turn clue, guess, reward, and loss details |

**Rate limiting (free Groq tier):** Add one of these to your `.env` to stay under 30 RPM:
```
GROQ_PACE_RPM=30
# or
GROQ_MIN_INTERVAL_SECONDS=2.05
```

---

## Project Layout

```
CS4100Project/
├── data/              # Word list, Q-learning checkpoints, LLM failure log
├── src/
│   ├── main.py        # Entry point
│   ├── game_engine.py # Game rules: turns, clues, guesses, win/loss
│   ├── board.py       # Board state
│   ├── llm_selfplay.py# Headless Groq self-play + Q-learning loop
│   ├── ai/            # Embeddings, Q-network, LLM client, SpymasterAgent
│   └── ui/            # PyQt6 interface
├── .env.example       # API key template — copy to .env
└── requirements.txt
```

---

## How the AI Works

**State:** The board is encoded as a 200-dimensional float vector — mean-pooled GloVe embeddings for team, opponent, neutral, and assassin word groups.

**Action:** Each candidate clue is a 101-dimensional vector (100-dim GloVe embedding + normalized clue number).

**Q-network:** A 3-layer neural network (301→128→64→1) approximates Q(state, action) — the expected reward for giving a clue in a given board state.

**Candidate generation:** `candidate_clues()` uses cosine similarity to narrow the vocabulary to ~50 reasonable clues, scoring each by team coverage minus a penalty for proximity to opponent or assassin words.

**Training signal:** After each operative turn, outcomes map to rewards:

| Outcome | Reward |
|---------|--------|
| Correct team word | +1.0 |
| Neutral word | −0.3 |
| Opponent word | −0.5 |
| Assassin word | −2.0 |

The Bellman target is computed and one gradient step is taken via MSE loss. Epsilon decays over time, shifting the agent from exploration toward exploitation.

---

## Game Rules (Quick Reference)

- **4 roles**: Red Spymaster, Red Operative, Blue Spymaster, Blue Operative.
- **Board**: 25 words — 9 starting team, 8 other team, 7 neutral, 1 assassin.
- **Spymaster turn**: Give one word clue + a number. The clue must not appear on the board.
- **Operative turn**: Guess up to (number + 1) words. Must guess at least once.
  - Correct team word → keep guessing.
  - Neutral or opponent word → turn ends.
  - Assassin → your team loses immediately.
- **Win**: Reveal all of your team's words first.

---

## Official Dataset Sources

- **Word list**: [thomasahle/codenames wordlist](https://github.com/thomasahle/codenames/blob/master/wordlist) — bundled in `data/words.txt`.
- **Key card distribution**: Generated per [official Codenames rules](https://www.ultraboardgames.com/codenames/game-rules.php).

---

## License

Educational and personal use only. Codenames is a trademark of Czech Games Edition.
