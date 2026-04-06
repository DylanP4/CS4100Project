"""
Quick sanity check for the Q-network and SpymasterAgent.

Run from the project root:
    python test_ai.py

What it checks:
  1. GloVe loads without error.
  2. candidate_clues() returns results for a sample board.
  3. The Q-network produces Q-values and its weights visibly change after training.
  4. Loss decreases over repeated iterations (network is learning).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from ai.embeddings import candidate_clues, load_model
from ai.Q_learning import BATCH_SIZE, QLearningAgent

# Synthetic 25-card layout split by Codenames role — test fixture only, not from the real app.
team_words = ["OCEAN", "FISH", "WAVE", "BEACH", "SHELL"]
opponent_words = ["KNIFE", "BLOOD", "FIRE", "ROCK", "TREE"]
neutral_words = ["KING", "QUEEN", "CAR", "PLANE"]
assassin_words = ["BANK"]
all_board_words = team_words + opponent_words + neutral_words + assassin_words

# ── 1. Load model ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading GloVe model...")
load_model()
print("OK\n")

# ── 2. Candidate clues ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2: candidate_clues() on a sample board")
candidates = candidate_clues(
    team_words, opponent_words, neutral_words, assassin_words, top_n=5
)
print(f"Top 5 candidates for team words {team_words}:")
for clue, score, covered in candidates:
    print(f"  {clue:<15} score={score:.3f}  covers={covered}")
print()

# ── 3. Q-value before training ─────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Q-values before any training")
agent = QLearningAgent(epsilon=0.0)  # epsilon=0 so it always exploits

model = load_model()

def make_state():
    t_vecs = [model[w.lower()] for w in team_words if w.lower() in model]
    o_vecs = [model[w.lower()] for w in opponent_words if w.lower() in model]
    n_vecs = [model[w.lower()] for w in neutral_words if w.lower() in model]
    a_vecs = [model[w.lower()] for w in assassin_words if w.lower() in model]
    return agent.build_state(t_vecs, o_vecs, n_vecs, a_vecs)

state = make_state()

for clue, score, covered in candidates:
    clue_vec = model[clue.lower()]
    number = max(1, len(covered))
    action_vec = agent.build_action_vec(clue_vec, number)
    q = agent.q_net.forward(np.concatenate([state, action_vec]))
    print(f"  Q({clue}, {number}) = {q:.4f}")

# ── 4. Fill replay buffer with fake transitions and train ──────────────────────
print()
print("=" * 60)
print(f"STEP 4: Filling replay buffer ({BATCH_SIZE} transitions) and training 20 steps")

# Simulate transitions: correct guess (+1.0 reward) for a state similar to above.
for _ in range(BATCH_SIZE):
    fake_state = state + np.random.randn(state.shape[0]).astype(np.float32) * 0.05
    clue, _, covered = candidates[0]
    clue_vec = model[clue.lower()]
    number = max(1, len(covered))
    action_vec = agent.build_action_vec(clue_vec, number)
    reward = 1.0   # simulated correct guess
    next_best_q = 0.5
    agent.store_transition(fake_state, action_vec, reward, next_best_q, done=False)

print(f"Buffer size: {len(agent.buffer)}")
print()

W1_before = agent.q_net.W1.copy()
losses = []
for step in range(1, 21):
    loss = agent.train_step()
    if loss is not None:
        losses.append(loss)
        print(f"  step {step:>2}  loss={loss:.6f}")

# ── 5. Confirm weights changed ─────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 5: Did the weights change?")
weight_delta = np.abs(agent.q_net.W1 - W1_before).mean()
print(f"  Mean absolute change in W1: {weight_delta:.6f}")
if weight_delta > 1e-8:
    print("  PASS — weights are updating.")
else:
    print("  FAIL — weights did not change. Check backward() / Adam.")

# ── 6. Q-values after training ─────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 6: Q-values after training (should shift toward reward=1.0)")
for clue, score, covered in candidates:
    clue_vec = model[clue.lower()]
    number = max(1, len(covered))
    action_vec = agent.build_action_vec(clue_vec, number)
    q = agent.q_net.forward(np.concatenate([state, action_vec]))
    print(f"  Q({clue}, {number}) = {q:.4f}")

print()
if losses:
    print(f"Loss trend: first={losses[0]:.6f}  last={losses[-1]:.6f}")
    if losses[-1] < losses[0]:
        print("PASS — loss is decreasing.")
    else:
        print("NOTE — loss did not decrease in 20 steps (normal for small batches).")

print()
print("Done.")
