"""
Q-learning agent for the Codenames AI spymaster.

Architecture
------------
State  : 400-dim vector — mean(team) + mean(opponent) + mean(neutral) + mean(assassin)
         (100d each; empty groups use zeros so the net sees role structure.)
Action : 101-dim vector  — clue word embedding (100d) + normalized number (1d)
Input  : 501-dim         — state concatenated with action
Output : scalar Q-value  — expected cumulative reward for (state, action)

Network: 501 → 128 → 64 → 1  (ReLU activations, Adam optimizer, pure numpy)

Training loop (called from main_window after each human guess):
  1. agent.build_state(...)         → state vector
  2. agent.select_action(state, candidates)   → (clue, number, action_vec)
  3. human guesses → reward computed
  4. agent.store_transition(state, action_vec, reward, next_best_q, done)
  5. agent.train_step()             → one gradient update
  6. agent.decay_epsilon()          → reduce exploration over time
"""

import pickle
from pathlib import Path

import numpy as np

# ── Hyperparameters ────────────────────────────────────────────────────────────
GAMMA = 0.95            # discount factor for future rewards
ALPHA = 1e-3            # learning rate
EPSILON_START = 1.0     # initial exploration rate (100% random at first)
EPSILON_MIN = 0.1       # floor — always keep 10% exploration
EPSILON_DECAY = 0.995   # multiplied each episode
BATCH_SIZE = 32
BUFFER_CAPACITY = 10_000
TARGET_UPDATE_FREQ = 100  # sync target network every N gradient steps

# ── Dimensions ─────────────────────────────────────────────────────────────────
EMBED_DIM = 100         # GloVe 100d
STATE_DIM = EMBED_DIM * 4   # 400: team, opponent, neutral, assassin pools
ACTION_DIM = EMBED_DIM + 1  # 101: clue vec + normalized number
INPUT_DIM = STATE_DIM + ACTION_DIM  # 501

# ── Rewards ────────────────────────────────────────────────────────────────────
REWARD_CORRECT = 1.0
REWARD_NEUTRAL = 0.0
REWARD_OPPONENT = -1.0
REWARD_ASSASSIN = -2.0


# ── Replay Buffer ──────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Circular buffer storing (state, action_vec, reward, next_best_q, done) transitions.

    next_best_q is the pre-computed max Q-value over all candidate actions in the
    next state, stored at transition time so training doesn't need to re-run
    candidate_clues on old board states.
    """

    def __init__(self, capacity: int = BUFFER_CAPACITY):
        self._capacity = capacity
        self._buf: list = []
        self._idx = 0

    def push(
        self,
        state: np.ndarray,
        action_vec: np.ndarray,
        reward: float,
        next_best_q: float,
        done: bool,
    ):
        transition = (state, action_vec, float(reward), float(next_best_q), float(done))
        if len(self._buf) < self._capacity:
            self._buf.append(transition)
        else:
            self._buf[self._idx] = transition
        self._idx = (self._idx + 1) % self._capacity

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self._buf), batch_size, replace=False)
        batch = [self._buf[i] for i in indices]
        states, actions, rewards, next_best_qs, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_best_qs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self._buf)


# ── Q-Network (pure numpy) ─────────────────────────────────────────────────────

class QNetwork:
    """
    Fully-connected network: INPUT_DIM (state+action) → 128 → 64 → 1.
    Uses ReLU activations and an Adam optimizer.
    Implemented in pure numpy — no PyTorch/TensorFlow required.
    """

    def __init__(self):
        # He initialization — recommended for ReLU networks.
        self.W1 = np.random.randn(INPUT_DIM, 128) * np.sqrt(2.0 / INPUT_DIM)
        self.b1 = np.zeros(128)
        self.W2 = np.random.randn(128, 64) * np.sqrt(2.0 / 128)
        self.b2 = np.zeros(64)
        self.W3 = np.random.randn(64, 1) * np.sqrt(2.0 / 64)
        self.b3 = np.zeros(1)

        # Adam optimizer moment estimates (one per parameter tensor).
        self._t = 0
        self._beta1, self._beta2, self._eps_adam = 0.9, 0.999, 1e-8
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._cache: dict = {}

    def _params(self) -> dict:
        return {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "W3": self.W3, "b3": self.b3,
        }

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray | float:
        """
        x: (batch, INPUT_DIM) or (INPUT_DIM,).
        Returns Q-values: (batch, 1) for batches, or a scalar for single inputs.
        """
        squeeze = x.ndim == 1
        if squeeze:
            x = x[np.newaxis, :]  # (1, INPUT_DIM)

        z1 = x @ self.W1 + self.b1          # (batch, 128)
        a1 = np.maximum(0, z1)              # ReLU
        z2 = a1 @ self.W2 + self.b2         # (batch, 64)
        a2 = np.maximum(0, z2)              # ReLU
        q = a2 @ self.W3 + self.b3          # (batch, 1)

        self._cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return float(q[0, 0]) if squeeze else q

    # ── Backward pass (MSE loss) ───────────────────────────────────────────────

    def backward(self, targets: np.ndarray, lr: float = ALPHA):
        """
        Compute gradients via backpropagation and apply an Adam update.
        targets: (batch, 1) array of Bellman target Q-values.
        """
        c = self._cache
        batch = c["x"].shape[0]

        q = c["a2"] @ self.W3 + self.b3     # recompute output (batch, 1)
        d_q = 2 * (q - targets) / batch     # MSE gradient

        # Layer 3 gradients
        dW3 = c["a2"].T @ d_q               # (64, 1)
        db3 = d_q.sum(axis=0)               # (1,)

        # Layer 2 gradients
        d_a2 = d_q @ self.W3.T              # (batch, 64)
        d_z2 = d_a2 * (c["z2"] > 0)        # ReLU gradient
        dW2 = c["a1"].T @ d_z2             # (128, 64)
        db2 = d_z2.sum(axis=0)             # (64,)

        # Layer 1 gradients
        d_a1 = d_z2 @ self.W2.T            # (batch, 128)
        d_z1 = d_a1 * (c["z1"] > 0)        # ReLU gradient
        dW1 = c["x"].T @ d_z1             # (INPUT_DIM, 128)
        db1 = d_z1.sum(axis=0)            # (128,)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
        self._adam_update(grads, lr)

    def _adam_update(self, grads: dict, lr: float):
        self._t += 1
        params = self._params()
        for key, grad in grads.items():
            self._m[key] = self._beta1 * self._m[key] + (1 - self._beta1) * grad
            self._v[key] = self._beta2 * self._v[key] + (1 - self._beta2) * grad ** 2
            m_hat = self._m[key] / (1 - self._beta1 ** self._t)
            v_hat = self._v[key] / (1 - self._beta2 ** self._t)
            params[key] -= lr * m_hat / (np.sqrt(v_hat) + self._eps_adam)


# ── Q-Learning Agent ───────────────────────────────────────────────────────────

class QLearningAgent:
    """
    Wraps QNetwork + ReplayBuffer and exposes a simple interface for the UI:

        state      = agent.build_state(team_vecs, opp_vecs, neu_vecs, ass_vecs)
        clue, num, action_vec = agent.select_action(state, candidates)
        # ... human guesses, reward computed ...
        next_best_q = agent.best_q(next_state, next_candidates)
        agent.store_transition(state, action_vec, reward, next_best_q, done)
        loss = agent.train_step()
        agent.decay_epsilon()
    """

    def __init__(self, epsilon: float = EPSILON_START):
        self.q_net = QNetwork()
        self.target_net = QNetwork()
        self._sync_target()
        self.buffer = ReplayBuffer()
        self.epsilon = epsilon
        self._steps = 0

    def _sync_target(self):
        """Copy q_net weights into target_net for stable Bellman targets."""
        for key in ("W1", "b1", "W2", "b2", "W3", "b3"):
            setattr(self.target_net, key, getattr(self.q_net, key).copy())

    # ── State / action constructors ────────────────────────────────────────────

    def build_state(
        self,
        team_vecs: list[np.ndarray],
        opponent_vecs: list[np.ndarray],
        neutral_vecs: list[np.ndarray],
        assassin_vecs: list[np.ndarray],
    ) -> np.ndarray:
        """
        Build a 400-dim state: mean embedding per role (unrevealed cards only).
        """
        def _mean(vecs: list[np.ndarray]) -> np.ndarray:
            return np.mean(vecs, axis=0) if vecs else np.zeros(EMBED_DIM)

        parts = [
            _mean(team_vecs),
            _mean(opponent_vecs),
            _mean(neutral_vecs),
            _mean(assassin_vecs),
        ]
        return np.concatenate(parts).astype(np.float32)

    def build_action_vec(self, clue_vec: np.ndarray, number: int) -> np.ndarray:
        """Build a 101-dim action vector: clue embedding + normalized number."""
        norm_num = np.array([number / 9.0], dtype=np.float32)
        return np.concatenate([clue_vec.astype(np.float32), norm_num])

    # ── Action selection ───────────────────────────────────────────────────────

    def best_q(
        self,
        state: np.ndarray,
        candidates: list[tuple[str, float, list[str], np.ndarray]],
    ) -> float:
        """
        Return the highest Q-value over all candidate actions in this state.
        Used to compute Bellman targets when storing transitions.
        candidates: list of (clue_word, embed_score, covered_words, clue_vec).
        """
        if not candidates:
            return 0.0
        qs = []
        for _, _, covered, clue_vec in candidates:
            number = max(1, len(covered))
            action_vec = self.build_action_vec(clue_vec, number)
            inp = np.concatenate([state, action_vec])
            qs.append(self.target_net.forward(inp))
        return float(max(qs))

    def select_action(
        self,
        state: np.ndarray,
        candidates: list[tuple[str, float, list[str], np.ndarray]],
    ) -> tuple[str, int, np.ndarray] | None:
        """
        Epsilon-greedy action selection.

        candidates : list of (clue_word, embed_score, covered_words, clue_vec)
                     as returned by candidate_clues() with clue_vec appended.
        Returns (clue_word, number, action_vec), or None if no candidates.
        """
        if not candidates:
            return None

        if np.random.rand() < self.epsilon:
            # Explore: pick a random candidate.
            clue_word, _, covered, clue_vec = candidates[np.random.randint(len(candidates))]
        else:
            # Exploit: pick the candidate with the highest Q-value.
            best_q_val = -np.inf
            best = candidates[0]
            for candidate in candidates:
                _, _, covered, clue_vec = candidate
                number = max(1, len(covered))
                action_vec = self.build_action_vec(clue_vec, number)
                q = self.q_net.forward(np.concatenate([state, action_vec]))
                if q > best_q_val:
                    best_q_val = q
                    best = candidate
            clue_word, _, covered, clue_vec = best

        number = max(1, len(covered))
        action_vec = self.build_action_vec(clue_vec, number)
        return clue_word, number, action_vec

    # ── Training ───────────────────────────────────────────────────────────────

    def store_transition(
        self,
        state: np.ndarray,
        action_vec: np.ndarray,
        reward: float,
        next_best_q: float,
        done: bool,
    ):
        self.buffer.push(state, action_vec, reward, next_best_q, done)

    def train_step(self) -> float | None:
        """
        Sample a mini-batch and do one gradient update.
        Returns the MSE loss, or None if the buffer isn't full enough yet.

        Bellman target: r + γ * max_Q(s', a')  if not done
                        r                        if done
        """
        if len(self.buffer) < BATCH_SIZE:
            return None

        states, actions, rewards, next_best_qs, dones = self.buffer.sample(BATCH_SIZE)

        # Bellman targets using the stored next-state best Q (from target_net).
        targets = (rewards + GAMMA * next_best_qs * (1.0 - dones)).reshape(-1, 1)

        inputs = np.concatenate([states, actions], axis=1)
        self.q_net.forward(inputs)
        self.q_net.backward(targets)

        self._steps += 1
        if self._steps % TARGET_UPDATE_FREQ == 0:
            self._sync_target()

        # Return current loss for logging.
        q_pred = self.q_net.forward(inputs)
        return float(np.mean((q_pred - targets) ** 2))

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | Path, log_suffix: str = ""):
        data = {
            "params": {k: getattr(self.q_net, k) for k in ("W1", "b1", "W2", "b2", "W3", "b3")},
            "epsilon": self.epsilon,
            "steps": self._steps,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        if log_suffix:
            print(f"[AI] Model saved | {log_suffix} → {path}")
        else:
            print(f"[AI] Model saved to {path}")

    def load(self, path: str | Path) -> bool:
        with open(path, "rb") as f:
            data = pickle.load(f)
        w1 = data["params"]["W1"]
        if w1.shape[0] != INPUT_DIM:
            print(
                f"[AI] Ignoring incompatible checkpoint (input dim {w1.shape[0]}, "
                f"need {INPUT_DIM}) — starting fresh weights."
            )
            return False
        for k, v in data["params"].items():
            setattr(self.q_net, k, v)
        self.epsilon = data["epsilon"]
        self._steps = data["steps"]
        self._sync_target()
        print(f"[AI] Model loaded from {path}")
        return True
