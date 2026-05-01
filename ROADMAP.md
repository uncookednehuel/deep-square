# Deep-Square Roadmap (EnvPool + Self-Play PPO + CUDA + Rendering)

This roadmap is written for your current codebase and learning goals. It includes concrete implementation details and the reasoning behind each step, so you can learn the "why" while you build.

## Snapshot of the current project
- [environment.py](environment.py): single-player `gymnasium` env on an 8x8 grid, action masking is already implemented.
- [train.py](train.py): PPO with a single actor-critic network; runs one environment on CPU/GPU.
- [world.py](world.py): incomplete stub for future map generation.

This is a good start: you already have action masking, PPO logic, and a simple grid state. The next step is to make the game multi-player, then scale it up with EnvPool and CUDA.

---

## Phase 0: Clarify the game model (before any speed work)
**Goal:** lock down the rules, state encoding, and rewards so self-play has a stable learning target.

### 0.1 Define the two-player state explicitly
**Why:** A shared policy needs a consistent view of "me vs opponent" regardless of which player is acting.

Recommended channel layout (5 channels, 8x8 each):
1. My cities (1 or 0)
2. Opponent cities (1 or 0)
3. Neutral cities (1 or 0)
4. My units (1 or 0)
5. Opponent units (1 or 0)

Optional extras (later): tech level, city level, turn count, or a "current player" scalar.

### 0.2 Define the action semantics
**Why:** If action semantics change between Python and EnvPool, you will spend a lot of time debugging.

Keep your existing flattened action space:
- Action = `(src_x, src_y, action_type, tgt_x, tgt_y)`
- Flatten to a single integer in a fixed order.

Document the exact flattening order (the order of dimensions), so the C++ EnvPool version matches exactly.

### 0.3 Make reward shaping minimal and consistent
**Why:** Self-play learns from relative advantage, and overly dense rewards can be exploited in weird ways.

Start with a sparse, clean signal:
- +1 for capturing an enemy or neutral city
- -1 for losing your own city (if you add that later)
- Small step penalty (like -0.01) to prefer shorter wins

Avoid large shaped rewards until the game rules are stable.

---

## Phase 1: Two-player environment (still in Python)
**Goal:** a correct multi-agent environment with deterministic rules before you port to EnvPool.

### 1.1 Convert to a turn-based two-player loop
**Why:** Turn-based games are easier to debug than simultaneous actions.

Implementation details:
- Add `current_player` (0 or 1) to the environment state.
- On each `step(action)`, apply the action for `current_player`, then switch turns.
- The observation returned should always be from the perspective of `current_player`.
- `get_valid_mask()` must be player-specific, based on that player's units and cities.

### 1.2 Add a "perspective transform"
**Why:** A single neural net can be used by both players if observations are always from "my" perspective.

Pseudo-code idea:
```python
# state shape: (channels, 8, 8)
# channels: p0 cities, p1 cities, neutral cities, p0 units, p1 units

def to_player_view(state, player_id):
    if player_id == 0:
        return state
    # swap p0 and p1 channels
    p0_city, p1_city, neutral, p0_unit, p1_unit = state
    return np.stack([p1_city, p0_city, neutral, p1_unit, p0_unit], axis=0)
```

### 1.3 Introduce a simple "opponent" in Python
**Why:** You need a baseline to verify the two-player logic before PPO.

Start with one of these:
- Random valid moves (uses `get_valid_mask`)
- Rule-based (capture city if possible, else random)

If the PPO agent cannot beat this, the environment likely has a bug.

---

## Phase 2: Rebuild the environment in EnvPool
**Goal:** move the environment logic into EnvPool for fast, parallel rollouts.

EnvPool is fast because environments run in C++ and are vectorized. The cost is that you must implement the environment in C++ and build a Python extension.

### 2.1 Plan the API boundary
**Why:** You want the Python training loop to be stable while the backend changes.

Keep the Python API the same:
- `obs = env.reset()` returns shape `(num_envs, C, 8, 8)`
- `obs, reward, terminated, truncated, info = env.step(action_batch)`
- `action_batch` is a vector of `int32` action IDs
- `action_mask` can be an `info` field or a separate method

### 2.2 Create a C++ EnvPool skeleton
**Why:** This is where the real speedup comes from.

High-level pieces you will implement:
- A `State` struct holding the grid, units, cities, and current player
- `Reset()` to create a new random map and initial units
- `Step(action)` to apply the current player's action, swap players, and produce reward
- `Observation()` to write the observation to the output buffer
- `ActionMask()` (optional but strongly recommended) to produce a boolean mask

Keep the observation and action format identical to the Python version.

### 2.3 Build and register the EnvPool extension
**Why:** This makes your environment available via `envpool.make`.

General steps (details depend on EnvPool's custom env template):
- Create a `cpp/` folder with your env files and a `CMakeLists.txt`
- Use EnvPool's custom env template for build settings and module registration
- Expose the environment with a unique name like `PolytopiaMini-v0`
- Build the extension and import it in Python

If you have not used C++ before, treat this as a learning mini-project. Start with a minimal, working env (even if only 1 action type works) and build from there.

### 2.4 Validate parity with Python
**Why:** Speed is meaningless if the rules drift.

Run a short script that:
- Seeds both Python and EnvPool versions
- Plays a fixed sequence of actions
- Confirms the resulting states match exactly

If they do not match, fix the EnvPool logic before training.

---

## Phase 3: Self-play PPO with adversarial networks
**Goal:** train a policy that improves by playing against itself and stronger past versions.

### 3.1 Decide your self-play strategy
**Why:** Self-play can collapse if the opponent is always identical.

Beginner-friendly approach:
- Maintain one "learner" network and one "opponent" network
- The opponent is a frozen snapshot of a previous learner
- Every N updates, replace the opponent if the learner beats it by a threshold

This is simple and stable, and you can expand to a league later.

### 3.2 Update the training loop for turn-based play
**Why:** PPO expects trajectories from a single policy; self-play requires careful bookkeeping.

Key rules:
- Use a shared policy for both players, or use two distinct policies.
- If using one shared policy, apply the perspective transform so the policy always sees "my" view.
- Store transitions only for the learning policy (or store both, but flip rewards for the opponent).

Pseudo-structure:
```python
for t in range(T):
    obs = to_player_view(raw_obs, current_player)
    mask = get_mask_for_current_player(...)

    if current_player == learner_id:
        action = learner.act(obs, mask)
        store_transition(...)
    else:
        action = opponent.act(obs, mask)

    raw_obs, reward, done = env.step(action)

    # If reward is from the current player perspective, flip when needed
```

### 3.3 Use GAE and batch rollouts
**Why:** GAE stabilizes PPO in sparse reward games.

Implementation details to add:
- Generalized Advantage Estimation (GAE): `adv = delta + gamma * lambda * next_adv`
- Collect rollouts from many parallel EnvPool envs (e.g., 128 or 256)
- Flatten `(T, N, ...)` into batches for the PPO update

### 3.4 Add an opponent pool (optional but strong)
**Why:** A single opponent snapshot can be exploited; a small pool adds robustness.

Simple pool strategy:
- Keep the last K opponent snapshots (e.g., K=5)
- Sample the opponent uniformly each episode
- Keep a win-rate table to avoid re-using weak opponents too often

---

## Phase 4: CUDA integration for maximum training speed
**Goal:** keep the GPU busy and minimize CPU-GPU transfer overhead.

### 4.1 Keep policy inference batched
**Why:** The GPU is fast only when work is large and contiguous.

Steps:
- Use many parallel envs (`num_envs` in EnvPool)
- Run a single forward pass for all envs at each step
- Avoid Python loops over envs

### 4.2 Avoid slow CPU-to-GPU transfers
**Why:** Transfers can dominate runtime.

Best practices:
- Convert observations to `torch` tensors once per step, not per env
- Use `torch.as_tensor(obs, device=device)` to avoid extra copies
- If you have to copy, use `non_blocking=True` with pinned memory

### 4.3 Use mixed precision carefully
**Why:** Mixed precision can speed up training but can destabilize PPO if gradients are noisy.

Suggested path:
- Train first in full precision until learning is stable
- Then experiment with `torch.cuda.amp.autocast` and `GradScaler`
- If the policy collapses, return to full precision

### 4.4 Enable compiler optimizations (optional)
**Why:** PyTorch 2+ can speed up models with `torch.compile`.

Steps:
- Wrap the model with `torch.compile(model)` once the code is stable
- Benchmark before and after; keep the faster version

---

## Phase 5: Rendering interface (text and UI options)
**Goal:** make the environment debuggable and fun to watch.

### 5.1 Text rendering inside `render()`
**Why:** Text output is lightweight and works everywhere.

Example idea:
- Use a 2D ASCII grid
- Map: `.` empty, `C` my city, `c` enemy city, `U` my unit, `u` enemy unit
- Print the grid plus current player and turn number

### 5.2 A simple windowed UI (suggested)
**Why:** Visual feedback makes debugging and demonstrations easier.

Recommended starter UI options:
1. **pygame**: easiest 2D grid rendering and input handling
2. **pyglet**: used by Gym, lightweight OpenGL window
3. **tkinter**: built-in, minimal dependencies

Minimal pygame approach:
- Each tile is a colored square (different colors for cities, units, neutral)
- Draw grid lines and basic icons
- Refresh at a fixed FPS to avoid slowing training

Run rendering in a separate "evaluation" script so training stays fast.

---

## Milestones (what "done" looks like)
1. **Correct two-player Python env** with action masking and perspective transform
2. **EnvPool build working** and passing parity tests vs Python env
3. **Self-play PPO** beats random and rule-based opponents
4. **CUDA training** runs with large batch rollouts and stable learning curves
5. **Text render + pygame viewer** for human inspection

---

## Suggested next implementation steps (concrete)
1. Update the Python env to two-player, add `current_player`, and implement `to_player_view`.
2. Write a small script that plays random-vs-random and validates rewards/termination.
3. Refactor PPO to support batched rollouts and GAE (even before EnvPool).
4. Start a minimal EnvPool C++ env that only supports move actions, then add the rest.
5. Add the self-play opponent snapshot logic once EnvPool rollouts are stable.

---

If you want, I can help you design the two-player state encoding or outline the EnvPool C++ scaffolding once you are ready to implement it.