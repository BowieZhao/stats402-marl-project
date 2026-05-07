# MARL Communication Project — STATS 402 Final Submission

**Authors:** Bowie Zhao (boheng.zhao@dukekunshan.edu.cn), Kewei Zhang (kewei.zhang@dukekunshan.edu.cn)

This repository contains the code, configuration, and analysis scripts for our project on **communication-driven multi-agent cooperation under partial observability**, evaluated on the MPE2 `simple_world_comm_v3` environment.

The full method (E1) reaches a peak smoothed team reward of **134**, compared to **101** (E3, fixed α) and **61** (E2, no communication). MAPPO outperforms DQN, DDPG, and PSO baselines under the same condition.

------

## 1. Environment & Dependencies

We use Python 3.10 with the following key packages. Install via:

```bash
conda create -n stats402 python=3.10
conda activate stats402
pip install torch numpy pandas matplotlib pillow
pip install pettingzoo[mpe]
pip install mpe2
```

**Dataset.** No external dataset is required. All data is generated on-the-fly by the MPE2 simulator. The environment used is `simple_world_comm_v3` from the [MPE2](https://github.com/Farama-Foundation/MPE2) library, accessible via `from mpe2 import simple_world_comm_v3`. We use the default configuration: `num_good=2, num_adversaries=4, num_forests=2, num_food=2, num_obstacles=1, max_cycles=50`.

------

## 2. Repository Structure

```
.
├── algorithms/              # Algorithm implementations
│   ├── mappo.py             # MAPPO with AdversaryAlphaActor + plan_bias
│   ├── dqn.py               # DQN baseline (role-shared Q-network)
│   ├── ddpg.py              # DDPG baseline (Gumbel-softmax discrete)
│   └── pso.py               # PSO baseline (parameter-vector swarm)
├── outputs/                 # Generated during training
│   ├── logs/                # CSV training logs (one per run)
│   └── models/              # Checkpoints per run (ep000500, ..., final)
├── config.py                # Hyperparameter dataclass
├── envs.py                  # WorldCommEnv wrapper (obs augmentation, reward stack, plan_bias)
├── experiment.py            # Runner: training loop, evaluation, logging
├── frozen_policy.py         # Optional frozen good-policy loader (not required)
├── main.py                  # Entry point with --algo / --condition / --seed dispatch
├── extract_alpha_real.py    # Loads checkpoints, runs eval, plots α distribution
├── visualize_policy.py      # Loads final ckpt, renders / saves GIFs of trajectories
├── study_a_plot.py          # Plots Study A (E1 vs E2 vs E3)
├── study_b_plot.py          # Plots Study B (MAPPO vs DQN/DDPG/PSO)
├── alpha_real_distribution.png    # Generated: α boxplot per checkpoint
├── alpha_real_mean_trajectory.png # Generated: α mean ± std over training
├── alpha_real_within_episode.png  # Generated: α per-step within episode
├── alpha_real_raw.npz       # Generated: raw α values for replotting
├── study_a_E1_vs_E2_vs_E3.png     # Generated: Study A figure
└── study_b_algo_comparison.png    # Generated: Study B figure
```

### File-by-File Function Description

**Core framework:**

- **`config.py`** — Defines the `Config` dataclass with all hyperparameters (PPO settings, reward coefficients, message dimension, α initialization, experiment condition `E1_full / E2_no_comm / E3_no_alpha`, etc.). Helper properties `message_enabled`, `plan_bias_enabled`, `alpha_is_learnable` derive condition flags from the `condition` field.
- **`envs.py`** — Wraps `simple_world_comm_v3` with our extensions: augmented observations (leader gets last-message one-hot + forest flags; adversary gets current-message one-hot), the structured reward stack (distance / capture / encircle / progress / action-matches-bias / thrash), the rule-based message decoder `compute_plan_bias()` for the ABCD semantics, and tracking of caught goods to mask them out from subsequent reward and observations. Provides `make_env(config)` factory.
- **`experiment.py`** — `Runner` class. Drives the training loop, calls `agent.select_actions / observe / update / end_episode / save`, runs periodic evaluation episodes, logs train/eval metrics to CSV, and saves model checkpoints every `save_every` episodes.
- **`main.py`** — Argparse entry point. Builds `Config`, creates env, dispatches to the right `Agent` class based on `--algo`, and starts the runner.

**Algorithms (`algorithms/`):**

- **`mappo.py`** — Our MAPPO implementation. Defines `ActorNet`, `AdversaryAlphaActor` (the α-gated mixing actor that combines own-policy logits with the message-derived `plan_bias`), `CriticNet`, `RoleGroup` (parameter-shared per role), and `MAPPOAgent` orchestrating training, GAE advantage estimation, and PPO updates with clipping.
- **`dqn.py`** — Independent DQN baseline. Each role has its own Q-network with shared parameters across agents in that role; ε-greedy exploration, replay buffer, hard target-network update.
- **`ddpg.py`** — DDPG baseline. Discrete actions handled via Gumbel-softmax on actor output; soft target updates; one-hot action critic input.
- **`pso.py`** — Particle swarm optimization over the parameters of the role-shared MLP policies. Each particle is one full parameter vector; fitness is team reward over one episode; PSO update happens once per swarm generation.

**Analysis & plotting:**

- **`extract_alpha_real.py`** — Loads each saved adversary checkpoint, runs 20 evaluation episodes through it, records the actual α value (sigmoid(W·h + b), not just the bias term) at every adversary step, and produces three figures: α distribution per checkpoint (boxplot), α mean trajectory with ±1σ band, and within-episode α traces. Also saves raw data to `alpha_real_raw.npz`.
- **`visualize_policy.py`** — Loads the final checkpoint and runs N evaluation episodes with rendering. With `--save_gif` flag, saves animated GIFs of the resulting trajectories. Prints message symbols (A/B/C/D), α values, and catch events at each step.
- **`study_a_plot.py`** — Reads CSV logs for E1, E2, E3 and produces the Study A figure: smoothed learning curves (window=50) + peak performance bar chart.
- **`study_b_plot.py`** — Reads CSV logs for MAPPO, DQN, DDPG, PSO and produces the Study B algorithm-comparison figure with the same conventions.

------

## 3. Reproducing the Results

### Step 1: Train MAPPO under three conditions (Study A)

```bash
python main.py --algo mappo --condition E1_full     --total_episodes 4000 --seed 42 --run_name final
python main.py --algo mappo --condition E2_no_comm  --total_episodes 4000 --seed 42 --run_name final
python main.py --algo mappo --condition E3_no_alpha --total_episodes 4000 --seed 42 --run_name final
```

Each run takes approximately 14 minutes on CPU. Outputs land in:

- `outputs/logs/final_mappo_<condition>_s42.csv`
- `outputs/models/final_mappo_<condition>_s42/{ep000500, ..., final}/`

### Step 2: Train baselines for algorithm comparison (Study B)

```bash
python main.py --algo dqn  --condition E1_full --total_episodes 4000 --seed 42 --run_name baseline
python main.py --algo ddpg --condition E1_full --total_episodes 4000 --seed 42 --run_name baseline
python main.py --algo pso  --condition E1_full --total_episodes 4000 --seed 42 --run_name baseline
```

### Step 3: Plot Study A and Study B

Edit the hardcoded `LOG_DIR` at the top of each plot script to point to your `outputs/logs/` path, then:

```bash
python study_a_plot.py
python study_b_plot.py
```

### Step 4: Analyze learned α (state-dependent trust)

```bash
python extract_alpha_real.py --run_dir outputs/models/final_mappo_E1_full_s42
```

Produces `alpha_real_distribution.png` (the headline α figure), plus `alpha_real_mean_trajectory.png`, `alpha_real_within_episode.png`, and the raw `alpha_real_raw.npz`.

### Step 5 (optional): Visualize trained behavior

```bash
python visualize_policy.py --save_gif --n_episodes 5
```

Generates GIFs named `episode_<N>_seed<X>_catches<Y>.gif`. The number of catches in the filename helps pick the most successful trajectories.

------

## 4. Experiment Conditions

| Condition     | Message channel | plan_bias decoder | α status     |
| ------------- | --------------- | ----------------- | ------------ |
| `E1_full`     | enabled         | enabled           | learnable    |
| `E2_no_comm`  | zeroed          | disabled          | N/A          |
| `E3_no_alpha` | enabled         | enabled           | fixed at 0.5 |

These conditions are selected via the `--condition` flag and are interpreted by `Config` properties consumed throughout `envs.py` and `algorithms/mappo.py`.

------

## 5. Notes

- Single-seed results (seed=42) are reported. Multi-seed averaging is left for future work due to compute constraints.
- The reward stack contains a teaching term `R_action_matches_plan_bias` that is structurally biased toward conditions with active communication; we discuss this trade-off in the paper's Limitations section.
- The MPE2 leader's discrete action space is 20 (4 messages × 5 moves). Our `envs.py` decodes this back into separate message + move channels for obs augmentation and reward computation.
