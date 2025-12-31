"""
RL + GA for initial population generation.

- RL (Double Q-Learning) learns to place machines (quadrant + rotation).
- Each episode builds ONE full layout (chromosome).
- Reward = -fitness(layout) from GeneticAlgorithm.fitness_function.
- After training, we sample layouts from the greedy policy and use them
  as part of the GA's initial population.

Features:
- 4 rotation types: 0, 90, 180, 270 degrees (4 quadrants x 4 rotations = 16 actions).
- Q-table save/load so you can reuse the trained agent without retraining.
- Utility to compare GA convergence with and without RL initial population.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from func.datastruct import Point, Machine
from algo.gen_algo_init import GeneticAlgorithm


# =====================================================================
# 1. Double Q-Learning agent with save/load
# =====================================================================

class DoubleQLearningAgent:
    """
    Tabular Double Q-Learning with two Q-tables (QA and QB).

    - Actions selected epsilon-greedily using QA + QB.
    - Updates randomly pick one table to update using the other as bootstrap.
    - Can save/load Q-tables to/from a .npz file for reuse.
    """
    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 0.01,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.QA = np.zeros((state_size, action_size), dtype=np.float32)
        self.QB = np.zeros((state_size, action_size), dtype=np.float32)

    # ----------------- Action selection -----------------

    def _greedy_action(self, state: int) -> int:
        q_sum = self.QA[state] + self.QB[state]
        return int(np.argmax(q_sum))

    def get_action(self, state: int) -> int:
        """Epsilon-greedy action (used in training)."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return self._greedy_action(state)

    def get_greedy_action(self, state: int) -> int:
        """Pure exploitation (used during initial population generation)."""
        return self._greedy_action(state)

    # ----------------- Learning update -----------------

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        Double Q-Learning update:

        With 0.5 probability, update QA using QB as bootstrap;
        otherwise, update QB using QA as bootstrap.
        """
        if done:
            target = reward
        else:
            if np.random.rand() < 0.5:
                # Update QA: argmax over QA, bootstrap with QB
                next_a = int(np.argmax(self.QA[next_state]))
                target = reward + self.gamma * self.QB[next_state, next_a]
            else:
                # Update QB: argmax over QB, bootstrap with QA
                next_a = int(np.argmax(self.QB[next_state]))
                target = reward + self.gamma * self.QA[next_state, next_a]

        if np.random.rand() < 0.5:
            td_error = target - self.QA[state, action]
            self.QA[state, action] += self.lr * td_error
        else:
            td_error = target - self.QB[state, action]
            self.QB[state, action] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ----------------- Save / load Q-tables -----------------

    def save_qtable(self, filepath: str):
        """
        Save Q-tables and hyperparameters to a .npz file.
        """
        np.savez(
            filepath,
            QA=self.QA,
            QB=self.QB,
            state_size=self.state_size,
            action_size=self.action_size,
            lr=self.lr,
            gamma=self.gamma,
            epsilon=self.epsilon,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay=self.epsilon_decay,
        )
        print(f"[DoubleQLearningAgent] Saved Q-table to {filepath}")

    @classmethod
    def load_from_qtable(cls, filepath: str) -> "DoubleQLearningAgent":
        """
        Load Q-tables and hyperparameters from a .npz file.
        """
        data = np.load(filepath, allow_pickle=True)
        state_size = int(data["state_size"])
        action_size = int(data["action_size"])
        lr = float(data["lr"])
        gamma = float(data["gamma"])
        epsilon_start = float(data["epsilon_start"])
        epsilon_end = float(data["epsilon_end"])
        epsilon_decay = float(data["epsilon_decay"])

        agent = cls(
            state_size=state_size,
            action_size=action_size,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
        )
        agent.QA = data["QA"]
        agent.QB = data["QB"]
        agent.epsilon = float(data["epsilon"])

        print(f"[DoubleQLearningAgent] Loaded Q-table from {filepath}")
        return agent


# =====================================================================
# 2. Environment that builds robot layouts sequentially
# =====================================================================

class LayoutRLEnvironment:
    """
    Environment:

    - Episode = build one chromosome (layout) for all machines.
    - State = index of current machine being placed (0..num_machines-1),
              plus a terminal state = num_machines.
    - Action = choose one of several placement patterns:
        * Quadrant (4 choices).
        * Rotation in {0, 90, 180, 270} degrees (4 choices).
      => 4 * 4 = 16 discrete actions by default.

    At the end of the episode:
        reward = -fitness(chromosome)  (lower fitness => higher reward).
    """

    def __init__(self, ga: GeneticAlgorithm, action_size: int = 16):
        self.ga = ga
        self.machines: List[Machine] = ga.machines
        self.num_machines = len(self.machines)

        self.action_size = action_size  # e.g. 16
        self.state_size = self.num_machines + 1  # +1 terminal
        self.terminal_state = self.num_machines

        self.workspace_bounds = ga.workspace_bounds
        self.min_x, self.max_x, self.min_y, self.max_y = self.workspace_bounds

        # per-episode state
        self.current_chromosome: Optional[np.ndarray] = None
        self.current_machine_idx: int = 0
        self.done: bool = False
        self.last_fitness: float = float("inf")

        # 4 quadrants
        self.quadrants = [
            (self.min_x, 0.0,        self.min_y, 0.0),        # Q1
            (0.0,        self.max_x, self.min_y, 0.0),        # Q2
            (self.min_x, 0.0,        0.0,        self.max_y), # Q3
            (0.0,        self.max_x, 0.0,        self.max_y), # Q4
        ]

        # 4 rotations
        self.rot_options = [0.0, 90.0, 180.0, 270.0]

    # ----------------- Core API -----------------

    def reset(self) -> int:
        """Start a new episode."""
        self.current_chromosome = np.zeros(self.num_machines * 3, dtype=np.float32)
        self.current_machine_idx = 0
        self.done = False
        self.last_fitness = float("inf")
        return 0  # initial state: placing machine 0

    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Take an action (place current machine), then:
        - If layout complete: compute fitness, return final reward.
        - Else: move to next machine (intermediate reward = 0).
        """
        if self.done:
            raise RuntimeError("step() called on finished episode")

        self._place_machine(self.current_machine_idx, action)
        self.current_machine_idx += 1

        # If all machines are placed, evaluate layout
        if self.current_machine_idx >= self.num_machines:
            fitness = self.ga.fitness_function(self.current_chromosome)
            self.last_fitness = fitness

            if not np.isfinite(fitness):
                reward = -1e6
            else:
                # scale fitness so rewards are not huge
                reward = -float(fitness) / 100.0

            self.done = True
            next_state = self.terminal_state
            return next_state, reward, True

        # Still placing machines, no immediate reward
        next_state = self.current_machine_idx
        reward = 0.0
        done = False
        return next_state, reward, done

    # ----------------- Helpers -----------------

    def _place_machine(self, machine_idx: int, action: int):
        """
        Interpret discrete action as:
          - quadrant_id in {0,1,2,3}
          - rotation in {0, 90, 180, 270}
        """
        machine = self.machines[machine_idx]
        w, h = machine.width, machine.height

        # Clamp action
        action = int(action)
        action = max(0, min(self.action_size - 1, action))

        # Map to quadrant and rotation id
        # With 16 actions: 4 quadrants x 4 rotations
        quadrant_id = action // 4       # 0..3
        rot_id = action % 4             # 0..3

        # Safety on quadrant
        quadrant_id = max(0, min(3, quadrant_id))
        qx1, qx2, qy1, qy2 = self.quadrants[quadrant_id]

        # Sample (x, y) inside that quadrant with margins
        x = np.random.uniform(qx1 + w / 2.0, qx2 - w / 2.0)
        y = np.random.uniform(qy1 + h / 2.0, qy2 - h / 2.0)

        rotation = self.rot_options[rot_id]

        idx = machine_idx * 3
        self.current_chromosome[idx:idx+3] = np.array([x, y, rotation], dtype=np.float32)


# =====================================================================
# 3. Training loop (with optional Q-table reuse)
# =====================================================================

def train_layout_rl_agent(
    machines: List[Machine],
    sequence: List[int],
    robot_position: Point,
    workspace_bounds: Tuple[float, float, float, float],
    num_episodes: int = 10000,
    max_steps_per_episode: Optional[int] = None,
    rl_action_size: int = 16,
    qtable_path: Optional[str] = None,
    reuse_if_exists: bool = True,
) -> Tuple[DoubleQLearningAgent, Dict]:
    """
    Train a Double Q-Learning agent to construct good layouts.

    If qtable_path is provided and reuse_if_exists=True and the file exists,
    we load the agent from disk and skip training.

    Returns:
        agent     : trained (or loaded) DoubleQLearningAgent
        info_dict : training statistics and metadata
    """
    ga = GeneticAlgorithm(
        machines=machines,
        sequence=sequence,
        robot_position=robot_position,
        workspace_bounds=workspace_bounds,
    )

    env = LayoutRLEnvironment(ga, action_size=rl_action_size)

    # If Q-table exists and we want to reuse it, load and skip training
    if qtable_path is not None and reuse_if_exists and os.path.exists(qtable_path):
        agent = DoubleQLearningAgent.load_from_qtable(qtable_path)

        # Sanity check: shapes must match current environment
        if agent.state_size != env.state_size or agent.action_size != rl_action_size:
            raise ValueError(
                f"Loaded Q-table has state_size={agent.state_size}, action_size={agent.action_size}, "
                f"but env has state_size={env.state_size}, action_size={rl_action_size}."
            )

        info = {
            "episode_best_fitness": [],
            "global_best_fitness": None,
            "global_best_layout": None,
            "loaded_from_qtable": True,
            "qtable_path": qtable_path,
        }
        return agent, info

    # Otherwise, create a fresh agent and train it
    agent = DoubleQLearningAgent(
        state_size=env.state_size,
        action_size=rl_action_size,
        lr=0.01,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.999,
    )

    episode_best_fitness = []
    global_best_fitness = float("inf")
    global_best_layout: Optional[np.ndarray] = None

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        total_reward = 0.0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if max_steps_per_episode is not None and steps >= max_steps_per_episode:
                done = True

        fit = env.last_fitness
        episode_best_fitness.append(float(fit))

        if np.isfinite(fit) and fit < global_best_fitness:
            global_best_fitness = fit
            global_best_layout = env.current_chromosome.copy()

        agent.decay_epsilon()

        if (ep + 1) % 100 == 0:
            print(
                f"[RL InitPop] Episode {ep+1}/{num_episodes} "
                f"Fitness={fit:.2f}  Best={global_best_fitness:.2f}  "
                f"Eps={agent.epsilon:.3f}"
            )

    # Optionally save Q-table
    if qtable_path is not None:
        agent.save_qtable(qtable_path)

    info = {
        "episode_best_fitness": episode_best_fitness,
        "global_best_fitness": global_best_fitness,
        "global_best_layout": global_best_layout,
        "loaded_from_qtable": False,
        "qtable_path": qtable_path,
    }
    return agent, info


# =====================================================================
# 4. Generate RL-optimized initial population for GA
# =====================================================================

def generate_initial_population_with_rl(
    machines: List[Machine],
    sequence: List[int],
    robot_position: Point,
    workspace_bounds: Tuple[float, float, float, float],
    ga_population_size: int = 500,
    num_rl_solutions: int = 300,
    num_episodes: int = 10000,
    qtable_path: Optional[str] = None,
    reuse_if_exists: bool = True,
) -> Tuple[List[np.ndarray], DoubleQLearningAgent, Dict]:
    """
    Train (or load) an RL agent and use it to generate an initial
    population for the Genetic Algorithm:

    - Train Double Q-Learning to build layouts (or load from Q-table).
    - Sample `num_rl_solutions` layouts using the greedy policy.
    - Fill the rest of the GA population with random layouts for diversity.
    """
    # 1. Train or load agent
    agent, train_info = train_layout_rl_agent(
        machines=machines,
        sequence=sequence,
        robot_position=robot_position,
        workspace_bounds=workspace_bounds,
        num_episodes=num_episodes,
        rl_action_size=16,          # 4 quadrants x 4 rotations
        qtable_path=qtable_path,
        reuse_if_exists=reuse_if_exists,
    )

    # 2. GA instance for fitness / random population
    ga = GeneticAlgorithm(
        machines=machines,
        sequence=sequence,
        robot_position=robot_position,
        workspace_bounds=workspace_bounds,
    )

    env = LayoutRLEnvironment(ga, action_size=agent.action_size)

    # Greedy when generating initial solutions
    agent.epsilon = 0.0

    rl_layouts: List[np.ndarray] = []

    while len(rl_layouts) < num_rl_solutions:
        state = env.reset()
        done = False

        while not done:
            action = agent.get_greedy_action(state)
            next_state, reward, done = env.step(action)
            state = next_state

        if np.isfinite(env.last_fitness):
            rl_layouts.append(env.current_chromosome.copy())

    remaining = ga_population_size - len(rl_layouts)

    if remaining < 0:
        initial_population = rl_layouts[:ga_population_size]
    else:
        random_layouts = ga.create_initial_population()
        if len(random_layouts) < remaining:
            raise ValueError(
                f"GA created only {len(random_layouts)} layouts, "
                f"but {remaining} are required to fill the population."
            )
        initial_population = rl_layouts + random_layouts[:remaining]

    if len(initial_population) != ga_population_size:
        raise ValueError(
            f"Initial population size mismatch: "
            f"{len(initial_population)} != {ga_population_size}"
        )

    return initial_population, agent, train_info


# =====================================================================
# 5. Compare GA convergence: random vs RL initial population
# =====================================================================

def compare_ga_convergence_with_and_without_rl(
    machines: List[Machine],
    sequence: List[int],
    robot_position: Point,
    workspace_bounds: Tuple[float, float, float, float],
    ga_population_size: int = 500,
    num_rl_solutions: int = 300,
    num_episodes: int = 10000,
    qtable_path: Optional[str] = "layout_qtable_initpop.npz",
    reuse_if_exists: bool = True,
    save_plot_path: str = "ga_convergence_compare.png",
):
    """
    Run GA twice:
    1) initial_population=None (pure random).
    2) initial_population generated by RL agent.

    Then plot best fitness vs generation for both runs and save the figure.
    """
    # --- GA baseline: random initial population ---
    print("\n[Baseline] Running GA with random initial population...")
    ga_random = GeneticAlgorithm(
        machines=machines,
        sequence=sequence,
        robot_position=robot_position,
        workspace_bounds=workspace_bounds,
    )
    final_machines_rand, best_fit_rand, results_rand = ga_random.optimize(
        initial_population=None
    )

    # --- GA with RL-based initial population ---
    print("\n[RL Init] Generating initial population with RL agent...")
    initial_pop_rl, agent, train_info = generate_initial_population_with_rl(
        machines=machines,
        sequence=sequence,
        robot_position=robot_position,
        workspace_bounds=workspace_bounds,
        ga_population_size=ga_population_size,
        num_rl_solutions=num_rl_solutions,
        num_episodes=num_episodes,
        qtable_path=qtable_path,
        reuse_if_exists=reuse_if_exists,
    )

    print("[RL Init] Running GA with RL-generated initial population...")
    ga_rl = GeneticAlgorithm(
        machines=machines,
        sequence=sequence,
        robot_position=robot_position,
        workspace_bounds=workspace_bounds,
    )
    final_machines_rl, best_fit_rl, results_rl = ga_rl.optimize(
        initial_population=initial_pop_rl
    )

    # --- Plot convergence ---
    best_hist_rand = results_rand["best_fitness_history"]
    best_hist_rl = results_rl["best_fitness_history"]

    gens_rand = range(len(best_hist_rand))
    gens_rl = range(len(best_hist_rl))

    plt.figure()
    plt.plot(gens_rand, best_hist_rand, label="GA random init")
    plt.plot(gens_rl, best_hist_rl, label="GA RL init")
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title("GA convergence: random vs RL-based initial population")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=150)
    print(f"[Compare] Saved convergence plot to {save_plot_path}")

    return {
        "baseline_results": results_rand,
        "rl_results": results_rl,
        "train_info": train_info,
        "plot_path": save_plot_path,
    }


# =====================================================================
# 6. Example usage
# =====================================================================

if __name__ == "__main__":
    # Example: replace these with your real machines / sequence if needed.
    machines_example = [
        Machine(id=1, shape='rectangle', width=4.0, height=3.0,
                access_point=Point(1.5, 0.0)),
        Machine(id=2, shape='l_shape', width=5.0, height=4.0,
                access_point=Point(-1.0, 1.0),
                l_cutout_width=2.0, l_cutout_height=2.0),
        Machine(id=3, shape='rectangle', width=3.5, height=2.5,
                access_point=Point(0.0, -1.0)),
        Machine(id=4, shape='l_shape', width=4.5, height=3.5,
                access_point=Point(1.0, -0.5),
                l_cutout_width=1.5, l_cutout_height=1.5),
        Machine(id=5, shape='l_shape', width=3.0, height=5.5,
                access_point=Point(1.0, -1.0),
                l_cutout_width=1.0, l_cutout_height=2.0),
        Machine(id=6, shape='l_shape', width=5.0, height=3.0,
                access_point=Point(1.0, -1.0),
                l_cutout_width=2.0, l_cutout_height=2.0),
        Machine(id=7, shape='l_shape', width=3.5, height=2.5,
                access_point=Point(1.0, -1.0),
                l_cutout_width=1.0, l_cutout_height=2.0),
        Machine(id=8, shape='rectangle', width=3.5, height=2.5,
                access_point=Point(0.0, 0.0)),
    ]

    seq_example = [1, 2, 3, 4, 5, 6, 7, 8]
    robot_pos = Point(0.0, 0.0)
    bounds = (-15.0, 15.0, -15.0, 15.0)

    pop_size = 500
    rl_share = 300
    qtable_file = "layout_qtable_initpop.npz"

    # This will:
    # 1) run GA baseline (random init),
    # 2) train/load RL agent,
    # 3) generate RL initial population,
    # 4) run GA with RL init,
    # 5) save convergence plot.
    compare_ga_convergence_with_and_without_rl(
        machines=machines_example,
        sequence=seq_example,
        robot_position=robot_pos,
        workspace_bounds=bounds,
        ga_population_size=pop_size,
        num_rl_solutions=rl_share,
        num_episodes=10000,
        qtable_path=qtable_file,
        reuse_if_exists=True,
        save_plot_path="ga_convergence_compare.png",
    )