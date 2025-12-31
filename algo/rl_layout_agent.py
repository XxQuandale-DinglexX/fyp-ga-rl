import numpy as np
import random
from copy import deepcopy
from typing import List, Tuple

from func.datastruct import Point, Machine
from func.collision_detector import CollisionDetector


class LayoutRLAgent:
    """
    Tabular Q-learning agent that *builds* a workcell layout by placing machines
    one by one on a discretized grid, with rotations {0, 90, 180, 270}.

    - State: index of the machine currently being placed (0..N-1)
    - Action: (grid_cell_x, grid_cell_y, rotation_index)
    - Reward: reduction in total robot tour length (old_length - new_length),
              with big penalty for collisions.
    - Training: epsilon-greedy
    - Inference: greedy (argmax Q) to get "optimized" layouts.
    """

    def __init__(
        self,
        machines: List[Machine],
        sequence: List[int],
        robot_position: Point,
        workspace_bounds: Tuple[float, float, float, float],
        num_cells_x: int = 8,
        num_cells_y: int = 8,
        collision_penalty: float = -200.0,
    ):
        self.machines = machines
        self.sequence = sequence
        self.robot_position = robot_position
        self.workspace_bounds = workspace_bounds

        self.num_machines = len(machines)
        self.num_cells_x = num_cells_x
        self.num_cells_y = num_cells_y
        self.num_positions = num_cells_x * num_cells_y
        self.num_rotations = 4  # 0, 90, 180, 270 deg

        # State: machine index (0..N-1)
        self.state_size = self.num_machines
        # Action: every (cell, rotation) combination
        self.action_size = self.num_positions * self.num_rotations

        # Q-table: shape [state_size, action_size]
        self.Q = np.zeros((self.state_size, self.action_size), dtype=np.float32)

        # Grid geometry
        min_x, max_x, min_y, max_y = self.workspace_bounds
        self.min_x, self.max_x = min_x, max_x
        self.min_y, self.max_y = min_y, max_y
        self.cell_w = (max_x - min_x) / self.num_cells_x
        self.cell_h = (max_y - min_y) / self.num_cells_y

        self.collision_penalty = collision_penalty

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _decode_action(self, action_id: int):
        """Convert action index -> (x, y, rotation_deg)."""
        pos_id = action_id // self.num_rotations
        rot_id = action_id % self.num_rotations

        i = pos_id % self.num_cells_x
        j = pos_id // self.num_cells_x

        x = self.min_x + (i + 0.5) * self.cell_w
        y = self.min_y + (j + 0.5) * self.cell_h
        rotation = [0.0, 90.0, 180.0, 270.0][rot_id]

        return x, y, rotation

    def _compute_path_length(self, machines: List[Machine]) -> float:
        """
        Compute total robot tour length according to visiting sequence.
        """
        id2m = {m.id: m for m in machines}
        points = [self.robot_position]

        for mid in self.sequence:
            points.append(id2m[mid].get_access_point_world())

        points.append(self.robot_position)

        total = 0.0
        for i in range(len(points) - 1):
            total += points[i].distance_to(points[i + 1])

        return total
    
    def _violates_workspace_bounds_machine(self, machine) -> bool:
        """True if any corner of machine is outside workspace bounds."""
        min_x, max_x, min_y, max_y = self.workspace_bounds
        for corner in machine.get_corners():
            if not (min_x <= corner.x <= max_x and min_y <= corner.y <= max_y):
                return True
        return False

    def _overlaps_center(self, machine) -> bool:
        """True if machine polygon contains the center (0,0)."""
        return CollisionDetector.point_in_polygon(
            Point(0.0, 0.0),
            machine.get_corners()
        )
    
    def _has_collision_with_placed(
        self,
        machines: List[Machine],
        idx: int,
        placed_indices: List[int],
    ) -> bool:
        """Check if machines[idx] collides with any already-placed machines."""
        poly_i = machines[idx].get_corners()
        for j in placed_indices:
            if j == idx:
                continue
            poly_j = machines[j].get_corners()
            if CollisionDetector.polygons_intersect(poly_i, poly_j):
                return True
        return False

    def _encode_layout_to_chromosome(self, machines: List[Machine]) -> np.ndarray:
        """
        Optional helper: encode layout into GA chromosome later:
        [x1, y1, rot1, x2, y2, rot2, ...]
        """
        chrom = []
        # assume stored order is used
        for m in machines:
            chrom.extend([m.position.x, m.position.y, m.rotation])
        return np.array(chrom, dtype=np.float32)
    
    def _violates_constraints(
        self,
        machines,
        idx: int,
        placed_indices,
    ) -> bool:
        """
        Unified constraint check:
        - out of workspace bounds
        - overlaps center (0,0)
        - collides with already placed machines
        """
        m = machines[idx]
        if self._violates_workspace_bounds_machine(m):
            return True
        if self._overlaps_center(m):
            return True
        if self._has_collision_with_placed(machines, idx, placed_indices):
            return True
        return False
    
    # ------------------------------------------------------------------
    # RL utilities
    # ------------------------------------------------------------------

    def _epsilon_greedy_action(self, state: int, epsilon: float) -> int:
        """Standard ε-greedy selection on Q[state]."""
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        q_row = self.Q[state]
        max_q = np.max(q_row)
        best_actions = np.flatnonzero(q_row == max_q)
        return int(np.random.choice(best_actions))

    # ------------------------------------------------------------------
    # Training: pure RL (no GA involved here)
    # ------------------------------------------------------------------

    def train(
        self,
        num_episodes: int = 5000,
        alpha: float = 0.05,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        verbose: bool = True,
    ):
        """
        Train Q-table via tabular Q-learning.

        Each episode:
        - Start with all machines at robot_position (conceptually "unplaced").
        - For machine index 0..N-1:
            * choose an action (grid cell + rotation)
            * if collides with already placed machines -> big negative reward
            * else -> reward = old_length - new_length
        - Greedy behavior will later place machines where path length grows as little
          as possible while avoiding collisions.
        """
        for ep in range(num_episodes):
            # epsilon decay
            frac = ep / max(1, num_episodes - 1)
            epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

            # fresh copy of machines
            machines_ep = [deepcopy(m) for m in self.machines]
            for m in machines_ep:
                m.position = deepcopy(self.robot_position)
                m.rotation = 0.0

            placed_indices: List[int] = []
            old_length = self._compute_path_length(machines_ep)
            last_reward = 0.0

            # place each machine in order 0..N-1 (state = machine_idx)
            for state in range(self.num_machines):
                machine_idx = state
                m = machines_ep[machine_idx]

                action = self._epsilon_greedy_action(state, epsilon)
                x, y, rot = self._decode_action(action)

                old_pos = deepcopy(m.position)
                old_rot = m.rotation

                # apply tentative placement
                m.position = Point(x, y)
                m.rotation = rot

                # collision only against already placed machines
                if self._violates_constraints(machines_ep, machine_idx, placed_indices):
                    # invalid action: out of bounds, overlaps center, or collision
                    m.position = old_pos
                    m.rotation = old_rot

                    reward = self.collision_penalty
                    new_length = old_length
                else:
                    # valid placement
                    new_length = self._compute_path_length(machines_ep)
                    reward = old_length - new_length
                    if machine_idx not in placed_indices:
                        placed_indices.append(machine_idx)

                # Q-learning update
                next_state = min(state + 1, self.state_size - 1)
                best_next = np.max(self.Q[next_state])
                td_target = reward + gamma * best_next
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += alpha * td_error

                old_length = new_length
                last_reward = reward

            if verbose and (ep + 1) % max(1, num_episodes // 10) == 0:
                print(
                    f"[RL] Episode {ep + 1}/{num_episodes}, "
                    f"eps={epsilon:.3f}, last_reward={last_reward:.3f}, "
                    f"placed={len(placed_indices)}/{self.num_machines}"
                )

    # ------------------------------------------------------------------
    # Inference: greedy layouts
    # ------------------------------------------------------------------

    def generate_layout_greedy(self, rng_jitter: float = 0.1):
        """
        Build a layout using greedy actions (argmax Q) and return:

        - machines_layout: List[Machine] with updated position & rotation
        - route_length: total tour length of that layout
        - chromosome: np.ndarray [x1, y1, rot1, ...] (for GA later if needed)
        """
        machines_ep = [deepcopy(m) for m in self.machines]
        for m in machines_ep:
            m.position = deepcopy(self.robot_position)
            m.rotation = 0.0

        placed_indices: List[int] = []

        for state in range(self.state_size):
            machine_idx = state
            m = machines_ep[machine_idx]

            q_row = self.Q[state]
            max_q = np.max(q_row)
            best_actions = np.flatnonzero(q_row >= max_q - 1e-8)

            # tie-breaking randomness for diversity
            if rng_jitter > 0.0 and len(best_actions) > 1:
                action = int(np.random.choice(best_actions))
            else:
                action = int(best_actions[0])

            tried = set()
            while True:
                tried.add(action)
                x, y, rot = self._decode_action(action)

                old_pos = deepcopy(m.position)
                old_rot = m.rotation

                m.position = Point(x, y)
                m.rotation = rot

                if not self._violates_constraints(machines_ep, machine_idx, placed_indices):
                    if machine_idx not in placed_indices:
                        placed_indices.append(machine_idx)
                    break  # valid placement

                # collision – revert & try a different action
                m.position = old_pos
                m.rotation = old_rot

                if len(tried) == self.action_size:
                    # no valid action found
                    print(f"[RL] Warning: no valid action for machine {m.id}, leaving at origin")
                    break

                candidates = [a for a in range(self.action_size) if a not in tried]
                action = random.choice(candidates)

        length = self._compute_path_length(machines_ep)
        chrom = self._encode_layout_to_chromosome(machines_ep)
        return machines_ep, length, chrom

    def evaluate_greedy(
        self,
        num_rollouts: int = 200,
        rng_jitter: float = 0.1,
    ):
        """
        Run greedy rollout several times (with small jitter) and return:
        - best_length
        - mean_length
        - best_chromosome
        """
        lengths = []
        best_length = None
        best_chrom = None

        for _ in range(num_rollouts):
            _, L, chrom = self.generate_layout_greedy(rng_jitter=rng_jitter)
            lengths.append(L)
            if best_length is None or L < best_length:
                best_length = L
                best_chrom = chrom

        return best_length, float(np.mean(lengths)), best_chrom

    # ------------------------------------------------------------------
    # Save / load Q-table
    # ------------------------------------------------------------------

    def save_q_table(self, path: str):
        np.save(path, self.Q)

    def load_q_table(self, path: str):
        self.Q = np.load(path)
