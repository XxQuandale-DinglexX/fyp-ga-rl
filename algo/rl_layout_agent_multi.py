import numpy as np
import random
from copy import deepcopy
from typing import List, Tuple, Dict
from collections import defaultdict

from func.datastruct import Point, Machine
from func.collision_detector import CollisionDetector


class MultiProblemTabularRLAgent:
    """
    Tabular Q-learning that works across multiple 8-machine problems.
    
    Key difference from single-problem agent:
    - State includes discretized machine features (size, shape, etc.)
    - Q-table is a dictionary mapping (feature_state, action) -> Q-value
    - Can generalize across different machine configurations
    """

    def __init__(
        self,
        num_cells_x: int = 10,
        num_cells_y: int = 10,
        collision_penalty: float = -150.0,
        num_machines: int = 8,
        # Feature discretization bins
        width_bins: int = 4,
        height_bins: int = 4,
        step_bins: int = 8,
    ):
        self.num_cells_x = num_cells_x
        self.num_cells_y = num_cells_y
        self.num_positions = num_cells_x * num_cells_y
        self.num_rotations = 4
        self.action_size = self.num_positions * self.num_rotations
        self.num_machines = num_machines
        
        self.collision_penalty = collision_penalty
        
        # Feature discretization
        self.width_bins = width_bins
        self.height_bins = height_bins
        self.step_bins = step_bins
        
        # Q-table as dictionary: (state_tuple, action) -> Q-value
        self.Q: Dict[Tuple, float] = defaultdict(float)
        
        # Problem-specific fields (set per problem)
        self.machines: List[Machine] = []
        self.sequence: List[int] = []
        self.robot_position: Point = Point(0, 0)
        self.workspace_bounds = (-15.0, 15.0, -15.0, 15.0)
        self.min_x = self.max_x = self.min_y = self.max_y = 0.0
        self.cell_w = self.cell_h = 1.0
        
        # Training statistics
        self.state_visit_counts = defaultdict(int)

    def set_problem(
        self,
        machines: List[Machine],
        sequence: List[int],
        robot_position: Point,
        workspace_bounds: Tuple[float, float, float, float],
    ):
        """Set the current problem to work on"""
        self.machines = machines
        self.sequence = sequence
        self.robot_position = robot_position
        self.workspace_bounds = workspace_bounds
        
        min_x, max_x, min_y, max_y = workspace_bounds
        self.min_x, self.max_x = min_x, max_x
        self.min_y, self.max_y = min_y, max_y
        self.cell_w = (max_x - min_x) / self.num_cells_x
        self.cell_h = (max_y - min_y) / self.num_cells_y

    # ------------------------------------------------------------------
    # Feature extraction and discretization
    # ------------------------------------------------------------------

    def _discretize_machine_features(self, machine: Machine) -> Tuple[int, int, int]:
        """
        Convert machine to discrete features:
        - shape_bin: 0 (rectangle) or 1 (L-shape)
        - width_bin: 0 to width_bins-1
        - height_bin: 0 to height_bins-1
        """
        shape_bin = 1 if machine.shape == "l_shape" else 0
        
        # Normalize dimensions (assume typical range 2-6 units)
        width_norm = np.clip((machine.width - 2.0) / 4.0, 0, 1)
        height_norm = np.clip((machine.height - 2.0) / 4.0, 0, 1)
        
        width_bin = min(int(width_norm * self.width_bins), self.width_bins - 1)
        height_bin = min(int(height_norm * self.height_bins), self.height_bins - 1)
        
        return shape_bin, width_bin, height_bin

    def _make_state(
        self,
        step_idx: int,
        machines_ep: List[Machine],
        placed_count: int,
    ) -> Tuple:
        """
        Create state tuple from current situation.
        State = (step_bin, shape, width_bin, height_bin, placed_count_bin)
        """
        # Discretize step progress
        step_norm = step_idx / max(1, self.num_machines - 1)
        step_bin = min(int(step_norm * self.step_bins), self.step_bins - 1)
        
        # Current machine features
        shape_bin, width_bin, height_bin = self._discretize_machine_features(
            machines_ep[step_idx]
        )
        
        # Placement progress
        placed_bin = min(placed_count, self.num_machines)
        
        return (step_bin, shape_bin, width_bin, height_bin, placed_bin)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _decode_action(self, action_id: int):
        """Convert action index -> (x, y, rotation_deg)"""
        pos_id = action_id // self.num_rotations
        rot_id = action_id % self.num_rotations

        i = pos_id % self.num_cells_x
        j = pos_id // self.num_cells_x

        x = self.min_x + (i + 0.5) * self.cell_w
        y = self.min_y + (j + 0.5) * self.cell_h
        rotation = [0.0, 90.0, 180.0, 270.0][rot_id]

        return x, y, rotation

    def _compute_path_length(self, machines: List[Machine], placed_steps: List[int]) -> float:
        """Compute path length through placed machines"""
        if not placed_steps:
            return 0.0
        
        id2m = {m.id: m for m in machines}
        placed_step_set = set(placed_steps)
        points = [self.robot_position]
        
        for step_idx, machine_id in enumerate(self.sequence):
            if step_idx in placed_step_set:
                points.append(id2m[machine_id].get_access_point_world())
        
        points.append(self.robot_position)
        
        total = 0.0
        for i in range(len(points) - 1):
            total += points[i].distance_to(points[i + 1])
        return total

    def _violates_workspace_bounds(self, machine: Machine) -> bool:
        min_x, max_x, min_y, max_y = self.workspace_bounds
        for corner in machine.get_corners():
            if not (min_x <= corner.x <= max_x and min_y <= corner.y <= max_y):
                return True
        return False

    def _overlaps_origin(self, machine: Machine) -> bool:
        return CollisionDetector.point_in_polygon(Point(0.0, 0.0), machine.get_corners())

    def _has_collision(self, machines: List[Machine], idx: int, placed_indices: List[int]) -> bool:
        poly_i = machines[idx].get_corners()
        for j in placed_indices:
            if j == idx:
                continue
            poly_j = machines[j].get_corners()
            if CollisionDetector.polygons_intersect(poly_i, poly_j):
                return True
        return False

    def _violates_constraints(self, machines: List[Machine], idx: int, placed_indices: List[int]) -> bool:
        m = machines[idx]
        if self._violates_workspace_bounds(m):
            return True
        if self._overlaps_origin(m):
            return True
        if self._has_collision(machines, idx, placed_indices):
            return True
        return False

    # ------------------------------------------------------------------
    # RL utilities
    # ------------------------------------------------------------------

    def _get_q_value(self, state: Tuple, action: int) -> float:
        """Get Q-value with default 0.0"""
        return self.Q.get((state, action), 0.0)

    def _get_best_action(self, state: Tuple) -> int:
        """Get action with highest Q-value for state"""
        q_values = [self._get_q_value(state, a) for a in range(self.action_size)]
        max_q = max(q_values)
        best_actions = [a for a in range(self.action_size) if q_values[a] >= max_q - 1e-8]
        return random.choice(best_actions)

    def _epsilon_greedy_action(self, state: Tuple, epsilon: float) -> int:
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        return self._get_best_action(state)

    # ------------------------------------------------------------------
    # Training across multiple problems
    # ------------------------------------------------------------------

    def train_on_problem(
        self,
        num_episodes: int = 500,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.3,
        verbose: bool = False,
    ):
        """
        Train on the currently set problem.
        This updates the shared Q-table.
        """
        success_count = 0
        
        for ep in range(num_episodes):
            machines_ep = [deepcopy(m) for m in self.machines]
            for m in machines_ep:
                m.position = deepcopy(self.robot_position)
                m.rotation = 0.0

            placed: List[int] = []
            
            for step in range(self.num_machines):
                state = self._make_state(step, machines_ep, len(placed))
                self.state_visit_counts[state] += 1
                
                action = self._epsilon_greedy_action(state, epsilon)
                x, y, rot = self._decode_action(action)

                m = machines_ep[step]
                old_pos = deepcopy(m.position)
                old_rot = m.rotation

                m.position = Point(x, y)
                m.rotation = rot

                if self._violates_constraints(machines_ep, step, placed):
                    # Invalid - revert
                    m.position = old_pos
                    m.rotation = old_rot
                    reward = self.collision_penalty
                else:
                    # Valid
                    new_length = self._compute_path_length(machines_ep, placed + [step])
                    old_length = self._compute_path_length(machines_ep, placed)
                    reward = old_length - new_length
                    placed.append(step)

                # Q-learning update
                next_state = self._make_state(
                    min(step + 1, self.num_machines - 1),
                    machines_ep,
                    len(placed)
                )
                
                current_q = self._get_q_value(state, action)
                max_next_q = max(
                    self._get_q_value(next_state, a) 
                    for a in range(self.action_size)
                )
                
                td_target = reward + gamma * max_next_q
                self.Q[(state, action)] = current_q + alpha * (td_target - current_q)

            if len(placed) == self.num_machines:
                success_count += 1

            if verbose and (ep + 1) % max(1, num_episodes // 10) == 0:
                print(f"  Episode {ep + 1}/{num_episodes}, "
                      f"eps={epsilon:.3f}, placed={len(placed)}/{self.num_machines}, "
                      f"success_rate={success_count/(ep+1)*100:.1f}%")

        return success_count / num_episodes

    def train_many_problems(
        self,
        problem_generator,
        num_problems: int = 100,
        episodes_per_problem: int = 500,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 0.5,
        epsilon_end: float = 0.1,
        verbose: bool = True,
    ):
        """
        Train across many problems, building a shared Q-table.
        """
        if verbose:
            print("="*70)
            print("MULTI-PROBLEM TABULAR RL TRAINING")
            print("="*70)
            print(f"Training on {num_problems} problems")
            print(f"Episodes per problem: {episodes_per_problem}")
            print(f"Total episodes: {num_problems * episodes_per_problem}")
            print()
        
        problem_success_rates = []
        
        for p in range(num_problems):
            # Get problem
            machines, seq, robot_pos, bounds = problem_generator(p + 1)
            self.set_problem(machines, seq, robot_pos, bounds)
            
            # Decay epsilon across problems
            frac = p / max(1, num_problems - 1)
            epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
            
            if verbose and (p + 1) % 10 == 0:
                print(f"\n--- Problem {p + 1}/{num_problems} (eps={epsilon:.3f}) ---")
            
            success_rate = self.train_on_problem(
                num_episodes=episodes_per_problem,
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
                verbose=verbose and (p + 1) % 10 == 0,
            )
            
            problem_success_rates.append(success_rate)
            
            if verbose and (p + 1) % 10 == 0:
                recent_success = np.mean(problem_success_rates[-10:])
                print(f"  Problem success rate: {success_rate*100:.1f}%")
                print(f"  Recent avg (last 10): {recent_success*100:.1f}%")
        
        if verbose:
            print("\n" + "="*70)
            print("TRAINING COMPLETE")
            print("="*70)
            print(f"Q-table size: {len(self.Q)} state-action pairs")
            print(f"Unique states visited: {len(self.state_visit_counts)}")
            print(f"Overall success rate: {np.mean(problem_success_rates)*100:.1f}%")
            print(f"Final 20 problems: {np.mean(problem_success_rates[-20:])*100:.1f}%")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate_layout_greedy(self, max_attempts: int = 100) -> Tuple[List[Machine], float, np.ndarray, bool]:
        """Generate layout using learned Q-table"""
        machines_ep = [deepcopy(m) for m in self.machines]
        for m in machines_ep:
            m.position = deepcopy(self.robot_position)
            m.rotation = 0.0

        placed: List[int] = []
        success = True

        for step in range(self.num_machines):
            state = self._make_state(step, machines_ep, len(placed))
            
            # Get Q-values for all actions
            q_values = [(a, self._get_q_value(state, a)) for a in range(self.action_size)]
            q_values.sort(key=lambda x: x[1], reverse=True)
            
            placed_successfully = False
            
            for action, q_val in q_values[:max_attempts]:
                x, y, rot = self._decode_action(action)
                
                m = machines_ep[step]
                m.position = Point(x, y)
                m.rotation = rot
                
                if not self._violates_constraints(machines_ep, step, placed):
                    placed.append(step)
                    placed_successfully = True
                    break
            
            if not placed_successfully:
                success = False
                machines_ep[step].position = deepcopy(self.robot_position)
                machines_ep[step].rotation = 0.0

        length = self._compute_path_length(machines_ep, placed)
        
        chrom = []
        for m in machines_ep:
            chrom.extend([m.position.x, m.position.y, m.rotation])
        
        return machines_ep, length, np.array(chrom, dtype=np.float32), success

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save Q-table to file"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'Q': dict(self.Q),
                'state_visit_counts': dict(self.state_visit_counts),
            }, f)

    def load(self, path: str):
        """Load Q-table from file"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.Q = defaultdict(float, data['Q'])
            self.state_visit_counts = defaultdict(int, data.get('state_visit_counts', {}))


# ------------------------------------------------------------------
# Training script
# ------------------------------------------------------------------

def train_multi_problem_tabular():
    """Example training script"""
    
    def generate_problem(problem_idx: int):
        base_seed = 1000 + problem_idx
        random.seed(base_seed)
        np.random.seed(base_seed)
        
        workspace_bounds = (-18, 18, -18, 18)  # Larger workspace
        robot_position = Point(0, 0)
        
        machines = []
        for i in range(8):
            m_id = i + 1
            shape = "l_shape" if i % 2 == 0 else "rectangle"
            
            width = random.uniform(2.5, 4.5)
            height = random.uniform(2.0, 4.0)
            
            ap_x = random.uniform(-1.2, 1.2)
            ap_y = random.uniform(-1.2, 1.2)
            
            if shape == "l_shape":
                l_cutout_width = random.uniform(1.0, width / 2.5)
                l_cutout_height = random.uniform(1.0, height / 2.5)
                machines.append(Machine(
                    id=m_id, shape="l_shape", width=width, height=height,
                    access_point=Point(ap_x, ap_y),
                    l_cutout_width=l_cutout_width, l_cutout_height=l_cutout_height
                ))
            else:
                machines.append(Machine(
                    id=m_id, shape="rectangle", width=width, height=height,
                    access_point=Point(ap_x, ap_y)
                ))
        
        sequence = list(range(1, 9))
        return machines, sequence, robot_position, workspace_bounds
    
    # Create agent
    agent = MultiProblemTabularRLAgent(
        num_cells_x=12,
        num_cells_y=12,
        collision_penalty=-100.0,
        num_machines=8,
        width_bins=4,
        height_bins=4,
        step_bins=8,
    )
    
    # Train
    agent.train_many_problems(
        problem_generator=generate_problem,
        num_problems=200,
        episodes_per_problem=800,
        alpha=0.1,
        gamma=0.95,
        epsilon_start=0.5,
        epsilon_end=0.05,
        verbose=True,
    )
    
    # Save
    agent.save("tabular_rl_multi_problem.pkl")
    print("\nSaved: tabular_rl_multi_problem.pkl")
    
    # Test
    print("\n" + "="*70)
    print("TESTING ON NEW PROBLEMS")
    print("="*70)
    
    success_count = 0
    for i in range(20):
        machines, seq, robot_pos, bounds = generate_problem(10000 + i)
        agent.set_problem(machines, seq, robot_pos, bounds)
        
        _, length, _, success = agent.generate_layout_greedy(max_attempts=100)
        if success:
            success_count += 1
            print(f"Test {i+1}: SUCCESS (length={length:.2f})")
        else:
            print(f"Test {i+1}: FAILED")
    
    print(f"\nTest success rate: {success_count}/20 ({success_count*5}%)")


if __name__ == "__main__":
    train_multi_problem_tabular()