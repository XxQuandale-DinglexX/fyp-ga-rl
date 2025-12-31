import random
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Deque, Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from func.datastruct import Point, Machine
from func.collision_detector import CollisionDetector


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNLayoutRLAgent:
    """
    Refined DQN agent with improved collision handling and state representation.
    
    Key improvements:
    - Proper collision penalty and state transitions
    - Tracks placement success/failure
    - Better reward shaping
    - Validation metrics
    """

    def __init__(
        self,
        num_cells_x: int = 10,
        num_cells_y: int = 10,
        collision_penalty: float = -100.0,
        invalid_placement_penalty: float = -50.0,
        step_penalty: float = -1.0,
        device: Optional[str] = None,
        replay_capacity: int = 100_000,
        batch_size: int = 256,
        lr: float = 1e-3,
        gamma: float = 0.95,
        target_update_every: int = 1000,
        hidden: int = 256,
        seed: int = 0,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_cells_x = num_cells_x
        self.num_cells_y = num_cells_y
        self.num_positions = num_cells_x * num_cells_y
        self.num_rotations = 4
        self.action_dim = self.num_positions * self.num_rotations

        self.collision_penalty = collision_penalty
        self.invalid_placement_penalty = invalid_placement_penalty
        self.step_penalty = step_penalty
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_every = target_update_every

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Problem-dependent fields
        self.machines: List[Machine] = []
        self.sequence: List[int] = []
        self.robot_position: Point = Point(0, 0)
        self.workspace_bounds = (-15.0, 15.0, -15.0, 15.0)
        self.min_x = self.max_x = self.min_y = self.max_y = 0.0
        self.cell_w = self.cell_h = 1.0
        self.num_machines = 0

        # Enhanced state: step + current machine + placed summary + prev AP + occupancy hints
        self.state_dim = 1 + 7 + 6 + 2 + 4
        # Added 4 features for spatial occupancy hints (quadrant densities)

        self.q = QNet(self.state_dim, self.action_dim, hidden=hidden).to(self.device)
        self.q_tgt = QNet(self.state_dim, self.action_dim, hidden=hidden).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.q_tgt.eval()

        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.replay: Deque[Transition] = deque(maxlen=replay_capacity)
        self.train_steps = 0
        
        # Metrics tracking
        self.episode_metrics = {
            'success_rate': [],
            'avg_path_length': [],
            'collision_rate': [],
            'placement_rate': []
        }

    # ------------------------- Problem setup -------------------------

    def set_problem(
        self,
        machines: List[Machine],
        sequence: List[int],
        robot_position: Point,
        workspace_bounds: Tuple[float, float, float, float],
    ):
        self.machines = machines
        self.sequence = sequence
        self.robot_position = robot_position
        self.workspace_bounds = workspace_bounds
        self.num_machines = len(machines)

        min_x, max_x, min_y, max_y = workspace_bounds
        self.min_x, self.max_x = min_x, max_x
        self.min_y, self.max_y = min_y, max_y
        self.cell_w = (max_x - min_x) / self.num_cells_x
        self.cell_h = (max_y - min_y) / self.num_cells_y

    # ------------------------- Geometry helpers -------------------------

    def _decode_action(self, action_id: int):
        pos_id = action_id // self.num_rotations
        rot_id = action_id % self.num_rotations
        i = pos_id % self.num_cells_x
        j = pos_id // self.num_cells_x

        x = self.min_x + (i + 0.5) * self.cell_w
        y = self.min_y + (j + 0.5) * self.cell_h
        rotation = [0.0, 90.0, 180.0, 270.0][rot_id]
        return x, y, rotation

    def _compute_path_length(self, machines: List[Machine], placed_steps: List[int]) -> float:
        """
        Compute path length through successfully placed machines.
        
        Args:
            machines: List of all machines
            placed_steps: List of step indices that were successfully placed
        """
        if not placed_steps:
            return 0.0
        
        id2m = {m.id: m for m in machines}
        placed_step_set = set(placed_steps)
        points = [self.robot_position]
        
        # Visit machines in sequence order, but only if their step was placed
        for step_idx, machine_id in enumerate(self.sequence):
            if step_idx in placed_step_set:
                points.append(id2m[machine_id].get_access_point_world())
        
        points.append(self.robot_position)
        
        total = 0.0
        for i in range(len(points) - 1):
            total += points[i].distance_to(points[i + 1])
        return total

    def _violates_workspace_bounds(self, machine: Machine) -> bool:
        """Check if machine extends beyond workspace"""
        min_x, max_x, min_y, max_y = self.workspace_bounds
        for corner in machine.get_corners():
            if not (min_x <= corner.x <= max_x and min_y <= corner.y <= max_y):
                return True
        return False

    def _overlaps_origin(self, machine: Machine) -> bool:
        """Check if machine overlaps the origin point (0,0)"""
        return CollisionDetector.point_in_polygon(Point(0.0, 0.0), machine.get_corners())

    def _has_collision(self, machines: List[Machine], idx: int, placed_indices: List[int]) -> bool:
        """Check if machine at idx collides with any already placed machine"""
        poly_i = machines[idx].get_corners()
        for j in placed_indices:
            if j == idx:
                continue
            poly_j = machines[j].get_corners()
            if CollisionDetector.polygons_intersect(poly_i, poly_j):
                return True
        return False

    def _check_constraints(self, machine: Machine, machines: List[Machine], 
                          idx: int, placed_indices: List[int]) -> Tuple[bool, str]:
        """
        Check all constraints and return (is_valid, violation_type)
        violation_type: 'none', 'bounds', 'origin', 'collision'
        """
        if self._violates_workspace_bounds(machine):
            return False, 'bounds'
        
        if self._overlaps_origin(machine):
            return False, 'origin'
        
        if self._has_collision(machines, idx, placed_indices):
            return False, 'collision'
        
        return True, 'none'

    # ------------------------- State featurization -------------------------

    def _norm_x(self, x: float) -> float:
        span = self.max_x - self.min_x
        return (x - self.min_x) / (span if span > 0 else 1.0)

    def _norm_y(self, y: float) -> float:
        span = self.max_y - self.min_y
        return (y - self.min_y) / (span if span > 0 else 1.0)

    def _machine_features(self, m: Machine) -> np.ndarray:
        """Extract normalized features from a machine"""
        span_x = (self.max_x - self.min_x) if (self.max_x - self.min_x) > 0 else 1.0
        span_y = (self.max_y - self.min_y) if (self.max_y - self.min_y) > 0 else 1.0

        shape_is_l = 1.0 if m.shape == "l_shape" else 0.0
        w = float(m.width) / span_x
        h = float(m.height) / span_y

        # Normalize access point (assuming typical range [-width/2, width/2])
        apx = (float(m.access_point.x) / float(m.width) + 0.5) if m.width > 0 else 0.5
        apy = (float(m.access_point.y) / float(m.height) + 0.5) if m.height > 0 else 0.5
        apx = np.clip(apx, 0.0, 1.0)
        apy = np.clip(apy, 0.0, 1.0)

        cutw = float(m.l_cutout_width) / span_x if hasattr(m, 'l_cutout_width') else 0.0
        cuth = float(m.l_cutout_height) / span_y if hasattr(m, 'l_cutout_height') else 0.0

        return np.array([shape_is_l, w, h, apx, apy, cutw, cuth], dtype=np.float32)

    def _occupancy_features(self, machines: List[Machine], placed_indices: List[int]) -> np.ndarray:
        """Compute spatial occupancy in four quadrants"""
        if not placed_indices:
            return np.zeros(4, dtype=np.float32)
        
        # Count machines in each quadrant
        counts = [0, 0, 0, 0]  # [Q1: +x,+y], [Q2: -x,+y], [Q3: -x,-y], [Q4: +x,-y]
        
        for idx in placed_indices:
            pos = machines[idx].position
            if pos.x >= 0 and pos.y >= 0:
                counts[0] += 1
            elif pos.x < 0 and pos.y >= 0:
                counts[1] += 1
            elif pos.x < 0 and pos.y < 0:
                counts[2] += 1
            else:  # pos.x >= 0 and pos.y < 0
                counts[3] += 1
        
        total = sum(counts)
        if total > 0:
            return np.array([c / total for c in counts], dtype=np.float32)
        return np.zeros(4, dtype=np.float32)

    def _placed_summary(self, machines: List[Machine], placed_indices: List[int]) -> np.ndarray:
        """Compute aggregate statistics of placed machines"""
        if not placed_indices:
            return np.zeros(6, dtype=np.float32)

        xs = [machines[i].position.x for i in placed_indices]
        ys = [machines[i].position.y for i in placed_indices]
        rots = [machines[i].rotation for i in placed_indices]

        mean_x = self._norm_x(float(np.mean(xs)))
        mean_y = self._norm_y(float(np.mean(ys)))
        
        span_x_val = float(np.max(xs) - np.min(xs)) if len(xs) > 1 else 0.0
        span_y_val = float(np.max(ys) - np.min(ys)) if len(ys) > 1 else 0.0
        
        span_x = span_x_val / ((self.max_x - self.min_x) if (self.max_x - self.min_x) > 0 else 1.0)
        span_y = span_y_val / ((self.max_y - self.min_y) if (self.max_y - self.min_y) > 0 else 1.0)
        
        mean_rot = (float(np.mean(rots)) % 360.0) / 360.0
        placed_frac = len(placed_indices) / max(1, self.num_machines)

        return np.array([placed_frac, mean_x, mean_y, span_x, span_y, mean_rot], dtype=np.float32)

    def _make_state(
        self,
        step_idx: int,
        machines_ep: List[Machine],
        placed_indices: List[int],
        prev_ap_world: Point,
    ) -> np.ndarray:
        """Construct state vector"""
        step_frac = np.array([step_idx / max(1, self.num_machines)], dtype=np.float32)
        mfeat = self._machine_features(machines_ep[step_idx])
        psum = self._placed_summary(machines_ep, placed_indices)
        prev = np.array([self._norm_x(prev_ap_world.x), self._norm_y(prev_ap_world.y)], dtype=np.float32)
        occup = self._occupancy_features(machines_ep, placed_indices)
        
        return np.concatenate([step_frac, mfeat, psum, prev, occup], axis=0)

    # ------------------------- Action selection -------------------------

    def _epsilon_greedy(self, s: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            x = torch.from_numpy(s).float().to(self.device).unsqueeze(0)
            qvals = self.q(x).squeeze(0)
            return int(torch.argmax(qvals).item())

    # ------------------------- Training on one episode -------------------------

    def _run_episode(self, eps: float) -> Tuple[float, int, int]:
        """
        Run one training episode.
        Returns: (final_path_length, num_placed, num_collisions)
        """
        machines_ep = [deepcopy(m) for m in self.machines]
        for m in machines_ep:
            m.position = deepcopy(self.robot_position)
            m.rotation = 0.0

        placed: List[int] = []
        prev_ap_world = deepcopy(self.robot_position)
        num_collisions = 0
        
        for step in range(self.num_machines):
            s = self._make_state(step, machines_ep, placed, prev_ap_world)
            a = self._epsilon_greedy(s, eps)
            x, y, rot = self._decode_action(a)

            m = machines_ep[step]
            old_pos = deepcopy(m.position)
            old_rot = m.rotation

            # Try placing machine
            m.position = Point(x, y)
            m.rotation = rot

            is_valid, violation_type = self._check_constraints(m, machines_ep, step, placed)
            
            if not is_valid:
                # Invalid placement - revert and penalize
                m.position = old_pos
                m.rotation = old_rot
                
                if violation_type == 'collision':
                    r = self.collision_penalty
                    num_collisions += 1
                elif violation_type == 'origin':
                    r = self.collision_penalty  # Origin overlap is serious
                    num_collisions += 1
                else:  # bounds violation
                    r = self.invalid_placement_penalty
                
                # State doesn't advance properly for next machine
                done = False  # Episode continues but this machine failed
                
                # Next state: same step, same placed list, same prev_ap
                s2 = self._make_state(step, machines_ep, placed, prev_ap_world)
                
            else:
                # Valid placement!
                placed.append(step)
                
                # Calculate path improvement
                new_length = self._compute_path_length(machines_ep, placed)
                old_length = self._compute_path_length(machines_ep, placed[:-1]) if len(placed) > 1 else 0.0
                
                # Reward = path length reduction - small step penalty
                r = (old_length - new_length) + self.step_penalty
                
                # Update previous access point
                prev_ap_world = machines_ep[step].get_access_point_world()
                
                # Check if episode is done
                done = (step == self.num_machines - 1)
                
                # Next state
                if not done:
                    s2 = self._make_state(step + 1, machines_ep, placed, prev_ap_world)
                else:
                    s2 = s  # Terminal state
                    # Bonus for completing all placements
                    if len(placed) == self.num_machines:
                        r += 50.0  # Completion bonus

            # Store transition
            self.replay.append(Transition(s=s, a=a, r=float(r), s2=s2, done=done))

            # Optimize
            self._optimize_step()
            
            # If invalid and we've tried many times, skip this machine
            # (This prevents getting stuck, but penalizes the episode)
            if not is_valid and step not in placed:
                # Continue to next machine but don't add to placed
                pass

        final_length = self._compute_path_length(machines_ep, placed)
        return final_length, len(placed), num_collisions

    # ------------------------- DQN optimize -------------------------

    def _optimize_step(self):
        if len(self.replay) < self.batch_size:
            return

        batch = random.sample(self.replay, self.batch_size)
        s = torch.from_numpy(np.stack([t.s for t in batch])).float().to(self.device)
        a = torch.tensor([t.a for t in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor([t.r for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.from_numpy(np.stack([t.s2 for t in batch])).float().to(self.device)
        done = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_sa = self.q(s).gather(1, a)

        # Target: r + gamma * max_a' Q_tgt(s',a') * (1-done)
        with torch.no_grad():
            q_next = self.q_tgt(s2).max(dim=1, keepdim=True)[0]
            target = r + self.gamma * q_next * (1.0 - done)

        loss = nn.functional.smooth_l1_loss(q_sa, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_every == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())

    # ------------------------- Public training -------------------------

    def train_many_problems(
        self,
        problem_generator,
        num_problems: int = 100,
        episodes_per_problem: int = 200,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        verbose_every: int = 50,
    ):
        """Train across multiple problems with metrics tracking"""
        total_episodes = num_problems * episodes_per_problem
        ep_count = 0

        for p in range(1, num_problems + 1):
            machines, seq, robot_pos, bounds = problem_generator(p)
            self.set_problem(machines, seq, robot_pos, bounds)
            
            problem_success = []
            problem_lengths = []
            problem_collisions = []
            problem_placements = []

            for e in range(episodes_per_problem):
                frac = ep_count / max(1, total_episodes - 1)
                eps = epsilon_start + frac * (epsilon_end - epsilon_start)

                final_len, num_placed, num_collisions = self._run_episode(eps)
                
                success = (num_placed == self.num_machines)
                problem_success.append(success)
                problem_lengths.append(final_len)
                problem_collisions.append(num_collisions)
                problem_placements.append(num_placed / max(1, self.num_machines))
                
                ep_count += 1

                if (ep_count % verbose_every) == 0:
                    recent_success = np.mean(problem_success[-50:]) * 100 if len(problem_success) >= 50 else 0
                    recent_collisions = np.mean(problem_collisions[-50:]) if len(problem_collisions) >= 50 else 0
                    print(f"[DQN] ep={ep_count}/{total_episodes} | eps={eps:.3f} | "
                          f"len={final_len:.1f} | placed={num_placed}/{self.num_machines} | "
                          f"success_rate={recent_success:.1f}% | avg_coll={recent_collisions:.2f}")
            
            # Store problem-level metrics
            self.episode_metrics['success_rate'].append(np.mean(problem_success))
            self.episode_metrics['avg_path_length'].append(np.mean(problem_lengths))
            self.episode_metrics['collision_rate'].append(np.mean(problem_collisions))
            self.episode_metrics['placement_rate'].append(np.mean(problem_placements))

    # ------------------------- Greedy inference -------------------------

    def generate_layout_greedy(self, max_attempts_per_machine: int = 10) -> Tuple[List[Machine], float, np.ndarray, bool]:
        """
        Greedy rollout with proper failure handling.
        Returns: (machines, path_length, chromosome, success)
        """
        machines_ep = [deepcopy(m) for m in self.machines]
        for m in machines_ep:
            m.position = deepcopy(self.robot_position)
            m.rotation = 0.0

        placed: List[int] = []
        prev_ap_world = deepcopy(self.robot_position)
        success = True

        for step in range(self.num_machines):
            s = self._make_state(step, machines_ep, placed, prev_ap_world)

            with torch.no_grad():
                x = torch.from_numpy(s).float().to(self.device).unsqueeze(0)
                qvals = self.q(x).squeeze(0).cpu().numpy()

            # Get top actions
            sorted_actions = np.argsort(qvals)[::-1]
            
            placed_successfully = False
            for attempt, a in enumerate(sorted_actions[:max_attempts_per_machine]):
                xw, yw, rot = self._decode_action(int(a))

                m = machines_ep[step]
                m.position = Point(xw, yw)
                m.rotation = rot

                is_valid, _ = self._check_constraints(m, machines_ep, step, placed)
                
                if is_valid:
                    placed.append(step)
                    prev_ap_world = machines_ep[step].get_access_point_world()
                    placed_successfully = True
                    break
            
            if not placed_successfully:
                # Could not place this machine
                success = False
                # Keep machine at robot position (invalid but marked as failed)
                machines_ep[step].position = deepcopy(self.robot_position)
                machines_ep[step].rotation = 0.0

        length = self._compute_path_length(machines_ep, placed)
        chrom = []
        for m in machines_ep:
            chrom.extend([m.position.x, m.position.y, m.rotation])
        
        return machines_ep, length, np.array(chrom, dtype=np.float32), success

    # ------------------------- Save / load -------------------------

    def save(self, path: str):
        torch.save({
            "q": self.q.state_dict(),
            "q_tgt": self.q_tgt.state_dict(),
            "train_steps": self.train_steps,
            "metrics": self.episode_metrics
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q.load_state_dict(ckpt["q"])
        self.q_tgt.load_state_dict(ckpt["q_tgt"])
        self.q_tgt.eval()
        self.train_steps = ckpt.get("train_steps", 0)
        if "metrics" in ckpt:
            self.episode_metrics = ckpt["metrics"]
    
    def get_metrics_summary(self) -> dict:
        """Get summary of training metrics"""
        if not self.episode_metrics['success_rate']:
            return {}
        
        return {
            'final_success_rate': self.episode_metrics['success_rate'][-1],
            'avg_success_rate': np.mean(self.episode_metrics['success_rate']),
            'final_avg_path_length': self.episode_metrics['avg_path_length'][-1],
            'final_collision_rate': self.episode_metrics['collision_rate'][-1],
            'final_placement_rate': self.episode_metrics['placement_rate'][-1]
        }