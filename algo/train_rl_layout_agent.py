import numpy as np

from func.datastruct import Point, Machine
from algo.rl_layout_agent import LayoutRLAgent
from visual.visualize import visualize_layout


def build_problem():
    machines = [
    Machine(id=1, shape='l_shape', width=5.21, height=4.33,
            access_point=Point(1.48, -1.12),
            l_cutout_width=2.01, l_cutout_height=1.72),

    Machine(id=2, shape='rectangle', width=4.87, height=3.12,
            access_point=Point(0.92, -0.44)),

    Machine(id=3, shape='l_shape', width=4.56, height=5.46,
            access_point=Point(-0.72, 1.89),
            l_cutout_width=1.58, l_cutout_height=2.74),

    Machine(id=4, shape='rectangle', width=3.44, height=2.61,
            access_point=Point(0.13, 0.89)),

    Machine(id=5, shape='l_shape', width=5.94, height=3.77,
            access_point=Point(1.04, -0.88),
            l_cutout_width=2.23, l_cutout_height=1.32),

    Machine(id=6, shape='rectangle', width=4.32, height=2.48,
            access_point=Point(-0.63, -0.41)),

    Machine(id=7, shape='l_shape', width=3.88, height=4.91,
            access_point=Point(0.55, 1.22),
            l_cutout_width=1.25, l_cutout_height=2.01),

    Machine(id=8, shape='rectangle', width=3.27, height=2.74,
            access_point=Point(0.15, -0.52)),
    ]


    # visiting sequence (e.g., in ID order or your custom process order)
    sequence = [m.id for m in machines]

    robot_position = Point(0.0, 0.0)
    workspace_bounds = (-15.0, 15.0, -15.0, 15.0)

    return machines, sequence, robot_position, workspace_bounds


def main():
    machines, sequence, robot_pos, bounds = build_problem()

    agent = LayoutRLAgent(
        machines=machines,
        sequence=sequence,
        robot_position=robot_pos,
        workspace_bounds=bounds,
        num_cells_x=10,
        num_cells_y=10,
    )

    print("Training RL layout agent (no GA)...")
    agent.train(
        num_episodes=500000,
        alpha=0.05,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        verbose=True,
    )

    # Evaluate greedy layouts after training
    best_L, mean_L, best_chrom = agent.evaluate_greedy(
        num_rollouts=500,
        rng_jitter=0.1,
    )
    print(f"[RL] Greedy evaluation: best={best_L:.3f}, mean={mean_L:.3f}")
    print(f"[RL] Example chromosome (for GA later) shape={best_chrom.shape}")

    agent.save_q_table("qtable_layout_rl_new.npy")

    # Option B: sample multiple greedy layouts and pick the best
    num_rollouts = 500
    best_machines = None
    best_length = None
    best_chrom = None

    for _ in range(num_rollouts):
        machines_layout, length, chrom = agent.generate_layout_greedy(rng_jitter=0.1)
        if best_length is None or length < best_length:
            best_length = length
            best_machines = machines_layout
            best_chrom = chrom

    print(f"[RL] Best greedy layout length = {best_length:.3f}")
    print(f"[RL] Best chromosome shape = {best_chrom.shape}")

    # ------------------------------------------------------------------
    # Visualize best layout using your existing visualize_layout()
    # ------------------------------------------------------------------
    # visualize_layout(
    #     machines=best_machines,
    #     robot_position=robot_pos,
    #     sequence=sequence,
    #     workspace_bounds=bounds,
    # )


if __name__ == "__main__":
    main()