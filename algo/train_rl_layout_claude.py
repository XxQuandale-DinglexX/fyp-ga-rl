import random
import numpy as np
import matplotlib.pyplot as plt
from func.datastruct import Point, Machine
from algo.rl_layout_claude import DQNLayoutRLAgent

def generate_problem(problem_idx: int):
    base_seed = 1000 + problem_idx
    random.seed(base_seed)
    np.random.seed(base_seed)
    
    workspace_bounds = (-15, 15, -15, 15)
    robot_position = Point(0, 0)
    
    # Generate 6-8 machines (variable for more diversity)
    num_machines = random.randint(6, 8)
    machines = []
    
    for i in range(num_machines):
        m_id = i + 1
        shape = "l_shape" if i % 2 == 0 else "rectangle"
        
        # Slightly smaller machines to ensure feasibility
        width = random.uniform(2.5, 5.0)
        height = random.uniform(2.0, 4.5)
        
        ap_x = random.uniform(-1.5, 1.5)
        ap_y = random.uniform(-1.5, 1.5)
        
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
    
    sequence = list(range(1, num_machines + 1))
    return machines, sequence, robot_position, workspace_bounds


def validate_agent(agent, num_test_problems=20):
    """Test agent on held-out problems"""
    print("\n" + "="*60)
    print("VALIDATION ON HELD-OUT PROBLEMS")
    print("="*60)
    
    successes = 0
    path_lengths = []
    placement_rates = []
    
    for i in range(num_test_problems):
        # Use different seed range for validation
        machines, seq, robot_pos, bounds = generate_problem(10000 + i)
        agent.set_problem(machines, seq, robot_pos, bounds)
        
        machines_result, length, _, success = agent.generate_layout_greedy(
            max_attempts_per_machine=20
        )
        
        if success:
            successes += 1
            path_lengths.append(length)
        
        # Count how many were actually placed
        num_placed = sum(1 for m in machines_result 
                        if not (m.position.x == 0 and m.position.y == 0))
        placement_rates.append(num_placed / len(machines))
    
    print(f"Success Rate: {successes}/{num_test_problems} ({successes/num_test_problems*100:.1f}%)")
    if path_lengths:
        print(f"Avg Path Length (successful): {np.mean(path_lengths):.2f} ± {np.std(path_lengths):.2f}")
    print(f"Avg Placement Rate: {np.mean(placement_rates)*100:.1f}%")
    print("="*60 + "\n")
    
    return {
        'success_rate': successes / num_test_problems,
        'avg_path_length': np.mean(path_lengths) if path_lengths else float('inf'),
        'placement_rate': np.mean(placement_rates)
    }


def plot_training_curves(agent, save_path='training_curves.png'):
    """Visualize training progress"""
    metrics = agent.episode_metrics
    
    if not metrics['success_rate']:
        print("No metrics to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Success rate
    axes[0, 0].plot(metrics['success_rate'])
    axes[0, 0].set_title('Success Rate per Problem')
    axes[0, 0].set_xlabel('Problem Index')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].grid(True)
    
    # Path length
    axes[0, 1].plot(metrics['avg_path_length'])
    axes[0, 1].set_title('Average Path Length per Problem')
    axes[0, 1].set_xlabel('Problem Index')
    axes[0, 1].set_ylabel('Path Length')
    axes[0, 1].grid(True)
    
    # Collision rate
    axes[1, 0].plot(metrics['collision_rate'])
    axes[1, 0].set_title('Collision Rate per Problem')
    axes[1, 0].set_xlabel('Problem Index')
    axes[1, 0].set_ylabel('Collisions per Episode')
    axes[1, 0].grid(True)
    
    # Placement rate
    axes[1, 1].plot(metrics['placement_rate'])
    axes[1, 1].set_title('Placement Rate per Problem')
    axes[1, 1].set_xlabel('Problem Index')
    axes[1, 1].set_ylabel('Placement Rate')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved to {save_path}")
    plt.close()


def main():
    print("="*60)
    print("DQN LAYOUT AGENT TRAINING")
    print("="*60)
    
    agent = DQNLayoutRLAgent(
        num_cells_x=10,
        num_cells_y=10,
        collision_penalty=-150.0,
        invalid_placement_penalty=-50.0,
        step_penalty=-1.0,
        replay_capacity=100_000,
        batch_size=256,
        lr=1e-3,
        gamma=0.95,
        target_update_every=1000,
        hidden=256,
        seed=0,
    )
    
    # Training configuration
    num_problems = 300           # Reduced to allow more episodes per problem
    episodes_per_problem = 300   # Increased for better convergence
    checkpoint_every = 50        # Save checkpoint every N problems
    
    print(f"\nTraining Configuration:")
    print(f"  Total problems: {num_problems}")
    print(f"  Episodes per problem: {episodes_per_problem}")
    print(f"  Total episodes: {num_problems * episodes_per_problem}")
    print(f"  Checkpoint frequency: every {checkpoint_every} problems")
    print()
    
    # Train with periodic validation
    for checkpoint_idx in range(0, num_problems, checkpoint_every):
        remaining = min(checkpoint_every, num_problems - checkpoint_idx)

        print(f"\n--- Training problems {checkpoint_idx+1} to {checkpoint_idx + remaining} ---")

        def sub_generator(local_p: int):
            # local_p is 1..remaining
            global_problem_idx = checkpoint_idx + local_p
            return generate_problem(global_problem_idx)

        eps_start = max(1.0 - (checkpoint_idx / max(1, num_problems - 1)), 0.05)

        agent.train_many_problems(
            problem_generator=sub_generator,
            num_problems=remaining,
            episodes_per_problem=episodes_per_problem,
            epsilon_start=eps_start,
            epsilon_end=0.05,
            verbose_every=100,
        )

        val_metrics = validate_agent(agent, num_test_problems=20)

        checkpoint_path = f"dqn_layout_checkpoint_{checkpoint_idx + remaining}.pt"
        agent.save(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    # Final save
    agent.save("dqn_layout_final.pt")
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    # Print final metrics
    print("\nFinal Training Metrics:")
    train_metrics = agent.get_metrics_summary()
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Final validation
    print("\nFinal Validation:")
    final_val = validate_agent(agent, num_test_problems=50)
    
    # Plot learning curves
    plot_training_curves(agent)
    
    print("\nModel saved as: dqn_layout_final.pt")
    print("Training curves saved as: training_curves.png")


if __name__ == "__main__":
    main()