"""
train_rl_agent.py
Train a single Q-learning agent that controls GA parameters across multiple 
randomly generated machine-layout problems, with automatic epoch optimization.

Usage (from project root):
    python -m algo.train_rl_agent
Or:
    python algo/train_rl_agent.py
"""
from typing import List, Tuple, Dict
import numpy as np
import random
from func.datastruct import Point, Machine
from algo.agent import QLearningAgent
from algo.gen_algo_rl_test import RLGeneticAlgorithm


# ============================================================
# 1. Random Problem Generator
# ============================================================
def generate_random_machine(
    machine_id: int,
    min_size: float = 2.0,
    max_size: float = 6.0,
    l_shape_probability: float = 0.4
) -> Machine:
    """Generate a single random machine with realistic parameters."""
    
    # Decide shape
    is_l_shape = random.random() < l_shape_probability
    shape = "l_shape" if is_l_shape else "rectangle"
    
    # Random dimensions
    width = random.uniform(min_size, max_size)
    height = random.uniform(min_size, max_size)
    
    # Random access point (relative to machine center)
    access_x = random.uniform(-width/3, width/3)
    access_y = random.uniform(-height/3, height/3)
    access_point = Point(access_x, access_y)
    
    if is_l_shape:
        # L-shape cutout parameters
        l_cutout_width = random.uniform(1.0, min(width * 0.5, 2.5))
        l_cutout_height = random.uniform(1.0, min(height * 0.5, 2.5))
        
        return Machine(
            id=machine_id,
            shape=shape,
            width=width,
            height=height,
            access_point=access_point,
            l_cutout_width=l_cutout_width,
            l_cutout_height=l_cutout_height
        )
    else:
        return Machine(
            id=machine_id,
            shape=shape,
            width=width,
            height=height,
            access_point=access_point
        )


def generate_random_problem(
    num_machines: int = 8,
    workspace_size: float = 15.0,
    **kwargs
) -> Tuple[List[Machine], List[int], Point, Tuple[float, float, float, float]]:
    """Generate a complete random machine layout problem."""
    
    machines = [
        generate_random_machine(i + 1, **kwargs) 
        for i in range(num_machines)
    ]
    
    sequence = list(range(1, num_machines + 1))
    robot_position = Point(0, 0)
    workspace_bounds = (-workspace_size, workspace_size, -workspace_size, workspace_size)
    
    return machines, sequence, robot_position, workspace_bounds


def build_training_problems(
    num_problems: int = 20,
    num_machines_range: Tuple[int, int] = (6, 10),
    variety_configs: List[Dict] = None
) -> List[Tuple[List[Machine], List[int], Point, Tuple[float, float, float, float]]]:
    """
    Generate multiple random training problems with variety.
    
    Args:
        num_problems: Total number of problems to generate
        num_machines_range: (min, max) number of machines per problem
        variety_configs: List of config dicts for different problem types
    """
    
    if variety_configs is None:
        # Default variety: small, medium, large, mixed
        variety_configs = [
            {"min_size": 2.0, "max_size": 3.5, "l_shape_probability": 0.3},  # Small machines
            {"min_size": 3.0, "max_size": 5.0, "l_shape_probability": 0.4},  # Medium machines
            {"min_size": 4.5, "max_size": 6.5, "l_shape_probability": 0.5},  # Large machines
            {"min_size": 2.0, "max_size": 6.0, "l_shape_probability": 0.6},  # Mixed with many L-shapes
        ]
    
    problems = []
    
    for i in range(num_problems):
        # Randomly select configuration
        config = random.choice(variety_configs)
        
        # Randomly select number of machines
        num_machines = random.randint(*num_machines_range)
        
        # Generate problem
        problem = generate_random_problem(
            num_machines=num_machines,
            **config
        )
        
        problems.append(problem)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_problems} problems...")
    
    print(f"\nTotal problems generated: {len(problems)}")
    return problems


# ============================================================
# 2. Training with Early Stopping & Convergence Detection
# ============================================================
def train_agent_with_optimization(
    num_problems: int = 20,
    max_epochs: int = 10,
    min_epochs: int = 3,
    convergence_window: int = 3,
    convergence_threshold: float = 0.02,
    q_table_path: str = "rl_ga_q_table_3.npy",
    state_size: int = 9,
    action_size: int = 8,
    num_machines_range: Tuple[int, int] = (6, 10),
) -> Tuple[QLearningAgent, Dict]:
    """
    Train RL agent with automatic epoch optimization via convergence detection.
    
    Args:
        num_problems: Number of random problems to generate
        max_epochs: Maximum training epochs
        min_epochs: Minimum epochs before checking convergence
        convergence_window: Number of epochs to check for convergence
        convergence_threshold: Max relative change to consider converged
        q_table_path: Path to save final Q-table
        state_size: State space dimension
        action_size: Action space dimension
        num_machines_range: (min, max) machines per problem
    
    Returns:
        Trained agent and training statistics
    """
    
    print(f"\n{'='*60}")
    print(f"REINFORCEMENT LEARNING AGENT TRAINING")
    print(f"{'='*60}")
    print(f"Generating {num_problems} random problems...")
    print(f"Max epochs: {max_epochs}, Min epochs: {min_epochs}")
    print(f"Convergence window: {convergence_window}, Threshold: {convergence_threshold}")
    print(f"{'='*60}\n")
    
    # Generate diverse training problems
    problems = build_training_problems(
        num_problems=num_problems,
        num_machines_range=num_machines_range
    )
    
    # Initialize agent
    agent = QLearningAgent(state_size=state_size, action_size=action_size)
    
    # Training metrics
    epoch_metrics = []
    fitness_history = []
    
    print(f"\n{'='*60}")
    print("Starting Training...")
    print(f"{'='*60}\n")
    
    for epoch in range(max_epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{max_epochs}")
        print(f"{'='*60}")
        
        epoch_fitnesses = []
        epoch_distances = []
        
        # Shuffle problems each epoch for better generalization
        random.shuffle(problems)
        
        for idx, (machines, sequence, robot_position, workspace_bounds) in enumerate(problems):
            # Progress indicator
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"Problem {idx + 1}/{len(problems)}...", end="\r")
            
            # Run GA with RL agent
            ga = RLGeneticAlgorithm(
                machines=machines,
                sequence=sequence,
                robot_position=robot_position,
                workspace_bounds=workspace_bounds,
                rl_agent=agent,
            )
            
            final_layout, best_fitness, results = ga.optimize()
            
            epoch_fitnesses.append(best_fitness)
            epoch_distances.append(results.get('total_distance', float('inf')))
        
        # Calculate epoch statistics
        avg_fitness = np.mean(epoch_fitnesses)
        std_fitness = np.std(epoch_fitnesses)
        avg_distance = np.mean(epoch_distances)
        
        epoch_metrics.append({
            'epoch': epoch + 1,
            'avg_fitness': avg_fitness,
            'std_fitness': std_fitness,
            'avg_distance': avg_distance,
            'epsilon': agent.epsilon
        })
        
        fitness_history.append(avg_fitness)
        
        print(f"\n")
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Avg Fitness: {avg_fitness:.4f} (±{std_fitness:.4f})")
        print(f"  Avg Distance: {avg_distance:.4f}")
        print(f"  Epsilon: {agent.epsilon:.4f}")
        
        # Check for convergence after minimum epochs
        if epoch >= min_epochs:
            if len(fitness_history) >= convergence_window:
                recent_fitness = fitness_history[-convergence_window:]
                fitness_change = (max(recent_fitness) - min(recent_fitness)) / (abs(np.mean(recent_fitness)) + 1e-10)
                
                print(f"  Convergence Check: {fitness_change:.4f} (threshold: {convergence_threshold})")
                
                if fitness_change < convergence_threshold:
                    print(f"\n{'='*60}")
                    print(f"CONVERGENCE DETECTED after {epoch + 1} epochs!")
                    print(f"Training stopped early.")
                    print(f"{'='*60}\n")
                    break
    
    # Save Q-table
    np.save(q_table_path, agent.q_table)
    print(f"\nQ-table saved to: {q_table_path}")
    
    # Final statistics
    training_stats = {
        'total_epochs': len(epoch_metrics),
        'final_avg_fitness': epoch_metrics[-1]['avg_fitness'],
        'final_epsilon': agent.epsilon,
        'epoch_metrics': epoch_metrics,
        'converged': len(epoch_metrics) < max_epochs
    }
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total Epochs: {training_stats['total_epochs']}")
    print(f"Final Avg Fitness: {training_stats['final_avg_fitness']:.4f}")
    print(f"Converged: {training_stats['converged']}")
    print(f"{'='*60}\n")
    
    return agent, training_stats


# ============================================================
# 3. Load trained agent
# ============================================================
def load_trained_agent(
    q_table_path: str = "rl_ga_q_table_3.npy",
    state_size: int = 9,
    action_size: int = 8,
    epsilon: float = 0.0,
) -> QLearningAgent:
    """Load a trained QLearningAgent from saved Q-table."""
    q_table = np.load(q_table_path)
    agent = QLearningAgent(state_size=state_size, action_size=action_size)
    agent.q_table = q_table
    agent.epsilon = epsilon
    print(f"Loaded trained agent from: {q_table_path}")
    return agent


# ============================================================
# 4. Test trained agent on new problem
# ============================================================
def test_trained_agent(
    q_table_path: str = "rl_ga_q_table_3.npy",
    num_test_problems: int = 10
):
    """Test the trained agent on new random problems."""
    
    print(f"\n{'='*60}")
    print("TESTING TRAINED AGENT")
    print(f"{'='*60}\n")
    
    agent = load_trained_agent(q_table_path=q_table_path, epsilon=0.0)
    
    test_problems = build_training_problems(num_problems=num_test_problems)
    
    test_fitnesses = []
    test_distances = []
    
    for idx, (machines, sequence, robot_position, workspace_bounds) in enumerate(test_problems):
        print(f"Test Problem {idx + 1}/{num_test_problems}...", end="\r")
        
        ga = RLGeneticAlgorithm(
            machines=machines,
            sequence=sequence,
            robot_position=robot_position,
            workspace_bounds=workspace_bounds,
            rl_agent=agent,
        )
        
        final_layout, best_fitness, results = ga.optimize()
        
        test_fitnesses.append(best_fitness)
        test_distances.append(results.get('total_distance', float('inf')))
    
    print(f"\n")
    print(f"Test Results:")
    print(f"  Avg Fitness: {np.mean(test_fitnesses):.4f} (±{np.std(test_fitnesses):.4f})")
    print(f"  Avg Distance: {np.mean(test_distances):.4f}")
    print(f"{'='*60}\n")


# ============================================================
# 5. Main
# ============================================================
if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    random.seed(42)
    np.random.seed(42)
    
    # Train agent with automatic optimization
    trained_agent, stats = train_agent_with_optimization(
        num_problems=20,           # Number of random problems
        max_epochs=10,             # Maximum epochs
        min_epochs=3,              # Minimum before convergence check
        convergence_window=3,      # Epochs to check for convergence
        convergence_threshold=0.02, # Convergence sensitivity
        q_table_path="rl_ga_q_table_3.npy",
        num_machines_range=(6, 10) # Machine count variety
    )
    
    # Optional: Test on new problems
    print("\nTesting trained agent on new problems...")
    test_trained_agent(
        q_table_path="rl_ga_q_table_3.npy",
        num_test_problems=10
    )