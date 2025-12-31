from func.datastruct import Point, Machine
from algo.gen_algo_test import GeneticAlgorithm
from algo.agent import QLearningAgent
from algo.gen_algo_rl_test import RLGeneticAlgorithm
import numpy as np

def main():    
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

    # machines = [
    # Machine(id=1, shape='rectangle', width=4.0, height=3.0, 
    #         access_point=Point(1.5, 0)),
    # Machine(id=2, shape='l_shape', width=5.0, height=4.0, 
    #         access_point=Point(-1.0, 1.0),
    #         l_cutout_width=2.0, l_cutout_height=2.0),
    # Machine(id=3, shape='rectangle', width=3.5, height=2.5, 
    #         access_point=Point(0, -1.0)),
    # Machine(id=4, shape='l_shape', width=4.5, height=3.5, 
    #         access_point=Point(1.0, -0.5),
    #         l_cutout_width=1.5, l_cutout_height=1.5),
    # Machine(id=5, shape='l_shape', width=3.5, height=5.5, 
    #         access_point=Point(1.0, -1.0),
    #         l_cutout_width=1, l_cutout_height=2),
    # Machine(id=6, shape='l_shape', width=3.5, height=5.5, 
    #         access_point=Point(1.0, -1.0),
    #         l_cutout_width=1, l_cutout_height=2),
    # Machine(id=7, shape='l_shape', width=3.5, height=5.5, 
    #         access_point=Point(1.0, -1.0),
    #         l_cutout_width=1, l_cutout_height=2),
    # Machine(id=8, shape='rectangle', width=3.5, height=2.5, 
    #         access_point=Point(0, -1.0)),
    # ]
    
    # Define sequence of operations
    sequence = [1, 2, 3, 4, 5, 6, 7, 8]  # Robot visits machines in this order
    
    # Robot starting position (center of workspace)
    robot_position = Point(0, 0)
    
    # Workspace boundaries (min_x, max_x, min_y, max_y)
    workspace_bounds = (-15, 15, -15, 15)
    
    # LOAD TRAINED AGENT
    trained_agent = QLearningAgent(state_size=9, action_size=8)
    trained_agent.q_table = np.load("rl_ga_q_table_2.npy") 
    trained_agent.epsilon = 0.1

    print("=== Robotic Workcell Layout Optimization ===")
    print(f"Machines: {len(machines)}")
    print(f"Sequence: {sequence}")
    print(f"Workspace: {workspace_bounds}")
    print()
    
    # Create and run optimization
    runs = 5
    optimizer_list = [GeneticAlgorithm]
    
    # Dictionary to store results for each optimizer
    comparison_results = {
        'GeneticAlgorithm': {
            'best_fitness': [],
            'execution_time': [],
            'total_distance': [],
            'final_machines': []
        },
        'RLGeneticAlgorithm': {
            'best_fitness': [],
            'execution_time': [],
            'total_distance': [],
            'final_machines': []
        }
    }
    
    for OptimizerClass in optimizer_list:
        optimizer_name = OptimizerClass.__name__
        print(f"\n{'='*70}")
        print(f"Running {optimizer_name}")
        print(f"{'='*70}\n")
        
        for run in range(runs):
            print(f"\n--- Run {run + 1}/{runs} ---")
            
            # If optimizer is RL-GA, inject trained RL agent
            if OptimizerClass.__name__ == "RLGeneticAlgorithm":
                optimizer = OptimizerClass(
                    machines=machines,
                    sequence=sequence,
                    robot_position=robot_position,
                    workspace_bounds=workspace_bounds,
                    rl_agent=trained_agent
                )
            else:
                optimizer = OptimizerClass(
                    machines=machines,
                    sequence=sequence,
                    robot_position=robot_position,
                    workspace_bounds=workspace_bounds
                )
            
            # Run optimization
            optimized_machines, best_fitness, results = optimizer.optimize()
            
            # Store results
            comparison_results[optimizer_name]['best_fitness'].append(best_fitness)
            comparison_results[optimizer_name]['execution_time'].append(results['execution_time'])
            comparison_results[optimizer_name]['total_distance'].append(results['total_distance'])
            comparison_results[optimizer_name]['final_machines'].append(optimized_machines)
            
            # Display individual run results
            print("\n=== RUN RESULTS ===")
            print(f"Best Fitness: {best_fitness:.2f}")
            print(f"Total Distance: {results['total_distance']:.2f} units")
            print(f"Execution Time: {results['execution_time']:.2f} seconds")
            print(f"Generations: {results['generations']}")
    
    # Calculate and display comparison statistics
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    for optimizer_name in comparison_results.keys():
        results = comparison_results[optimizer_name]
        
        avg_fitness = np.mean(results['best_fitness'])
        std_fitness = np.std(results['best_fitness'])
        min_fitness = np.min(results['best_fitness'])
        max_fitness = np.max(results['best_fitness'])
        
        avg_time = np.mean(results['execution_time'])
        std_time = np.std(results['execution_time'])
        
        avg_distance = np.mean(results['total_distance'])
        std_distance = np.std(results['total_distance'])
        
        print(f"\n{optimizer_name}:")
        print(f"  Best Fitness:")
        print(f"    Average: {avg_fitness:.2f} ± {std_fitness:.2f}")
        print(f"    Min: {min_fitness:.2f}, Max: {max_fitness:.2f}")
        print(f"  Execution Time:")
        print(f"    Average: {avg_time:.2f} ± {std_time:.2f} seconds")
        print(f"  Total Distance:")
        print(f"    Average: {avg_distance:.2f} ± {std_distance:.2f} units")
    
    # Performance comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    ga_avg_fitness = np.mean(comparison_results['GeneticAlgorithm']['best_fitness'])
    rl_avg_fitness = np.mean(comparison_results['RLGeneticAlgorithm']['best_fitness'])
    
    ga_avg_time = np.mean(comparison_results['GeneticAlgorithm']['execution_time'])
    rl_avg_time = np.mean(comparison_results['RLGeneticAlgorithm']['execution_time'])
    
    fitness_improvement = ((ga_avg_fitness - rl_avg_fitness) / ga_avg_fitness) * 100
    time_difference = ((rl_avg_time - ga_avg_time) / ga_avg_time) * 100
    
    print(f"\nFitness Improvement: {fitness_improvement:+.2f}%")
    if fitness_improvement > 0:
        print(f"  → RL-GA achieved {fitness_improvement:.2f}% better fitness")
    else:
        print(f"  → Standard GA achieved {abs(fitness_improvement):.2f}% better fitness")
    
    print(f"\nTime Difference: {time_difference:+.2f}%")
    if time_difference > 0:
        print(f"  → RL-GA took {time_difference:.2f}% longer")
    else:
        print(f"  → RL-GA was {abs(time_difference):.2f}% faster")
    
    # Find best overall solution
    best_ga_idx = np.argmin(comparison_results['GeneticAlgorithm']['best_fitness'])
    best_rl_idx = np.argmin(comparison_results['RLGeneticAlgorithm']['best_fitness'])
    
    best_ga_fitness = comparison_results['GeneticAlgorithm']['best_fitness'][best_ga_idx]
    best_rl_fitness = comparison_results['RLGeneticAlgorithm']['best_fitness'][best_rl_idx]
    
    print(f"\nBest Solution Found:")
    if best_rl_fitness < best_ga_fitness:
        print(f"  → RL-GA (Run {best_rl_idx + 1}): Fitness = {best_rl_fitness:.2f}")
        best_machines = comparison_results['RLGeneticAlgorithm']['final_machines'][best_rl_idx]
    else:
        print(f"  → Standard GA (Run {best_ga_idx + 1}): Fitness = {best_ga_fitness:.2f}")
        best_machines = comparison_results['GeneticAlgorithm']['final_machines'][best_ga_idx]
    
    print("\nBest Machine Positions:")
    for machine in best_machines:
        print(f"Machine {machine.id}: "
              f"Position=({machine.position.x:.2f}, {machine.position.y:.2f}), "
              f"Rotation={machine.rotation:.1f}°")
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Best Fitness Comparison
        axes[0].bar(['GA', 'RL-GA'], 
                    [ga_avg_fitness, rl_avg_fitness],
                    yerr=[np.std(comparison_results['GeneticAlgorithm']['best_fitness']),
                          np.std(comparison_results['RLGeneticAlgorithm']['best_fitness'])],
                    capsize=5, color=['#3498db', '#e74c3c'], alpha=0.7)
        axes[0].set_ylabel('Average Best Fitness')
        axes[0].set_title('Fitness Comparison (Lower is Better)')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Execution Time Comparison
        axes[1].bar(['GA', 'RL-GA'], 
                    [ga_avg_time, rl_avg_time],
                    yerr=[np.std(comparison_results['GeneticAlgorithm']['execution_time']),
                          np.std(comparison_results['RLGeneticAlgorithm']['execution_time'])],
                    capsize=5, color=['#3498db', '#e74c3c'], alpha=0.7)
        axes[1].set_ylabel('Average Execution Time (seconds)')
        axes[1].set_title('Execution Time Comparison')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('optimizer_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\nComparison chart saved as 'optimizer_comparison.png'")
        plt.show()
    except ImportError:
        print("\nMatplotlib not available for visualization")
    
    return comparison_results, best_machines

if __name__ == "__main__":
    comparison_results, best_machines = main()