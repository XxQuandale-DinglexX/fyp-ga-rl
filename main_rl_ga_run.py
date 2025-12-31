import os
import numpy as np
import random
import time
from func.datastruct import Point, Machine
from algo.gen_algo_init import GeneticAlgorithm
from algo.rl_layout_agent import LayoutRLAgent


def generate_problem(problem_idx: int):
    """Generate one problem instance with 8 machines"""
    base_seed = 1000 + problem_idx
    random.seed(base_seed)
    np.random.seed(base_seed)

    workspace_bounds = (-15, 15, -15, 15)
    robot_position = Point(0, 0)

    machines = []
    for i in range(8):
        m_id = i + 1
        shape = "l_shape" if i % 2 == 0 else "rectangle"
        
        width = random.uniform(3.0, 6.0)
        height = random.uniform(2.5, 5.0)
        
        ap_x = random.uniform(-1.5, 1.5)
        ap_y = random.uniform(-1.5, 1.5)

        if shape == "l_shape":
            l_cutout_width = random.uniform(1.0, width / 2.0)
            l_cutout_height = random.uniform(1.0, height / 2.0)
            machine = Machine(
                id=m_id, shape="l_shape", width=width, height=height,
                access_point=Point(ap_x, ap_y),
                l_cutout_width=l_cutout_width,
                l_cutout_height=l_cutout_height,
            )
        else:
            machine = Machine(
                id=m_id, shape="rectangle", width=width, height=height,
                access_point=Point(ap_x, ap_y),
            )

        machines.append(machine)

    sequence = list(range(1, 9))
    return machines, sequence, robot_position, workspace_bounds


def create_rl_initial_population(agent: LayoutRLAgent, pop_size: int):
    """
    Create initial population using trained RL agent.
    Returns list of chromosomes.
    """
    population = []
    for _ in range(pop_size):
        _, _, chrom = agent.generate_layout_greedy(rng_jitter=0.1)
        population.append(chrom)
    return population


def run_ga_with_init(machines, sequence, robot_pos, bounds, 
                      initial_population, run_name: str):
    """
    Run GA optimization with given initial population.
    Returns (best_fitness, results_dict)
    """
    ga = GeneticAlgorithm(
        machines=machines,
        sequence=sequence,
        robot_position=robot_pos,
        workspace_bounds=bounds,  # Match init pop size
    )
    
    start_time = time.time()
    optimized_machines, best_fitness, results = ga.optimize(
        initial_population=initial_population
    )
    end_time = time.time()
    
    results['execution_time'] = end_time - start_time
    results['optimized_machines'] = optimized_machines
    
    return best_fitness, results


def main():
    """
    Compare GA with two different initialization strategies:
    1. Random initialization (baseline)
    2. RL-based initialization (our method)
    """
    
    # Configuration
    num_problems = 1
    runs_per_problem = 1  # Increase if you want multiple runs per problem
    pop_size = 200
    
    qtable_path = "qtable_layout_rl_new.npy"
    if not os.path.exists(qtable_path):
        raise FileNotFoundError(
            f"{qtable_path} not found. Run RL training script first."
        )
    
    print("="*80)
    print("GA COMPARISON: Random Init vs RL Init")
    print("="*80)
    print(f"Number of problems: {num_problems}")
    print(f"Runs per problem: {runs_per_problem}")
    print(f"Population size: {pop_size}")
    print(f"RL Q-table: {qtable_path}")
    print()
    
    os.makedirs("results", exist_ok=True)
    
    # Storage for all results
    all_results = {
        'random_init': {
            'best_fitness': [],
            'final_fitness': [],
            'execution_time': [],
            'total_distance': [],
            'generations': [],
        },
        'rl_init': {
            'best_fitness': [],
            'final_fitness': [],
            'execution_time': [],
            'total_distance': [],
            'generations': [],
        }
    }
    
    # Main comparison loop
    for problem_idx in range(num_problems):
        print(f"\n{'='*80}")
        print(f"PROBLEM {problem_idx + 1}/{num_problems}")
        print(f"{'='*80}")
        
        machines, sequence, robot_pos, bounds = generate_problem(problem_idx)
        
        # Create RL agent for this problem
        agent = LayoutRLAgent(
            machines=machines,
            sequence=sequence,
            robot_position=robot_pos,
            workspace_bounds=bounds,
            num_cells_x=8,
            num_cells_y=8,
        )
        agent.load_q_table(qtable_path)
        
        for run in range(runs_per_problem):
            if runs_per_problem > 1:
                print(f"\n--- Run {run + 1}/{runs_per_problem} ---")
            
            # Set seed for reproducibility
            seed = 42 + problem_idx * 100 + run
            random.seed(seed)
            np.random.seed(seed)
            
            # --- Method 1: Random Initialization (Baseline) ---
            print("\n[1] GA with Random Initialization...")
            
            ga_random = GeneticAlgorithm(
                machines=machines,
                sequence=sequence,
                robot_position=robot_pos,
                workspace_bounds=bounds
            )
            random_init_pop = ga_random.create_initial_population()
            
            # Evaluate initial population
            initial_fitness_random = []
            for chrom in random_init_pop:
                initial_fitness_random.append(ga_random.fitness_function(chrom))
            best_initial_random = min(initial_fitness_random)
            
            # Run GA
            best_fitness_random, results_random = run_ga_with_init(
                machines, sequence, robot_pos, bounds, 
                random_init_pop, "Random Init"
            )
            
            print(f"  Initial best: {best_initial_random:.2f}")
            print(f"  Final best:   {best_fitness_random:.2f}")
            print(f"  Improvement:  {best_initial_random - best_fitness_random:.2f}")
            print(f"  Time:         {results_random['execution_time']:.2f}s")
            print(f"  Generations:  {results_random['generations']}")
            
            # --- Method 2: RL Initialization ---
            print("\n[2] GA with RL Initialization...")
            
            # Reset seed for fair comparison
            random.seed(seed + 1000)
            np.random.seed(seed + 1000)
            
            # Create RL-initialized population
            rl_init_pop = create_rl_initial_population(agent, pop_size)
            
            # Evaluate initial population
            ga_rl = GeneticAlgorithm(
                machines=machines,
                sequence=sequence,
                robot_position=robot_pos,
                workspace_bounds=bounds
            )
            initial_fitness_rl = []
            for chrom in rl_init_pop:
                initial_fitness_rl.append(ga_rl.fitness_function(chrom))
            best_initial_rl = min(initial_fitness_rl)
            
            # Run GA
            best_fitness_rl, results_rl = run_ga_with_init(
                machines, sequence, robot_pos, bounds,
                rl_init_pop, "RL Init"
            )
            
            print(f"  Initial best: {best_initial_rl:.2f}")
            print(f"  Final best:   {best_fitness_rl:.2f}")
            print(f"  Improvement:  {best_initial_rl - best_fitness_rl:.2f}")
            print(f"  Time:         {results_rl['execution_time']:.2f}s")
            print(f"  Generations:  {results_rl['generations']}")
            
            # --- Comparison ---
            print("\n[COMPARISON]")
            print(f"  Random Init Final: {best_fitness_random:.2f}")
            print(f"  RL Init Final:     {best_fitness_rl:.2f}")
            
            improvement = ((best_fitness_random - best_fitness_rl) / best_fitness_random) * 100
            if improvement > 0:
                print(f"  ✅ RL Init is {improvement:.1f}% better")
            else:
                print(f"  ❌ Random Init is {abs(improvement):.1f}% better")
            
            # Store results
            all_results['random_init']['best_fitness'].append(best_initial_random)
            all_results['random_init']['final_fitness'].append(best_fitness_random)
            all_results['random_init']['execution_time'].append(results_random['execution_time'])
            all_results['random_init']['total_distance'].append(results_random['total_distance'])
            all_results['random_init']['generations'].append(results_random['generations'])
            
            all_results['rl_init']['best_fitness'].append(best_initial_rl)
            all_results['rl_init']['final_fitness'].append(best_fitness_rl)
            all_results['rl_init']['execution_time'].append(results_rl['execution_time'])
            all_results['rl_init']['total_distance'].append(results_rl['total_distance'])
            all_results['rl_init']['generations'].append(results_rl['generations'])
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*80)
    print("FINAL SUMMARY ACROSS ALL PROBLEMS")
    print("="*80)
    
    random_final = np.array(all_results['random_init']['final_fitness'])
    rl_final = np.array(all_results['rl_init']['final_fitness'])
    
    random_initial = np.array(all_results['random_init']['best_fitness'])
    rl_initial = np.array(all_results['rl_init']['best_fitness'])
    
    print("\nInitial Population Quality:")
    print(f"  Random Init - Mean: {np.mean(random_initial):.2f}, Median: {np.median(random_initial):.2f}")
    print(f"  RL Init     - Mean: {np.mean(rl_initial):.2f}, Median: {np.median(rl_initial):.2f}")
    init_improvement = ((np.mean(random_initial) - np.mean(rl_initial)) / np.mean(random_initial)) * 100
    print(f"  → RL init is {init_improvement:.1f}% better initially")
    
    print("\nFinal Optimized Results:")
    print(f"  Random Init - Mean: {np.mean(random_final):.2f}, Median: {np.median(random_final):.2f}")
    print(f"  RL Init     - Mean: {np.mean(rl_final):.2f}, Median: {np.median(rl_final):.2f}")
    final_improvement = ((np.mean(random_final) - np.mean(rl_final)) / np.mean(random_final)) * 100
    print(f"  → RL init is {final_improvement:.1f}% better after GA optimization")
    
    print("\nExecution Time:")
    random_time = np.array(all_results['random_init']['execution_time'])
    rl_time = np.array(all_results['rl_init']['execution_time'])
    print(f"  Random Init - Mean: {np.mean(random_time):.2f}s")
    print(f"  RL Init     - Mean: {np.mean(rl_time):.2f}s")
    
    print("\nGenerations:")
    random_gens = np.array(all_results['random_init']['generations'])
    rl_gens = np.array(all_results['rl_init']['generations'])
    print(f"  Random Init - Mean: {np.mean(random_gens):.0f}")
    print(f"  RL Init     - Mean: {np.mean(rl_gens):.0f}")
    
    # Statistical significance test
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(random_final, rl_final)
    print(f"\nStatistical Significance (paired t-test):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"  → Difference is statistically significant (p < 0.05)")
    else:
        print(f"  → Difference is NOT statistically significant (p >= 0.05)")
    
    # Save numerical results
    np.savez(
        "results/ga_comparison_results.npz",
        random_initial=random_initial,
        rl_initial=rl_initial,
        random_final=random_final,
        rl_final=rl_final,
        random_time=random_time,
        rl_time=rl_time,
        random_gens=random_gens,
        rl_gens=rl_gens,
    )
    print("\n✓ Results saved to results/ga_comparison_results.npz")
    
    # ========== VISUALIZATIONS ==========
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Boxplot: Final Fitness
        ax = axes[0, 0]
        bp = ax.boxplot(
            [random_final, rl_final],
            labels=["Random Init", "RL Init"],
            showmeans=True,
            meanline=False,
            showfliers=True,
            notch=True,
            patch_artist=True
        )
        colors = ["skyblue", "lightcoral"]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_ylabel("Final Best Fitness")
        ax.set_title("Final Fitness After GA Optimization\n(Lower is Better)")
        ax.grid(axis="y", alpha=0.3)
        
        # 2. Boxplot: Initial vs Final
        ax = axes[0, 1]
        positions = [1, 2, 4, 5]
        bp = ax.boxplot(
            [random_initial, random_final, rl_initial, rl_final],
            positions=positions,
            labels=["Random\nInitial", "Random\nFinal", "RL\nInitial", "RL\nFinal"],
            showmeans=True,
            patch_artist=True
        )
        colors_combined = ["lightblue", "skyblue", "lightcoral", "red"]
        for patch, color in zip(bp['boxes'], colors_combined):
            patch.set_facecolor(color)
        ax.set_ylabel("Best Fitness")
        ax.set_title("Initial vs Final Fitness Comparison")
        ax.grid(axis="y", alpha=0.3)
        
        # 3. Histogram: Improvement Distribution
        ax = axes[1, 0]
        random_improvement = random_initial - random_final
        rl_improvement = rl_initial - rl_final
        ax.hist(random_improvement, bins=20, alpha=0.6, label="Random Init", color="skyblue")
        ax.hist(rl_improvement, bins=20, alpha=0.6, label="RL Init", color="lightcoral")
        ax.set_xlabel("Fitness Improvement (Initial - Final)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Fitness Improvements")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        
        # 4. Scatter: Initial vs Final
        ax = axes[1, 1]
        ax.scatter(random_initial, random_final, alpha=0.6, label="Random Init", color="skyblue")
        ax.scatter(rl_initial, rl_final, alpha=0.6, label="RL Init", color="lightcoral")
        
        # Add diagonal line (no improvement)
        min_val = min(random_initial.min(), rl_initial.min(), random_final.min(), rl_final.min())
        max_val = max(random_initial.max(), rl_initial.max(), random_final.max(), rl_final.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label="No Improvement")
        
        ax.set_xlabel("Initial Best Fitness")
        ax.set_ylabel("Final Best Fitness")
        ax.set_title("Initial vs Final Fitness Correlation")
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("results/ga_comparison_plots.png", dpi=300, bbox_inches="tight")
        print("✓ Plots saved to results/ga_comparison_plots.png")
        plt.show()
        
    except ImportError:
        print("\n(matplotlib/scipy not available, skipping visualization)")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    results = main()