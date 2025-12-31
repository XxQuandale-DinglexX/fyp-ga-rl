import os
import numpy as np
import random
from func.datastruct import Point, Machine
from algo.gen_algo_init import GeneticAlgorithm
from algo.rl_layout_agent import LayoutRLAgent  # Tabular (single problem)
# Uncomment for DQN:
from algo.rl_layout_claude import DQNLayoutRLAgent
# Uncomment for multi-problem tabular:
from algo.rl_layout_agent_multi import MultiProblemTabularRLAgent


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


def evaluate_population_fitness(ga: GeneticAlgorithm, population: np.ndarray):
    """Evaluate a whole population with GA's fitness_function"""
    fitness_scores = []
    for chrom in population:
        f = ga.fitness_function(chrom)
        fitness_scores.append(f)

    finite = [f for f in fitness_scores if f != float("inf")]
    avg_fitness = float(np.mean(finite)) if finite else float("inf")
    best_fitness = float(np.min(fitness_scores))
    success_rate = len(finite) / len(fitness_scores) if fitness_scores else 0.0
    
    return best_fitness, avg_fitness, success_rate


def test_single_problem_tabular():
    """
    Test the single-problem tabular RL agent.
    NOTE: This agent was trained on ONE specific problem,
    so it will only work well on THAT problem!
    """
    print("="*70)
    print("TESTING SINGLE-PROBLEM TABULAR RL")
    print("="*70)
    print("⚠️  This agent was trained on problem_idx=0 only!")
    print()
    
    # Use the SAME problem it was trained on
    problem_idx = 0
    machines, sequence, robot_pos, bounds = generate_problem(problem_idx)
    
    ga = GeneticAlgorithm(
        machines=machines,
        sequence=sequence,
        robot_position=robot_pos,
        workspace_bounds=bounds,
    )
    pop_size = 500  # Smaller for quick test
    
    # --- GA random baseline ---
    ga_random_pop = np.stack(ga.create_initial_population(), axis=0)
    ga_best, ga_avg, ga_success = evaluate_population_fitness(ga, ga_random_pop)
    
    # --- RL agent ---
    agent = LayoutRLAgent(
        machines=machines,
        sequence=sequence,
        robot_position=robot_pos,
        workspace_bounds=bounds,
        num_cells_x=8,
        num_cells_y=8,
    )
    
    qtable_path = "qtable_layout_rl_new.npy"
    if not os.path.exists(qtable_path):
        print(f"❌ {qtable_path} not found!")
        print("   Run training first: python train_single_problem_rl.py")
        return
    
    agent.load_q_table(qtable_path)
    
    # Generate RL population
    rl_pop = []
    for i in range(pop_size):
        _, _, chrom = agent.generate_layout_greedy(rng_jitter=0.1)
        rl_pop.append(chrom)
    rl_pop = np.stack(rl_pop, axis=0)
    rl_best, rl_avg, rl_success = evaluate_population_fitness(ga, rl_pop)
    
    # Results
    print(f"\n[RESULTS ON TRAINING PROBLEM]")
    print(f"  GA Random:")
    print(f"    Best:    {ga_best:.3f}")
    print(f"    Average: {ga_avg:.3f}")
    print(f"    Success: {ga_success*100:.1f}%")
    print(f"  RL Init:")
    print(f"    Best:    {rl_best:.3f}")
    print(f"    Average: {rl_avg:.3f}")
    print(f"    Success: {rl_success*100:.1f}%")
    
    if rl_best < ga_best:
        print(f"\n  ✅ RL is {((ga_best - rl_best)/ga_best*100):.1f}% better!")
    else:
        print(f"\n  ❌ RL is worse by {((rl_best - ga_best)/ga_best*100):.1f}%")
    
    # Test on NEW problem (should fail)
    print(f"\n[TESTING ON NEW PROBLEM (problem_idx=999)]")
    machines_new, sequence_new, robot_pos_new, bounds_new = generate_problem(999)
    
    # This will fail because agent expects SAME machines!
    print("  ⚠️  This will likely fail - single-problem agent doesn't generalize")


def test_multi_problem_tabular():
    """
    Test the multi-problem tabular RL agent.
    This should work on different problems.
    """
    print("="*70)
    print("TESTING MULTI-PROBLEM TABULAR RL")
    print("="*70)
    
    from algo.rl_layout_agent_multi import MultiProblemTabularRLAgent
    
    # Load trained agent
    agent = MultiProblemTabularRLAgent(
        num_cells_x=12,
        num_cells_y=12,
        collision_penalty=-100.0,
        num_machines=8,
    )
    
    model_path = "tabular_rl_multi_problem.pkl"
    if not os.path.exists(model_path):
        print(f"❌ {model_path} not found!")
        print("   Run training first: python train_multi_problem_tabular.py")
        return
    
    agent.load(model_path)
    print(f"✓ Loaded model with {len(agent.Q)} state-action pairs")
    print()
    
    # Test on multiple NEW problems
    num_problems = 20
    ga_best_list = []
    rl_best_list = []
    rl_success_list = []
    
    for problem_idx in range(num_problems):
        machines, sequence, robot_pos, bounds = generate_problem(10000 + problem_idx)
        
        # Set problem for RL agent
        agent.set_problem(machines, sequence, robot_pos, bounds)
        pop_size = 500
        # GA setup
        ga = GeneticAlgorithm(
            machines=machines,
            sequence=sequence,
            robot_position=robot_pos,
            workspace_bounds=bounds
        )
        
        # GA random
        ga_random_pop = np.stack(ga.create_initial_population(), axis=0)
        ga_best, ga_avg, ga_success = evaluate_population_fitness(ga, ga_random_pop)
        
        # RL init
        rl_pop = []
        for _ in range(pop_size):
            _, _, chrom, success = agent.generate_layout_greedy(max_attempts=100)
            rl_pop.append(chrom)
        rl_pop = np.stack(rl_pop, axis=0)
        rl_best, rl_avg, rl_success = evaluate_population_fitness(ga, rl_pop)
        
        ga_best_list.append(ga_best)
        rl_best_list.append(rl_best)
        rl_success_list.append(rl_success)
        
        if (problem_idx + 1) % 5 == 0:
            print(f"Problem {problem_idx+1}: GA={ga_best:.1f}, RL={rl_best:.1f}, "
                  f"RL_success={rl_success*100:.0f}%")
    
    # Summary
    ga_best_arr = np.array(ga_best_list)
    rl_best_arr = np.array(rl_best_list)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"GA Random Init:")
    print(f"  Mean best:   {np.mean(ga_best_arr):.2f}")
    print(f"  Median best: {np.median(ga_best_arr):.2f}")
    print(f"  Std:         {np.std(ga_best_arr):.2f}")
    
    print(f"\nRL Init:")
    print(f"  Mean best:   {np.mean(rl_best_arr):.2f}")
    print(f"  Median best: {np.median(rl_best_arr):.2f}")
    print(f"  Std:         {np.std(rl_best_arr):.2f}")
    print(f"  Avg success: {np.mean(rl_success_list)*100:.1f}%")
    
    improvement = (np.mean(ga_best_arr) - np.mean(rl_best_arr)) / np.mean(ga_best_arr) * 100
    if improvement > 0:
        print(f"\n✅ RL is {improvement:.1f}% better on average!")
    else:
        print(f"\n❌ RL is {abs(improvement):.1f}% worse on average")
    
    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        data = [ga_best_arr, rl_best_arr]
        
        bp = plt.boxplot(
            data,
            labels=["GA Random Init", "RL Init"],
            showmeans=True,
            meanline=False,
            showfliers=True,
            notch=True,
            patch_artist=True
        )
        
        colors = ["skyblue", "orange"]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.title("Best Fitness: GA Random vs RL Initial Population")
        plt.ylabel("Best Fitness (Lower is Better)")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig("rl_vs_ga_comparison.png", dpi=150)
        plt.show()
        print("\n✓ Plot saved as rl_vs_ga_comparison.png")
    except ImportError:
        print("\n(matplotlib not available, skipping plot)")


def test_dqn():
    """
    Test the DQN agent across multiple problems.
    """
    print("="*70)
    print("TESTING DQN AGENT")
    print("="*70)
    
    from algo.rl_layout_claude import DQNLayoutRLAgent
    
    # Load trained agent
    agent = DQNLayoutRLAgent(
        num_cells_x=12,
        num_cells_y=12,
        collision_penalty=-80.0,
        invalid_placement_penalty=-30.0,
        step_penalty=-0.5,
    )
    
    model_path = "dqn_final.pt"
    if not os.path.exists(model_path):
        print(f"❌ {model_path} not found!")
        print("   Run training first with fixed_dqn_training.py")
        return
    
    agent.load(model_path)
    print(f"✓ Loaded DQN model (train_steps={agent.train_steps})")
    print()
    
    # Test on multiple problems
    num_problems = 20
    ga_best_list = []
    rl_best_list = []
    rl_success_list = []
    
    for problem_idx in range(num_problems):
        machines, sequence, robot_pos, bounds = generate_problem(10000 + problem_idx)
        
        # Set problem for DQN
        agent.set_problem(machines, sequence, robot_pos, bounds)
        
        # GA setup
        pop_size = 100
        ga = GeneticAlgorithm(
            machines=machines,
            sequence=sequence,
            robot_position=robot_pos,
            workspace_bounds=bounds,
            population_size = pop_size,
        )
        
        # GA random
        ga_random_pop = np.stack(ga.create_initial_population(), axis=0)
        ga_best, ga_avg, ga_success = evaluate_population_fitness(ga, ga_random_pop)
        
        # DQN init
        rl_pop = []
        success_count = 0
        for _ in range(pop_size):
            _, _, chrom, success = agent.generate_layout_greedy(max_attempts_per_machine=50)
            rl_pop.append(chrom)
            if success:
                success_count += 1
        
        rl_pop = np.stack(rl_pop, axis=0)
        rl_best, rl_avg, rl_success = evaluate_population_fitness(ga, rl_pop)
        
        ga_best_list.append(ga_best)
        rl_best_list.append(rl_best)
        rl_success_list.append(success_count / pop_size)
        
        if (problem_idx + 1) % 5 == 0:
            print(f"Problem {problem_idx+1}: GA={ga_best:.1f}, DQN={rl_best:.1f}, "
                  f"DQN_success={success_count}/{pop_size}")
    
    # Summary (same as tabular)
    ga_best_arr = np.array(ga_best_list)
    rl_best_arr = np.array(rl_best_list)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"GA Random Init:")
    print(f"  Mean best:   {np.mean(ga_best_arr):.2f}")
    print(f"  Median best: {np.median(ga_best_arr):.2f}")
    
    print(f"\nDQN Init:")
    print(f"  Mean best:   {np.mean(rl_best_arr):.2f}")
    print(f"  Median best: {np.median(rl_best_arr):.2f}")
    print(f"  Avg success: {np.mean(rl_success_list)*100:.1f}%")
    
    improvement = (np.mean(ga_best_arr) - np.mean(rl_best_arr)) / np.mean(ga_best_arr) * 100
    if improvement > 0:
        print(f"\n✅ DQN is {improvement:.1f}% better on average!")
    else:
        print(f"\n❌ DQN is {abs(improvement):.1f}% worse on average")
    
    # Plot
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        data = [ga_best_arr, rl_best_arr]
        
        bp = plt.boxplot(
            data,
            labels=["GA Random Init", "DQN Init"],
            showmeans=True,
            meanline=False,
            showfliers=True,
            notch=True,
            patch_artist=True
        )
        
        colors = ["skyblue", "lightgreen"]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.title("Best Fitness: GA Random vs DQN Initial Population")
        plt.ylabel("Best Fitness (Lower is Better)")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig("dqn_vs_ga_comparison.png", dpi=150)
        plt.show()
        print("\n✓ Plot saved as dqn_vs_ga_comparison.png")
    except ImportError:
        pass


def main():
    """Run appropriate test based on what's available"""
    print("="*70)
    print("RL vs GA INITIALIZATION COMPARISON")
    print("="*70)
    print()
    print("Select which agent to test:")
    print("  1. Single-problem tabular RL (qtable_layout_rl_new.npy)")
    print("  2. Multi-problem tabular RL (tabular_rl_multi_problem.pkl)")
    print("  3. DQN agent (dqn_final.pt)")
    print()
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        test_single_problem_tabular()
    elif choice == "2":
        test_multi_problem_tabular()
    elif choice == "3":
        test_dqn()
    else:
        print("Invalid choice!")
        print("\nAuto-detecting available models...")
        
        if os.path.exists("dqn_final.pt"):
            print("Found DQN model, testing...")
            test_dqn()
        elif os.path.exists("tabular_rl_multi_problem.pkl"):
            print("Found multi-problem tabular model, testing...")
            test_multi_problem_tabular()
        elif os.path.exists("qtable_layout_rl_new.npy"):
            print("Found single-problem tabular model, testing...")
            test_single_problem_tabular()
        else:
            print("❌ No trained models found!")
            print("\nPlease train a model first:")
            print("  - For DQN: python fixed_dqn_training.py")
            print("  - For multi-problem tabular: python train_multi_problem_tabular.py")
            print("  - For single-problem tabular: python train_single_problem_rl.py")


if __name__ == "__main__":
    main()