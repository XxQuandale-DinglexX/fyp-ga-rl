import os
import numpy as np
import random
from func.datastruct import Point, Machine
from algo.gen_algo_init import GeneticAlgorithm          # adjust path if needed
from algo.rl_layout_agent import LayoutRLAgent      # your RL class
# from algo.rl_layout_agent_multi import MultiProblemTabularRLAgent      # your RL class


# --- same build_problem you already have ---
def generate_problem(problem_idx: int):
    """
    Generate one problem instance with 8 machines.
    We vary machine sizes, shapes and access points per problem.
    """
    # For reproducibility of problem definitions
    base_seed = 1000 + problem_idx
    random.seed(base_seed)
    np.random.seed(base_seed)

    workspace_bounds = (-15, 15, -15, 15)
    robot_position = Point(0, 0)

    machines = []
    for i in range(8):
        m_id = i + 1

        # Alternate shapes for variety
        shape = "l_shape" if i % 2 == 0 else "rectangle"

        # Random-ish but reasonable machine sizes
        width = random.uniform(3.0, 6.0)
        height = random.uniform(2.5, 5.0)

        # Access point somewhere near machine center
        ap_x = random.uniform(-1.5, 1.5)
        ap_y = random.uniform(-1.5, 1.5)

        if shape == "l_shape":
            # L-cutout smaller than total width/height
            l_cutout_width = random.uniform(1.0, width / 2.0)
            l_cutout_height = random.uniform(1.0, height / 2.0)
            machine = Machine(
                id=m_id,
                shape="l_shape",
                width=width,
                height=height,
                access_point=Point(ap_x, ap_y),
                l_cutout_width=l_cutout_width,
                l_cutout_height=l_cutout_height,
            )
        else:
            machine = Machine(
                id=m_id,
                shape="rectangle",
                width=width,
                height=height,
                access_point=Point(ap_x, ap_y),
            )

        machines.append(machine)

    # Simple sequence: visit machines 1..8
    sequence = list(range(1, 9))

    return machines, sequence, robot_position, workspace_bounds


def evaluate_population_fitness(ga: GeneticAlgorithm, population: np.ndarray):
    """Evaluate a whole population with GA's fitness_function."""
    fitness_scores = []
    for chrom in population:
        f = ga.fitness_function(chrom)
        fitness_scores.append(f)

    finite = [f for f in fitness_scores if f != float("inf")]
    avg_fitness = float(np.mean(finite)) if finite else float("inf")
    best_fitness = float(np.min(fitness_scores))
    return best_fitness, avg_fitness


def main():
    num_problems = 100
    ga_best_list = []
    rl_best_list = []
    for problem_idx in range(num_problems):
        print("Run number =", problem_idx+1)
        machines, sequence, robot_pos, bounds = generate_problem(problem_idx)

        # ---------------- GA setup ----------------
        ga = GeneticAlgorithm(
            machines=machines,
            sequence=sequence,
            robot_position=robot_pos,
            workspace_bounds=bounds,
        )
        pop_size = 500  # 500 by default

        # --- baseline: GA random initial population ---
        ga_random_pop_list = ga.create_initial_population()      # List[np.ndarray]
        ga_random_pop = np.stack(ga_random_pop_list, axis=0)     # (pop_size, chrom_len)
        ga_best, ga_avg = evaluate_population_fitness(ga, ga_random_pop)
        # print(f"[GA random init] best={ga_best:.3f}, avg={ga_avg:.3f}")

        # ---------------- RL setup (load trained agent) ----------------
        agent = LayoutRLAgent(
            machines=machines,
            sequence=sequence,
            robot_position=robot_pos,
            workspace_bounds=bounds,
            num_cells_x=8,
            num_cells_y=8,
        )


        qtable_path = "qtable_layout_rl_new.npy"   # same as in your training script
        # qtable_path = "tabular_rl_multi_problem.pkl"
        if not os.path.exists(qtable_path):
            raise FileNotFoundError(
                f"{qtable_path} not found. Run your RL training script first to create it."
            )

        agent.load_q_table(qtable_path)

        # --- build RL initial population using generate_layout_greedy ---
        rl_pop = []
        for _ in range(pop_size):
            _, _, chrom = agent.generate_layout_greedy(rng_jitter=0.1)
            rl_pop.append(chrom)
        rl_pop = np.stack(rl_pop, axis=0)
        rl_best, rl_avg = evaluate_population_fitness(ga, rl_pop)
        # (f"[RL init pop]   best={rl_best:.3f}, avg={rl_avg:.3f}")

        # Simple comparison
        print("\n[COMPARISON]")
        print(f"  GA random best = {ga_best:.3f}")
        print(f"  RL init best   = {rl_best:.3f}")
        if rl_best < ga_best:
            print("  ✅ RL initial population is BETTER than random GA initial population.")
        else:
            print("  ❌ RL initial population is NOT yet better than GA random (you may need more RL training).")
        
        ga_best_list.append(ga_best)
        rl_best_list.append(rl_best)
    
    import matplotlib.pyplot as plt

    ga_best_arr = np.array(ga_best_list)
    rl_best_arr = np.array(rl_best_list)

    plt.figure(figsize=(10, 6))

    # Create data for boxplot
    data = [ga_best_arr, rl_best_arr]

    plt.boxplot(
        data,
        labels=["GA Random Init (Best Fitness)", "RL Init (Best Fitness)"],
        showmeans=True,           # Show the mean as a dot
        meanline=False,
        showfliers=True,          # Show outliers
        notch=True,               # notch to show confidence interval around median
        patch_artist=True         # allow color fill
    )

    # Color fill the boxes
    colors = ["skyblue", "orange"]
    for patch, color in zip(plt.gca().artists, colors):
        patch.set_facecolor(color)

    plt.title("Whisker Plot of Best Fitness: GA Random vs RL Initial Population")
    plt.ylabel("Best Fitness (Lower is Better)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()