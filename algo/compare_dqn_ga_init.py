import os
import numpy as np
import random
import matplotlib.pyplot as plt

from func.datastruct import Point, Machine
from algo.gen_algo_init import GeneticAlgorithm
from algo.rl_layout_claude import DQNLayoutRLAgent


def generate_problem(problem_idx: int):
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
                l_cutout_width=l_cutout_width, l_cutout_height=l_cutout_height
            )
        else:
            machine = Machine(
                id=m_id, shape="rectangle", width=width, height=height,
                access_point=Point(ap_x, ap_y)
            )
        machines.append(machine)

    sequence = list(range(1, 9))
    return machines, sequence, robot_position, workspace_bounds


def evaluate_population_fitness(ga: GeneticAlgorithm, population: np.ndarray):
    fitness_scores = [ga.fitness_function(chrom) for chrom in population]
    finite = [f for f in fitness_scores if f != float("inf")]
    avg_fitness = float(np.mean(finite)) if finite else float("inf")
    best_fitness = float(np.min(fitness_scores))
    return best_fitness, avg_fitness


def main():
    num_problems = 100

    # IMPORTANT: in your GA class, create_initial_population uses self.population_size
    # If you want pop_size=500, you should set ga.population_size = 500 before calling it.
    pop_size = 500

    # ---- load DQN once ----
    dqn_path = "dqn_layout_final.pt"
    if not os.path.exists(dqn_path):
        raise FileNotFoundError(f"Missing {dqn_path}. Train DQN first.")

    agent = DQNLayoutRLAgent(num_cells_x=10, num_cells_y=10)
    agent.load(dqn_path)
    print(f"[DQN] Loaded model from {dqn_path}")

    ga_best_list = []
    rl_best_list = []
    rl_success_list = []  # how often greedy rollout placed all machines

    for problem_idx in range(num_problems):
        print(f"\nRun number = {problem_idx + 1}")

        machines, sequence, robot_pos, bounds = generate_problem(problem_idx)

        # --------- GA random init ----------
        ga = GeneticAlgorithm(
            machines=machines,
            sequence=sequence,
            robot_position=robot_pos,
            workspace_bounds=bounds,
        )
        ga.population_size = pop_size  # make GA actually generate 500

        ga_random_pop_list = ga.create_initial_population()
        ga_random_pop = np.stack(ga_random_pop_list, axis=0)
        ga_best, ga_avg = evaluate_population_fitness(ga, ga_random_pop)

        # --------- DQN init ----------
        agent.set_problem(machines, sequence, robot_pos, bounds)

        rl_pop = []
        success_count = 0

        for _ in range(pop_size):
            _, _, chrom, success = agent.generate_layout_greedy(max_attempts_per_machine=30)
            rl_pop.append(chrom)
            success_count += int(success)

        rl_pop = np.stack(rl_pop, axis=0)
        rl_best, rl_avg = evaluate_population_fitness(ga, rl_pop)

        rl_success_rate = success_count / pop_size
        rl_success_list.append(rl_success_rate)

        print("[COMPARISON]")
        print(f"  GA random best = {ga_best:.3f}")
        print(f"  DQN init best  = {rl_best:.3f}")
        print(f"  DQN success    = {rl_success_rate*100:.1f}% (placed all machines)")

        ga_best_list.append(ga_best)
        rl_best_list.append(rl_best)

    # --------- boxplot ----------
    ga_best_arr = np.array(ga_best_list)
    rl_best_arr = np.array(rl_best_list)

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [ga_best_arr, rl_best_arr],
        labels=["GA Random Init (Best Fitness)", "DQN Init (Best Fitness)"],
        showmeans=True,
        notch=True,
        showfliers=True
    )
    plt.title("Whisker Plot of Best Fitness: GA Random vs DQN Initial Population")
    plt.ylabel("Best Fitness (Lower is Better)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # --------- also plot success rates ----------
    plt.figure(figsize=(10, 4))
    plt.plot(np.array(rl_success_list) * 100.0)
    plt.title("DQN Greedy Success Rate per Problem")
    plt.xlabel("Problem Index")
    plt.ylabel("Success Rate (%)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("\nSummary:")
    print(f"Avg GA best:  {np.mean(ga_best_arr):.3f}")
    print(f"Avg DQN best: {np.mean(rl_best_arr):.3f}")
    print(f"Avg DQN success rate: {np.mean(rl_success_list)*100:.1f}%")


if __name__ == "__main__":
    main()
