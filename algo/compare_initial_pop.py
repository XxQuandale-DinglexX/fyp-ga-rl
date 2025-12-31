import numpy as np
import random
from typing import List, Tuple
from func.datastruct import Machine, Point
from func.collision_detector import CollisionDetector
from gen_algo_init import GeneticAlgorithm


# ------------------------------------------------------------
# ORIGINAL INITIAL POPULATION (your current version)
# ------------------------------------------------------------
def create_initial_population_original(
    machines: List[Machine],
    workspace_bounds: Tuple[float, float, float, float],
    population_size: int
) -> List[np.ndarray]:

    population = []
    min_x, max_x, min_y, max_y = workspace_bounds

    for _ in range(population_size):
        chromosome = []
        quadrants = [
            (min_x, 0, min_y, 0),
            (0, max_x, min_y, 0),
            (min_x, 0, 0, max_y),
            (0, max_x, 0, max_y)
        ]

        for i, m in enumerate(machines):
            if i < 4:
                qx1, qx2, qy1, qy2 = quadrants[i]
                x = random.uniform(qx1 + m.width/2, qx2 - m.width/2)
                y = random.uniform(qy1 + m.height/2, qy2 - m.height/2)
            else:
                x = random.uniform(min_x + m.width/2, max_x - m.width/2)
                y = random.uniform(min_y + m.height/2, max_y - m.height/2)

            rot = random.uniform(0, 360)
            chromosome.extend([x, y, rot])

        population.append(np.array(chromosome))

    return population


# ------------------------------------------------------------
# IMPROVED INITIAL POPULATION (robust version)
# ------------------------------------------------------------
def create_initial_population_fixed(
    machines: List[Machine],
    workspace_bounds: Tuple[float, float, float, float],
    population_size: int,
    max_tries: int = 50
) -> List[np.ndarray]:

    population = []
    min_x, max_x, min_y, max_y = workspace_bounds

    def valid_layout(decoded: List[Machine]) -> bool:
        # no center overlap
        for m in decoded:
            if CollisionDetector.point_in_polygon(Point(0, 0), m.get_corners()):
                return False
        # no collisions
        for i in range(len(decoded)):
            for j in range(i + 1, len(decoded)):
                if CollisionDetector.polygons_intersect(
                    decoded[i].get_corners(), decoded[j].get_corners()
                ):
                    return False
        # inside bounds
        for m in decoded:
            for p in m.get_corners():
                if not (min_x <= p.x <= max_x and min_y <= p.y <= max_y):
                    return False
        return True

    for _ in range(population_size):
        chromosome = None

        for _try in range(max_tries):
            genes = []
            decoded = []

            for base in machines:
                x = random.uniform(min_x, max_x)
                y = random.uniform(min_y, max_y)
                rot = random.uniform(0, 360)

                m = Machine(
                    id=base.id,
                    shape=base.shape,
                    width=base.width,
                    height=base.height,
                    access_point=base.access_point,
                    l_cutout_width=base.l_cutout_width,
                    l_cutout_height=base.l_cutout_height,
                    position=Point(x, y),
                    rotation=rot
                )

                decoded.append(m)
                genes.extend([x, y, rot])

            if valid_layout(decoded):
                chromosome = np.array(genes)
                break

        if chromosome is None:
            chromosome = np.array(genes)  # fallback

        population.append(chromosome)

    return population


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------
def evaluate_population(
    population: List[np.ndarray],
    machines: List[Machine],
    workspace_bounds,
    fitness_fn
):
    valid_count = 0
    best_fitness = float("inf")

    for ch in population:
        decoded = []
        for i, base in enumerate(machines):
            x, y, rot = ch[3*i:3*i+3]
            decoded.append(
                Machine(
                    id=base.id,
                    shape=base.shape,
                    width=base.width,
                    height=base.height,
                    access_point=base.access_point,
                    l_cutout_width=base.l_cutout_width,
                    l_cutout_height=base.l_cutout_height,
                    position=Point(x, y),
                    rotation=rot
                )
            )

        fitness = fitness_fn(ch)
        if fitness != float("inf"):
            valid_count += 1
            best_fitness = min(best_fitness, fitness)

    return valid_count, best_fitness


# ------------------------------------------------------------
# MAIN COMPARISON
# ------------------------------------------------------------
def run_comparison(
    machines,
    workspace_bounds,
    population_size,
    fitness_fn
):
    pop_orig = create_initial_population_original(
        machines, workspace_bounds, population_size
    )
    pop_fix = create_initial_population_fixed(
        machines, workspace_bounds, population_size
    )

    valid_o, best_o = evaluate_population(
        pop_orig, machines, workspace_bounds, fitness_fn
    )
    valid_f, best_f = evaluate_population(
        pop_fix, machines, workspace_bounds, fitness_fn
    )

    print("=== INITIAL POPULATION COMPARISON ===")
    print(f"Population size: {population_size}\n")

    print("Original initializer:")
    print(f"  Valid layouts     : {valid_o}/{population_size}")
    print(f"  Best fitness      : {best_o:.2f}\n")

    print("Improved initializer:")
    print(f"  Valid layouts     : {valid_f}/{population_size}")
    print(f"  Best fitness      : {best_f:.2f}")


# ------------------------------------------------------------
# USAGE
# ------------------------------------------------------------

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

sequence = [1, 2, 3, 4, 5, 6, 7, 8]
workspace_bounds = (-15, 15, -15, 15)
robot_position = Point(0,0)
ga = GeneticAlgorithm(
    machines=machines,
    sequence=sequence,
    robot_position=robot_position,
    workspace_bounds=workspace_bounds,
)

for n in range(100,1001,100):
    run_comparison(machines, workspace_bounds, population_size=n, fitness_fn=ga.fitness_function)