from func.datastruct import Point, Machine
from algo.agent import QLearningAgent
import numpy as np
import random
from typing import List, Tuple, Optional, Dict
import time
from algo.gen_algo_test_copy import GeneticAlgorithm

class RLGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self, machines: List[Machine], sequence: List[int], 
                 robot_position: Point, workspace_bounds: Tuple[float, float, float, float]):
        super().__init__(machines, sequence, robot_position, workspace_bounds)
        
        # Replace plain GA mutation with an RL-controlled strategy
        self.current_mutation_strategy = self.mutate  
        
        # RL agent: 9 states, 8 actions (parameter tuning + mutation strategy)
        self.rl_agent = QLearningAgent(state_size=9, action_size=8)
        
        # For RL reward computation
        self.best_fitness_prev = float('inf')
    
    def discretize_state(self, diversity: float, improvement: float) -> int:
        """Map continuous values into discrete RL states"""
        # Diversity bins
        if diversity < 5:
            div_bin = 0
        elif diversity < 15:
            div_bin = 1
        else:
            div_bin = 2
        
        # Improvement bins
        if improvement <= 0:
            imp_bin = 0
        elif improvement < 5:
            imp_bin = 1
        else:
            imp_bin = 2
        
        # Combine into single state index (max ~9 states)
        return div_bin * 3 + imp_bin
    
    def apply_bounds(self, chromosome: np.ndarray) -> np.ndarray:
        """Apply workspace bounds to chromosome"""
        min_x, max_x, min_y, max_y = self.workspace_bounds
        
        for i in range(0, len(chromosome), 3):
            machine_idx = i // 3
            machine_width = self.machines[machine_idx].width
            machine_height = self.machines[machine_idx].height
            
            chromosome[i] = np.clip(chromosome[i], 
                                  min_x + machine_width/2, 
                                  max_x - machine_width/2)
            chromosome[i+1] = np.clip(chromosome[i+1], 
                                    min_y + machine_height/2, 
                                    max_y - machine_height/2)
            chromosome[i+2] = chromosome[i+2] % 360
        
        return chromosome
    
    def gaussian_mutation(self, chromosome: np.ndarray) -> np.ndarray:
        """Standard Gaussian mutation"""
        return self.mutate(chromosome)

    def cauchy_mutation(self, chromosome: np.ndarray) -> np.ndarray:
            """Cauchy mutation (heavy-tailed distribution)"""
            mutated = chromosome.copy()
            
            for i in range(0, len(chromosome), 3):
                if random.random() < self.mutation_rate:
                    # Cauchy distribution has heavier tails than Gaussian
                    scale = 2.0 if i % 3 != 2 else 20.0
                    mutated[i] += np.random.standard_cauchy() * scale
                    mutated[i+1] += np.random.standard_cauchy() * scale
                    mutated[i+2] += np.random.standard_cauchy() * scale
            
            return self.apply_bounds(mutated)
    
    def polynomial_mutation(self, chromosome: np.ndarray) -> np.ndarray:
        """Polynomial mutation"""
        mutated = chromosome.copy()
        eta = 20.0  # Distribution index
        
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                u = random.random()
                if u <= 0.5:
                    delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                
                mutated[i] += delta * 0.1 * abs(chromosome[i])
        
        return self.apply_bounds(mutated)
    
    def apply_action(self, action: int, population: List[np.ndarray]):
        """Apply RL-chosen action"""
        if action == 0:  # increase mutation rate
            self.mutation_rate = min(0.5, self.mutation_rate * 1.2)
        elif action == 1:  # decrease mutation rate
            self.mutation_rate = max(0.01, self.mutation_rate * 0.8)
        elif action == 2:  # increase crossover rate
            self.crossover_rate = min(1.0, self.crossover_rate * 1.1)
        elif action == 3:  # decrease crossover rate
            self.crossover_rate = max(0.2, self.crossover_rate * 0.9)
        elif action == 4:  # apply local search to elites
            elite_count = int(self.elitism_rate * self.population_size)
            for i in range(elite_count):
                population[i] = self.local_search(population[i], max_iterations=5)
        elif action == 5:  # switch to Gaussian mutation
            self.current_mutation_strategy = self.gaussian_mutation
        elif action == 6:  # switch to Cauchy mutation
            self.current_mutation_strategy = self.cauchy_mutation
        elif action == 7:  # switch to Polynomial mutation
            self.current_mutation_strategy = self.polynomial_mutation

    def optimize(self) -> Tuple[List[Machine], float, Dict]:
        print("Starting Reinforcement Learning - Genetic Algorithm")
        population = self.create_initial_population()
        best_solution = None
        best_fitness = float('inf')

        self.best_fitness_history = []
        self.avg_fitness_history = []
        rl_log = []
        start_time = time.time()
        
        # Initialize tracking variables
        prev_state = None
        prev_action = None
        prev_best_fitness = float('inf')

        for generation in range(self.generations):
            # SINGLE fitness evaluation per generation
            fitness_scores = [self.fitness_function(ch) for ch in population]
            best_idx = np.argmin(fitness_scores)
            best_fitness_curr = fitness_scores[best_idx]

            # Calculate current metrics
            diversity = self.calculate_diversity(population)
            improvement = prev_best_fitness - best_fitness_curr
            state = self.discretize_state(diversity, improvement)

            # RL Update: Learn from PREVIOUS generation's transition
            if generation > 0:
                reward = (prev_best_fitness - best_fitness_curr) / (abs(prev_best_fitness) + 1e-6)
                self.rl_agent.update(prev_state, prev_action, reward, state)
            
            # Select and apply action for THIS generation
            action = self.rl_agent.get_action(state)
            self.apply_action(action, population)

            # Logging
            rl_log.append({
                'generation': generation,
                'state': state,
                'action': action,
                'reward': float(reward) if generation > 0 else 0.0,
                'diversity': float(diversity),
                'improvement': float(improvement),
                'best_fitness': float(best_fitness_curr),
                'mutation_rate': float(self.mutation_rate),
                'crossover_rate': float(self.crossover_rate)
            })

            # Track best solution
            if best_fitness_curr < best_fitness:
                best_fitness = best_fitness_curr
                best_solution = population[best_idx].copy()

            avg_fitness = np.mean([f for f in fitness_scores if f != float('inf')])
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)

            # Create new population (GA operations)
            new_population = []
            elite_count = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(fitness_scores)[:elite_count]
            for idx in elite_indices:
                new_population.append(population[idx].copy())

            while len(new_population) < self.population_size:
                p1 = self.tournament_selection(population, fitness_scores)
                p2 = self.tournament_selection(population, fitness_scores)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.current_mutation_strategy(c1)
                c2 = self.current_mutation_strategy(c2)
                new_population.extend([c1, c2])

            population = new_population[:self.population_size]
            
            # Store for next iteration
            prev_state = state
            prev_action = action
            prev_best_fitness = best_fitness_curr

            if generation % 20 == 0:
                print(f"Gen {generation}: Best={best_fitness:.2f}, Avg={avg_fitness:.2f}, Diversity={diversity:.2f}")
                recent = rl_log[-1]
                print(f"[RL] Gen {recent['generation']}: Action={recent['action']}, "
                    f"Reward={recent['reward']:.3f}, MutRate={recent['mutation_rate']:.3f}, "
                    f"CrossRate={recent['crossover_rate']:.3f}")

        end_time = time.time()

        final_machines = self.decode_chromosome(best_solution)
        results = {
            'total_distance': self.calculate_total_distance(final_machines),
            'generations': self.generations,
            'execution_time': end_time - start_time,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'final_fitness': best_fitness,
            'q_table': self.rl_agent.q_table,
            'rl_log': rl_log
        }

        return final_machines, best_fitness, results