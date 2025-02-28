from core.Individual import Individual
from managersAndHelpers.GeneticOperators import *
class PopulationManager:
    def __init__(self, population_size: int):
        self.population_size = population_size
        self.population: list[Individual] = []
        self.best_solution: Individual = None
        self.solution_list: list[float] = []


    def initialize_population(self, spaces: list[float], values: list[float], space_limit: float) -> None:
        """
        Initialize the population with random individuals.
        
        :param spaces: List of space requirements for each item.
        :param values: List of values for each item.
        :param space_limit: Maximum space limit for the knapsack problem.
        """
        for _ in range(self.population_size):
            individual = Individual(spaces, values, space_limit)  # Create a new individual
            individual.generate_chromosome()  # Generate a random chromosome for the individual
            self.population.append(individual)  # Add the individual to the population
        print(self.population[0])
        self.best_solution = self.population[0]  # Initialize the best solution as the first individual

    def evaluate_population(self) -> None:
        """Evaluate fitness of all individuals."""
        for individual in self.population:
            individual.evaluate()

    def sort_population(self) -> None:
        """Sort population by fitness (descending)."""
        self.population.sort(key=lambda x: x.evaluation_score, reverse=True)

    def update_best_solution(self, candidate: Individual) -> None:
        """Update the best solution if the candidate is better."""
        if candidate.evaluation_score > self.best_solution.evaluation_score:
            self.best_solution = candidate

    def sum_evaluations(self) -> float:
        """Sum of all evaluation scores in the population."""
        return sum(ind.evaluation_score for ind in self.population)

    def generate_new_population(self, current_population: int, total_score: float, mutation_rate: float, elitism_chance: float,  genetic_operators: GeneticOperators) -> list[int]:
        """Generate a new population through elitism, selection, crossover, and mutation."""
        # **Elitism:** Select the top-performing individuals to carry over.
        elite_count = max(1, int(elitism_chance * len(current_population)))
        new_population = current_population[:elite_count]

        #**Selection and Crossover:** Generate offspring to fill the population
        num_offspring = (len(self.population) - elite_count) // 2  # Number of pairs
        for _ in range(num_offspring):
            # **Selection:** Pick two parents
            parent1 = genetic_operators.select_parent(self.population, total_score)
            parent2 = genetic_operators.select_parent(self.population, total_score)

            # **Crossover:** Generate two new children
            child1, child2 = genetic_operators.apply_crossover(parent1, parent2)

            # **Mutation:** Randomly mutate children
            genetic_operators.apply_mutation(child1, mutation_rate)
            genetic_operators.apply_mutation(child2, mutation_rate)

            # Add children to the new population
            new_population.extend([child1, child2])

        return new_population