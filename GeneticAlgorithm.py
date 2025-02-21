from Individual import Individual 

from SelectionMethods import *
from Visualization import *
from CrossOverMethods import *
from MutationMethods import *
from ReplacementMethods import *
from MigrationMethods import *
from GeneticOperators import *

from typing import Callable

class GeneticAlgorithm():
    def __init__(self, 
        population_size: int, 
        selection_method: str = "roulette", 
        crossover_method: str = "single_point", 
        mutation_method: str = "bit_flip", 
        standard_execution: bool = False, 
        migration_method: str = "star_migration_bidirectional", 
        primary_replacement_method: str = "best", 
        secundary_replacement_method: str = "random",
        migration_args: tuple = None ):
        """
        Initialize the Genetic Algorithm with the given parameters.
        
        :param population_size: Number of individuals in the population.
        :param selection_method: Method used for selecting parents (default: "roulette").
        :param crossover_method: Method used for crossover (default: "single_point").
        :param elitism_chance: Percentage of top individuals to carry over to the next generation (default: 0.05).
        """
        self.population_size: int = population_size  # Size of the population
        self.selection_method: str = selection_method  # Selection method to use
        self.crossover_method: str = crossover_method  # Crossover method to use
        self.mutation_method: str = mutation_method # mutation method to use
        self.migration_method: str = migration_method # migration method to use
        self.primary_replacement_method: str = primary_replacement_method # The primary individual to migrate 
        self.secundary_replacement_method: str = secundary_replacement_method # The secundary individual to migrate 
        self.standard_execution: bool  = standard_execution # Use island model or normal model
        self.population: list[Individual] = []  # List to hold the population of individuals
        self.generation: int = 0  # Current generation number
        self.best_solution: Individual = None  # Best solution found so far
        self.solution_list: list[float] = []  # List to store the best solution scores over generations
        self.islands: list[list[Individual]] = []  # List of populations for island model
        self.migration_args = migration_args if migration_args is not None else () 

        self.genetic_operators = GeneticOperators(
            selection_method=selection_method,
            crossover_method=crossover_method,
            mutation_method=mutation_method,
            migration_method=migration_method,
            primary_replacement_method=primary_replacement_method,
            secondary_replacement_method=secundary_replacement_method,  # Correct typo here
            migration_args=self.migration_args,
        )
       

        self.migration_methods = {
            "ring_migration": MigrationMethods.ring_migration,
            "random_migration": MigrationMethods.random_migration,
            "adaptive_migration": MigrationMethods.adaptive_migration,
            "star_migration_bidirectional": MigrationMethods.star_migration_bidirectional,
            "star_migration_unidirectional": MigrationMethods.star_migration_unidirectional,
            "tournament_migration": MigrationMethods.tournament_migration
        }

        self.replacement_methods = {
            "random": ReplacementMethods.random_individual_replacement,
            "best": ReplacementMethods.best_individual_replacement,
            "worst": ReplacementMethods.worst_individual_replacement,
        }


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
        self.best_solution = self.population[0]  # Initialize the best solution as the first individual

    def sort_population(self) -> None:
        """Sort the population based on evaluation scores in descending order."""
        self.population.sort(key=lambda ind: ind.evaluation_score, reverse=True)

    def update_best_individual(self, individual: Individual) -> None:
        """
        Update the best solution if the given individual has a better score.
        
        :param individual: The individual to compare with the current best solution.
        """
        if individual.evaluation_score > self.best_solution.evaluation_score:
            self.best_solution = individual

    def sum_avaliation(self, population=None) -> float:
        """Calculate the total evaluation score of the population."""
        if population is None:
            population = self.population
        return sum(ind.evaluation_score for ind in population)
    
    def select_parent(self, total_score: float, population: list[Individual] = None) -> Individual:
        """
        Select a parent based on the chosen selection method.
        
        :param total_score: The total evaluation score of the population (used for some selection methods).
        :return: The selected parent.
        """

        if population is None:
            population = self.population
        if self.selection_method not in self.selection_methods:
            raise ValueError(f"Invalid selection method: {self.selection_method}")

        # Call the appropriate selection method dynamically
        method = self.selection_methods[self.selection_method]
        if method in {SelectionMethods.roulette_selection, SelectionMethods.sus_selection}:
            return method(population, total_score)
        else:
            return method(population)



    def visualize_generation(self) -> None:
        """
        Print the best individual of the current generation.
        """
        best = self.population[0]
        print(f"G:{best.generation} -> Score: {best.evaluation_score} Chromosome: {best.chromosome}")

    def calculate_diversity(self, population: list[Individual] = None) -> float:
        """
        Get the diversity (how individuals are different from each other)
        """
        if population is None:
            population = self.population
        pop_size = len(population)
        if pop_size < 2:
             return 0
        diversity = 0
        for i in range(len(self.population)):
            for j in range(i+1, len(self.population)):# Start from i+1 to avoid double counting
                # Calculate how many genes differ between the two individuals' chromosomes
                diversity += sum(c1 != c2 for c1, c2 in zip(
                    self.population[i].chromosome,
                    self.population[j].chromosome))
        # Normalize diversity by the total number of unique individual pairs (combinations)
        return diversity / (self.population_size * (self.population_size - 1) / 2)

    def calculate_mutation_rate(self, adaptative_mutation : bool = True, mutation_rate : float = 0.5) -> float:

        diversity = self.calculate_diversity()
        if (adaptative_mutation):
            return mutation_rate * (1 - diversity) # Higher diversity -> Lower mutation
        else: 
            return mutation_rate



    def solve(self, 
              mutation_rate: float, 
              num_generations: int, 
              spaces: list[float], 
              values: list[float], 
              space_limit: float, 
              generate_graphic: bool = True, 
              adaptative_mutation: bool = True, 
              elitism_chance: float | int = 0.05, 
              num_islands: int = 4, 
              migration_interval: int = 5, 
              num_migrants: int = 2) -> tuple[list[int], list[int]]:
        """
        Run the genetic algorithm to solve an optimization problem, such as the knapsack problem.

        Parameters:
        :param mutation_rate: (float) The probability of mutation for each individual.
        :param num_generations: (int) The number of generations to run the algorithm.
        :param spaces: (list) List of space requirements for each item.
        :param values: (list) List of values for each item.
        :param space_limit: (float) Maximum space limit for the knapsack problem.
        :param generate_graphic: (bool) Whether to generate a visualization of the results. Default is True.
        :param adaptative_mutation: (bool) Whether to use adaptive mutation rates. Default is True.
        :param elitism_chance: (float) Percentage of top individuals to carry over to the next generation. Default is 0.05 (5%).
        :param num_islands: (int) Number of islands (sub-populations) for the island model. Default is 4.
        :param migration_interval: (int) Number of generations between migrations. Default is 5.
        :param num_migrants: (int) Number of individuals to migrate between islands. Default is 2.
        
        Returns:
        :return: A tuple containing the best solution chromosome and the list of generation scores.
        """
        if self.standard_execution:
            generation_scores, avg_scores = self.execute_standard(
                mutation_rate, num_generations, spaces, values, space_limit, adaptative_mutation, elitism_chance
            )
        else:
            generation_scores, avg_scores = self.execute_island_model(
                mutation_rate, num_generations, adaptative_mutation, elitism_chance,
                num_islands, migration_interval, num_migrants, spaces, values, space_limit
            )

        print(f"\nBest solution -> G: {self.best_solution.generation}, "
            f"Score: {self.best_solution.evaluation_score}, "
            f"Chromosome: {self.best_solution.chromosome}")
        
        if generate_graphic:
            Visualization.plot_generation_scores(generation_scores, avg_scores)
        
        return self.best_solution, generation_scores

    def execute_standard(self, 
                        mutation_rate: float,
                        num_generations: int,
                        spaces :list[float],
                        values :list[float],
                        space_limit :float,
                        adaptative_mutation :bool,
                        elitism_chance :float) -> tuple[list[int], list[float]] :
        """
        Execute the standard genetic algorithm without island-based evolution.
        
        :param mutation_rate: Probability of mutation per individual.
        :param num_generations: Number of generations to run.
        :param spaces: Space requirements for each item.
        :param values: Values assigned to each item.
        :param space_limit: Maximum capacity for the knapsack problem.
        :param adaptative_mutation: Whether to use adaptive mutation rates.
        :param elitism_chance: Proportion of elite individuals carried to the next generation.
        :return: A list of best generation scores and average scores per generation.
        """
        self.initialize_and_evaluate(spaces, values, space_limit)
        self.sort_population()
        self.best_solution = self.population[0]
        self.solution_list = [self.best_solution.evaluation_score]
        self.visualize_generation()

        # Initialize lists to track generation scores
        generation_scores = [self.best_solution.evaluation_score]
        avg_scores = [self.sum_avaliation() / self.population_size]

        # Iterate through the generations
        for _ in range(num_generations):
            # Calculate adaptive mutation rate if applicable
            adapted_mutation_rate = self.calculate_mutation_rate(adaptative_mutation, mutation_rate)
            total_score = self.sum_avaliation()
            avg_scores.append(total_score / self.population_size)

            # Generate a new population for the next generation
            new_population = self.generate_new_population(self.population, total_score, adapted_mutation_rate, elitism_chance)
            self.population = new_population

            # Evaluate and sort the new population
            self.evaluate_population()
            self.sort_population()
            self.visualize_generation()

            # Update the best solution found so far
            best_current = self.population[0]
            self.update_best_individual(best_current)
            generation_scores.append(best_current.evaluation_score)

        # Return the scores for each generation
        return generation_scores, avg_scores

    def execute_island_model(self, 
                            mutation_rate: float,
                            num_generations: int,
                            adaptative_mutation: bool,
                            elitism_chance: float,
                            num_islands: int,
                            migration_interval: int,
                            num_migrants: int,
                            spaces: list[float],
                            values: list[float],
                            space_limit: float) -> tuple[list[int], list[float]]:
        """
        Execute the genetic algorithm using an island model approach.
        
        :param mutation_rate: Probability of mutation per individual.
        :param num_generations: Number of generations to run.
        :param adaptative_mutation: Whether to use adaptive mutation rates.
        :param elitism_chance: Proportion of elite individuals carried to the next generation.
        :param num_islands: Number of isolated populations.
        :param migration_interval: Generations between migrations.
        :param num_migrants: Number of individuals migrating between islands.
        :param spaces: Space requirements for each item.
        :param values: Values assigned to each item.
        :param space_limit: Maximum capacity for the knapsack problem.
        :return: A list of best generation scores and average scores per generation.
        """
        self.initialize_and_evaluate(spaces, values, space_limit)

        # Split the population into isolated islands
        self.split_into_islands(num_islands)
        self.sort_population()
        self.best_solution = self.population[0]
        self.solution_list = [self.best_solution.evaluation_score]

        # Initialize lists to track generation scores
        generation_scores = [self.best_solution.evaluation_score]
        total_avg = self.sum_avaliation() / self.population_size
        avg_scores = [total_avg]

        # Iterate through the generations
        for gen in range(num_generations):
            # Evolve each island individually
            for island_idx in range(len(self.islands)):
                self.islands[island_idx] = self.evolve_island(
                    self.islands[island_idx], mutation_rate, adaptative_mutation, elitism_chance
                )
                # Update the best individual for the island
                self.update_best_individual(self.islands[island_idx][0])

            # Perform migration between islands at the specified interval
            if (gen + 1) % migration_interval == 0:
                self.apply_migration_method(num_migrants)

            # Track the best score of the generation from all islands
            current_gen_best = max(island[0].evaluation_score for island in self.islands)
            generation_scores.append(current_gen_best)

            # Calculate the average score across all islands
            total_score_all = sum(sum(ind.evaluation_score for ind in island) for island in self.islands)
            avg_scores.append(total_score_all / sum(len(island) for island in self.islands))

        # Update the best solution across all islands
        self.update_final_island_best()

        # Return the scores for each generation
        return generation_scores, avg_scores

    def initialize_and_evaluate(self, spaces: list[float], values: list[float], space_limit: float) -> None:
        """Initialize the population and evaluate each individual."""
        self.initialize_population(spaces, values, space_limit)
        self.evaluate_population()

    def evaluate_population(self) -> None:
        """Evaluate all individuals in the current population."""
        for individual in self.population:
            individual.evaluate()

    def sort_population(self) -> None:
        """Sort the population based on evaluation scores."""
        self.population.sort(key=lambda x: x.evaluation_score, reverse=True)

    def generate_new_population(self, current_population: int, total_score: float, mutation_rate: float, elitism_chance: float) -> list[int]:
        """Generate a new population through elitism, selection, crossover, and mutation."""
        # **Elitism:** Select the top-performing individuals to carry over.
        elite_count = max(1, int(elitism_chance * len(current_population)))
        new_population = current_population[:elite_count]

        #**Selection and Crossover:** Generate offspring to fill the population
        num_offspring = (len(self.population) - elite_count) // 2  # Number of pairs
        for _ in range(num_offspring):
            # **Selection:** Pick two parents
            parent1 = self.genetic_operators.select_parent(self.population, total_score)
            parent2 = self.genetic_operators.select_parent(self.population, total_score)

            # **Crossover:** Generate two new children
            child1, child2 = self.genetic_operators.apply_crossover(parent1, parent2)

            # **Mutation:** Randomly mutate children
            self.genetic_operators.apply_mutation(child1, mutation_rate)
            self.genetic_operators.apply_mutation(child2, mutation_rate)

            # Add children to the new population
            new_population.extend([child1, child2])

        return new_population

    def evolve_island(self, island: int, mutation_rate: float, adaptative_mutation: bool, elitism_chance: float):
        """
        Evolves an island's population for one generation.

        Steps:
        1. Temporarily replaces the main population with the island's population.
        2. Calculates the total fitness score of the island's individuals.
        3. Adjusts mutation rate if adaptive mutation is enabled.
        4. Generates a new population for the island using selection, crossover, and mutation.
        5. Evaluates and sorts the new population based on fitness.
        6. Restores the original population and returns the evolved island.

        :param island: The sub-population (island) to evolve.
        :param mutation_rate: The base probability of mutation for each individual.
        :param adaptative_mutation: Whether to adjust mutation rate dynamically.
        :param elitism_chance: The percentage of top individuals retained unaltered.
        :return: The evolved island's population.
        """
        original_population = self.population  # Store the original population
        self.population = island.copy()  # Work on the island's population

        total_score = self.sum_avaliation()  # Compute total fitness score
        adapted_rate = self.calculate_mutation_rate(adaptative_mutation, mutation_rate)  # Adjust mutation rate if needed
        new_population = self.generate_new_population(self.population, total_score, adapted_rate, elitism_chance)

        self.population = new_population  # Replace with the new evolved population
        self.evaluate_population()  # Evaluate fitness
        self.sort_population()  # Sort by best fitness

        evolved_island = self.population  # Store the evolved island
        self.population = original_population  # Restore original population
        return evolved_island

    def split_into_islands(self, num_islands: int) -> None:
        """
        Splits the main population into multiple islands (sub-populations).

        :param num_islands: The number of islands to create.
        """
        island_size = self.population_size // num_islands # Determine size per island by dividing the total population size by the number of islands
        self.islands = []
        for i in range(num_islands):
            start = i * island_size # Calculate the starting index for the current island's population
            end = start + island_size   # Calculate the ending index for the current island's population
            if i == num_islands - 1:  # Ensure the last island gets any remaining individuals
                end = self.population_size
            self.islands.append(self.population[start:end])  # Append the slice of the population corresponding to the current island to the islands list

    def update_best_individual(self, candidate: Individual) -> None:
        """
        Updates the best solution found so far if the candidate is better.

        :param candidate: The individual being compared to the current best.
        """
        if candidate.evaluation_score > self.best_solution.evaluation_score:
            self.best_solution = candidate

    def update_final_island_best(self) -> None:
        """
        After all generations, updates the best solution by considering all islands.
        """
        all_individuals = [ind for island in self.islands for ind in island]  # Flatten all islands into one list
        all_individuals.sort(key=lambda x: x.evaluation_score, reverse=True)  # Sort by best fitness
        self.best_solution = all_individuals[0]  # Pick the top individual
        self.solution_list.append(self.best_solution.evaluation_score)  # Store the best solution score



    def apply_migration_method(self, num_migrants: int) -> None:
        """
        Apply the specified migration method to the island model.
        """
        if self.migration_method not in self.migration_methods:
            raise ValueError(f"Invalid replacement method: {self.migration_method}")
        migration_function = self.migration_methods[self.migration_method]
        # Get the function's parameters
        sig = inspect.signature(migration_function)
        params = list(sig.parameters.values())
        
        # Calculate how many additional parameters the function expects after the first 5
        # (islands, num_migrants, replacement_methods, primary, secondary)
        num_extra_params = max(len(params) - 5, 0)
        args_to_pass = self.migration_args[:num_extra_params]  # Ensure non-negative
        
        # Slice migration_args to match the number of extra parameters
        args_to_pass = self.migration_args[:num_extra_params]
        migration_function(self.islands,
                            num_migrants,
                            self.replacement_methods,
                            self.primary_replacement_method,
                            self.secundary_replacement_method,
                            *args_to_pass)