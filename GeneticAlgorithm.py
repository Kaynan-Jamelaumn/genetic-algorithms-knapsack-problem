from core.Individual import Individual 

from mapMethods.SelectionMethods import *
from core.Visualization import *
from mapMethods.CrossOverMethods import *
from mapMethods.MutationMethods import *
from mapMethods.ReplacementMethods import *
from mapMethods.MigrationMethods import *
from managersAndHelpers.GeneticOperators import *
from managersAndHelpers.PopulationManager import *
from managersAndHelpers.IslandManager import *


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
        self.population_manager = PopulationManager(population_size)


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

        self.island_manager = IslandManager(self.population_manager, self.genetic_operators, self.migration_methods, self.replacement_methods, self.calculate_mutation_rate)



    def visualize_generation(self) -> None:
        """
        Print the best individual of the current generation.
        """
        best = self.population_manager.population[0]
        print(f"G:{best.generation} -> Score: {best.evaluation_score} Chromosome: {best.chromosome}")

    def calculate_diversity(self, population: list[Individual] = None) -> float:
        """
        Get the diversity (how individuals are different from each other)
        """
        if population is None:
            population = self.population_manager.population
        pop_size = len(population)
        if pop_size < 2:
             return 0
        diversity = 0
        for i in range(len(self.population_manager.population)):
            for j in range(i+1, len(self.population_manager.population)):# Start from i+1 to avoid double counting
                # Calculate how many genes differ between the two individuals' chromosomes
                diversity += sum(c1 != c2 for c1, c2 in zip(
                    self.population_manager.population[i].chromosome,
                    self.population_manager.population[j].chromosome))
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

        print(f"\nBest solution -> G: {self.population_manager.best_solution.generation}, "
            f"Score: {self.population_manager.best_solution.evaluation_score}, "
            f"Chromosome: {self.population_manager.best_solution.chromosome}")
        
        if generate_graphic:
            Visualization.plot_generation_scores(generation_scores, avg_scores)
        
        return self.population_manager.best_solution, generation_scores

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
        self.population_manager.sort_population()
        self.population_manager.best_solution = self.population_manager.population[0]
        self.solution_list = [self.population_manager.best_solution.evaluation_score]
        self.visualize_generation()

        # Initialize lists to track generation scores
        generation_scores = [self.population_manager.best_solution.evaluation_score]
        avg_scores = [self.population_manager.sum_evaluations() / self.population_size]

        # Iterate through the generations
        for _ in range(num_generations):
            # Calculate adaptive mutation rate if applicable
            adapted_mutation_rate = self.calculate_mutation_rate(adaptative_mutation, mutation_rate)
            total_score = self.population_manager.sum_evaluations()
            avg_scores.append(total_score / self.population_size)

            # Generate a new population for the next generation
            new_population = self.population_manager.generate_new_population(self.population_manager.population, total_score, adapted_mutation_rate, elitism_chance, self.genetic_operators)
            self.population_manager.population = new_population

            # Evaluate and sort the new population
            self.population_manager.evaluate_population()
            self.population_manager.sort_population()
            self.visualize_generation()

            # Update the best solution found so far
            best_current = self.population_manager.population[0]
            self.population_manager.update_best_solution(best_current)
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
        self.island_manager.split_into_islands(num_islands)
        self.population_manager.sort_population()
        self.population_manager.best_solution = self.population_manager.population[0]
        self.solution_list = [self.population_manager.best_solution.evaluation_score]

        # Initialize lists to track generation scores
        generation_scores = [self.population_manager.best_solution.evaluation_score]
        total_avg = self.population_manager.sum_evaluations() / self.population_size
        avg_scores = [total_avg]

        # Iterate through the generations
        for gen in range(num_generations):
            # Evolve each island individually
            for island_idx in range(len(self.island_manager.islands)):
                self.island_manager.islands[island_idx] = self.island_manager.evolve_island(
                    self.island_manager.islands[island_idx], mutation_rate, adaptative_mutation, elitism_chance
                )
                # Update the best individual for the island
                self.population_manager.update_best_solution(self.island_manager.islands[island_idx][0])

            # Perform migration between islands at the specified interval
            if (gen + 1) % migration_interval == 0:
                self.island_manager.apply_migration_method(num_migrants, self.genetic_operators.migration_method, self.genetic_operators.primary_replacement_method, self.genetic_operators.secondary_replacement_method,  self.genetic_operators.migration_args)

            # Track the best score of the generation from all islands
            current_gen_best = max(island[0].evaluation_score for island in self.island_manager.islands)
            generation_scores.append(current_gen_best)

            # Calculate the average score across all islands
            total_score_all = sum(sum(ind.evaluation_score for ind in island) for island in self.island_manager.islands)
            avg_scores.append(total_score_all / sum(len(island) for island in self.island_manager.islands))

        # Update the best solution across all islands
        self.island_manager.update_final_island_best()

        # Return the scores for each generation
        return generation_scores, avg_scores

    def initialize_and_evaluate(self, spaces: list[float], values: list[float], space_limit: float) -> None:
        """Initialize the population and evaluate each individual."""
        self.population_manager.initialize_population(spaces, values, space_limit)
        self.population_manager.evaluate_population()


   