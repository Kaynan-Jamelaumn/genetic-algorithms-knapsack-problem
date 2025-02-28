import inspect
class IslandManager:
    def __init__(self, population_manager, genetic_operators, migration_methods, replacement_methods, calculate_mutation_rate_method):
        self.population_manager = population_manager
        self.genetic_operators = genetic_operators
        self.migration_methods = migration_methods
        self.replacement_methods = replacement_methods
        self.calculate_mutation_rate = calculate_mutation_rate_method
        self.islands = []

    def split_into_islands(self, num_islands: int) -> None:
        """
        Splits the main population into multiple islands (sub-populations).

        :param num_islands: The number of islands to create.
        """
        island_size = self.population_manager.population_size // num_islands # Determine size per island by dividing the total population size by the number of islands
        self.islands = []
        for i in range(num_islands):
            start = i * island_size # Calculate the starting index for the current island's population
            end = start + island_size  # Calculate the ending index for the current island's population
            if i == num_islands - 1: # Ensure the last island gets any remaining individuals
                end = self.population_manager.population_size
            self.islands.append(self.population_manager.population[start:end])  # Append the slice of the population corresponding to the current island to the islands list

    def evolve_island(self, island: list, mutation_rate: float, adaptative_mutation: bool, elitism_chance: float) -> list:
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
        original_population = self.population_manager.population
        self.population_manager.population = island.copy()

        total_score = self.population_manager.sum_evaluations()
        adapted_rate = self.calculate_mutation_rate(adaptative_mutation, mutation_rate)
        new_population = self.population_manager.generate_new_population(
            self.population_manager.population, total_score, adapted_rate, elitism_chance, self.genetic_operators
        )

        self.population_manager.population = new_population
        self.population_manager.evaluate_population()
        self.population_manager.sort_population()

        evolved_island = self.population_manager.population
        self.population_manager.population = original_population
        return evolved_island

    def apply_migration_method(self, num_migrants: int, migration_method: str, primary_replacement_method: str, secundary_replacement_method: str, migration_args: tuple) -> None:
        """
        Apply the specified migration method to the island model.
        """
        if migration_method not in self.migration_methods:
            raise ValueError(f"Invalid migration method: {migration_method}")
        migration_function = self.migration_methods[migration_method]
                # Get the function's parameters
        sig = inspect.signature(migration_function)
        params = list(sig.parameters.values())

        # Calculate how many additional parameters the function expects after the first 5
        # (islands, num_migrants, replacement_methods, primary, secondary)
        num_extra_params = max(len(params) - 5, 0)

        # Slice migration_args to match the number of extra parameters
        args_to_pass = migration_args[:num_extra_params]
        migration_function(
            self.islands,
            num_migrants,
            self.replacement_methods,
            primary_replacement_method,
            secundary_replacement_method,
            *args_to_pass
        )

    def update_final_island_best(self) -> None:
        """
        After all generations, updates the best solution by considering all islands.
        """
        all_individuals = [ind for island in self.islands for ind in island]  # Flatten all islands into one list
        all_individuals.sort(key=lambda x: x.evaluation_score, reverse=True)  # Sort by best fitness
        self.population_manager.best_solution = all_individuals[0]  # Pick the top individual
        self.population_manager.solution_list.append(self.population_manager.best_solution.evaluation_score)  # Store the best solution score