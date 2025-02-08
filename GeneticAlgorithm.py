from Individual import Individual 
from random import random, choice, sample 
from math import exp  # Import exp for Boltzmann selection

class GeneticAlgorithm():
    def __init__(self, population_size: int, selection_method: str ="roulette", crossover_method:str ="single_point", elitism_chance: float| int = 0.05):
        """
        Initialize the Genetic Algorithm with the given parameters.
        
        :param population_size: Number of individuals in the population.
        :param selection_method: Method used for selecting parents (default: "roulette").
        :param crossover_method: Method used for crossover (default: "single_point").
        :param elitism_chance: Percentage of top individuals to carry over to the next generation (default: 0.05).
        """
        self.population_size = population_size  # Size of the population
        self.selection_method = selection_method  # Selection method to use
        self.crossover_method = crossover_method  # Crossover method to use
        self.population = []  # List to hold the population of individuals
        self.generation = 0  # Current generation number
        self.best_solution = None  # Best solution found so far
        self.solution_list = []  # List to store the best solution scores over generations
        self.elitsm_chance = elitism_chance  # Percentage of elites to preserve

        # Mapping of selection methods to their corresponding functions
        self.selection_methods = {
            "roulette": self.roulette_selection,
            "tournament": self.tournament_selection,
            "rank": self.rank_selection,
            "truncation": self.truncation_selection,
            "sus": self.sus_selection,
            "steady_state": self.steady_state_selection,
            "random": self.random_selection,
            "boltzmann": self.boltzmann_selection,
            "linear_ranking": self.linear_ranking_selection,
            "exponential_ranking": self.exponential_ranking_selection,
            "mu_lambda": self.mu_lambda_selection,
            "metropolis_hastings": self.metropolis_hastings_selection,
            "rss": self.remainder_stochastic_sampling
        }

        # Mapping of crossover methods to their corresponding functions
        self.crossover_methods = {
            "single_point": lambda p1, p2: p1.crossover(p2),
            "uniform": lambda p1, p2: p1.uniform_crossover(p2),
            "two_point": lambda p1, p2: p1.two_point_crossover(p2),
            "arithmetic": lambda p1, p2: p1.arithmetic_crossover(p2),
            "half_uniform": lambda p1, p2: p1.half_uniform_crossover(p2)
        }

    def initialize_population(self, spaces, values, space_limit) -> None:
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

    def update_best_individual(self, individual) -> None:
        """
        Update the best solution if the given individual has a better score.
        
        :param individual: The individual to compare with the current best solution.
        """
        if individual.evaluation_score > self.best_solution.evaluation_score:
            self.best_solution = individual

    def sum_avaliation(self) -> float:
        """Calculate the total evaluation score of the population."""
        return sum(ind.evaluation_score for ind in self.population)
    
    def apply_crossover(self, parent1, parent2) -> tuple[Individual, Individual]:
        """
        Apply crossover between two parents to produce two children.
        
        :param parent1: The first parent.
        :param parent2: The second parent.
        :return: Two new individuals (children) resulting from the crossover.
        """
        if self.crossover_method not in self.crossover_methods:
            raise ValueError(f"Invalid crossover method: {self.crossover_method}")
        return self.crossover_methods[self.crossover_method](parent1, parent2)

    def select_parent(self, total_score) -> Individual:
        """
        Select a parent based on the chosen selection method.
        
        :param total_score: The total evaluation score of the population (used for some selection methods).
        :return: The selected parent.
        """
        if self.selection_method not in self.selection_methods:
            raise ValueError(f"Invalid selection method: {self.selection_method}")
        
        # Call the appropriate selection method dynamically
        method = self.selection_methods[self.selection_method]
        
        # Some methods require `total_score`, so we pass it when needed
        return method(total_score) if method in {self.roulette_selection, self.sus_selection} else method()

    def roulette_selection(self, total_score) -> Individual:
        """
        Roulette wheel selection: Selects individuals based on their fitness proportion.
        
        :param total_score: The total evaluation score of the population.
        :return: The selected individual.
        """
        selected_value = random() * total_score
        running_sum = 0
        for individual in self.population:
            running_sum += individual.evaluation_score
            if running_sum >= selected_value:
                return individual
        return self.population[-1]

    def tournament_selection(self, k=3) -> Individual:
        """
        Tournament selection: Selects the best individual from a random subset of the population.
        
        :param k: Number of competitors in the tournament (default: 3).
        :return: The selected individual.
        """
        competitors = sample(self.population, k)
        return max(competitors, key=lambda ind: ind.evaluation_score)
    
    def rank_selection(self) -> Individual:
        """
        Rank selection: Selects individuals based on their rank in the sorted population.
        
        :return: The selected individual.
        """
        sorted_population = sorted(self.population, key=lambda ind: ind.evaluation_score)
        ranks = list(range(1, len(sorted_population) + 1))
        total_rank = sum(ranks)
        selected_value = random() * total_rank
        running_sum = 0
        for i, individual in enumerate(sorted_population):
            running_sum += ranks[i]
            if running_sum >= selected_value:
                return individual
        return sorted_population[-1]
    
    def truncation_selection(self, percentage=0.5) -> Individual:
        """
        Truncation selection: Selects individuals from the top percentage of the population.
        
        :param percentage: The percentage of the population to consider (default: 0.5).
        :return: The selected individual.
        """
        cutoff = int(len(self.population) * percentage)
        return choice(self.population[:cutoff])

    def sus_selection(self, total_score) -> Individual:
        """
        Stochastic Universal Sampling (SUS): Selects individuals evenly spaced across the population.
        
        :param total_score: The total evaluation score of the population.
        :return: The selected individual.
        """
        pointer_distance = total_score / self.population_size
        start_point = random() * pointer_distance
        points = [start_point + i * pointer_distance for i in range(self.population_size)]

        selected = []
        running_sum = 0
        index = 0
        for individual in self.population:
            running_sum += individual.evaluation_score
            while index < len(points) and running_sum >= points[index]:
                selected.append(individual)
                index += 1
        return choice(selected)
    
    def boltzmann_selection(self, temperature=1.0) -> Individual:
        """
        Boltzmann selection: Selects individuals based on their fitness adjusted by a temperature parameter.
        
        :param temperature: The temperature parameter (default: 1.0).
        :return: The selected individual.
        """
        exp_scores = [exp(ind.evaluation_score / temperature) for ind in self.population]
        total_exp_score = sum(exp_scores)
        
        selected_value = random() * total_exp_score
        running_sum = 0
        
        for i, individual in enumerate(self.population):
            running_sum += exp_scores[i]
            if running_sum >= selected_value:
                return individual
        return self.population[-1]
    
    def linear_ranking_selection(self) -> Individual:
        """
        Linear ranking selection: Selects individuals based on their linearly assigned ranks.
        
        :return: The selected individual.
        """
        sorted_population = sorted(self.population, key=lambda ind: ind.evaluation_score)
        population_size = len(sorted_population)
        
        ranks = [i + 1 for i in range(population_size)]  # Assign ranks 1 to N
        total_rank = sum(ranks)
        
        selected_value = random() * total_rank
        running_sum = 0
        
        for i, individual in enumerate(sorted_population):
            running_sum += ranks[i]
            if running_sum >= selected_value:
                return individual
        return sorted_population[-1]

    def exponential_ranking_selection(self, base=1.5) -> Individual:
        """
        Exponential ranking selection: Selects individuals based on their exponentially assigned ranks.
        
        :param base: The base for the exponential function (default: 1.5).
        :return: The selected individual.
        """
        sorted_population = sorted(self.population, key=lambda ind: ind.evaluation_score)
        population_size = len(sorted_population)

        ranks = [base ** i for i in range(population_size)]
        total_rank = sum(ranks)

        selected_value = random() * total_rank
        running_sum = 0

        for i, individual in enumerate(sorted_population):
            running_sum += ranks[i]
            if running_sum >= selected_value:
                return individual
        return sorted_population[-1]

    def mu_lambda_selection(self, mu_ratio=0.5) -> Individual:
        """
        (μ, λ) selection: Selects individuals from the top μ portion of the population.
        
        :param mu_ratio: The ratio of the population to consider (default: 0.5).
        :return: The selected individual.
        """
        mu = int(self.population_size * mu_ratio)
        return choice(self.population[:mu])

    def metropolis_hastings_selection(self) -> Individual:
        """
        Metropolis-Hastings selection: A probabilistic selection method inspired by MCMC.
        
        :return: The selected individual.
        """
        candidate1 = choice(self.population)
        candidate2 = choice(self.population)
        
        if candidate2.evaluation_score > candidate1.evaluation_score:
            return candidate2  # Always pick the better one
        
        # If candidate2 is worse, pick it with a probability based on the difference
        probability = candidate2.evaluation_score / (candidate1.evaluation_score + 1e-10)  # Avoid division by zero
        return candidate2 if random() < probability else candidate1

    def remainder_stochastic_sampling(self, total_score) -> Individual:
        """
        Remainder Stochastic Sampling (RSS): A combination of deterministic and probabilistic selection.
        
        :param total_score: The total evaluation score of the population.
        :return: The selected individual.
        """
        expected_counts = [(ind, (ind.evaluation_score / total_score) * self.population_size) for ind in self.population]

        # Deterministic selection (floor values)
        selected = []
        for individual, expected in expected_counts:
            selected.extend([individual] * int(expected))  # Add deterministic selections
        
        # Fill remaining slots probabilistically
        remaining_spots = self.population_size - len(selected)
        probabilities = [(expected % 1) for _, expected in expected_counts]  # Get fractional parts
        probabilistic_choices = [ind for ind, _ in sorted(zip(self.population, probabilities), key=lambda x: x[1], reverse=True)]
        
        selected.extend(sample(probabilistic_choices, remaining_spots))
        return choice(selected)

    def steady_state_selection(self) -> Individual:
        """
        Steady-state selection: Selects individuals from the top portion of the population.
        
        :return: The selected individual.
        """
        return choice(self.population[:int(self.population_size * 0.2)])
    
    def random_selection(self) -> Individual:
        """
        Random selection: Selects a random individual from the population.
        
        :return: The selected individual.
        """
        return choice(self.population)

    def visualize_generation(self) -> None:
        """
        Print the best individual of the current generation.
        """
        best = self.population[0]
        print(f"G:{best.generation} -> Score: {best.evaluation_score} Chromosome: {best.chromosome}")

    def solve(self, mutation_rate, num_generations, spaces, values, space_limit) -> list:
        """
        Run the genetic algorithm to solve the problem.
        
        :param mutation_rate: The probability of mutation for each individual.
        :param num_generations: The number of generations to run the algorithm.
        :param spaces: List of space requirements for each item.
        :param values: List of values for each item.
        :param space_limit: Maximum space limit for the knapsack problem.
        :return: The best solution chromosome found.
        """
        self.initialize_population(spaces, values, space_limit)
        for individual in self.population:
            individual.evaluate()
        
        self.sort_population()
        self.best_solution = self.population[0]
        self.solution_list.append(self.best_solution.evaluation_score)
        self.visualize_generation()

        for _ in range(num_generations):
            total_score = self.sum_avaliation()
            new_population = []

            # Preserve the top N individuals (Elitism)
            elite_count = max(1, int(self.elitism_chance * self.population_size))
            elites = self.population[:elite_count]
            new_population.extend(elites)

            # Generate the rest of the population
            for _ in range((self.population_size - elite_count) // 2):
                parent1 = self.select_parent(total_score)
                parent2 = self.select_parent(total_score)
                child1, child2 = self.apply_crossover(parent1, parent2)
                
                child1.mutation(mutation_rate)
                child2.mutation(mutation_rate)
                
                new_population.append(child1)
                new_population.append(child2)

            self.population = new_population

            for individual in self.population:
                individual.evaluate()
            
            self.sort_population()
            self.visualize_generation()
            best = self.population[0]
            self.solution_list.append(best.evaluation_score)
            self.update_best_individual(best)

        print(f"\nBest solution -> G: {self.best_solution.generation}, "
              f"Score: {self.best_solution.evaluation_score}, "
              f"Chromosome: {self.best_solution.chromosome}")
        
        return self.best_solution.chromosome