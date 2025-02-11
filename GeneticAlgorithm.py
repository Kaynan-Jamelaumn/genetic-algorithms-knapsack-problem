from Individual import Individual 

from SelectionMethods import *
from Visualization import *
from CrossOverMethods import *
from MutationMethods import *

class GeneticAlgorithm():
    def __init__(self, population_size: int, selection_method: str ="roulette", crossover_method:str ="single_point", mutation_method:str ="bit_flip"):
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
        self.mutation_method = mutation_method # mutation method to use
        self.population = []  # List to hold the population of individuals
        self.generation = 0  # Current generation number
        self.best_solution = None  # Best solution found so far
        self.solution_list = []  # List to store the best solution scores over generations

        # Mapping of selection methods to their corresponding functions
        self.selection_methods = {
            "roulette": SelectionMethods.roulette_selection,
            "tournament": SelectionMethods.tournament_selection,
            "rank": SelectionMethods.rank_selection,
            "truncation": SelectionMethods.truncation_selection,
            "sus": SelectionMethods.sus_selection,
            "steady_state": SelectionMethods.steady_state_selection,
            "random": SelectionMethods.random_selection,
            "boltzmann": SelectionMethods.boltzmann_selection,
            "linear_ranking": SelectionMethods.linear_ranking_selection,
            "exponential_ranking": SelectionMethods.exponential_ranking_selection,
            "mu_lambda": SelectionMethods.mu_lambda_selection,
            "metropolis_hastings": SelectionMethods.metropolis_hastings_selection,
            "rss": SelectionMethods.remainder_stochastic_sampling
        }

        self.crossover_methods = {
            "single_point": CrossoverMethods.single_point_crossover,
            "uniform": CrossoverMethods.uniform_crossover,
            "two_point": CrossoverMethods.two_point_crossover,
            "arithmetic": CrossoverMethods.arithmetic_crossover,
            "half_uniform": CrossoverMethods.half_uniform_crossover
        }

        # Mapping of mutation methods
        self.mutation_methods = {
            "bit_flip": MutationMethods.bit_flip,
            "swap_mutation": MutationMethods.swap_mutation,
            "scramble_mutation": MutationMethods.scramble_mutation,
            "random": MutationMethods.random_mutation,
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
        
        if method in {SelectionMethods.roulette_selection, SelectionMethods.sus_selection}:
            return method(self.population, total_score)
        else:
            return method(self.population)

    def apply_mutation(self, individual, mutation_chance):
        """
        Apply the specified mutation method to the individual.
        """
        if self.mutation_method not in self.mutation_methods:
            raise ValueError(f"Invalid mutation method: {self.mutation_method}")
        mutation_function = self.mutation_methods[self.mutation_method]
          
        if self.mutation_method == "swap_mutation" or self.mutation_method == "scramble_mutation":
            mutation_function(individual)
        else:
            mutation_function(individual, mutation_chance)

        

    def visualize_generation(self) -> None:
        """
        Print the best individual of the current generation.
        """
        best = self.population[0]
        print(f"G:{best.generation} -> Score: {best.evaluation_score} Chromosome: {best.chromosome}")

    def calculate_diversity(self):
        """
        Get the diversity (how individuals are different from each other)
        """
        diversity = 0
        for i in range(len(self.population)):
            for j in range(i+1, len(self.population)):
                diversity += sum(c1 != c2 for c1, c2 in zip(
                    self.population[i].chromosome,
                    self.population[j].chromosome))
        return diversity / (self.population_size * (self.population_size - 1) / 2)

    def calculate_mutation_rate(self, adaptative_mutation : bool = True, mutation_rate : float = 0.5):

        diversity = self.calculate_diversity()
        if (adaptative_mutation):
            return mutation_rate * (1 - diversity) # Higher diversity -> Lower mutation
        else: 
            return  mutation_rate



    def solve(self, mutation_rate, num_generations, spaces, values, space_limit, generate_graphic=True, adaptative_mutation: bool = True,  elitism_chance: float| int = 0.05) -> list:
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


        # Store the best score for each generation
        generation_scores = [self.best_solution.evaluation_score]
        avg_scores = [self.sum_avaliation() / self.population_size] 
        
        for _ in range(num_generations):
             # Calculate diversity and adjust mutation rate
            adapted_mutation_rate = self.calculate_mutation_rate(adaptative_mutation, mutation_rate)


            total_score = self.sum_avaliation()
            avg_scores.append(total_score / self.population_size)  #Store avg fitness
            new_population = []

            # Preserve the top N individuals (Elitism)
            elite_count = max(1, int(elitism_chance * self.population_size))
            elites = self.population[:elite_count]
            new_population.extend(elites)

            # Generate the rest of the population
            for _ in range((self.population_size - elite_count) // 2):
                parent1 = self.select_parent(total_score)
                parent2 = self.select_parent(total_score)
                child1, child2 = self.apply_crossover(parent1, parent2)
                
                self.apply_mutation(child1, adapted_mutation_rate)
                self.apply_mutation(child2, adapted_mutation_rate)
                
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
            generation_scores.append(best.evaluation_score)

        print(f"\nBest solution -> G: {self.best_solution.generation}, "
              f"Score: {self.best_solution.evaluation_score}, "
              f"Chromosome: {self.best_solution.chromosome}")
        
        
        if generate_graphic:
            Visualization.plot_generation_scores(generation_scores, avg_scores)
        
        return self.best_solution.chromosome,  generation_scores
    