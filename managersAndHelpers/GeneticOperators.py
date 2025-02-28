from mapMethods.SelectionMethods import *
from mapMethods.CrossOverMethods import *
from mapMethods.MutationMethods import *
from mapMethods.MigrationMethods import *
from mapMethods.ReplacementMethods import *
from typing import Callable, Optional

class GeneticOperators:
    def __init__(
        self,
        selection_method: str = "roulette",
        crossover_method: str = "single_point",
        mutation_method: str = "bit_flip",
        migration_method: str = "star_migration_bidirectional",
        primary_replacement_method: str = "best",
        secondary_replacement_method: str = "random",
        migration_args: tuple = ()
    ):
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.migration_method = migration_method
        self.primary_replacement_method = primary_replacement_method
        self.secondary_replacement_method = secondary_replacement_method
        self.migration_args = migration_args

        # Selection methods mapping
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

        # Crossover methods mapping
        self.crossover_methods = {
            "single_point": CrossoverMethods.single_point_crossover,
            "uniform": CrossoverMethods.uniform_crossover,
            "two_point": CrossoverMethods.two_point_crossover,
            "arithmetic": CrossoverMethods.arithmetic_crossover,
            "half_uniform": CrossoverMethods.half_uniform_crossover
        }

        # Mutation methods mapping
        self.mutation_methods = {
            "bit_flip": MutationMethods.bit_flip,
            "swap_mutation": MutationMethods.swap_mutation,
            "scramble_mutation": MutationMethods.scramble_mutation,
            "random": MutationMethods.random_mutation,
            "inversion_mutation" MutationMethods.inversion_mutation
        }

    def select_parent(self, population: list, total_score: Optional[float] = None) -> Individual:
        """
        Select a parent based on the chosen selection method.
        
        :param total_score: The total evaluation score of the population (used for some selection methods).
        :return: The selected parent.
        """
        if self.selection_method not in self.selection_methods:
            raise ValueError(f"Invalid selection method: {self.selection_method}")

        method = self.selection_methods[self.selection_method]
        if method in {SelectionMethods.roulette_selection, SelectionMethods.sus_selection}:
            return method(population, total_score)
        else:
            return method(population)

    def apply_crossover(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        """
        Apply crossover between two parents to produce two children.
        
        :param parent1: The first parent.
        :param parent2: The second parent.
        :return: Two new individuals (children) resulting from the crossover.
        """
        if self.crossover_method not in self.crossover_methods:
            raise ValueError(f"Invalid crossover method: {self.crossover_method}")
        return self.crossover_methods[self.crossover_method](parent1, parent2)

    def apply_mutation(self, individual: Individual, mutation_chance: float) -> None:
        """
        Apply the specified mutation method to the individual.
        """
        if self.mutation_method not in self.mutation_methods:
            raise ValueError(f"Invalid mutation method: {self.mutation_method}")
        
        mutation_function = self.mutation_methods[self.mutation_method]
        if self.mutation_method in {"swap_mutation", "scramble_mutation"}:
            mutation_function(individual)
        else:
            mutation_function(individual, mutation_chance)

