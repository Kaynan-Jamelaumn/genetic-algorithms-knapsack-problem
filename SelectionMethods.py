from random import random, choice, sample 
from math import exp  # Import exp for Boltzmann selection



class SelectionMethods:
    @staticmethod
    def roulette_selection(population, total_score, *args):
        """
        Roulette wheel selection: Selects individuals based on their fitness proportion.
        
        :param total_score: The total evaluation score of the population.
        :return: The selected individual.
        """
        selected_value = random() * total_score
        running_sum = 0
        for individual in population:
            running_sum += individual.evaluation_score
            if running_sum >= selected_value:
                return individual
        return population[-1]

    @staticmethod
    def tournament_selection(population, k=3):
        """
        Tournament selection: Selects the best individual from a random subset of the population.
        
        :param k: Number of competitors in the tournament (default: 3).
        :return: The selected individual.
        """
        competitors = sample(population, k)
        return max(competitors, key=lambda ind: ind.evaluation_score)

    @staticmethod
    def rank_selection(population):
        """
        Rank selection: Selects individuals based on their rank in the sorted population.
        
        :return: The selected individual.
        """
        sorted_population = sorted(population, key=lambda ind: ind.evaluation_score)
        ranks = list(range(1, len(sorted_population) + 1))
        total_rank = sum(ranks)
        selected_value = random() * total_rank
        running_sum = 0
        for i, individual in enumerate(sorted_population):
            running_sum += ranks[i]
            if running_sum >= selected_value:
                return individual
        return sorted_population[-1]

    @staticmethod
    def truncation_selection(population, percentage=0.5):
        """
        Truncation selection: Selects individuals from the top percentage of the population.
        
        :param percentage: The percentage of the population to consider (default: 0.5).
        :return: The selected individual.
        """
        cutoff = int(len(population) * percentage)
        return choice(population[:cutoff])

    @staticmethod
    def sus_selection(population, total_score, population_size):
        """
        Stochastic Universal Sampling (SUS): Selects individuals evenly spaced across the population.
        
        :param total_score: The total evaluation score of the population.
        :return: The selected individual.
        """
        pointer_distance = total_score / population_size
        start_point = random() * pointer_distance
        points = [start_point + i * pointer_distance for i in range(population_size)]

        selected = []
        running_sum = 0
        index = 0
        for individual in population:
            running_sum += individual.evaluation_score
            while index < len(points) and running_sum >= points[index]:
                selected.append(individual)
                index += 1
        return choice(selected)

    @staticmethod
    def boltzmann_selection(population, temperature=1.0):
        """
        Boltzmann selection: Selects individuals based on their fitness adjusted by a temperature parameter.
        
        :param temperature: The temperature parameter (default: 1.0).
        :return: The selected individual.
        """
        exp_scores = [exp(ind.evaluation_score / temperature) for ind in population]
        total_exp_score = sum(exp_scores)
        
        selected_value = random() * total_exp_score
        running_sum = 0
        
        for i, individual in enumerate(population):
            running_sum += exp_scores[i]
            if running_sum >= selected_value:
                return individual
        return population[-1]

    @staticmethod
    def linear_ranking_selection(population):
        """
        Linear ranking selection: Selects individuals based on their linearly assigned ranks.
        
        :return: The selected individual.
        """
        sorted_population = sorted(population, key=lambda ind: ind.evaluation_score)
        population_size = len(sorted_population)
        
        ranks = [i + 1 for i in range(population_size)]
        total_rank = sum(ranks)
        
        selected_value = random() * total_rank
        running_sum = 0
        
        for i, individual in enumerate(sorted_population):
            running_sum += ranks[i]
            if running_sum >= selected_value:
                return individual
        return sorted_population[-1]

    @staticmethod
    def exponential_ranking_selection(population, base=1.5):
        """
        Exponential ranking selection: Selects individuals based on their exponentially assigned ranks.
        
        :param base: The base for the exponential function (default: 1.5).
        :return: The selected individual.
        """
        sorted_population = sorted(population, key=lambda ind: ind.evaluation_score)
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

    @staticmethod
    def mu_lambda_selection(population, population_size, mu_ratio=0.5):
        """
        (μ, λ) selection: Selects individuals from the top μ portion of the population.
        
        :param mu_ratio: The ratio of the population to consider (default: 0.5).
        :return: The selected individual.
        """
        mu = int(population_size * mu_ratio)
        return choice(population[:mu])

    @staticmethod
    def metropolis_hastings_selection(population):
        """
        Metropolis-Hastings selection: A probabilistic selection method inspired by MCMC.
        
        :return: The selected individual.
        """
        candidate1 = choice(population)
        candidate2 = choice(population)
        
        if candidate2.evaluation_score > candidate1.evaluation_score:
            return candidate2
        
        probability = candidate2.evaluation_score / (candidate1.evaluation_score + 1e-10)
        return candidate2 if random() < probability else candidate1

    @staticmethod
    def remainder_stochastic_sampling(population, total_score, population_size):
        """
        Remainder Stochastic Sampling (RSS): A combination of deterministic and probabilistic selection.
        
        :param total_score: The total evaluation score of the population.
        :return: The selected individual.
        """
        expected_counts = [(ind, (ind.evaluation_score / total_score) * population_size) for ind in population]

        selected = []
        for individual, expected in expected_counts:
            selected.extend([individual] * int(expected))
        
        remaining_spots = population_size - len(selected)
        probabilities = [(expected % 1) for _, expected in expected_counts]
        probabilistic_choices = [ind for ind, _ in sorted(zip(population, probabilities), key=lambda x: x[1], reverse=True)]
        
        selected.extend(sample(probabilistic_choices, remaining_spots))
        return choice(selected)

    @staticmethod
    def steady_state_selection(population, population_size):
        """
        Steady-state selection: Selects individuals from the top portion of the population.
        
        :return: The selected individual.
        """
        return choice(population[:int(population_size * 0.2)])

    @staticmethod
    def random_selection(population):
        """
        Random selection: Selects a random individual from the population.
        
        :return: The selected individual.
        """
        return choice(population)