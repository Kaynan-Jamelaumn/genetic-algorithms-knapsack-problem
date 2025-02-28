from __future__ import annotations  # Enable forward references for type hints
from random import randint, random, sample

class Individual:
    def __init__(self, space: list[float], price: list[float], total_space: int, generation: int = 0):
        """
        Initialize an individual in the population.
        
        :param space: List of space requirements for each item.
        :param price: List of prices (values) for each item.
        :param total_space: Maximum space limit for the knapsack problem.
        :param generation: The generation number of the individual (default: 0).
        """
        self.space = space  # Space requirements for items
        self.price = price  # Prices (values) of items
        self.total_space = total_space  # Maximum space limit
        self.evaluation_score = 0  # Fitness score of the individual
        self.generation = generation  # Generation number
        self.chromosome = []  # Binary chromosome representing the solution
        self.space_value = []  # List of tuples (space, price) for selected items

    def generate_chromosome(self) ->  list[int]:
        """
        Generate a random binary chromosome for the individual.
        
        :return: The generated chromosome as a list of 0s and 1s.
        """
        self.chromosome = [randint(0, 1) for _ in range(len(self.space))]  # Randomly select items (0 or 1)
        return self.chromosome

    def evaluate(self) -> float:
        """
        Evaluate the fitness of the individual based on the knapsack constraints.
        
        :return: The fitness score of the individual.
        """
        # Create a list of (space, price) tuples for selected items (chromosome value == 1)
        self.space_value = [(self.space[i], self.price[i]) for i in range(len(self.chromosome)) if self.chromosome[i] == 1]

        # Check if the total space exceeds the limit
        if sum(s for s, _ in self.space_value) > self.total_space:
            self.evaluation_score = 1  # Penalize invalid solutions
        else:
            self.evaluation_score = sum(p for _, p in self.space_value)  # Sum of prices for valid solutions
        
        return self.evaluation_score

    def crossover(self, individual: Individual) -> tuple[Individual, Individual]:
        """
        Perform single-point crossover between two individuals.
        
        :param individual: The other parent for crossover.
        :return: Two new children resulting from the crossover.
        """
        split = randint(0, len(self.chromosome) - 1)  # Randomly choose a split point
        
        # Create two children with the next generation number
        child1 = Individual(self.space, self.price, self.total_space, self.generation + 1)
        child2 = Individual(self.space, self.price, self.total_space, self.generation + 1)

        # Combine chromosomes from both parents at the split point
        child1.chromosome = self.chromosome[:split] + individual.chromosome[split:]
        child2.chromosome = individual.chromosome[:split] + self.chromosome[split:]

        return child1, child2

    def uniform_crossover(self, individual: Individual) -> tuple[Individual, Individual]:
        """
        Perform uniform crossover between two individuals.
        
        :param individual: The other parent for crossover.
        :return: Two new children resulting from the crossover.
        """
        # Create two children with the next generation number
        child1 = Individual(self.space, self.price, self.total_space, self.generation + 1)
        child2 = Individual(self.space, self.price, self.total_space, self.generation + 1)

        # Randomly select genes from either parent for each child
        child1.chromosome = [self.chromosome[i] if random() < 0.5 else individual.chromosome[i] for i in range(len(self.chromosome))]
        child2.chromosome = [individual.chromosome[i] if random() < 0.5 else self.chromosome[i] for i in range(len(self.chromosome))]

        return child1, child2

    def two_point_crossover(self, individual: Individual) -> tuple[Individual, Individual]:
        """
        Perform two-point crossover between two individuals.
        
        :param individual: The other parent for crossover.
        :return: Two new children resulting from the crossover.
        """
        # Randomly choose two split points
        point1, point2 = sorted([randint(0, len(self.chromosome) - 1) for _ in range(2)])

        # Create two children with the next generation number
        child1 = Individual(self.space, self.price, self.total_space, self.generation + 1)
        child2 = Individual(self.space, self.price, self.total_space, self.generation + 1)

        # Combine chromosomes from both parents between the two points
        child1.chromosome = (self.chromosome[:point1] +
                             individual.chromosome[point1:point2] +
                             self.chromosome[point2:])
        child2.chromosome = (individual.chromosome[:point1] +
                             self.chromosome[point1:point2] +
                             individual.chromosome[point2:])

        return child1, child2

    def arithmetic_crossover(self, individual: Individual, alpha: float = 0.5) -> tuple[Individual, Individual]:
        """
        Perform arithmetic crossover (blending genes) between two individuals.
        
        :param individual: The other parent for crossover.
        :param alpha: The blending factor (default: 0.5).
        :return: Two new children resulting from the crossover.
        """
        # Create two children with the next generation number
        child1 = Individual(self.space, self.price, self.total_space, self.generation + 1)
        child2 = Individual(self.space, self.price, self.total_space, self.generation + 1)
        
        # Blend genes using the alpha parameter
        child1.chromosome = [(alpha * self.chromosome[i] + (1 - alpha) * individual.chromosome[i]) for i in range(len(self.chromosome))]
        child2.chromosome = [(alpha * individual.chromosome[i] + (1 - alpha) * self.chromosome[i]) for i in range(len(self.chromosome))]
        
        # Convert blended genes back to binary (0 or 1)
        child1.chromosome = [1 if gene >= 0.5 else 0 for gene in child1.chromosome]
        child2.chromosome = [1 if gene >= 0.5 else 0 for gene in child2.chromosome]
        
        return child1, child2

    def half_uniform_crossover(self, individual: Individual) -> tuple[Individual, Individual]:
        """
        Perform half-uniform crossover between two individuals.
        
        :param individual: The other parent for crossover.
        :return: Two new children resulting from the crossover.
        """
        # Create two children with the next generation number
        child1 = Individual(self.space, self.price, self.total_space, self.generation + 1)
        child2 = Individual(self.space, self.price, self.total_space, self.generation + 1)

        # Identify differing genes between the two parents
        differing_genes = [i for i in range(len(self.chromosome)) if self.chromosome[i] != individual.chromosome[i]]
        swap_count = len(differing_genes) // 2  # Swap half of the differing genes
        swap_indices = set(randint(0, len(differing_genes) - 1) for _ in range(swap_count))  # Randomly select genes to swap

        # Initialize children with parent chromosomes
        child1.chromosome = self.chromosome[:]
        child2.chromosome = individual.chromosome[:]

        # Swap selected genes between the children
        for i in swap_indices:
            idx = differing_genes[i]
            child1.chromosome[idx], child2.chromosome[idx] = child2.chromosome[idx], child1.chromosome[idx]
        
        return child1, child2

    def bit_flip(self, mutation_chance: float) -> None:
        """
        Perform mutation on the individual's chromosome.
        
        :param mutation_chance: Probability of flipping a gene (0 to 1).
        """
        # Flip genes with a probability of `mutation_chance`
        self.chromosome = [gene ^ 1 if random() < mutation_chance else gene for gene in self.chromosome]

    def swap_mutation(self) -> None:
        """
        Perform swap mutation by swapping two distinct genes in the chromosome.
        """
        if len(self.chromosome) > 1:
            idx1, idx2 = sample(range(len(self.chromosome)), 2)
            self.chromosome[idx1], self.chromosome[idx2] = self.chromosome[idx2], self.chromosome[idx1]

    def scramble_mutation(self) -> None:
        """
        Perform scramble mutation by shuffling a random subsection of the chromosome.
        """
        if len(self.chromosome) > 1:
            start, end = sorted(sample(range(len(self.chromosome)), 2))
            if start < end:  # Ensure a valid range
                self.chromosome[start:end] = random.sample(self.chromosome[start:end], len(self.chromosome[start:end]))


    def inversion_mutation(self) -> None:
            if len(self.chromosome) > 1:
                start, end = sorted(random.sample(range(len(self.chromosome)), 2))
                self.chromosome[start:end] = self.chromosome[start:end][::-1]