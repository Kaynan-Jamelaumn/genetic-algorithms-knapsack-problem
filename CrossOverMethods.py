class CrossoverMethods:
    @staticmethod
    def single_point_crossover(parent1, parent2):
        return parent1.crossover(parent2)

    @staticmethod
    def uniform_crossover(parent1, parent2):
        return parent1.uniform_crossover(parent2)

    @staticmethod
    def two_point_crossover(parent1, parent2):
        return parent1.two_point_crossover(parent2)

    @staticmethod
    def arithmetic_crossover(parent1, parent2):
        return parent1.arithmetic_crossover(parent2)

    @staticmethod
    def half_uniform_crossover(parent1, parent2):
        return parent1.half_uniform_crossover(parent2)