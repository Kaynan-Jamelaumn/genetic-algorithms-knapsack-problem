import itertools  # Import itertools for generating combinations

class Knapsack:
    def __init__(self, volume_limit: float):
        """
        Initialize the Knapsack with a volume limit.
        
        :param volume_limit: The maximum volume capacity of the knapsack.
        """
        self.volume_limit = volume_limit  # Set the volume limit for the knapsack

    def find_best_combination(self, products):
        """
        Find the best combination of products that maximizes value without exceeding the volume limit.
        
        :param products: A list of product objects, each with `get_volume()` and `get_price()` methods.
        :return: A tuple containing the best combination of products and the total value of that combination.
        """
        best_combination = None  # Store the best combination of products
        max_value = 0  # Store the maximum value found

        # Iterate over all possible combination sizes (from 1 to the total number of products)
        for r in range(1, len(products) + 1):
            # Generate all combinations of size `r`
            for combination in itertools.combinations(products, r):
                total_volume = sum(product.get_volume() for product in combination)
                total_value = sum(product.get_price() for product in combination)

                # Check if the combination is valid (volume <= limit) and better than the current best
                if total_volume <= self.volume_limit and total_value > max_value:
                    best_combination = combination
                    max_value = total_value 

        return best_combination, max_value  # Return the best combination and its value