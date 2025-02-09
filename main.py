from GeneticAlgorithm import GeneticAlgorithm
from Product import Product
from random import seed
# Seed for reproducibility
#seed(42)

# List of products
products = [
    Product("Rice", 1.11, 4.75),
    Product("Beans", 1.25, 8.00),
    Product("Wheat Flour", 1.67, 5.50),
    Product("Sugar", 1.25, 3.50),
    Product("Salt", 0.46, 1.50),
    Product("Cooking Oil", 0.90, 4.50),
    Product("Coffee", 1.39, 16.00),
    Product("Milk", 1.00, 3.75),
    Product("Butter", 0.54, 11.50),
    Product("Bread", 1.75, 10.00),
    Product("Pasta", 1.33, 7.50),
    Product("Canned Goods", 0.33, 6.00),
    Product("Soap", 0.22, 3.00),
    Product("Toilet Paper", 3.24, 4.50),
    Product("Shampoo", 0.85, 6.75),
    Product("Eggs", 2.50, 5.00),
    Product("Cheese", 0.48, 12.00),
    Product("Chicken", 3.00, 15.00),
    Product("Apples", 2.00, 4.00),
    Product("Bananas", 1.20, 3.00),
    Product("Juice", 1.75, 8.50),
]

# Extracting spaces and prices from products
spaces = [product.volume for product in products]
prices = [product.price for product in products]

# Genetic Algorithm parameters
population_size = 50
mutation_rate = 0.05
num_generations = 100
total_space = 10  # Knapsack capacity
selection_method = "roulette"  # Can be changed to other methods
crossover_method = "single_point"  # Can be changed to other methods

# Initialize and run the genetic algorithm
ga = GeneticAlgorithm(population_size, selection_method, crossover_method)
best_solution, generation_scores  = ga.solve(mutation_rate, num_generations, spaces, prices, total_space)

# Visualize the final best solution
selected_products = [products[i].name for i, gene in enumerate(best_solution) if gene == 1]
selected_spaces = [products[i].volume for i, gene in enumerate(best_solution) if gene == 1]
selected_prices = [products[i].price for i, gene in enumerate(best_solution) if gene == 1]

print("\nSelected Products in the Best Solution:")
for product, space, price in zip(selected_products, selected_spaces, selected_prices):
    print(f"{product}: Space = {space}, Price = {price}")

print(f"\nTotal Space Used: {sum(selected_spaces)}")
print(f"Total Price: {sum(selected_prices)}")

print("\nFinal Best Solution:")
print(f"Chromosome: {best_solution}")
