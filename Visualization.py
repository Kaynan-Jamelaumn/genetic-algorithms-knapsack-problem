import matplotlib.pyplot as plt
import numpy as np

class Visualization:
    @staticmethod
    def plot_generation_scores(generation_scores, avg_scores=None, save_path=None):
        generations = np.arange(len(generation_scores))
        num_generations = len(generation_scores)
        
        plt.figure(figsize=(14, 7))  # Wider figure for better readability

        # Dynamically adjust marker density based on the number of generations
        if num_generations > 300:
            marker_step = num_generations // 30  # Show fewer markers for large datasets
            alpha_value = 0.4  # More transparency for very large data
        elif num_generations > 100:
            marker_step = num_generations // 100  # Moderate marker density
            alpha_value = 0.6
        else:
            marker_step = 1  # Show all markers for small datasets
            alpha_value = 1.0

        # Plot best scores with transparency
        plt.plot(generations, generation_scores, linestyle='-', color='b', alpha=alpha_value, label='Best Score')
        plt.scatter(generations[::marker_step], generation_scores[::marker_step], color='b', label='Sampled Best Scores', marker='o')

        # Plot average scores with a dashed red line
        if avg_scores:
            plt.plot(generations, avg_scores, linestyle='--', color='r', alpha=alpha_value, label='Average Score')
            plt.scatter(generations[::marker_step], avg_scores[::marker_step], color='r', label='Sampled Average Scores', marker='s')

        # Add best score annotation
        best_gen = np.argmax(generation_scores)
        best_score = generation_scores[best_gen]
        plt.annotate(f'Best: {best_score:.2f}', 
                     xy=(best_gen, best_score), 
                     xytext=(best_gen, best_score + 5), 
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     fontsize=10)

        plt.title('Evolution of the Best Solution Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Evaluation Score')
        plt.legend()
        plt.grid(True)

        # Save or Show
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()
