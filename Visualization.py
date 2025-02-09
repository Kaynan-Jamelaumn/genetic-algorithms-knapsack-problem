import matplotlib.pyplot as plt

class Visualization:
    @staticmethod
    def plot_generation_scores(generation_scores):
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(generation_scores)), generation_scores, marker='o', linestyle='-', color='b')
        plt.title('Evolution of the Best Solution Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Best Evaluation Score')
        plt.grid(True)
        plt.show()