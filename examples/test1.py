import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from benchmarkers.test_nn_benchmarker import TestNNBenchmarker
from optimizers.random_search import RandomSearchOptimizer
from benchmarkers.test_lr_benchmarker import LogisticRegressionBenchmarker
from optimizers.llm_optimizer import LLMOptimizer
import os
from dotenv import load_dotenv

load_dotenv()
opk = os.getenv('KEY')  # add your API key in .env file

# Assuming all necessary classes and functions have been imported
def run_optimizers(benchmarker, optimizers, iterations):
    results = []
    for name, optimizer in optimizers.items():
        result = optimizer.optimize(iterations, )
        for config, metrics in result:
            metrics.update({
                "optimizer": name,
                "benchmarker": type(benchmarker).__name__,
                "config": str(config)  # Convert config dict to string for easier handling in DataFrame
            })
            results.append(metrics)
    return results


# Initialize benchmarkers
benchmarker_nn = TestNNBenchmarker()
benchmarker_lr = LogisticRegressionBenchmarker()

# Initialize optimizers

optimizers_nn = {
    "RandomSearch": RandomSearchOptimizer(benchmarker_nn),
    "LLMOptimizer": LLMOptimizer(benchmarker_nn, opk)
}

optimizers_lr = {
    "RandomSearch": RandomSearchOptimizer(benchmarker_lr),
    "LLMOptimizer": LLMOptimizer(benchmarker_lr, opk)
}

# Run experiments
iterations = 5
results_nn = run_optimizers(benchmarker_nn, optimizers_nn, iterations)
results_lr = run_optimizers(benchmarker_lr, optimizers_lr, iterations)

# Combine and convert to DataFrame
all_results = results_nn + results_lr
df = pd.DataFrame(all_results)

# Save results to CSV (optional)
df.to_csv("optimizer_results.csv", index=False)

# Set up the plot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='optimizer', y='function_value', hue='benchmarker')
plt.title('Comparison of Optimizer Performance by Function Value')
plt.ylabel('Function Value (Error)')
plt.xlabel('Optimizer')
plt.legend(title='Benchmarker')

# Save the figure
plt.savefig('optimizer_performance_comparison.png', dpi=300)  # Saves the figure as a PNG file with high resolution

# Clear the figure after saving to free memory, especially useful if creating many plots in a loop
plt.clf()
