import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from benchmarkers.benchmarker import Benchmarker
import json
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
        for i, config, metrics in result:
            metrics.update({
                "dataset_name": dataset,
                "task_id": benchmarker.task_id,
                "iteration": i,
                "optimizer": name,
                "benchmarker": type(benchmarker).__name__,
                "config": str(config)  # Convert config dict to string for easier handling in DataFrame
            })
            results.append(metrics)
    return results



task_id_path = "task_ids.json"

with open(task_id_path, "r") as file:
    task_dict = json.load(file)


for dataset, id in task_dict.items():
    results = []
    for model in ["svm", "xgb"]:
        # Initialize benchmarkers
        benchmarker = Benchmarker(id, model)
        search_space = benchmarker.get_search_space()
        print(dataset)
        print(benchmarker.task_id)
        print(benchmarker.model_name)
        print(search_space)


        # Initialize optimizers
        optimizers = {
            "RandomSearch": RandomSearchOptimizer(benchmarker),
            "LLMOptimizer": LLMOptimizer(benchmarker, opk)
        }
        # Run experiments
        iterations = 5
        # Combine the results for one dataset
        results += (run_optimizers(benchmarker, optimizers, iterations))


    # Save results to CSV (optional)
    df = pd.DataFrame(results)
    df.to_csv(f"optimizer_results_for_dataset_{dataset}.csv", index=False)

    # Set up the plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='optimizer', y='validation_loss', hue='benchmarker')
    plt.title(f'Comparison of Optimizer Performance for dataset {dataset}')
    plt.ylabel('Function Value (Error)')
    plt.xlabel('Optimizer')
    plt.legend(title='Benchmarker')

    # Save the figure
    plt.savefig(f'optimizer_performance_comparison_for_dataset_{dataset}.png', dpi=300)  # Saves the figure as a PNG file with high resolution

    # Clear the figure after saving to free memory, especially useful if creating many plots in a loop
    plt.clf()
