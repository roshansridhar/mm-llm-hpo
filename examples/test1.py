import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from benchmarkers.benchmarker import Benchmarker
import json
from optimizers.random_search import RandomSearchOptimizer
from optimizers.llm_optimizer import LLMOptimizer
from optimizers.bayesian_optimizer import BayesianOptimizer
from optimizers.vision_optimizer import GPT4VisionOptimizer
from dotenv import load_dotenv

load_dotenv()
opk = os.getenv('KEY')  # add your API key in .env file

output_dir = "examples_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def run_optimizers(benchmarker, optimizers, iterations):
    results = []
    for name, optimizer in optimizers.items():
        result = optimizer.optimize(iterations)
        for iteration, config, metrics in result:
            metrics.update({
                "dataset_name": dataset,
                "task_id": benchmarker.task_id,
                "iteration": iteration,
                "optimizer": name,
                "benchmarker": benchmarker.model_name,
                "config": str(config)  # Convert config dict to string for easier handling in DataFrame
            })
            results.append(metrics)
    return results


task_id_path = "task_ids.json"
with open(task_id_path, "r") as file:
    task_dict = json.load(file)

for dataset, dataset_id in task_dict.items():
    results = []
    for model in ["svm", "xgb"]:
        # Initialize benchmarkers
        benchmarker = Benchmarker(dataset_id, model, output_dir)
        search_space = benchmarker.get_search_space()
        print(dataset)
        print(benchmarker.task_id)
        print(benchmarker.model_name)
        print(search_space)

        # Initialize optimizers
        optimizers = {
            "RandomSearch": RandomSearchOptimizer(benchmarker),
            "LLMOptimizer": LLMOptimizer(benchmarker, opk),
            "BayesianOptimizer": BayesianOptimizer(benchmarker),
            "GPT4VisionOptimizer": GPT4VisionOptimizer(benchmarker, opk),
        }
        # Run experiments
        iterations = 3
        # Combine the results for one dataset
        results += run_optimizers(benchmarker, optimizers, iterations)

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, f"optimizer_results_for_dataset_{dataset}.csv"), index=False)

    # Set up the plot
    df_min_loss = df.loc[df.groupby(["benchmarker", "optimizer"]).validation_loss.idxmin()]
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_min_loss, x='optimizer', y='validation_loss', hue='benchmarker')
    plt.title(f'Comparison of Optimizer Performance for dataset {dataset}')
    plt.ylabel('Function Value (Error)')
    plt.xlabel('Optimizer')
    plt.legend(title='Benchmarker')

    # Save the figure
    plt.savefig(os.path.join(output_dir, f'optimizer_performance_comparison_for_dataset_{dataset}.png'), dpi=300)
    plt.clf()
