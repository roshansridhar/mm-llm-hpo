import json
import os
import matplotlib.pyplot as plt


class BaseOptimizer:
    def __init__(self, benchmarker):
        self.benchmarker = benchmarker
        self.history = []
        self.hyperparameter_def = {
            "SupportVectorMachine": "Here is the description of hyperparameters:\nC: regularization parameter. The strength of the regularization is inversely proportional to C. The penalty is a squared l2 penalty.\n gamma: Kernel coefficient for rbf.",
            "XGBoost": "Here is the description of hyperparameters:\ncolsample_bytree: Subsample ratio of columns when constructing each tree.\n eta: Boosting learning rate.\n max_depth: Maximum tree depth for base learners.\n reg_lambda: L2 regularization term on weights."
            }

    def optimize(self, iterations, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def generate_prompt(self, history, additional_info="", is_vision=False):
        model_name = self.benchmarker.model_name
        search_space = self.benchmarker.get_search_space()
        if not history:
            prompt = f"You are helping to tune hyperparameters for a {model_name} model for the first time. Here is " \
                     f"the search space:\n "
            prompt += json.dumps(search_space, indent=2) + "\n"
            prompt += self.hyperparameter_def[model_name] + "\n"
            prompt += "We have a budget to try 10 configurations in total. The goal is to find the configuration that minimizes the validation loss with the given budget" + "\n"
            prompt += "Please suggest the initial configuration strictly in JSON format. Config:"
        else:
            if is_vision:
                prompt = f"You are helping tune hyperparameters for a {model_name} model. Here is what is tried so far:\n"
                for i, config, score in history:
                    prompt += f"Iteration: {i+1}, Config: {config}\n"
                prompt += "The validation loss for each iteration as shown in the image."
                prompt += f"Please provide the next configuration strictly in JSON format within the search space:\n"
                prompt += f"{json.dumps(search_space, indent=2)}\nConfig:"

            else:
                prompt = f"You are helping tune hyperparameters for a {model_name} model. Here is what is tried so far:\n"
                for i, config, score in history:
                    prompt += f"Config: {config}, Validation_loss: {score['validation_loss']}\n"
                prompt += additional_info
                prompt += f"Please provide the next configuration strictly in JSON format within the search space:\n"
                prompt += f"{json.dumps(search_space, indent=2)}\nConfig:"
        return prompt

    def plot_validation_loss(self):
        """Generates and saves a plot of validation loss over iterations."""
        if not self.history:
            return None

        # Extract validation loss and iterations
        iterations = [i+1 for i in list(range(len(self.history)))]
        validation_losses = [entry[2]['validation_loss'] for entry in self.history]

        # Plot validation loss over iterations
        plt.figure(figsize=(8, 5))
        plt.plot(iterations, validation_losses, marker='o', linestyle='-')
        plt.title('Validation Loss Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Validation Loss')
        plt.xticks(iterations)
        plt.grid(True)

        # Save plot to a temporary file and return the file path
        output_dir = self.benchmarker.output_dir
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"validation_loss_plot_{self.benchmarker.task_id}_{len(iterations)}.png")
        plt.savefig(plot_path)
        plt.close()
        return plot_path
