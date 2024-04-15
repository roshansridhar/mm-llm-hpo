import json


class BaseOptimizer:
    def __init__(self, benchmarker):
        self.benchmarker = benchmarker
        self.history = []

    def optimize(self, iterations, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def generate_prompt(self, history, additional_info=""):
        if not history:
            search_space = self.benchmarker.get_search_space()
            prompt = "You are helping to tune hyperparameters for the first time. Here is the search space:\n"
            prompt += json.dumps(search_space, indent=2) + "\n"
            prompt += "Please suggest the initial configuration in JSON format. Config:"
        else:
            prompt = "You are helping tune hyperparameters. Here is what we have tried so far:\n"
            for config, score in history:
                prompt += f"Config: {config}, Score: {score}\n"
            prompt += additional_info
            prompt += "Please provide the next configuration in JSON format. Config:"
        return prompt
