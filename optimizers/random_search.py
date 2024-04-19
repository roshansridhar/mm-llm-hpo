from optimizers.base_optimizer import BaseOptimizer
import numpy as np


class RandomSearchOptimizer(BaseOptimizer):
    def optimize(self, iterations, **kwargs):
        search_space = self.benchmarker.get_search_space()
        for _ in range(iterations):
            config = {}
            for param, values in search_space.items():
                if isinstance(values, tuple) and isinstance(values[0], float) and isinstance(values[1], float):
                    config[param] = np.random.uniform(values[0], values[1])
                elif isinstance(values, tuple) and isinstance(values[0], int) and isinstance(values[1], int):
                    config[param] = np.random.randint(values[0], values[1])
                elif isinstance(values, list):
                    config[param] = np.random.choice(values)
            score = self.benchmarker.evaluate(config)
            self.history.append((_, config, score))
        return self.history
