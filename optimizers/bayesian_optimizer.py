from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade, Scenario
from optimizers.base_optimizer import BaseOptimizer
from ConfigSpace import ConfigurationSpace


class BayesianOptimizer(BaseOptimizer):
    def __init__(self, benchmarker):
        super().__init__(benchmarker)
        # The search space is retrieved here but used in the optimize method.
        self.search_space = self.benchmarker.get_search_space()

    def optimize(self, iterations, **kwargs):
        # Dynamically set the number of trials based on the iterations argument
        scenario_config = {
            'configspace': ConfigurationSpace(self.search_space),
            'deterministic': True,
            'n_trials': iterations
        }
        scenario = Scenario(**scenario_config)

        # Initialize SMAC with the scenario and the target function
        smac = HyperparameterOptimizationFacade(scenario, self.benchmarker.evaluate)

        # Optimize the hyperparameters
        best_config = smac.optimize()

        # Evaluate the best found configuration
        best_score = self.benchmarker.evaluate(best_config)

        # Store the result in history
        self.history.append((iterations, best_config, best_score))

        return self.history
