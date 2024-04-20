from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
from hpobench.benchmarks.ml.svm_benchmark import SVMBenchmark
from loguru import logger


class Benchmarker:
    def __init__(self, task_id, model_name="xgb", output_dir="."):
        self.task_id = task_id
        self.output_dir = output_dir
        self.config_space = None
        self.search_space = None
        if model_name == "xgb":
            self.model_name = "XGBoost"
            self.benchmarker = XGBoostBenchmark(task_id=task_id)
            self.fidelity = {"n_estimators": 2000, "subsample": 1}
        elif model_name == "svm":
            self.model_name = "SupportVectorMachine"
            self.benchmarker = SVMBenchmark(task_id=task_id)
            self.fidelity = {"subsample": 1}
        else:
            raise Exception("Need a model name: xgb or svm")

    def get_search_space(self, seed=1):
        self.config_space = self.benchmarker.get_configuration_space(seed)
        self.search_space = dict((i.name, (i.lower, i.upper)) for i in self.config_space.values())
        return self.search_space

    def evaluate(self, config, seed=1):
        result_dict = self.benchmarker.objective_function(configuration=config, fidelity=self.fidelity, rng=seed)

        logger.debug(f"task id: {str(self.task_id)} \n config: {config} \n result_dict: {result_dict}")
        train_loss = result_dict['info']["train_loss"]
        val_loss = result_dict['info']["test_loss"]
        eval_cost = result_dict['cost']
        loss_log = {"train_loss": train_loss,
                    "validation_loss": val_loss,
                    "cost": eval_cost}

        return loss_log
