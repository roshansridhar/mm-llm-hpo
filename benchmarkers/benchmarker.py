import time
import re
from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
from hpobench.benchmarks.ml.svm_benchmark import SVMBenchmark
from loguru import logger



class Benchmarker:
    def __init__(self, task_id, model_name="xgb"):
        self.task_id = task_id
        self.config_space = None
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

    def parse_config_space(self, config_space_str):
        cs_range = re.split(", Default", re.split('Range: ', config_space_str)[-1])[0]
        low_and_high = re.findall(r"[-+]?(?:\d*\.*\d+)", cs_range)
        cs_list = []
        for number in low_and_high:
            try:
                cs_list.append(int(number))
            except ValueError:
                cs_list.append(float(number))
        cs_tuple = tuple(cs_list)
        return cs_tuple

    def get_search_space(self, seed=1):
        self.config_space = self.benchmarker.get_configuration_space(seed)
        config_space_dict = dict(self.config_space)
        search_space_out = {}
        for hp_name, space in config_space_dict.items():
            # hp_dict = {}
            # hp_dict["hyperparameter_name"] = hp_name
            # hp_dict["range"] = self.parse_config_space(str(space))
            # search_space_out.append(hp_dict)
            search_space_out[hp_name] = self.parse_config_space(str(space))

        return search_space_out

    def evaluate(self, config, if_test=False, rng=1):
        start_time = time.time()
        result_dict = self.benchmarker.objective_function(
            configuration=config, fidelity=self.fidelity, rng=rng)

        logger.debug(f"task id: {str(self.task_id)} \n config: {config} \n result_dict: {result_dict}")
        train_loss = result_dict['info']["train_loss"]
        val_loss = result_dict['info']["val_loss"]
        eval_time = time.time() - start_time
        loss_log = {"train_loss": train_loss,
                    "validation_loss": val_loss}

        return loss_log
