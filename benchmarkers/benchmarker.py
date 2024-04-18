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
            self.fidelity = {"n_estimators": 256, "subsample": 1}
        elif model_name == "svm":
            self.model_name = "SupportVectorMachine"
            self.benchmarker = SVMBenchmark(task_id=task_id)
            self.fidelity = {"subsample": 1}
        else:
            raise Exception("Need a model name: xgb or svm")

    def parse_config_space(self, config_space_str):
        cs_range = re.split(", Default", re.split('Range: ', config_space_str)[-1])[0]
        return cs_range

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

    def evaluate(self, config, rng=1):
        start_time = time.time()
        result_dict = self.benchmarker.objective_function(
            configuration=config, fidelity=self.fidelity, rng=rng)

        logger.debug(f"task id: {str(self.task_id)} \n config: {config} \n result_dict: {result_dict}")
        train_loss = result_dict['info']["train_loss"]
        valid_loss = result_dict['function_value']
        eval_time = time.time() - start_time
        loss_log = {"train_loss": train_loss,
                    "validation_loss": valid_loss}

        return loss_log

        #
# class LogisticRegressionBenchmarker:
#     def __init__(self):
#         # Simulate a small dataset for classification
#         X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
#         self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(X, y, test_size=0.33, random_state=42)
#
#     def get_search_space(self):
#         # Defines the hyperparameter search space for logistic regression
#         return {
#             'C': (0.01, 10),  # Regularization strength
#             'solver': ['liblinear', 'lbfgs']  # Algorithm to use in the optimization problem
#         }
#
#     def evaluate(self, config):
#         start_time = time.time()
#
#         # Create a logistic regression model with specified hyperparameters
#         model = LogisticRegression(
#             C=config.get('C', 1.0),
#             solver=config.get('solver', 'liblinear'),
#             max_iter=100,
#             random_state=42
#         )
#
#         # Fit the model
#         model_fit_start = time.time()
#         model.fit(self.train_X, self.train_y)
#         model_fit_time = time.time() - model_fit_start
#
#         # Evaluate model accuracy on training and validation sets
#         train_acc = accuracy_score(self.train_y, model.predict(self.train_X))
#         val_acc = accuracy_score(self.val_y, model.predict(self.val_X))
#
#         # Additional metrics for detailed performance evaluation
#         train_f1 = f1_score(self.train_y, model.predict(self.train_X), average='macro')
#         val_f1 = f1_score(self.val_y, model.predict(self.val_X), average='macro')
#         train_bal_acc = balanced_accuracy_score(self.train_y, model.predict(self.train_X))
#         val_bal_acc = balanced_accuracy_score(self.val_y, model.predict(self.val_X))
#
#         total_eval_time = time.time() - start_time
#
#         return {
#             'function_value': 1 - val_acc,  # Optimization target (minimizing the error)
#             'cost': total_eval_time,
#             'info': {
#                 'train_loss': 1 - train_acc,
#                 'val_loss': 1 - val_acc,
#                 'model_cost': model_fit_time,
#                 'train_scores': {
#                     'f1': train_f1,
#                     'acc': train_acc,
#                     'bal_acc': train_bal_acc
#                 },
#                 'valid_scores': {
#                     'f1': val_f1,
#                     'acc': val_acc,
#                     'bal_acc': val_bal_acc
#                 },
#                 'train_costs': model_fit_time,  # Time taken to compute performance metrics over the training set
#                 'valid_costs': total_eval_time - model_fit_time
#                 # Time taken to compute performance metrics over the validation set
#             }
#         }
#
#
# # Example instantiation and usage
# benchmarker = LogisticRegressionBenchmarker()
# config = {
#     'C': 0.05,
#     'solver': 'liblinear'
# }
# result = benchmarker.evaluate(config)
# print(result)




def runXGBoostHPO(task_id, config, benchmark):
    result_dict = benchmark.objective_function(configuration=config,
                                               fidelity={"n_estimators": 128, "dataset_fraction": 1}, rng=1)

    logger.debug(f"task id: {str(task_id)} \n result_dict: {result_dict}")
    train_loss = result_dict['info']["train_loss"]
    valid_loss = result_dict['function_value']
    return train_loss, valid_loss


def runSVMHPO(task_id, config, benchmark):
    result_dict = benchmark.objective_function(configuration=config,
                                               fidelity={"dataset_fraction": 1}, rng=1)

    logger.debug(f"task id: {str(task_id)} \n result_dict: {result_dict}")
    train_loss = result_dict['info']["train_loss"]
    valid_loss = result_dict['function_value']
    return train_loss, valid_loss

#
# xgb_training_log = {}
# svm_training_log = {}
# for task in [75227, 266, 261, 75099]:
#     xgbb = XGBoostBenchmark(task_id=task)
#     svmb = SupportVectorMachine(task_id=task)
#     xgbcs = xgbb.get_configuration_space()
#     svmcs = svmb.get_configuration_space()
#
#     config_xgb = xgbcs.sample_configuration()
#
#     logger.debug(f"task id: {str(task)} \n ML model: XGBoost \n configuration space: {xgbcs}")
#     logger.debug(f"task id: {str(task)} \n ML model: SVM \n configuration space: {svmcs}")
#     # define config_xgb and config_svm for the four tasks here
#
#     xgb_train_loss, xgb_valid_loss = runXGBoostHPO(task, config_xgb, xgbb)
#     svm_train_loss, svm_valid_loss = runSVMHPO(task, config_svm, svmb)
#
#     xgb_training_log[task] = {"configuration_space": xgbcs,
#                               "config": config_xgb,
#                               "train_loss": xgb_train_loss,
#                               "valid_loss": xgb_valid_loss}
#     svm_training_log[task] = {"configuration_space": svmcs,
#                               "config": config_svm,
#                               "train_loss": svm_train_loss,
#                               "valid_loss": svm_valid_loss}
#
# # we can do 20 or 33 hold out set