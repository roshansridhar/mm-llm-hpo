import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score


class LogisticRegressionBenchmarker:
    def __init__(self):
        # Simulate a small dataset for classification
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(X, y, test_size=0.33, random_state=42)

    def get_search_space(self):
        # Defines the hyperparameter search space for logistic regression
        return {
            'C': (0.01, 10),  # Regularization strength
            'solver': ['liblinear', 'lbfgs']  # Algorithm to use in the optimization problem
        }

    def evaluate(self, config):
        start_time = time.time()

        # Create a logistic regression model with specified hyperparameters
        model = LogisticRegression(
            C=config.get('C', 1.0),
            solver=config.get('solver', 'liblinear'),
            max_iter=100,
            random_state=42
        )

        # Fit the model
        model_fit_start = time.time()
        model.fit(self.train_X, self.train_y)
        model_fit_time = time.time() - model_fit_start

        # Evaluate model accuracy on training and validation sets
        train_acc = accuracy_score(self.train_y, model.predict(self.train_X))
        val_acc = accuracy_score(self.val_y, model.predict(self.val_X))

        # Additional metrics for detailed performance evaluation
        train_f1 = f1_score(self.train_y, model.predict(self.train_X), average='macro')
        val_f1 = f1_score(self.val_y, model.predict(self.val_X), average='macro')
        train_bal_acc = balanced_accuracy_score(self.train_y, model.predict(self.train_X))
        val_bal_acc = balanced_accuracy_score(self.val_y, model.predict(self.val_X))

        total_eval_time = time.time() - start_time

        return {
            'function_value': 1 - val_acc,  # Optimization target (minimizing the error)
            'cost': total_eval_time,
            'info': {
                'train_loss': 1 - train_acc,
                'val_loss': 1 - val_acc,
                'model_cost': model_fit_time,
                'train_scores': {
                    'f1': train_f1,
                    'acc': train_acc,
                    'bal_acc': train_bal_acc
                },
                'valid_scores': {
                    'f1': val_f1,
                    'acc': val_acc,
                    'bal_acc': val_bal_acc
                },
                'train_costs': model_fit_time,  # Time taken to compute performance metrics over the training set
                'valid_costs': total_eval_time - model_fit_time
                # Time taken to compute performance metrics over the validation set
            }
        }


# Example instantiation and usage
benchmarker = LogisticRegressionBenchmarker()
config = {
    'C': 0.05,
    'solver': 'liblinear'
}
result = benchmarker.evaluate(config)
print(result)
