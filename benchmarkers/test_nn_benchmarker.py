import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score


class TestNNBenchmarker:
    def __init__(self):
        # Simulate a small dataset
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(X, y, test_size=0.33, random_state=42)
    
    def get_search_space(self):
        return {
            'alpha': (1e-5, 1),
            'learning_rate_init': (1e-5, 1),
            'batch_size': (10, 100),
            'hidden_layer_sizes': (50, 100)
        }
    
    def evaluate(self, config):
        start_time = time.time()
        
        # Create a simple MLP model
        print(config)
        model = MLPClassifier(
            alpha=config.get('alpha', 0.0001),
            learning_rate_init=config.get('learning_rate_init', 0.001),
            batch_size=int(config.get('batch_size', 50)),
            hidden_layer_sizes=(int(config.get('hidden_layer_sizes', 50)),),
            max_iter=100,
            random_state=42
        )
        
        # Fit model
        model_fit_start = time.time()
        model.fit(self.train_X, self.train_y)
        model_fit_time = time.time() - model_fit_start
        
        # Evaluate accuracy
        train_acc = accuracy_score(self.train_y, model.predict(self.train_X))
        val_acc = accuracy_score(self.val_y, model.predict(self.val_X))
        
        # Evaluate additional metrics
        train_f1 = f1_score(self.train_y, model.predict(self.train_X), average='macro')
        val_f1 = f1_score(self.val_y, model.predict(self.val_X), average='macro')
        train_bal_acc = balanced_accuracy_score(self.train_y, model.predict(self.train_X))
        val_bal_acc = balanced_accuracy_score(self.val_y, model.predict(self.val_X))
        
        total_eval_time = time.time() - start_time
        
        return {
            'function_value': 1 - val_acc,
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
                'train_costs': model_fit_time,
                'valid_costs': total_eval_time - model_fit_time
            }
        }


# Example usage
# benchmarker = TestNNBenchmarker()
# config = {
#     'alpha': 0.0001,
#     'learning_rate_init': 0.001,
#     'batch_size': 50,
#     'hidden_layer_sizes': 50
# }
# result = benchmarker.evaluate(config)
# print(result)
