import itertools
from typing import Dict, List, Callable
import torch.nn as nn
from torch.optim import AdamW, SGD, RMSprop
import json
from datetime import datetime
import torch
from copy import deepcopy
import os
from trainer import StarModel


class StarModelTuner:
    def __init__(self, csv_file: str, base_output_dir: str = "./tuning_results"):
        self.csv_file = csv_file
        self.base_output_dir = base_output_dir
        self.results = []
        os.makedirs(base_output_dir, exist_ok=True)

    def define_parameter_grid(self, param_grid: Dict[str, List]) -> List[Dict]:
        """
        Creates combinations of hyperparameters to test

        param_grid = {
            'learning_rate': [1e-5, 3e-5, 5e-5],
            'batch_size': [16, 32],
            'num_train_epochs': [3, 5],
            'max_length': [128, 256]
        }
        """
        keys = param_grid.keys()
        combinations = itertools.product(*param_grid.values())
        return [dict(zip(keys, combo)) for combo in combinations]

    def custom_loss_functions(self) -> Dict[str, Callable]:
        """Define different loss functions to test"""
        return {
            "cross_entropy": nn.CrossEntropyLoss(),
            "weighted_cross_entropy": nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, 2.0, 1.0])
            ),  # Adjust weights as needed
            "focal_loss": FocalLoss(alpha=0.25, gamma=2),
            "label_smoothing": nn.CrossEntropyLoss(label_smoothing=0.1),
        }

    def train_and_evaluate(
        self, params: Dict, loss_fn_name: str, loss_fn: Callable
    ) -> Dict:
        """Train model with given parameters and loss function"""
        # Create a unique output directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(self.base_output_dir, f"run_{timestamp}")
        # Initialize model with current parameters
        model = StarModel(
            output_dir=run_output_dir,
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            num_train_epochs=params["num_train_epochs"],
            max_length=params["max_length"],
        )

        # Override the compute_loss method with current loss function
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("label")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

        model.compute_loss = compute_loss.__get__(model)
        # Train and evaluate
        model.preprocess_data(self.csv_file)
        model.initialize_model()
        model.setup_trainer()
        training_output = model.trainer.train()
        # Get evaluation metrics
        eval_results = model.trainer.evaluate()
        # Combine parameters, loss function, and results
        run_results = {
            "parameters": params,
            "loss_function": loss_fn_name,
            "eval_results": eval_results,
            "training_loss": training_output.training_loss,
            "output_dir": run_output_dir,
        }
        return run_results

    def run_tuning(self, param_grid: Dict[str, List]):
        """Run full hyperparameter tuning experiment"""
        parameter_combinations = self.define_parameter_grid(param_grid)
        loss_functions = self.custom_loss_functions()
        for params in parameter_combinations:
            for loss_name, loss_fn in loss_functions.items():
                try:
                    results = self.train_and_evaluate(params, loss_name, loss_fn)
                    self.results.append(results)
                    self.save_results()
                except Exception as e:
                    print(
                        f"Error with parameters {params} and loss {loss_name}: {str(e)}"
                    )
                    continue

    def save_results(self):
        """Save results to JSON file"""
        output_file = os.path.join(self.base_output_dir, "tuning_results.json")
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

    def get_best_model(self, metric="eval_accuracy"):
        """Get the best performing model based on specified metric"""
        best_result = max(self.results, key=lambda x: x["eval_results"][metric])
        return best_result


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# Example usage
if __name__ == "__main__":
    param_grid = {
        "learning_rate": [3e-5, 5e-5],
        "batch_size": [16, 32],
        "num_train_epochs": [3, 5],
        "max_length": [128, 256],
    }
    tuner = StarModelTuner(csv_file="data/exported_df_no_outliers.csv")
    tuner.run_tuning(param_grid)
    # Get best model
    best_model = tuner.get_best_model()
    print("Best performing model configuration:")
    print(json.dumps(best_model, indent=2))
