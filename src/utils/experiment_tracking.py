# src/utils/experiment_tracking.py
import os
import json
import time
import torch
from datetime import datetime


class ExperimentTracker:
    """
    Utility for tracking experiments, logging metrics, and saving model checkpoints.
    """
    
    def __init__(self, experiment_name, base_dir="experiments"):
        """
        Initialize the experiment tracker.
        
        Args:
            experiment_name (str): Name of the experiment
            base_dir (str): Base directory for saving experiment data
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(base_dir, f"{experiment_name}_{self.timestamp}")
        
        # Create experiment directory
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "metrics"), exist_ok=True)
        
        # Initialize metrics
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "test_metrics": {},
            "metapath_importance": {},
            "hyperparameters": {},
            "training_time": 0
        }
        
        self.start_time = time.time()
        
    def log_hyperparameters(self, hyperparams):
        """Log model hyperparameters."""
        self.metrics["hyperparameters"] = hyperparams
        self._save_metrics()
    
    def log_train_loss(self, epoch, loss):
        """Log training loss for an epoch."""
        self.metrics["train_loss"].append({"epoch": epoch, "loss": loss})
        self._save_metrics()
    
    def log_val_loss(self, epoch, loss):
        """Log validation loss for an epoch."""
        self.metrics["val_loss"].append({"epoch": epoch, "loss": loss})
        self._save_metrics()
    
    def log_test_metrics(self, metrics):
        """Log test set evaluation metrics."""
        self.metrics["test_metrics"] = metrics
        self._save_metrics()
    
    def log_metapath_importance(self, metapath, importance):
        """Log importance scores for discovered metapaths."""
        if metapath not in self.metrics["metapath_importance"]:
            self.metrics["metapath_importance"][metapath] = []
        
        self.metrics["metapath_importance"][metapath].append(importance)
        self._save_metrics()
    
    def save_checkpoint(self, model, optimizer, epoch):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.experiment_dir, "checkpoints", f"checkpoint_epoch_{epoch}.pt"
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
    
    def finish(self):
        """Finalize experiment tracking."""
        end_time = time.time()
        self.metrics["training_time"] = end_time - self.start_time
        self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics to JSON file."""
        metrics_path = os.path.join(self.experiment_dir, "metrics", "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
