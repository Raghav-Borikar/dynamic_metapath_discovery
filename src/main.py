# src/main.py
import os
import torch
import argparse
import logging
from datetime import datetime

# Import project modules
from models.base_model import DynamicMetapathGNN
from models.relation_attention import RelationAttention
from models.metapath_attention import MetapathAttention
from models.expansion_criteria import ExpansionCriteria
from utils.experiment_tracking import ExperimentTracker


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("project.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def setup_environment(args):
    """Set up the development environment."""
    logger = logging.getLogger(__name__)
    
    # Create necessary directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    logger.info(f"Environment setup complete. Using device: {args.device}")
    return args.device


def create_model_architecture(args):
    """Create the model architecture with attention mechanisms."""
    logger = logging.getLogger(__name__)
    
    # Define node and edge types (these would come from your dataset in practice)
    node_types = ["author", "paper", "conference", "term"]
    edge_types = ["writes", "written_by", "publishes", "published_in", "contains", "contained_in"]
    
    # Define feature dimensions for each node type
    feature_dims = {
        "author": 100,
        "paper": 200,
        "conference": 50,
        "term": 300
    }
    
    # Create model components
    relation_attention = RelationAttention(
        hidden_dim=args.hidden_dim,
        num_heads=args.attention_heads,
        edge_types=edge_types
    )
    
    metapath_attention = MetapathAttention(
        hidden_dim=args.hidden_dim,
        num_heads=args.attention_heads
    )
    
    expansion_criteria = ExpansionCriteria(
        max_path_length=args.max_path_length,
        importance_threshold=args.importance_threshold,
        max_paths=args.max_expansion_paths
    )
    
    # Create the full model
    model = DynamicMetapathGNN(
        node_types=node_types,
        edge_types=edge_types,
        feature_dims=feature_dims,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        max_path_length=args.max_path_length,
        attention_heads=args.attention_heads
    )
    
    logger.info(f"Model architecture created with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def main():
    """Main function to run Week 1 tasks."""
    parser = argparse.ArgumentParser(description="Dynamic Metapath Discovery - Week 1")
    
    # Environment settings
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use")
    
    # Model architecture settings
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--output_dim", type=int, default=64, help="Output dimension size")
    parser.add_argument("--attention_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--max_path_length", type=int, default=3, help="Maximum metapath length")
    parser.add_argument("--importance_threshold", type=float, default=0.1, 
                        help="Threshold for metapath importance")
    parser.add_argument("--max_expansion_paths", type=int, default=5, 
                        help="Maximum number of paths to expand")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Dynamic Metapath Discovery Project - Week 1")
    
    # Set up environment
    device = setup_environment(args)
    
    # Create experiment tracker
    experiment_tracker = ExperimentTracker(
        experiment_name="dynamic_metapath_week1",
        base_dir=args.output_dir
    )
    
    # Log hyperparameters
    experiment_tracker.log_hyperparameters(vars(args))
    
    # Create model architecture
    model = create_model_architecture(args)
    model.to(device)
    
    # Save initial model architecture
    experiment_tracker.save_checkpoint(model, None, epoch=0)
    
    logger.info("Week 1 tasks completed successfully")
    logger.info(f"Model architecture saved to {args.output_dir}")
    
    # Print summary of completed tasks
    print("\n" + "="*50)
    print("Week 1 Tasks Completed:")
    print("  - Project environment setup")
    print("  - Model architecture design with relation-level and metapath-level attention")
    print("  - Module interfaces created between components")
    print("  - Stopping criteria defined for dynamic metapath expansion")
    print("  - Experiment tracking infrastructure set up")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
