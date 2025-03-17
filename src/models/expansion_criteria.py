# src/models/expansion_criteria.py
import torch
import torch.nn as nn


class ExpansionCriteria(nn.Module):
    """
    Module defining criteria for stopping metapath expansion.
    """
    
    def __init__(self, max_path_length=3, importance_threshold=0.1, max_paths=5):
        """
        Initialize expansion criteria.
        
        Args:
            max_path_length (int): Maximum length of metapaths
            importance_threshold (float): Minimum importance score for expansion
            max_paths (int): Maximum number of paths to expand per iteration
        """
        super(ExpansionCriteria, self).__init__()
        
        self.max_path_length = max_path_length
        self.importance_threshold = nn.Parameter(torch.tensor(importance_threshold))
        self.max_paths = max_paths
        
    def should_expand(self, path, path_length, importance):
        """
        Determine if a path should be expanded further.
        
        Args:
            path: Current metapath
            path_length: Length of the current path
            importance: Importance score of the path
            
        Returns:
            bool: Whether to expand the path
        """
        # Check path length
        if path_length >= self.max_path_length:
            return False
        
        # Check importance threshold
        if importance < self.importance_threshold:
            return False
        
        return True
    
    def select_paths_to_expand(self, paths, importances):
        """
        Select which paths to expand in the next iteration.
        
        Args:
            paths: List of current metapaths
            importances: Importance scores for each path
            
        Returns:
            selected_paths: Paths selected for expansion
        """
        # Sort paths by importance
        sorted_indices = torch.argsort(importances, descending=True)
        
        # Select top paths that exceed the threshold
        selected_indices = []
        for i in sorted_indices:
            if len(selected_indices) >= self.max_paths:
                break
            if importances[i] >= self.importance_threshold:
                selected_indices.append(i)
        
        # Return selected paths
        return [paths[i] for i in selected_indices]
