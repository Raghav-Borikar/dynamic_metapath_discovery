# src/models/base_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicMetapathGNN(nn.Module):
    """
    Dynamic Metapath Graph Neural Network that discovers and utilizes metapaths
    through relation-level and metapath-level attention mechanisms.
    """
    
    def __init__(self, node_types, edge_types, feature_dims, hidden_dim=64, 
                 output_dim=64, max_path_length=3, attention_heads=8):
        """
        Initialize the Dynamic Metapath GNN model.
        
        Args:
            node_types (list): List of node types in the heterogeneous graph
            edge_types (list): List of edge types in the heterogeneous graph
            feature_dims (dict): Dictionary mapping node types to their feature dimensions
            hidden_dim (int): Dimension of hidden layers
            output_dim (int): Dimension of output embeddings
            max_path_length (int): Maximum length of metapaths to consider
            attention_heads (int): Number of attention heads
        """
        super(DynamicMetapathGNN, self).__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_path_length = max_path_length
        self.attention_heads = attention_heads
        
        # Node type specific embedding layers
        self.node_embeddings = nn.ModuleDict({
            ntype: nn.Linear(feature_dims[ntype], hidden_dim)
            for ntype in node_types
        })
        
        # Relation-level attention
        self.relation_attention = RelationAttention(
            hidden_dim, 
            attention_heads,
            edge_types
        )
        
        # Metapath-level attention
        self.metapath_attention = MetapathAttention(
            hidden_dim,
            attention_heads
        )
        
        # Final projection layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Expansion stopping criteria parameters
        self.path_importance_threshold = nn.Parameter(torch.tensor(0.1))
        self.max_expansion_paths = 5
        
    def forward(self, graph, node_features):
        """
        Forward pass of the model.
        
        Args:
            graph: Heterogeneous graph structure
            node_features: Dictionary of node features by type
            
        Returns:
            node_embeddings: Updated node embeddings
        """
        # Initial node embeddings
        initial_embeddings = {}
        for ntype in self.node_types:
            initial_embeddings[ntype] = self.node_embeddings[ntype](node_features[ntype])
        
        # Dynamic metapath expansion
        metapath_embeddings = []
        metapath_weights = []
        
        # Start with single-hop relations
        current_paths = [(src, edge, dst) for src in self.node_types 
                         for edge in self.edge_types 
                         for dst in self.node_types]
        
        # Iteratively expand metapaths up to max_path_length
        for path_length in range(1, self.max_path_length + 1):
            # Apply relation-level attention to current paths
            path_embeddings, path_importance = self.relation_attention(
                graph, initial_embeddings, current_paths
            )
            
            # Store embeddings and weights for metapath-level attention
            metapath_embeddings.extend(path_embeddings)
            metapath_weights.extend(path_importance)
            
            # Determine which paths to expand further based on importance
            if path_length < self.max_path_length:
                current_paths = self._expand_important_paths(
                    current_paths, path_importance
                )
        
        # Apply metapath-level attention to aggregate embeddings
        final_embeddings = self.metapath_attention(
            metapath_embeddings, metapath_weights
        )
        
        # Final projection
        output_embeddings = self.output_layer(final_embeddings)
        
        return output_embeddings
    
    def _expand_important_paths(self, current_paths, path_importance):
        """
        Determine which paths to expand further based on importance scores.
        
        Args:
            current_paths: List of current metapaths
            path_importance: Importance scores for each path
            
        Returns:
            expanded_paths: List of paths to expand in next iteration
        """
        # Sort paths by importance
        sorted_paths = [p for _, p in sorted(
            zip(path_importance, current_paths), 
            key=lambda x: x[0], 
            reverse=True
        )]
        
        # Select top paths that exceed the importance threshold
        important_paths = []
        for i, path in enumerate(sorted_paths):
            if i >= self.max_expansion_paths:
                break
            if path_importance[i] > self.path_importance_threshold:
                important_paths.append(path)
        
        # Expand selected paths by adding one more hop
        expanded_paths = []
        for path in important_paths:
            last_node_type = path[-1]
            for edge in self.edge_types:
                for dst in self.node_types:
                    # Check if edge is valid from last node type to dst
                    if self._is_valid_edge(last_node_type, edge, dst):
                        expanded_paths.append(path + (edge, dst))
        
        return expanded_paths
    
    def _is_valid_edge(self, src_type, edge_type, dst_type):
        """Check if edge type can connect source and destination node types."""
        # This would be implemented based on the graph schema
        # For now, assume all connections are possible
        return True
