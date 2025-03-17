# src/models/relation_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationAttention(nn.Module):
    """
    Relation-level attention mechanism for selecting important neighbors
    based on relation types.
    """
    
    def __init__(self, hidden_dim, num_heads, edge_types):
        """
        Initialize the relation attention module.
        
        Args:
            hidden_dim (int): Dimension of hidden representations
            num_heads (int): Number of attention heads
            edge_types (list): List of edge types in the graph
        """
        super(RelationAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.edge_types = edge_types
        
        # Per-relation projection matrices
        self.relation_projections = nn.ModuleDict({
            edge: nn.Linear(hidden_dim, hidden_dim)
            for edge in edge_types
        })
        
        # Multi-head attention parameters
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.attn_dropout = nn.Dropout(0.1)
        self.proj_dropout = nn.Dropout(0.1)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, graph, node_embeddings, metapaths):
        """
        Apply relation-level attention to node embeddings.
        
        Args:
            graph: Heterogeneous graph structure
            node_embeddings: Dictionary of node embeddings by type
            metapaths: List of metapaths to process
            
        Returns:
            path_embeddings: Embeddings for each metapath
            path_importance: Importance score for each metapath
        """
        path_embeddings = []
        path_importance = []
        
        for path in metapaths:
            # For simplicity, assume path is (src_type, edge_type, dst_type)
            src_type, edge_type, dst_type = path
            
            # Get source and destination nodes
            src_nodes = graph.get_nodes(src_type)
            dst_nodes = graph.get_neighbors(src_nodes, edge_type)
            
            # Get embeddings
            src_emb = node_embeddings[src_type]
            dst_emb = node_embeddings[dst_type]
            
            # Apply relation-specific projection
            rel_proj_dst = self.relation_projections[edge_type](dst_emb)
            
            # Multi-head attention
            q = self.query(src_emb).view(-1, self.num_heads, self.head_dim)
            k = self.key(rel_proj_dst).view(-1, self.num_heads, self.head_dim)
            v = self.value(rel_proj_dst).view(-1, self.num_heads, self.head_dim)
            
            # Scaled dot-product attention
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            # Apply attention weights to values
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.reshape(-1, self.hidden_dim)
            
            # Final projection
            path_emb = self.output_projection(attn_output)
            path_emb = self.proj_dropout(path_emb)
            
            # Calculate path importance as mean attention weight
            importance = attn_weights.mean().item()
            
            path_embeddings.append(path_emb)
            path_importance.append(importance)
        
        return path_embeddings, path_importance
