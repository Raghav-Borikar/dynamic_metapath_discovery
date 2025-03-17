# src/models/metapath_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetapathAttention(nn.Module):
    """
    Metapath-level attention mechanism for weighting different metapaths.
    """
    
    def __init__(self, hidden_dim, num_heads):
        """
        Initialize the metapath attention module.
        
        Args:
            hidden_dim (int): Dimension of hidden representations
            num_heads (int): Number of attention heads
        """
        super(MetapathAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Semantic attention network
        self.semantic_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        # Multi-head attention parameters
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.attn_dropout = nn.Dropout(0.1)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, metapath_embeddings, path_importance=None):
        """
        Apply metapath-level attention to aggregate embeddings from different metapaths.
        
        Args:
            metapath_embeddings: List of embeddings for each metapath
            path_importance: Optional pre-computed importance scores
            
        Returns:
            final_embeddings: Aggregated node embeddings
        """
        if not metapath_embeddings:
            raise ValueError("No metapath embeddings provided")
        
        # Stack embeddings
        stacked_embeddings = torch.stack(metapath_embeddings, dim=1)  # [num_nodes, num_metapaths, hidden_dim]
        
        # Calculate attention weights
        if path_importance is not None:
            # Use pre-computed importance scores
            weights = torch.tensor(path_importance).to(stacked_embeddings.device)
            weights = F.softmax(weights, dim=0)
            weights = weights.unsqueeze(0).expand(stacked_embeddings.shape[0], -1)
        else:
            # Calculate semantic attention weights
            semantic_scores = self.semantic_attention(stacked_embeddings)  # [num_nodes, num_metapaths, 1]
            weights = F.softmax(semantic_scores, dim=1)
        
        # Apply attention weights
        weighted_embeddings = stacked_embeddings * weights
        
        # Aggregate embeddings
        final_embeddings = weighted_embeddings.sum(dim=1)  # [num_nodes, hidden_dim]
        
        return final_embeddings
