import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class HeterogeneousGraphDataset:
    """Dataset class for heterogeneous graphs with preprocessing capabilities"""
    
    def __init__(self, name, raw_dir='./data/raw', processed_dir='./data/processed'):
        self.name = name
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        
        # Create directories
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        
        # Graph data structures
        self.node_types = []
        self.edge_types = []
        self.node_features = {}
        self.edge_indices = {}
        self.node_labels = {}
        
    def download(self):
        """Download dataset from source"""
        print(f"Downloading {self.name} dataset...")
        # Implementation depends on specific dataset source
        # For academic datasets like DBLP, ACM, etc.
        
    def process(self):
        """Process raw data into heterogeneous graph format"""
        print(f"Processing {self.name} dataset...")
        
        # Load raw data files
        # Example implementation for academic network:
        # - Load papers, authors, venues from CSV/JSON files
        # - Extract node features from text/metadata
        # - Create edge indices for different relation types
        
        # Set up node types and features
        # self.node_types = ['paper', 'author', 'venue']
        # self.node_features['paper'] = paper_features
        
        # Set up edge types and connections
        # self.edge_types = [('author', 'writes', 'paper'), ('paper', 'published_in', 'venue')]
        # self.edge_indices[('author', 'writes', 'paper')] = author_paper_edges
        
        # Save processed data
        self.save()
        
    def save(self):
        """Save processed data to disk"""
        processed_file = os.path.join(self.processed_dir, f"{self.name}_processed.pt")
        data_dict = {
            'node_types': self.node_types,
            'edge_types': self.edge_types,
            'node_features': self.node_features,
            'edge_indices': self.edge_indices,
            'node_labels': self.node_labels
        }
        torch.save(data_dict, processed_file)
        print(f"Saved processed data to {processed_file}")
        
    def load(self):
        """Load processed data if exists, otherwise process raw data"""
        processed_file = os.path.join(self.processed_dir, f"{self.name}_processed.pt")
        
        if os.path.exists(processed_file):
            print(f"Loading processed data from {processed_file}")
            data_dict = torch.load(processed_file)
            
            self.node_types = data_dict['node_types']
            self.edge_types = data_dict['edge_types']
            self.node_features = data_dict['node_features']
            self.edge_indices = data_dict['edge_indices']
            self.node_labels = data_dict['node_labels']
        else:
            print(f"Processed data not found. Processing raw data...")
            self.download()
            self.process()
            
    def get_node_features(self, node_type):
        """Get features for specific node type"""
        return self.node_features.get(node_type, None)
    
    def get_edge_indices(self, edge_type):
        """Get edge indices for specific relation type"""
        return self.edge_indices.get(edge_type, None)
    
    def get_node_labels(self, node_type):
        """Get labels for specific node type"""
        return self.node_labels.get(node_type, None)
    
    def create_train_test_split(self, node_type, test_size=0.2, val_size=0.1, random_state=42):
        """Create train/val/test split for node classification"""
        if node_type not in self.node_labels:
            raise ValueError(f"No labels available for node type {node_type}")
            
        labels = self.node_labels[node_type]
        num_nodes = labels.shape[0]
        indices = np.arange(num_nodes)
        
        # First split into train+val and test
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Then split train+val into train and val
        val_size_adjusted = val_size / (1 - test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_size_adjusted, random_state=random_state, 
            stratify=labels[train_val_idx]
        )
        
        return {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }
    
    def create_link_prediction_split(self, edge_type, test_size=0.2, val_size=0.1, random_state=42):
        """Create train/val/test split for link prediction"""
        if edge_type not in self.edge_indices:
            raise ValueError(f"No edges available for edge type {edge_type}")
            
        edges = self.edge_indices[edge_type]
        num_edges = edges.shape[1]
        indices = np.arange(num_edges)
        
        # Split edges into train+val and test
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        
        # Split train+val into train and val
        val_size_adjusted = val_size / (1 - test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_size_adjusted, random_state=random_state
        )
        
        # Create edge splits
        train_edges = edges[:, train_idx]
        val_edges = edges[:, val_idx]
        test_edges = edges[:, test_idx]
        
        return {
            'train': train_edges,
            'val': val_edges,
            'test': test_edges
        }


class GraphVisualizer:
    """Tools for visualizing heterogeneous graphs"""
    
    def __init__(self, dataset):
        self.dataset = dataset
        
    def visualize_graph_statistics(self):
        """Visualize basic statistics of the heterogeneous graph"""
        # Node type distribution
        node_counts = {ntype: len(self.dataset.get_node_features(ntype)) 
                      for ntype in self.dataset.node_types 
                      if self.dataset.get_node_features(ntype) is not None}
        
        # Edge type distribution
        edge_counts = {etype: self.dataset.get_edge_indices(etype).shape[1] 
                      for etype in self.dataset.edge_types
                      if self.dataset.get_edge_indices(etype) is not None}
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Node type distribution
        ax1.bar(node_counts.keys(), node_counts.values())
        ax1.set_title('Node Type Distribution')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Edge type distribution
        ax2.bar(edge_counts.keys(), edge_counts.values())
        ax2.set_title('Edge Type Distribution')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"./data/visualizations/{self.dataset.name}_statistics.png")
        plt.close()
        
    def visualize_subgraph(self, node_types=None, edge_types=None, max_nodes=100):
        """Visualize a subgraph with specified node and edge types"""
        if node_types is None:
            node_types = self.dataset.node_types
        if edge_types is None:
            edge_types = self.dataset.edge_types
            
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        node_offset = 0
        node_id_map = {}
        
        for ntype in node_types:
            features = self.dataset.get_node_features(ntype)
            if features is None:
                continue
                
            num_nodes = min(len(features), max_nodes)
            for i in range(num_nodes):
                node_id = f"{ntype}_{i}"
                G.add_node(node_id, type=ntype)
                node_id_map[(ntype, i)] = node_id
                
        # Add edges
        for etype in edge_types:
            src_type, rel, dst_type = etype
            if src_type not in node_types or dst_type not in node_types:
                continue
                
            edge_indices = self.dataset.get_edge_indices(etype)
            if edge_indices is None:
                continue
                
            for i in range(min(edge_indices.shape[1], 1000)):  # Limit number of edges
                src_idx, dst_idx = edge_indices[0, i], edge_indices[1, i]
                
                if (src_type, src_idx) in node_id_map and (dst_type, dst_idx) in node_id_map:
                    src_id = node_id_map[(src_type, src_idx)]
                    dst_id = node_id_map[(dst_type, dst_idx)]
                    G.add_edge(src_id, dst_id, type=rel)
        
        # Visualize
        plt.figure(figsize=(12, 10))
        
        # Node colors based on type
        node_types_unique = list(set(nx.get_node_attributes(G, 'type').values()))
        color_map = plt.cm.get_cmap('tab10', len(node_types_unique))
        node_colors = [node_types_unique.index(G.nodes[node]['type']) for node in G.nodes]
        
        # Draw the graph
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(
            G, pos, 
            node_color=node_colors, 
            cmap=color_map,
            node_size=100,
            with_labels=False,
            width=0.5,
            alpha=0.8
        )
        
        # Add legend
        plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', 
                                       markerfacecolor=color_map(i), markersize=10, label=ntype) 
                           for i, ntype in enumerate(node_types_unique)],
                  title="Node Types")
        
        plt.title(f"Subgraph Visualization of {self.dataset.name}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"./data/visualizations/{self.dataset.name}_subgraph.png")
        plt.close()
        
    def visualize_feature_distributions(self, node_type):
        """Visualize feature distributions for a specific node type"""
        features = self.dataset.get_node_features(node_type)
        if features is None:
            print(f"No features available for node type {node_type}")
            return
            
        # Convert to numpy for easier manipulation
        if isinstance(features, torch.Tensor):
            features = features.numpy()
            
        # Plot feature distributions
        n_features = min(features.shape[1], 10)  # Limit to 10 features
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 2*n_features))
        
        for i in range(n_features):
            if n_features == 1:
                ax = axes
            else:
                ax = axes[i]
                
            ax.hist(features[:, i], bins=30)
            ax.set_title(f'Feature {i} Distribution')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            
        plt.tight_layout()
        plt.savefig(f"./data/visualizations/{self.dataset.name}_{node_type}_features.png")
        plt.close()


# Example implementation for a specific dataset (DBLP)
class DBLPDataset(HeterogeneousGraphDataset):
    """DBLP dataset implementation"""
    
    def __init__(self, raw_dir='./data/raw', processed_dir='./data/processed'):
        super().__init__('DBLP', raw_dir, processed_dir)
        
    def download(self):
        """Download DBLP dataset"""
        super().download()
        # Specific implementation for DBLP
        # Could use requests to download from source URL
        
    def process(self):
        """Process DBLP dataset"""
        print("Processing DBLP dataset...")
        
        # Define node types
        self.node_types = ['author', 'paper', 'conference', 'term']
        
        # Define edge types (relations)
        self.edge_types = [
            ('author', 'writes', 'paper'),
            ('paper', 'published_in', 'conference'),
            ('paper', 'contains', 'term')
        ]
        
        # Process node features (simplified example)
        # In a real implementation, these would be loaded from files
        num_authors = 4057
        num_papers = 14328
        num_conferences = 20
        num_terms = 8789
        
        # Create dummy features for demonstration
        self.node_features['author'] = torch.randn(num_authors, 334)
        self.node_features['paper'] = torch.randn(num_papers, 4231)
        self.node_features['conference'] = torch.randn(num_conferences, 50)
        self.node_features['term'] = torch.randn(num_terms, 50)
        
        # Create dummy edge indices for demonstration
        self.edge_indices[('author', 'writes', 'paper')] = torch.randint(
            low=0, high=min(num_authors, num_papers), size=(2, 19645)
        )
        self.edge_indices[('paper', 'published_in', 'conference')] = torch.randint(
            low=0, high=min(num_papers, num_conferences), size=(2, 14328)
        )
        self.edge_indices[('paper', 'contains', 'term')] = torch.randint(
            low=0, high=min(num_papers, num_terms), size=(2, 88420)
        )
        
        # Create dummy labels for author classification
        self.node_labels['author'] = torch.randint(0, 4, (num_authors,))
        
        # Save processed data
        self.save()


# Usage example
if __name__ == "__main__":
    # Create visualization directory
    os.makedirs("./data/visualizations", exist_ok=True)
    
    # Initialize and load DBLP dataset
    dblp = DBLPDataset()
    dblp.load()
    
    # Create train/test splits for node classification
    author_splits = dblp.create_train_test_split('author')
    print(f"Author classification splits: {len(author_splits['train'])} train, "
          f"{len(author_splits['val'])} validation, {len(author_splits['test'])} test")
    
    # Create train/test splits for link prediction
    link_splits = dblp.create_link_prediction_split(('author', 'writes', 'paper'))
    print(f"Link prediction splits: {link_splits['train'].shape[1]} train, "
          f"{link_splits['val'].shape[1]} validation, {link_splits['test'].shape[1]} test")
    
    # Visualize graph statistics and structure
    visualizer = GraphVisualizer(dblp)
    visualizer.visualize_graph_statistics()
    visualizer.visualize_subgraph()
    visualizer.visualize_feature_distributions('author')


