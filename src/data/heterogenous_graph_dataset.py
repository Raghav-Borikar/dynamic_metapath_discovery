import os
import torch
import numpy as np
import pandas as pd
import urllib.request
import gzip
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class DBLPDataset:
    """
    DBLP heterogeneous graph dataset implementation
    """
    
    def __init__(self, raw_dir='./data/raw', processed_dir='./data/processed'):
        self.name = 'DBLP'
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        
        # DBLP XML file URLs
        self.dblp_xml_url = "https://dblp.org/xml/dblp.xml.gz"
        self.dblp_dtd_url = "https://dblp.org/xml/dblp.dtd"
        
        # Create directories
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs('./data/visualizations', exist_ok=True)
        
        # Graph data structures
        self.node_types = []
        self.edge_types = []
        self.node_features = {}
        self.edge_indices = {}
        self.node_labels = {}
        
    def download(self):
        """Download DBLP dataset if not exists"""
        print("Downloading DBLP dataset...")
        
        # Download XML file
        xml_gz_path = os.path.join(self.raw_dir, "dblp.xml.gz")
        xml_path = os.path.join(self.raw_dir, "dblp.xml")
        dtd_path = os.path.join(self.raw_dir, "dblp.dtd")
        
        if not os.path.exists(xml_path):
            if not os.path.exists(xml_gz_path):
                print(f"Downloading DBLP XML file from {self.dblp_xml_url}")
                urllib.request.urlretrieve(self.dblp_xml_url, xml_gz_path)
            
            # Extract the gzipped file
            print("Extracting XML file...")
            with gzip.open(xml_gz_path, 'rb') as f_in:
                with open(xml_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        # Download DTD file
        if not os.path.exists(dtd_path):
            print(f"Downloading DBLP DTD file from {self.dblp_dtd_url}")
            urllib.request.urlretrieve(self.dblp_dtd_url, dtd_path)
        
        print("Download complete.")
        
    def process(self, max_papers=10000):
        """Process DBLP dataset"""
        print("Processing DBLP dataset...")
        
        # Define node types
        self.node_types = ['author', 'paper', 'conference', 'term']
        
        # Define edge types (relations)
        self.edge_types = [
            ('author', 'writes', 'paper'),
            ('paper', 'written_by', 'author'),
            ('conference', 'publishes', 'paper'),
            ('paper', 'published_in', 'conference'),
            ('paper', 'contains', 'term'),
            ('term', 'contained_in', 'paper')
        ]
        
        # Check if XML file exists
        xml_path = os.path.join(self.raw_dir, "dblp.xml")
        if not os.path.exists(xml_path):
            self.download()
        
        # Initialize data structures
        authors = {}
        papers = {}
        conferences = {}
        terms = set()
        paper_author_edges = []
        paper_conf_edges = []
        paper_term_edges = []
        
        # Parse XML file
        print("Parsing XML file...")
        context = ET.iterparse(xml_path, events=('start', 'end'))
        
        paper_count = 0
        current_element = None
        
        for event, elem in tqdm(context, desc="Parsing XML"):
            if event == 'end' and elem.tag in ['article', 'inproceedings', 'proceedings']:
                paper_id = elem.get('key')
                title_elem = elem.find('title')
                
                if title_elem is not None and title_elem.text:
                    title = title_elem.text.lower()
                    papers[paper_id] = {'title': title, 'year': 0}
                    
                    # Extract year
                    year_elem = elem.find('year')
                    if year_elem is not None and year_elem.text:
                        papers[paper_id]['year'] = int(year_elem.text)
                    
                    # Process authors
                    for author_elem in elem.findall('author'):
                        if author_elem.text:
                            author_name = author_elem.text
                            author_id = author_name.replace(" ", "_").lower()
                            
                            # Store author
                            if author_id not in authors:
                                authors[author_id] = author_name
                            
                            # Create author-paper edge
                            paper_author_edges.append((author_id, paper_id))
                    
                    # Process conference/journal
                    conf_elem = elem.find('booktitle') or elem.find('journal')
                    if conf_elem is not None and conf_elem.text:
                        conf_name = conf_elem.text
                        conf_id = conf_name.replace(" ", "_").lower()
                        
                        # Store conference
                        if conf_id not in conferences:
                            conferences[conf_id] = conf_name
                        
                        # Create paper-conference edge
                        paper_conf_edges.append((paper_id, conf_id))
                    
                    # Process terms (simple tokenization)
                    if title:
                        # Simple preprocessing: lowercase, remove punctuation
                        title_processed = ''.join(c.lower() if c.isalnum() else ' ' for c in title)
                        title_terms = [t for t in title_processed.split() if len(t) > 3]  # Filter short terms
                        
                        for term in title_terms:
                            # Store term
                            if term not in terms:
                                terms.add(term)
                            
                            # Create paper-term edge
                            paper_term_edges.append((paper_id, term))
                    
                    paper_count += 1
                    if paper_count >= max_papers:
                        break
                
                # Clear element to save memory
                elem.clear()
            
            # Break if we've processed enough papers
            if paper_count >= max_papers:
                break
        
        print(f"Processed {paper_count} papers, {len(authors)} authors, {len(conferences)} conferences, {len(terms)} terms")
        
        # Create node ID mappings
        author_ids = {author_id: i for i, author_id in enumerate(authors.keys())}
        paper_ids = {paper_id: i for i, paper_id in enumerate(papers.keys())}
        conf_ids = {conf_id: i for i, conf_id in enumerate(conferences.keys())}
        term_ids = {term: i for i, term in enumerate(terms)}
        
        # Create node features
        print("Creating node features...")
        # Author features: one-hot encoding
        self.node_features['author'] = torch.eye(len(authors))
        
        # Paper features: year and TF-IDF of titles
        paper_titles = [papers[pid]['title'] for pid in papers.keys()]
        vectorizer = TfidfVectorizer(max_features=100)  # Limit features for memory
        tfidf_features = vectorizer.fit_transform(paper_titles).toarray()
        
        # Combine year and TF-IDF features
        paper_years = np.array([[papers[pid]['year']] for pid in papers.keys()]) / 2025.0  # Normalize
        paper_features = np.hstack([paper_years, tfidf_features])
        self.node_features['paper'] = torch.FloatTensor(paper_features)
        
        # Conference features: one-hot encoding
        self.node_features['conference'] = torch.eye(len(conferences))
        
        # Term features: one-hot encoding
        self.node_features['term'] = torch.eye(len(terms))
        
        # Create edge indices
        print("Creating edge indices...")
        
        # Author-Paper edges
        author_to_paper = []
        paper_to_author = []
        for author_id, paper_id in paper_author_edges:
            if author_id in author_ids and paper_id in paper_ids:
                author_to_paper.append([author_ids[author_id], paper_ids[paper_id]])
                paper_to_author.append([paper_ids[paper_id], author_ids[author_id]])
        
        self.edge_indices[('author', 'writes', 'paper')] = torch.LongTensor(author_to_paper).t() if author_to_paper else torch.zeros((2, 0), dtype=torch.long)
        self.edge_indices[('paper', 'written_by', 'author')] = torch.LongTensor(paper_to_author).t() if paper_to_author else torch.zeros((2, 0), dtype=torch.long)
        
        # Conference-Paper edges
        conf_to_paper = []
        paper_to_conf = []
        for paper_id, conf_id in paper_conf_edges:
            if paper_id in paper_ids and conf_id in conf_ids:
                conf_to_paper.append([conf_ids[conf_id], paper_ids[paper_id]])
                paper_to_conf.append([paper_ids[paper_id], conf_ids[conf_id]])
        
        self.edge_indices[('conference', 'publishes', 'paper')] = torch.LongTensor(conf_to_paper).t() if conf_to_paper else torch.zeros((2, 0), dtype=torch.long)
        self.edge_indices[('paper', 'published_in', 'conference')] = torch.LongTensor(paper_to_conf).t() if paper_to_conf else torch.zeros((2, 0), dtype=torch.long)
        
        # Paper-Term edges
        paper_to_term = []
        term_to_paper = []
        for paper_id, term in paper_term_edges:
            if paper_id in paper_ids and term in term_ids:
                paper_to_term.append([paper_ids[paper_id], term_ids[term]])
                term_to_paper.append([term_ids[term], paper_ids[paper_id]])
        
        self.edge_indices[('paper', 'contains', 'term')] = torch.LongTensor(paper_to_term).t() if paper_to_term else torch.zeros((2, 0), dtype=torch.long)
        self.edge_indices[('term', 'contained_in', 'paper')] = torch.LongTensor(term_to_paper).t() if term_to_paper else torch.zeros((2, 0), dtype=torch.long)
        
        # Create author labels (4 research areas)
        # For demonstration, assign random labels
        self.node_labels['author'] = torch.randint(0, 4, (len(authors),))
        
        # Save processed data
        self.save()
        
        print(f"Processing complete with {sum(len(edges) for edges in self.edge_indices.values())} total edges")
        
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


# Usage example
if __name__ == "__main__":
    # Initialize and load DBLP dataset
    dblp = DBLPDataset()
    dblp.load()
    
    # Print dataset statistics
    print("\nDBLP Dataset Statistics:")
    print(f"Node types: {dblp.node_types}")
    for ntype in dblp.node_types:
        if ntype in dblp.node_features:
            print(f"  {ntype}: {dblp.node_features[ntype].shape[0]} nodes, {dblp.node_features[ntype].shape[1]} features")
    
    print("\nEdge types:")
    for etype in dblp.edge_types:
        if etype in dblp.edge_indices:
            print(f"  {etype}: {dblp.edge_indices[etype].shape[1]} edges")
    
    # Create train/test splits for node classification
    author_splits = dblp.create_train_test_split('author')
    print(f"\nAuthor classification splits: {len(author_splits['train'])} train, "
          f"{len(author_splits['val'])} validation, {len(author_splits['test'])} test")
    
    # Create train/test splits for link prediction
    link_splits = dblp.create_link_prediction_split(('author', 'writes', 'paper'))
    print(f"\nLink prediction splits: {link_splits['train'].shape[1]} train, "
          f"{link_splits['val'].shape[1]} validation, {link_splits['test'].shape[1]} test")
