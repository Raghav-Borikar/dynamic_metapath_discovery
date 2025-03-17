# -*- coding: utf-8 -*-
"""data_loader.py

Data loader for DBLP heterogeneous graph dataset
"""

import dgl
import torch
import numpy as np
import pandas as pd
import os
import urllib.request
import gzip
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pickle


class DBLPGraphDataLoader:
    """
    Data loader for DBLP heterogeneous graph
    """
    def __init__(self, raw_dir='./data/raw', processed_dir='./data/processed'):
        self.dataset_name = "dblp"
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        
        # DBLP XML file URLs
        self.dblp_xml_url = "https://dblp.org/xml/dblp.xml.gz"
        self.dblp_dtd_url = "https://dblp.org/xml/dblp.dtd"
        
        # Node types and edge types
        self.node_types = ["author", "paper", "conference", "term"]
        self.edge_types = [
            ("author", "writes", "paper"),
            ("paper", "written_by", "author"),
            ("conference", "publishes", "paper"),
            ("paper", "published_in", "conference"),
            ("paper", "contains", "term"),
            ("term", "contained_in", "paper")
        ]
        
        # Create directories if they don't exist
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

    def download(self):
        """Download the DBLP dataset if not exists"""
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
        """
        Process the raw DBLP XML data into a heterogeneous graph
        
        Args:
            max_papers: Maximum number of papers to process (for memory constraints)
        """
        print("Processing DBLP dataset...")
        
        xml_path = os.path.join(self.raw_dir, "dblp.xml")
        if not os.path.exists(xml_path):
            self.download()
        
        # Dictionaries to store nodes and their features
        authors = {}  # author_id -> author_name
        papers = {}   # paper_id -> {title, year}
        conferences = {}  # conf_id -> conf_name
        terms = {}    # term_id -> term
        
        # Dictionaries to store edges
        author_paper_edges = []  # (author_id, paper_id)
        paper_conf_edges = []    # (paper_id, conf_id)
        paper_term_edges = []    # (paper_id, term_id)
        
        # Parse XML file
        print("Parsing XML file...")
        context = ET.iterparse(xml_path, events=('start', 'end'))
        
        paper_count = 0
        current_element = None
        
        for event, elem in tqdm(context, desc="Parsing XML"):
            if event == 'start':
                if elem.tag in ['article', 'inproceedings', 'proceedings', 'book', 'incollection']:
                    current_element = elem
            
            elif event == 'end':
                if elem.tag in ['article', 'inproceedings', 'proceedings', 'book', 'incollection']:
                    if current_element is not None:
                        # Process paper
                        paper_id = elem.get('key')
                        
                        # Extract title
                        title_elem = elem.find('title')
                        if title_elem is not None and title_elem.text:
                            title = title_elem.text
                            
                            # Extract year
                            year_elem = elem.find('year')
                            year = int(year_elem.text) if year_elem is not None and year_elem.text else 0
                            
                            # Extract venue/conference
                            venue = None
                            if elem.tag == 'article':
                                journal_elem = elem.find('journal')
                                if journal_elem is not None and journal_elem.text:
                                    venue = journal_elem.text
                            elif elem.tag in ['inproceedings', 'proceedings']:
                                booktitle_elem = elem.find('booktitle')
                                if booktitle_elem is not None and booktitle_elem.text:
                                    venue = booktitle_elem.text
                            
                            # Store paper
                            papers[paper_id] = {
                                'title': title,
                                'year': year
                            }
                            
                            # Process authors
                            for author_elem in elem.findall('author'):
                                if author_elem.text:
                                    author_name = author_elem.text
                                    author_id = author_name.replace(" ", "_").lower()
                                    
                                    # Store author
                                    if author_id not in authors:
                                        authors[author_id] = author_name
                                    
                                    # Create author-paper edge
                                    author_paper_edges.append((author_id, paper_id))
                            
                            # Process venue/conference
                            if venue:
                                conf_id = venue.replace(" ", "_").lower()
                                
                                # Store conference
                                if conf_id not in conferences:
                                    conferences[conf_id] = venue
                                
                                # Create paper-conference edge
                                paper_conf_edges.append((paper_id, conf_id))
                            
                            # Extract terms from title (simple tokenization)
                            if title:
                                # Simple preprocessing: lowercase, remove punctuation
                                title_processed = ''.join(c.lower() if c.isalnum() else ' ' for c in title)
                                title_terms = [t for t in title_processed.split() if len(t) > 3]  # Filter short terms
                                
                                for term in title_terms:
                                    term_id = term
                                    
                                    # Store term
                                    if term_id not in terms:
                                        terms[term_id] = term
                                    
                                    # Create paper-term edge
                                    paper_term_edges.append((paper_id, term_id))
                            
                            paper_count += 1
                            if paper_count >= max_papers:
                                break
                    
                    current_element = None
                
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
        term_ids = {term_id: i for i, term_id in enumerate(terms.keys())}
        
        # Create node features
        # For simplicity, we'll use one-hot encodings for authors, conferences, and terms
        # For papers, we'll use year as a feature
        author_features = torch.eye(len(authors))
        paper_features = torch.zeros((len(papers), 1))
        for i, (paper_id, paper_data) in enumerate(papers.items()):
            paper_features[i, 0] = paper_data['year'] / 2025.0  # Normalize by current year
        conf_features = torch.eye(len(conferences))
        term_features = torch.eye(len(terms))
        
        # Create edge indices
        author_to_paper = torch.zeros((2, len(author_paper_edges)), dtype=torch.long)
        paper_to_author = torch.zeros((2, len(author_paper_edges)), dtype=torch.long)
        conf_to_paper = torch.zeros((2, len(paper_conf_edges)), dtype=torch.long)
        paper_to_conf = torch.zeros((2, len(paper_conf_edges)), dtype=torch.long)
        paper_to_term = torch.zeros((2, len(paper_term_edges)), dtype=torch.long)
        term_to_paper = torch.zeros((2, len(paper_term_edges)), dtype=torch.long)
        
        for i, (author_id, paper_id) in enumerate(author_paper_edges):
            if author_id in author_ids and paper_id in paper_ids:
                author_to_paper[0, i] = author_ids[author_id]
                author_to_paper[1, i] = paper_ids[paper_id]
                paper_to_author[0, i] = paper_ids[paper_id]
                paper_to_author[1, i] = author_ids[author_id]
        
        for i, (paper_id, conf_id) in enumerate(paper_conf_edges):
            if paper_id in paper_ids and conf_id in conf_ids:
                paper_to_conf[0, i] = paper_ids[paper_id]
                paper_to_conf[1, i] = conf_ids[conf_id]
                conf_to_paper[0, i] = conf_ids[conf_id]
                conf_to_paper[1, i] = paper_ids[paper_id]
        
        for i, (paper_id, term_id) in enumerate(paper_term_edges):
            if paper_id in paper_ids and term_id in term_ids:
                paper_to_term[0, i] = paper_ids[paper_id]
                paper_to_term[1, i] = term_ids[term_id]
                term_to_paper[0, i] = term_ids[term_id]
                term_to_paper[1, i] = paper_ids[paper_id]
        
        # Create heterogeneous graph
        graph_data = {
            ('author', 'writes', 'paper'): (author_to_paper[0], author_to_paper[1]),
            ('paper', 'written_by', 'author'): (paper_to_author[0], paper_to_author[1]),
            ('conference', 'publishes', 'paper'): (conf_to_paper[0], conf_to_paper[1]),
            ('paper', 'published_in', 'conference'): (paper_to_conf[0], paper_to_conf[1]),
            ('paper', 'contains', 'term'): (paper_to_term[0], paper_to_term[1]),
            ('term', 'contained_in', 'paper'): (term_to_paper[0], term_to_paper[1])
        }
        
        g = dgl.heterograph(graph_data)
        
        # Set node features
        g.nodes['author'].data['feat'] = author_features
        g.nodes['paper'].data['feat'] = paper_features
        g.nodes['conference'].data['feat'] = conf_features
        g.nodes['term'].data['feat'] = term_features
        
        # Save node mappings for reference
        node_mappings = {
            'author': {i: author_id for author_id, i in author_ids.items()},
            'paper': {i: paper_id for paper_id, i in paper_ids.items()},
            'conference': {i: conf_id for conf_id, i in conf_ids.items()},
            'term': {i: term_id for term_id, i in term_ids.items()}
        }
        
        # Save metadata
        metadata = {
            'authors': authors,
            'papers': papers,
            'conferences': conferences,
            'terms': terms,
            'node_mappings': node_mappings
        }
        
        # Save processed graph
        print("Saving processed graph...")
        processed_file = os.path.join(self.processed_dir, f"{self.dataset_name}_graph.bin")
        dgl.save_graphs(processed_file, [g])
        
        # Save metadata
        metadata_file = os.path.join(self.processed_dir, f"{self.dataset_name}_metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        print("Processing complete.")
        return g, metadata

    def load(self):
        """Load the processed DBLP graph"""
        # Check if processed file exists, if not, process it
        processed_file = os.path.join(self.processed_dir, f"{self.dataset_name}_graph.bin")
        metadata_file = os.path.join(self.processed_dir, f"{self.dataset_name}_metadata.pkl")
        
        if not os.path.exists(processed_file) or not os.path.exists(metadata_file):
            print("Processed files not found. Processing DBLP dataset...")
            return self.process()
        
        # Load the processed graph
        print("Loading processed DBLP graph...")
        g, _ = dgl.load_graphs(processed_file)
        
        # Load metadata
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"Loaded DBLP graph with {g[0].num_nodes()} nodes and {g[0].num_edges()} edges")
        return g[0], metadata


if __name__ == "__main__":
    # Example usage
    loader = DBLPGraphDataLoader()
    graph, metadata = loader.load()
    
    # Print graph statistics
    print("\nGraph Statistics:")
    print(f"Number of nodes: {graph.num_nodes()}")
    print(f"Number of edges: {graph.num_edges()}")
    
    # Print node types and counts
    print("\nNode Types:")
    for ntype in graph.ntypes:
        print(f"  {ntype}: {graph.num_nodes(ntype)}")
    
    # Print edge types and counts
    print("\nEdge Types:")
    for etype in graph.etypes:
        print(f"  {etype}: {graph.num_edges(etype)}")
    
    # Print feature dimensions
    print("\nFeature Dimensions:")
    for ntype in graph.ntypes:
        if 'feat' in graph.nodes[ntype].data:
            print(f"  {ntype}: {graph.nodes[ntype].data['feat'].shape}")
