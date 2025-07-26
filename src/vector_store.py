"""
Vector store management using FAISS and sentence transformers
"""
import os
import pickle
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .config import Config

class VectorStore:
    """Handle embeddings and vector similarity search"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        print("Loading embedding model...")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        print(f"Embedding model loaded. Dimension: {self.dimension}")
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """Create embeddings for all chunks"""
        texts = [chunk['content'] for chunk in chunks]
        
        print(f"Creating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        self.chunks = chunks
        return embeddings
    
    def build_index(self, embeddings: np.ndarray):
        """Build FAISS index for fast similarity search"""
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        if self.index is None:
            print("❌ No index available!")
            return []
        
        if top_k is None:
            top_k = self.config.TOP_K_DOCS
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if similarity > self.config.SIMILARITY_THRESHOLD:
                result = self.chunks[idx].copy()
                result['similarity'] = float(similarity)
                results.append(result)
        
        return results
    
    def save_index(self, filepath: str = None):
        """Save FAISS index and chunks"""
        if filepath is None:
            filepath = self.config.INDEX_FILE
        
        if self.index:
            faiss.write_index(self.index, filepath)
            with open(filepath.replace('.bin', '_chunks.pkl'), 'wb') as f:
                pickle.dump(self.chunks, f)
            print(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str = None):
        """Load FAISS index and chunks"""
        if filepath is None:
            filepath = self.config.INDEX_FILE
        
        try:
            self.index = faiss.read_index(filepath)
            with open(filepath.replace('.bin', '_chunks.pkl'), 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"Index loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return False
    
    def setup_from_chunks(self, chunks: List[Dict[str, Any]]):
        """Complete setup from document chunks"""
        embeddings = self.create_embeddings(chunks)
        self.build_index(embeddings)