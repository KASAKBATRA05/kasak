import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Any, Tuple
import pickle
import os

class EmbeddingManager:
    """Manages document embeddings using Sentence Transformers and FAISS"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding manager
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.documents = []
        self.embeddings = None
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            self.is_initialized = True
        except Exception as e:
            raise Exception(f"Error initializing embedding model: {str(e)}")
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            raise Exception(f"Error creating embeddings: {str(e)}")
    
    def build_index(self, texts: List[str]) -> None:
        """
        Build FAISS index from texts
        
        Args:
            texts: List of text strings to index
        """
        try:
            if not texts:
                raise ValueError("No texts provided for indexing")
            
            # Store documents
            self.documents = texts
            
            # Create embeddings
            self.embeddings = self.create_embeddings(texts)
            
            # Build FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            
            # Add embeddings to index
            self.index.add(self.embeddings)
            
        except Exception as e:
            raise Exception(f"Error building index: {str(e)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            list: Search results with scores and content
        """
        try:
            if not self.index or not self.documents:
                raise ValueError("Index not built. Call build_index() first.")
            
            # Create query embedding
            query_embedding = self.create_embeddings([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, top_k)
            
            # Format results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):  # Valid index
                    results.append({
                        'content': self.documents[idx],
                        'score': float(score),
                        'rank': i + 1,
                        'index': int(idx)
                    })
            
            return results
            
        except Exception as e:
            raise Exception(f"Error searching: {str(e)}")
    
    def get_similarity_matrix(self) -> np.ndarray:
        """
        Get similarity matrix between all documents
        
        Returns:
            numpy.ndarray: Similarity matrix
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available. Build index first.")
        
        # Compute cosine similarity matrix
        similarity_matrix = np.dot(self.embeddings, self.embeddings.T)
        return similarity_matrix
    
    def find_most_similar(self, text: str, exclude_self: bool = True) -> Dict[str, Any]:
        """
        Find the most similar document to given text
        
        Args:
            text: Input text
            exclude_self: Whether to exclude exact matches
            
        Returns:
            dict: Most similar document info
        """
        results = self.search(text, top_k=2 if exclude_self else 1)
        
        if not results:
            return None
        
        # If excluding self and first result is very similar (>0.99), return second
        if exclude_self and len(results) > 1 and results[0]['score'] > 0.99:
            return results[1]
        else:
            return results[0]
    
    def cluster_documents(self, num_clusters: int = 5) -> Dict[str, Any]:
        """
        Cluster documents using K-means
        
        Args:
            num_clusters: Number of clusters
            
        Returns:
            dict: Clustering results
        """
        try:
            if self.embeddings is None:
                raise ValueError("No embeddings available. Build index first.")
            
            from sklearn.cluster import KMeans
            
            # Perform clustering
            kmeans = KMeans(n_clusters=min(num_clusters, len(self.documents)), random_state=42)
            cluster_labels = kmeans.fit_predict(self.embeddings)
            
            # Group documents by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    'index': i,
                    'content': self.documents[i]
                })
            
            return {
                'clusters': clusters,
                'labels': cluster_labels.tolist(),
                'centroids': kmeans.cluster_centers_
            }
            
        except Exception as e:
            raise Exception(f"Error clustering documents: {str(e)}")
    
    def save_index(self, filepath: str) -> None:
        """
        Save the index and documents to file
        
        Args:
            filepath: Path to save the index
        """
        try:
            if not self.index or not self.documents:
                raise ValueError("No index to save")
            
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save documents and embeddings
            data = {
                'documents': self.documents,
                'embeddings': self.embeddings,
                'model_name': self.model_name
            }
            
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            raise Exception(f"Error saving index: {str(e)}")
    
    def load_index(self, filepath: str) -> None:
        """
        Load index and documents from file
        
        Args:
            filepath: Path to load the index from
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load documents and embeddings
            with open(f"{filepath}.pkl", 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data['documents']
            self.embeddings = data['embeddings']
            self.model_name = data['model_name']
            
            # Initialize model if needed
            if not self.is_initialized:
                self.initialize()
                
        except Exception as e:
            raise Exception(f"Error loading index: {str(e)}")
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about indexed documents
        
        Returns:
            dict: Document statistics
        """
        if not self.documents:
            return {'error': 'No documents indexed'}
        
        total_docs = len(self.documents)
        total_chars = sum(len(doc) for doc in self.documents)
        total_words = sum(len(doc.split()) for doc in self.documents)
        avg_length = total_chars / total_docs if total_docs > 0 else 0
        
        return {
            'total_documents': total_docs,
            'total_characters': total_chars,
            'total_words': total_words,
            'average_document_length': round(avg_length, 2),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0
        }