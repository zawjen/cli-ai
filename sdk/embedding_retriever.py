from typing import List
import time 
import numpy as np  
import faiss 
from sentence_transformers import SentenceTransformer 

from sdk.query import Query
from sdk.time_logger import TimeLogger  


class EmbeddingRetriever:
    """Generates embeddings and retrieves the most relevant document chunks using FAISS."""
    def __init__(self, query: Query, document_chunks: List[str]=None):
        start_time = time.time()
                
        self.chunks = document_chunks
        self.query = query

        # Use a faster model variant while maintaining quality
        self.model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

        if self.query.doc.has_index:
            # Load existing FAISS index
            self.index = faiss.read_index(self.query.doc.index_path)
            TimeLogger.log("EmbeddingRetriever.__init__: faiss.read_index completed", start_time)
        else:
            # Batch encode chunks for better performance
            batch_size = 32
            self.embeddings = []
            for i in range(0, len(self.chunks), batch_size):
                batch = self.chunks[i:i + batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_tensor=False)
                self.embeddings.extend(batch_embeddings)
            self.embeddings = np.array(self.embeddings).astype('float32')

            # Use IndexIVFFlat for faster search with large datasets
            self.index = self.create_faiss_index()
            self.save_faiss_index()
            TimeLogger.log("EmbeddingRetriever.__init__: Embedding generation and FAISS indexing completed", start_time)
    

    def create_faiss_index(self):
        """Creates a FAISS index optimized for fast search."""
        dimension = self.embeddings.shape[1]
        
        # Number of centroids - rule of thumb is sqrt(N) where N is dataset size
        nlist = int(np.sqrt(len(self.embeddings)))
        
        # Create quantizer
        quantizer = faiss.IndexFlatL2(dimension)
        
        # Create IVF index with better search performance
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        # Train the index
        index.train(self.embeddings)
        
        # Add vectors to the index
        index.add(self.embeddings)
        
        # Set number of probes (higher = more accurate but slower)
        index.nprobe = 4
        
        return index
    

    def retrieve_context(self, query, top_k=4):
        """Finds the most relevant chunks using FAISS."""
        start_time = time.time()
        
        # Encode query without unnecessary conversions
        query_embedding = self.model.encode(query, convert_to_tensor=False).reshape(1, -1).astype('float32')
        
        # Search index
        distances, indices = self.index.search(query_embedding, top_k)
        
        TimeLogger.log("EmbeddingRetriever.retrieve_context: Context retrieval completed", start_time)
        
        if len(indices[0]) == 0:
            return [], []
            
        # Use numpy operations for faster sorting
        idx = np.argsort(distances[0])
        sorted_indices = indices[0][idx]
        
        # Return results
        return [self.chunks[i] for i in sorted_indices], sorted_indices.tolist()


    def save_faiss_index(self):
        """Saves the FAISS index to disk."""
        faiss.write_index(self.index, self.query.doc.index_path)