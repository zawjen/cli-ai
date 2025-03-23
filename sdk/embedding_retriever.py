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
        # Start timing the initialization process
        start_time = time.time()
                
        self.chunks = document_chunks
        self.query = query

        # Initialize the sentence transformer model with a pre-trained model
        # all-MiniLM-L6-v2 is a lightweight model good for generating text embeddings
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        if self.query.doc.has_index:
            # Load existing FAISS index
            self.index = faiss.read_index(self.query.doc.index_path)
        else:
            # Convert all document chunks to embeddings using the model
            # Store as numpy array for efficient computation
            self.embeddings = np.array(self.model.encode(self.chunks)).astype('float32')

            # Generate embeddings and create FAISS index for similarity search
            self.index = self.create_faiss_index()
            self.save_faiss_index()

        # Log the completion time of initialization
        TimeLogger.log("Embedding generation and FAISS indexing completed", start_time)
    

    def create_faiss_index(self):
        """Creates a FAISS index for efficient search."""
        # Get the dimensionality of the embeddings
        dimension = self.embeddings.shape[1]
        
        # Create a FAISS index that uses L2 (Euclidean) distance
        index = faiss.IndexFlatL2(dimension)
        
        # Add the document embeddings to the FAISS index
        index.add(self.embeddings)
        
        # Return the populated index
        return index
    

    def retrieve_context(self, query, top_k=4):
        """Finds the most relevant chunks using FAISS."""
        # Start timing the retrieval process
        start_time = time.time()
        
        # Convert query to embedding and ensure correct data type
        query_embedding = np.array(self.model.encode([query])).astype('float32')
        
        # Search the FAISS index for top_k most similar vectors
        # Returns distances and indices of nearest neighbors
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Log completion time of retrieval
        TimeLogger.log("Context retrieval completed", start_time)
        
        # Handle edge case where no matches are found
        if len(indices[0]) == 0:
            return [], []
        
        # Return the matching chunks and their indices
        # indices[0] is used because FAISS returns results in a 2D array
        return [self.chunks[i] for i in indices[0]], indices[0].tolist()


    def save_faiss_index(self):
        """Saves the FAISS index to disk."""
        faiss.write_index(self.index, self.query.doc.index_path)