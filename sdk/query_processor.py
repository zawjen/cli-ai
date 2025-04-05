
import json
import time
import numpy as np
from pathlib import Path

from sdk.query import Query
from sdk.time_logger import TimeLogger
from sdk.answer_generator import AnswerGenerator
from sdk.document_processor import DocumentProcessor
from sdk.embedding_retriever import EmbeddingRetriever

class QueryProcessor:
    """Loads queries and extracts the required query based on QID."""
    def __init__(self, query: Query):
        self.query: Query = query


    def generate_answer(self):
        doc_processor = None
        retriever = None

        doc_processor = DocumentProcessor(self.query.doc)
        retriever = EmbeddingRetriever(self.query, doc_processor.chunks)
        
        # Retrieve context
        context, source_chunks = retriever.retrieve_context(self.query.text)

        generator = AnswerGenerator(self.query, "\n".join(context))
        
        self.query.answer.text = generator.generate_answer()
        self.query.answer.source_chunks = source_chunks
        self.query.answer.source_text = context
        
        return self.query.answer
    
    def save_answer(self):
        self.query.save(self.query.doc.output_path)

    