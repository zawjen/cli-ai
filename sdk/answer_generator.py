from sdk.query import Query
from sdk.time_logger import TimeLogger
import time
from transformers import pipeline


class AnswerGenerator:
    """Uses an LLM (Flan-T5) to generate answers based on retrieved context."""
    def __init__(self, query: Query, context):
        # Store the query text that we want to answer
        self.query: Query = query
        
        # Store the retrieved context/passages that contain relevant information
        self.context = context
        
        # Initialize the Flan-T5 language model for text generation
        # We use the 'large' variant which offers a good balance of performance and speed
        # Other available sizes are:
        # - flan-t5-base: Smaller, faster but less capable
        # - flan-t5-xl: Larger, slower but more capable
        self.llm = pipeline("text2text-generation", model="google/flan-t5-large")
        #self.llm = pipeline("question-answering", model="deepset/roberta-base-squad2")

    def generate_answer(self):
        """Generates an answer using Flan-T5."""
        # Start timing the answer generation process
        start_time = time.time()

        # If no context was retrieved, return a default message
        if not self.context:
            return "No relevant information found in the document."

        # Construct the prompt by combining:
        # - Instruction to only use provided context
        # - The retrieved context passages
        # - The user's query
        # - A marker for where the answer should begin
        prompt = f"Answer the query based only on the given context.\n\nContext:\n{self.context}\n\nQuery: {self.query.text}"

        # Generate the answer using Flan-T5:
        # - max_length=100 limits response length to avoid overly verbose answers
        # - do_sample=True enables some randomness in generation
        response = self.llm(prompt, max_length=100)[0]["generated_text"].strip()

        # Log the completion time of answer generation
        TimeLogger.log("Answer generation completed", start_time)

        # Return the generated answer
        return response

