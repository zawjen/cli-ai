from sdk.pickle_file import PickleFile
from sdk.doc import Doc
from sdk.time_logger import TimeLogger
import time


class DocumentProcessor:
    """Loads the document, splits it into chunks, and processes text."""
    def __init__(self, doc: Doc, max_tokens=80):
        start_time = time.time()
        self.doc: Doc = doc
        self.max_tokens = max_tokens

        if self.doc.has_chunks:
             self.text = None
             self.chunks = self.load_chunks()
        else:
            self.text = self.load_document()
            self.chunks = self.chunk_text()
            self.save_chunks()

        TimeLogger.log("Document processing completed", start_time)


    def load_document(self):
        """Reads the document from file."""
        with open(self.doc.path, "r", encoding="utf-8") as file:
            return file.read()


    def chunk_text(self):
        """Splits the text into chunks based on token count."""
        # Split the entire text into lines using newline character as separator
        sentences = self.text.split("\n")
        
        # Initialize an empty list to store final chunks and a string for the current chunk
        chunks, chunk = [], ""
        
        # Iterate through each line (sentence) from the text
        for sentence in sentences:
            # Calculate if adding this sentence would exceed max_tokens
            # split() breaks the text into words, len() counts the words
            # Checks if current chunk words + new sentence words <= max_tokens
            if len(chunk.split()) + len(sentence.split()) <= self.max_tokens:
                # If within limit, add sentence to current chunk with a space
                chunk += sentence + " "
            else:
                # If would exceed limit:
                # 1. Add current chunk (stripped of trailing spaces) to chunks list
                chunks.append(chunk.strip())
                # 2. Start new chunk with this sentence
                chunk = sentence + " "
        
        # After loop ends, add any remaining chunk (if exists)
        if chunk:
            chunks.append(chunk.strip())
        
        # Return the list of all chunks
        return chunks


    def save_chunks(self):
        file = PickleFile()
        file.save(self.doc.chunks_path, self.chunks)


    def load_chunks(self):
        file = PickleFile()
        return file.load(self.doc.chunks_path)