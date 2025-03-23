class Answer:
    def __init__(self, text: str = None) -> None:
        self.text = text
        self.source_chunks = None
        self.source_text = None


    def to_dict(self):
        return {
            "text": self.text,
            "source_chunks": self.source_chunks,
            "source_text": self.source_text
        }