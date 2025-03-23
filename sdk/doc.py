import os
from sdk.config import Config


class Doc:
    def __init__(self, name) -> None:
        self.name = name
        self.config = Config()
        

    @property
    def path(self):
        return self.config.doc_path(self.name)
    

    @property
    def index_path(self):
        return self.config.index_path(self.name)
        

    @property
    def output_path(self):
        return self.config.output_path(self.name)
    

    @property
    def chunks_path(self):
        return self.config.chunks_path(self.name)
        

    @property
    def has_index(self):
        return os.path.exists(self.index_path)    
    

    @property
    def has_chunks(self):
        return os.path.exists(self.chunks_path)