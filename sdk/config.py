import os

class Config:
    def __init__(self) -> None:
        self.load()


    @property
    def data_dir(self):
        return "data"


    @property
    def datasets_dir(self):
        return f"{self.data_dir}/datasets/"


    @property
    def output_dir(self):
        return f"{self.data_dir}/output/"
    

    @property
    def index_dir(self):
        return f"{self.data_dir}/index/"    
    

    @property
    def chunks_dir(self):
        return f"{self.data_dir}/chunks/"
    

    def doc_path(self, file_name):
        return os.path.join(self.datasets_dir, file_name)
    

    def output_path(self, file_name):
        return os.path.join(self.output_dir, f"{file_name}.json")
        

    def index_path(self, file_name):
        return os.path.join(self.index_dir, f"{file_name}.fiass.index")
            
            
    def chunks_path(self, file_name):
        return os.path.join(self.chunks_dir, f"{file_name}.chunks.pkl")
    

    def load(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)
