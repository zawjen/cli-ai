from enum import Enum
import json

from sdk.answer import Answer
from sdk.doc import Doc
from sdk.json_file import JsonFile
from sdk.pickle_file import PickleFile
from sdk.pipeline import Pipeline


class QueryType(Enum):
    TEXT = "text"


class Query:
    def __init__(self, text: str, doc: str, pipeline: Pipeline):
        self.type = QueryType.TEXT
        self.text = text
        self.pipeline = pipeline
        self.doc = Doc(doc)
        self.answer = Answer()

    def to_dict(self):
        return {
            "type": "text",
            "query": self.text,
            "pipeline": self.pipeline.to_dict(),
            "answer": self.answer.to_dict()
        }
    
    def save(self, file_path):
        file = JsonFile()
        file.save(file_path, self.to_dict())
    
    
    def load(self, file_path):
        file = PickleFile()
        self = file.load(file_path)

        return self