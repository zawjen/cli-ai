from enum import Enum
import json

from sdk.answer import Answer
from sdk.doc import Doc
from sdk.json_file import JsonFile
from sdk.pickle_file import PickleFile


class QueryType(Enum):
    TEXT = "text"


class Query:
    def __init__(self, text: str, doc: str):
        self.type = QueryType.TEXT
        self.text = text
        self.doc = Doc(doc)
        self.answer = Answer()

    def to_dict(self):
        return {
            "type": "text",
            "query": self.text,
            "answer": self.answer.to_dict()
        }
    
    def save(self, file_path):
        file = JsonFile()
        file.save(file_path, self.to_dict())
    
    
    def load(self, file_path):
        file = PickleFile()
        self = file.load(file_path)

        return self