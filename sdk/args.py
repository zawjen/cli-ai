import argparse

from sdk.pipeline import Pipeline
from sdk.query import Query


class Args:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Process queries.")

        parser.add_argument("--text", required=False, type=str, help="A simple text query")
        parser.add_argument("--doc", required=False, type=str, help="Document to be used to answer query")
        parser.add_argument("--task", required=False, type=str, help="Type of pipeline task to use for processing", default="question-answering")
        parser.add_argument("--model", required=False, type=str, help="Model to use for the pipeline", default="distilbert-base-uncased-distilled-squad")
        
        self.args = parser.parse_args()
        self.pipeline = Pipeline(self.args.task, self.args.model)
        self.query = Query(self.args.text, self.args.doc, self.pipeline)
