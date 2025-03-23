import argparse

from sdk.query import Query


class Args:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Process queries.")
        parser.add_argument("--text", required=False, type=str, help="A simple text query")
        parser.add_argument("--doc", required=False, type=str, help="Document to be used to answer query")
        self.args = parser.parse_args()
        self.query = Query(self.args.text, self.args.doc)