import pickle

class PickleFile:
    def save(self, file_path: str, obj):
        """Save any Python object to disk using the fastest Pickle protocol."""
        with open(file_path, "wb") as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path: str):
        """Load any Python object from disk."""
        with open(file_path, "rb") as file:
            return pickle.load(file)
