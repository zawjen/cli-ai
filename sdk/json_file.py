import json

class JsonFile:
    def save(self, file_path: str, obj):
        """Save any Python object to a JSON file."""
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(obj, file, ensure_ascii=False, indent=4)

    def load(self, file_path: str):
        """Load any Python object from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)