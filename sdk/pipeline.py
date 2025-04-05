from sdk.pipeline_task import PipelineTask

class Pipeline:
    def __init__(self, task: str, model: str):
        self.task = task
        self.model = model


    @property
    def model_name(self):
        return PipelineTask[self.model.replace("-", "_").upper()]


    @property
    def task_type(self):
        return PipelineTask[self.task.replace("-", "_").upper()]


    @property
    def max_tokens(self):
        # Default if unknown model
        default_max = 512

        # Exact model names to max token lengths
        model_token_limits = {
            "distilbert-base-uncased-distilled-squad": 512,
            "bert-base-uncased": 512,
            "bert-large-uncased-whole-word-masking-finetuned-squad": 512,
            "deepset/roberta-base-squad2": 512,
            "gpt2": 1024,
            "gpt2-medium": 1024,
            "gpt2-large": 1024,
            "EleutherAI/gpt-neo-1.3B": 2048,
            "EleutherAI/gpt-neo-2.7B": 2048,
            "EleutherAI/gpt-j-6B": 2048,
            "gpt-neox-20b": 2048,
            "mistral-7b": 8192,
            "mistral-12b": 8192,
            "mistral-7b-instruct": 8192,
            "llama-2-7b": 4096,
            "llama-2-13b": 4096,
            "llama-2-70b": 4096,
            "t5-small": 512,
            "t5-base": 512,
            "t5-large": 512,
            "flan-t5-base": 512,
            "flan-t5-large": 512,
            "facebook/bart-large-cnn": 1024,
            "google/flan-t5-base": 512,
            "google/flan-t5-large": 512,
            "bloom": 2048,
            "bloom-3b": 2048,
            "falcon-7b": 2048,
            "falcon-40b": 2048,
            "command-r": 8192,
        }

        # Check exact model name match
        return model_token_limits.get(self.model, default_max)


    def to_dict(self):
        return {
            "task": self.task,
            "model": self.model
    }