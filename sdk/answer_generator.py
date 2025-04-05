import time
from transformers import pipeline, AutoTokenizer

from sdk.pipeline_task import PipelineTask
from sdk.query import Query
from sdk.time_logger import TimeLogger

class AnswerGenerator:
    def __init__(self, query: Query, context):
        self.query: Query = query
        self.context = context
        self.llm = pipeline(self.query.pipeline.task, model=query.pipeline.model)
        self.tokenizer = AutoTokenizer.from_pretrained(query.pipeline.model)

    @property
    def truncated_context(self):
        # Tokenize the context
        max_tokens = self.query.pipeline.max_tokens
        tokens = self.tokenizer.tokenize(self.context)

        # If tokens exceed max_tokens, truncate
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]

        # Convert tokens back to text
        return self.tokenizer.convert_tokens_to_string(tokens)


    def generate_answer(self):
        start_time = time.time()

        if not self.context:
            return "No relevant information found in the document."

        # Truncate context to ensure the total prompt doesn't exceed the model's limit
        truncated_context = self.truncated_context

        task_type = self.query.pipeline.task_type
        switcher = {
            PipelineTask.TEXT_CLASSIFICATION: self.handle_text_classification,
            PipelineTask.TOKEN_CLASSIFICATION: self.handle_token_classification,
            PipelineTask.QUESTION_ANSWERING: self.handle_question_answering,
            PipelineTask.TEXT_GENERATION: self.handle_text_generation,
            PipelineTask.TEXT2TEXT_GENERATION: self.handle_text2text_generation,
            PipelineTask.SUMMARIZATION: self.handle_summarization,
            PipelineTask.TRANSLATION: self.handle_translation,
            PipelineTask.CONVERSATIONAL: self.handle_conversational,
            PipelineTask.FEATURE_EXTRACTION: self.handle_feature_extraction,
            PipelineTask.ZERO_SHOT_CLASSIFICATION: self.handle_zero_shot_classification,
            PipelineTask.IMAGE_CLASSIFICATION: self.handle_image_classification,
            PipelineTask.OBJECT_DETECTION: self.handle_object_detection,
            PipelineTask.IMAGE_SEGMENTATION: self.handle_image_segmentation,
            PipelineTask.AUTOMATIC_SPEECH_RECOGNITION: self.handle_automatic_speech_recognition,
            PipelineTask.AUDIO_CLASSIFICATION: self.handle_audio_classification,
            PipelineTask.TEXT_TO_AUDIO: self.handle_text_to_audio,
            PipelineTask.TEXT_TO_IMAGE: self.handle_text_to_image,
            PipelineTask.IMAGE_TO_TEXT: self.handle_image_to_text,
            PipelineTask.DOCUMENT_QUESTION_ANSWERING: self.handle_document_question_answering,
        }

        handler = switcher.get(task_type, lambda: "Invalid task type")
        response = handler()

        TimeLogger.log("AnswerGenerator.generate_answer: Answer generation completed", start_time)
        return response

    def handle_text_classification(self):
            return "Handling text classification task."

    def handle_token_classification(self):
        return "Handling token classification task."

    def handle_question_answering(self):
        response = self.llm(question=self.query.text, context=self.truncated_context)["answer"].strip()
        return response

    def handle_text_generation(self):
        prompt = (
            f"Answer the query based only on the given context.\n\n"
            f"Context:\n{self.truncated_context}\n\n"
            f"Query: {self.query.text}"
        )
        response = self.llm(prompt, max_length=100)[0]["generated_text"].strip()
        return response

    def handle_text2text_generation(self):
        return "Handling text to text generation task."

    def handle_summarization(self):
        return "Handling summarization task."

    def handle_translation(self):
        return "Handling translation task."

    def handle_conversational(self):
        return "Handling conversational task."

    def handle_feature_extraction(self):
        return "Handling feature extraction task."

    def handle_zero_shot_classification(self):
        return "Handling zero shot classification task."

    def handle_image_classification(self):
        return "Handling image classification task."

    def handle_object_detection(self):
        return "Handling object detection task."

    def handle_image_segmentation(self):
        return "Handling image segmentation task."

    def handle_text_classification(self):
        return "Handling text classification task."

    def handle_token_classification(self):
        return "Handling token classification task."

    def handle_question_answering(self):
        response = self.llm(question=self.query.text, context=self.truncated_context)["answer"].strip()

        return response

    def handle_text_generation(self):
                
        prompt = (
            f"Answer the query based only on the given context.\n\n"
            f"Context:\n{self.truncated_context}\n\n"
            f"Query: {self.query.text}"
        )

        response = self.llm(prompt, max_length=100)[0]["generated_text"].strip()

        return response

    def handle_text2text_generation(self):
        return "Handling text to text generation task."

    def handle_summarization(self):
        return "Handling summarization task."

    def handle_translation(self):
        return "Handling translation task."

    def handle_conversational(self):
        return "Handling conversational task."

    def handle_feature_extraction(self):
        return "Handling feature extraction task."

    def handle_zero_shot_classification(self):
        return "Handling zero shot classification task."

    def handle_image_classification(self):
        return "Handling image classification task."

    def handle_object_detection(self):
        return "Handling object detection task."

    def handle_image_segmentation(self):
        return "Handling image segmentation task."

    def handle_automatic_speech_recognition(self):
        return "Handling automatic speech recognition task."

    def handle_audio_classification(self):
        return "Handling audio classification task."

    def handle_text_to_audio(self):
        return "Handling text to audio task."

    def handle_text_to_image(self):
        return "Handling text to image task."

    def handle_image_to_text(self):
        return "Handling image to text task."

    def handle_document_question_answering(self):
        return "Handling document question answering task."
