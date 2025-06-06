from enum import Enum

class PipelineTask(Enum):
    TEXT_CLASSIFICATION = "text-classification"
    TOKEN_CLASSIFICATION = "token-classification"
    QUESTION_ANSWERING = "question-answering"
    TEXT_GENERATION = "text-generation"
    TEXT2TEXT_GENERATION = "text2text-generation"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CONVERSATIONAL = "conversational"
    FEATURE_EXTRACTION = "feature-extraction"
    ZERO_SHOT_CLASSIFICATION = "zero-shot-classification"
    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"
    IMAGE_SEGMENTATION = "image-segmentation"
    AUTOMATIC_SPEECH_RECOGNITION = "automatic-speech-recognition"
    AUDIO_CLASSIFICATION = "audio-classification"
    TEXT_TO_AUDIO = "text-to-audio"
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_TEXT = "image-to-text"
    DOCUMENT_QUESTION_ANSWERING = "document-question-answering"



