"""
Types for Vertex Embeddings Requests
"""

from enum import Enum

from typing_extensions import TypedDict


class TaskType(str, Enum):
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    FACT_VERIFICATION = "FACT_VERIFICATION"
    CODE_RETRIEVAL_QUERY = "CODE_RETRIEVAL_QUERY"


class TextEmbeddingInput(TypedDict, total=False):
    content: str
    task_type: TaskType | None
    title: str | None


class TextEmbeddingBGEInput(TypedDict, total=False):
    prompt: str
    task_type: TaskType | None
    title: str | None


# Fine-tuned models require a different input format
# Ref: https://console.cloud.google.com/vertex-ai/model-garden?hl=en&project=adroit-crow-413218&pageState=(%22galleryStateKey%22:(%22f%22:(%22g%22:%5B%5D,%22o%22:%5B%5D),%22s%22:%22%22))
class TextEmbeddingFineTunedInput(TypedDict, total=False):
    inputs: str


class TextEmbeddingFineTunedParameters(TypedDict, total=False):
    max_new_tokens: int | None
    temperature: float | None
    top_p: float | None
    top_k: int | None


class EmbeddingParameters(TypedDict, total=False):
    auto_truncate: bool | None
    output_dimensionality: int | None


class VertexEmbeddingRequest(TypedDict, total=False):
    instances: list[TextEmbeddingInput] | list[TextEmbeddingBGEInput] | list[TextEmbeddingFineTunedInput]
    parameters: EmbeddingParameters | TextEmbeddingFineTunedParameters | None
    labels: dict[str, str] | None


# Example usage:
# example_request: VertexEmbeddingRequest = {
#     "instances": [
#         {
#             "content": "I would like embeddings for this text!",
#             "task_type": "RETRIEVAL_DOCUMENT",
#             "title": "document title"
#         }
#     ],
#     "parameters": {
#         "auto_truncate": True,
#         "output_dimensionality": None
#     }
# }
