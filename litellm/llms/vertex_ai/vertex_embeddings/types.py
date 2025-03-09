"""
Types for Vertex Embeddings Requests
"""

from enum import Enum
from typing import List, Optional, TypedDict, Union


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
    task_type: Optional[TaskType]
    title: Optional[str]


# Fine-tuned models require a different input format
# Ref: https://console.cloud.google.com/vertex-ai/model-garden?hl=en&project=adroit-crow-413218&pageState=(%22galleryStateKey%22:(%22f%22:(%22g%22:%5B%5D,%22o%22:%5B%5D),%22s%22:%22%22))
class TextEmbeddingFineTunedInput(TypedDict, total=False):
    inputs: str


class TextEmbeddingFineTunedParameters(TypedDict, total=False):
    max_new_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]


class EmbeddingParameters(TypedDict, total=False):
    auto_truncate: Optional[bool]
    output_dimensionality: Optional[int]


class VertexEmbeddingRequest(TypedDict, total=False):
    instances: Union[List[TextEmbeddingInput], List[TextEmbeddingFineTunedInput]]
    parameters: Optional[Union[EmbeddingParameters, TextEmbeddingFineTunedParameters]]


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
