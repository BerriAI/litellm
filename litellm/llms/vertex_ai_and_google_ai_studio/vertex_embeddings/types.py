"""
Types for Vertex Embeddings Requests
"""

from enum import Enum
from typing import List, Literal, Optional, TypedDict, Union


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


class EmbeddingParameters(TypedDict, total=False):
    auto_truncate: Optional[bool]
    output_dimensionality: Optional[int]


class VertexEmbeddingRequest(TypedDict, total=False):
    instances: List[TextEmbeddingInput]
    parameters: Optional[EmbeddingParameters]


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
