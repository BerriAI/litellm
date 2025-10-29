"""
Vertex AI BGE (BAAI General Embedding) Configuration

BGE models deployed on Vertex AI require different input format:
- Use "prompt" instead of "content" as the input field
"""

from typing import List, Optional, Union

from .types import (
    EmbeddingParameters,
    TaskType,
    TextEmbeddingBGEInput,
    VertexEmbeddingRequest,
)


class VertexBGEConfig:
    """
    Configuration and transformation logic for BGE models on Vertex AI.
    
    BGE (BAAI General Embedding) models use a different request format
    where the input field is named "prompt" instead of "content".
    """

    @staticmethod
    def is_bge_model(model: str) -> bool:
        """
        Check if the model is a BGE (BAAI General Embedding) model.
        
        Args:
            model: The model name
            
        Returns:
            bool: True if the model is a BGE model
        """
        return "bge" in model.lower()

    @staticmethod
    def transform_request(
        input: Union[list, str], optional_params: dict, model: str
    ) -> VertexEmbeddingRequest:
        """
        Transforms an OpenAI request to a Vertex BGE embedding request.
        
        BGE models use "prompt" instead of "content" as the input field.
        
        Args:
            input: The input text(s) to embed
            optional_params: Optional parameters for the request
            model: The model name
            
        Returns:
            VertexEmbeddingRequest: The transformed request
        """
        vertex_request: VertexEmbeddingRequest = VertexEmbeddingRequest()
        vertex_text_embedding_input_list: List[TextEmbeddingBGEInput] = []
        task_type: Optional[TaskType] = optional_params.get("task_type")
        title = optional_params.get("title")

        if isinstance(input, str):
            input = [input]

        for text in input:
            embedding_input = VertexBGEConfig._create_embedding_input(
                prompt=text, task_type=task_type, title=title
            )
            vertex_text_embedding_input_list.append(embedding_input)

        vertex_request["instances"] = vertex_text_embedding_input_list
        vertex_request["parameters"] = EmbeddingParameters(**optional_params)

        return vertex_request

    @staticmethod
    def _create_embedding_input(
        prompt: str,
        task_type: Optional[TaskType] = None,
        title: Optional[str] = None,
    ) -> TextEmbeddingBGEInput:
        """
        Creates a TextEmbeddingBGEInput object for BGE models.

        BGE models use "prompt" instead of "content" as the input field.

        Args:
            prompt: The prompt to be embedded
            task_type: The type of task to be performed
            title: The title of the document to be embedded

        Returns:
            TextEmbeddingBGEInput: A TextEmbeddingBGEInput object
        """
        text_embedding_input = TextEmbeddingBGEInput(prompt=prompt)
        if task_type is not None:
            text_embedding_input["task_type"] = task_type
        if title is not None:
            text_embedding_input["title"] = title
        return text_embedding_input

