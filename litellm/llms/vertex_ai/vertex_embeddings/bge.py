"""
Vertex AI BGE (BAAI General Embedding) Configuration

BGE models deployed on Vertex AI require different input/output format:
- Request: Use "prompt" instead of "content" as the input field
- Response: Embeddings are returned directly as arrays, not wrapped in objects

Model name handling:
- Model names like "bge/endpoint_id" are automatically transformed in common_utils._get_vertex_url()
- This module focuses on request/response transformation only
"""

from typing import List, Optional, Union

from litellm.types.utils import EmbeddingResponse, Usage

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
    
    Supported model patterns (after provider split in main.py):
    - "bge-small-en-v1.5" (model name)
    - "bge/204379420394258432" (endpoint ID pattern)
    
    Note: Model name transformation (bge/ -> numeric ID) is handled automatically
    in common_utils._get_vertex_url(). This class focuses on request/response format only.
    """

    @staticmethod
    def is_bge_model(model: str) -> bool:
        """
        Check if the model is a BGE (BAAI General Embedding) model.
        
        After provider split in main.py, supports:
        - "bge-small-en-v1.5" (model name)
        - "bge/204379420394258432" (endpoint ID pattern)
        
        Args:
            model: The model name after provider split
            
        Returns:
            bool: True if the model is a BGE model
        """
        model_lower = model.lower()
        # Check for "bge/" prefix (endpoint pattern) or "bge" in model name
        return model_lower.startswith("bge/") or "bge" in model_lower

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

    @staticmethod
    def transform_response(
        response: dict, model: str, model_response: EmbeddingResponse
    ) -> EmbeddingResponse:
        """
        Transforms a Vertex BGE embedding response to OpenAI format.
        
        BGE models return embeddings directly as arrays in predictions:
        {
          "predictions": [
            [0.002, 0.021, ...],
            [0.003, 0.022, ...]
          ]
        }
        
        Args:
            response: The raw response from Vertex AI
            model: The model name
            model_response: The EmbeddingResponse object to populate
            
        Returns:
            EmbeddingResponse: The transformed response in OpenAI format
            
        Raises:
            KeyError: If response doesn't contain 'predictions'
            ValueError: If predictions is not a list or contains invalid data
        """
        if "predictions" not in response:
            raise KeyError("Response missing 'predictions' field")
        
        _predictions = response["predictions"]
        
        if not isinstance(_predictions, list):
            raise ValueError(f"Expected 'predictions' to be a list, got {type(_predictions)}")

        embedding_response = []
        # BGE models don't return token counts, so we estimate or set to 0
        input_tokens = 0

        for idx, embedding_values in enumerate(_predictions):
            if not isinstance(embedding_values, list):
                raise ValueError(
                    f"Expected embedding at index {idx} to be a list, got {type(embedding_values)}"
                )
            
            embedding_response.append(
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": embedding_values,
                }
            )

        model_response.object = "list"
        model_response.data = embedding_response
        model_response.model = model
        usage = Usage(
            prompt_tokens=input_tokens, completion_tokens=0, total_tokens=input_tokens
        )
        setattr(model_response, "usage", usage)
        return model_response

