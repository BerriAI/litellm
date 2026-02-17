"""
Gemini Embedding Models Transformation for Vertex AI

Handles Gemini embedding models that require the :embedContent endpoint
with {"content": {"parts": [{"text": "..."}]}} format instead of the
standard :predict endpoint with {"instances": [...]} format.
"""

from typing import List, Union

from litellm.types.llms.vertex_ai import ContentType, PartType
from litellm.types.utils import EmbeddingResponse, Usage

# List of Gemini embedding models that require :embedContent endpoint
GEMINI_EMBEDDING_MODELS = {
    "gemini-embedding-001",
    "gemini-embedding-2-exp-11-2025",
    "text-embedding-005",
    "text-multilingual-embedding-002",
}


class VertexGeminiEmbeddingConfig:
    """
    Configuration and transformation for Gemini embedding models on Vertex AI.

    These models use the :embedContent endpoint instead of :predict.
    """

    @staticmethod
    def is_gemini_embedding_model(model: str) -> bool:
        """
        Check if the model is a Gemini embedding model that requires :embedContent endpoint.

        Args:
            model: The model name (may include routing prefixes like "vertex_ai/")

        Returns:
            bool: True if the model is a Gemini embedding model
        """
        # Strip any routing prefixes
        base_model = model.split("/")[-1]
        return base_model in GEMINI_EMBEDDING_MODELS

    @staticmethod
    def transform_openai_request_to_vertex_embedding_request(
        input: Union[list, str], optional_params: dict, model: str
    ) -> dict:
        """Alias for transform_request to match the standard interface."""
        return VertexGeminiEmbeddingConfig.transform_request(input, optional_params, model)
    
    @staticmethod
    def transform_request(
        input: Union[list, str], optional_params: dict, model: str
    ) -> dict:
        """
        Transforms an OpenAI request to a Gemini embedContent request format.

        Gemini embedding models use the :embedContent endpoint with format:
        {
            "content": {"parts": [{"text": "..."}]},
            "taskType": "...",
            "outputDimensionality": ...
        }

        Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings

        Args:
            input: Text input(s) to embed
            optional_params: Additional parameters (task_type, outputDimensionality, etc.)
            model: Model name

        Returns:
            dict: Gemini embedContent request format
        """
        if isinstance(input, str):
            input_list = [input]
        else:
            input_list = input

        # For single input, use the simple embedContent format
        if len(input_list) == 1:
            request: dict = {"content": ContentType(parts=[PartType(text=input_list[0])])}

            # Add task type if specified
            task_type = optional_params.get("task_type")
            if task_type:
                request["taskType"] = task_type

            # Add output dimensionality if specified
            output_dim = optional_params.get("outputDimensionality")
            if output_dim:
                request["outputDimensionality"] = output_dim

            return request
        else:
            # For multiple inputs, we need to call the endpoint multiple times
            # Store the full input list in a special key for the handler
            request: dict = {
                "content": ContentType(parts=[PartType(text=input_list[0])]),
                "_batch_inputs": input_list,  # Internal flag for handler
            }

            # Add task type if specified
            task_type = optional_params.get("task_type")
            if task_type:
                request["taskType"] = task_type

            # Add output dimensionality if specified
            output_dim = optional_params.get("outputDimensionality")
            if output_dim:
                request["outputDimensionality"] = output_dim

            return request

    @staticmethod
    def transform_vertex_response_to_openai(
        response: Union[dict, List[dict]], model: str, model_response: EmbeddingResponse
    ) -> EmbeddingResponse:
        """Alias for transform_response to match the standard interface."""
        return VertexGeminiEmbeddingConfig.transform_response(response, model, model_response)
    
    @staticmethod
    def transform_response(
        response: Union[dict, List[dict]], model: str, model_response: EmbeddingResponse
    ) -> EmbeddingResponse:
        """
        Transforms a Gemini embedContent response to OpenAI format.

        Gemini embedContent response format:
        {
            "embedding": {
                "values": [0.1, 0.2, ...]
            }
        }

        Or for multiple embeddings (from handler looping):
        [
            {"embedding": {"values": [...]}, ...},
            {"embedding": {"values": [...]}, ...}
        ]

        Args:
            response: Gemini embedContent response(s)
            model: Model name
            model_response: EmbeddingResponse object to populate

        Returns:
            EmbeddingResponse: OpenAI-compatible embedding response
        """
        embedding_response = []

        # Check if response is a list (multiple embeddings) or single embedding
        if isinstance(response, list):
            # Multiple embeddings
            for idx, item in enumerate(response):
                if "embedding" in item:
                    embedding_values = item["embedding"]["values"]
                    embedding_response.append(
                        {
                            "object": "embedding",
                            "index": idx,
                            "embedding": embedding_values,
                        }
                    )
        else:
            # Single embedding
            if "embedding" in response:
                embedding_values = response["embedding"]["values"]
                embedding_response.append(
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": embedding_values,
                    }
                )

        model_response.object = "list"
        model_response.data = embedding_response
        model_response.model = model

        # Gemini embedContent doesn't return token counts in the response
        # Use a basic estimation or set to 0
        input_tokens = 0
        usage = Usage(
            prompt_tokens=input_tokens, completion_tokens=0, total_tokens=input_tokens
        )
        setattr(model_response, "usage", usage)
        return model_response
