"""
Translates from OpenAI's `/v1/embeddings` to Nebius AI Studio's `/embeddings` route.
"""

from typing import Optional

import httpx

from litellm.llms.base_llm.embedding.transformation import (
    BaseEmbeddingConfig,
    LiteLLMLoggingObj,
)
from litellm.types.llms.nebius import NebiusAIEndpoint
from litellm.types.llms.openai import AllEmbeddingInputValues
from litellm.types.utils import EmbeddingResponse, Usage


class NebiusEmbeddingConfig(BaseEmbeddingConfig):
    def get_supported_openai_params(self, model: str) -> list:
        return [
            "input",
            "model",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return optional_params

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """
        Convert from OpenAI to Nebius embeddings format
        """
        # Nebius embeddings expect input in the format {"input": [text1, text2, ...]}
        return {
            "input": input,
            "model": model,
        }

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Construct the complete URL for the embeddings API
        """
        if not api_base:
            raise ValueError("api_base is required for Nebius embeddings")
        
        url = api_base.rstrip("/")
        
        if not url.endswith("/embeddings"):
            url = f"{url}{NebiusAIEndpoint.EMBEDDINGS.value}"
            
        return url

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
    ) -> EmbeddingResponse:
        """
        Convert from Nebius to OpenAI embedding response format
        """
        logging_obj.post_call(
            original_response=raw_response.text,
        )
        json_resp = raw_response.json()
        
        if model_response is None:
            model_response = EmbeddingResponse(model=model)
            
        # Parse the data array from the response
        embedding_response = []
        data = json_resp.get("data", [])
        
        for idx, item in enumerate(data):
            embedding_response.append(
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": item.get("embedding", []),
                }
            )
            
        model_response.object = "list"
        model_response.data = embedding_response
        
        # Set usage information if available in the response
        usage_data = json_resp.get("usage", {})
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        
        setattr(
            model_response,
            "usage",
            Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=prompt_tokens,
            ),
        )
        
        return model_response 