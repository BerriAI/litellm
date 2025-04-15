"""
Re rank api for Hosted VLLM

LiteLLM supports the re rank API format, no parameter transformation occurs
"""

from typing import Any, Dict, List, Optional, Union

import litellm
from litellm.llms.base import BaseLLM
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.hosted_vllm.rerank.transformation import HostedVLLMRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.rerank import RerankRequest, RerankResponse


class HostedVLLMRerank(BaseLLM):
    def rerank(
        self,
        model: str,
        api_key: str,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: Optional[bool] = True,
        max_chunks_per_doc: Optional[int] = None,
        api_base: Optional[str] = None,
        _is_async: Optional[bool] = False,
    ) -> RerankResponse:
        client = _get_httpx_client()

        request_data = RerankRequest(
            model=model,
            query=query,
            top_n=top_n,
            documents=documents,
            rank_fields=rank_fields,
            return_documents=return_documents,
        )

        # exclude None values from request_data
        request_data_dict = request_data.dict(exclude_none=True)
        if max_chunks_per_doc is not None:
            raise ValueError("Hosted VLLM does not support max_chunks_per_doc")

        # Get API base URL
        api_base = api_base or get_secret_str("HOSTED_VLLM_API_BASE")
        if api_base is None:
            raise ValueError("api_base must be provided for Hosted VLLM rerank")
        
        # Get API key
        api_key = api_key or get_secret_str("HOSTED_VLLM_API_KEY") or "fake-api-key"

        if _is_async:
            return self.async_rerank(request_data_dict, api_key, api_base)  # type: ignore # Call async method

        response = client.post(
            f"{api_base}/v1/rerank",
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {api_key}",
            },
            json=request_data_dict,
        )

        if response.status_code != 200:
            raise Exception(response.text)

        _json_response = response.json()

        return HostedVLLMRerankConfig()._transform_response(_json_response)

    async def async_rerank(  # New async method
        self,
        request_data_dict: Dict[str, Any],
        api_key: str,
        api_base: str,
    ) -> RerankResponse:
        client = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.HOSTED_VLLM
        )  # Use async client

        response = await client.post(
            f"{api_base}/v1/rerank",
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {api_key}",
            },
            json=request_data_dict,
        )

        if response.status_code != 200:
            raise Exception(response.text)

        _json_response = response.json()

        return HostedVLLMRerankConfig()._transform_response(_json_response) 