"""
Re rank api

LiteLLM supports the re rank API format, no paramter transformation occurs
"""

from typing import Any, Dict, List, Optional, Union

import httpx
from pydantic import BaseModel

import litellm
from litellm.llms.base import BaseLLM
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.rerank_api.types import RerankRequest, RerankResponse


class TogetherAIRerank(BaseLLM):
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
            raise ValueError("TogetherAI does not support max_chunks_per_doc")

        if _is_async:
            return self.async_rerank(request_data_dict, api_key)  # type: ignore # Call async method

        response = client.post(
            "https://api.together.xyz/v1/rerank",
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

        response = RerankResponse(
            id=_json_response.get("id"),
            results=_json_response.get("results"),
            meta=_json_response.get("meta") or {},
        )

        return response

    async def async_rerank(  # New async method
        self,
        request_data_dict: Dict[str, Any],
        api_key: str,
    ) -> RerankResponse:
        client = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.TOGETHER_AI
        )  # Use async client

        response = await client.post(
            "https://api.together.xyz/v1/rerank",
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

        return RerankResponse(
            id=_json_response.get("id"),
            results=_json_response.get("results"),
            meta=_json_response.get("meta") or {},
        )  # Return response

    pass
