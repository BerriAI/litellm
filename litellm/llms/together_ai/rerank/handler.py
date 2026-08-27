"""
Re rank api

LiteLLM supports the re rank API format, no paramter transformation occurs
"""

from typing import Any, Final

import litellm
from litellm.llms.base import BaseLLM
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.together_ai.rerank.transformation import TogetherAIRerankConfig
from litellm.types.rerank import RerankRequest, RerankResponse


def _rerank_url(api_base: str) -> str:
    return f"{api_base.rstrip('/')}/rerank"


class TogetherAIRerank(BaseLLM):
    def rerank(
        self,
        model: str,
        api_key: str,
        api_base: str,
        query: str,
        documents: list[str | dict[str, Any]],
        top_n: int | None = None,
        rank_fields: list[str] | None = None,
        return_documents: bool | None = True,
        max_chunks_per_doc: int | None = None,
        _is_async: bool | None = False,
    ) -> RerankResponse:
        client: Final = _get_httpx_client()

        request_data: Final = RerankRequest(
            model=model,
            query=query,
            top_n=top_n,
            documents=documents,
            rank_fields=rank_fields,
            return_documents=return_documents,
        )

        # exclude None values from request_data
        request_data_dict: Final = request_data.dict(exclude_none=True)
        if max_chunks_per_doc is not None:
            raise ValueError("TogetherAI does not support max_chunks_per_doc")

        if _is_async:
            return self.async_rerank(request_data_dict, api_key, api_base)

        response: Final = client.post(
            _rerank_url(api_base),
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {api_key}",
            },
            json=request_data_dict,
        )

        if response.status_code != 200:
            raise Exception(response.text)

        _json_response: Final = response.json()

        return TogetherAIRerankConfig()._transform_response(_json_response)

    async def async_rerank(  # New async method
        self,
        request_data_dict: dict[str, Any],
        api_key: str,
        api_base: str,
    ) -> RerankResponse:
        client: Final = get_async_httpx_client(llm_provider=litellm.LlmProviders.TOGETHER_AI)  # Use async client

        response: Final = await client.post(
            _rerank_url(api_base),
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {api_key}",
            },
            json=request_data_dict,
        )

        if response.status_code != 200:
            raise Exception(response.text)

        _json_response: Final = response.json()

        return TogetherAIRerankConfig()._transform_response(_json_response)
