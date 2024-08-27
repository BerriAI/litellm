"""
Re rank api

LiteLLM supports the re rank API format, no paramter transformation occurs
"""

import httpx
from pydantic import BaseModel

from litellm.llms.base import BaseLLM
from litellm.llms.custom_httpx.http_handler import (
    _get_async_httpx_client,
    _get_httpx_client,
)
from litellm.rerank_api.types import RerankRequest, RerankResponse


class CohereRerank(BaseLLM):
    def rerank(
        self,
        model: str,
        api_key: str,
        query: str,
        documents: list[str],
        top_n: int = 3,
    ) -> RerankResponse:
        client = _get_httpx_client()
        request_data = RerankRequest(
            model=model, query=query, top_n=top_n, documents=documents
        )

        response = client.post(
            "https://api.cohere.com/v1/rerank",
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"bearer {api_key}",
            },
            json=request_data.dict(),
        )

        return RerankResponse(**response.json())

    pass
