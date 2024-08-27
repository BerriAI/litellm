"""
Re rank api

LiteLLM supports the re rank API format, no paramter transformation occurs
"""

from typing import Any, Dict, List, Optional, Union

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
        documents: list[Union[str, Dict[str, Any]]],
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: Optional[bool] = True,
        max_chunks_per_doc: Optional[int] = None,
    ) -> RerankResponse:
        client = _get_httpx_client()

        request_data = RerankRequest(
            model=model,
            query=query,
            top_n=top_n,
            documents=documents,
            rank_fields=rank_fields,
            return_documents=return_documents,
            max_chunks_per_doc=max_chunks_per_doc,
        )

        request_data_dict = request_data.dict(exclude_none=True)

        response = client.post(
            "https://api.cohere.com/v1/rerank",
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"bearer {api_key}",
            },
            json=request_data_dict,
        )

        return RerankResponse(**response.json())

    pass
