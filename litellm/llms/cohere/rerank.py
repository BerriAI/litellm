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
        api_base: str,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: Optional[bool] = True,
        max_chunks_per_doc: Optional[int] = None,
        _is_async: Optional[bool] = False,  # New parameter
    ) -> RerankResponse:
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

        if _is_async:
            return self.async_rerank(request_data_dict, api_key, api_base)  # type: ignore # Call async method

        client = _get_httpx_client()
        response = client.post(
            api_base,
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"bearer {api_key}",
            },
            json=request_data_dict,
        )

        return RerankResponse(**response.json())

    async def async_rerank(
        self,
        request_data_dict: Dict[str, Any],
        api_key: str,
        api_base: str,
    ) -> RerankResponse:
        client = _get_async_httpx_client()

        response = await client.post(
            api_base,
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"bearer {api_key}",
            },
            json=request_data_dict,
        )

        return RerankResponse(**response.json())
