"""
Re rank api

LiteLLM supports the re rank API format, no paramter transformation occurs
"""

from typing import Any, Dict, List, Optional, Union

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base import BaseLLM
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.rerank_api.types import RerankRequest, RerankResponse


class CohereRerank(BaseLLM):
    def validate_environment(self, api_key: str, headers: Optional[dict]) -> dict:
        default_headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"bearer {api_key}",
        }

        if headers is None:
            return default_headers

        # If 'Authorization' is provided in headers, it overrides the default.
        if "Authorization" in headers:
            default_headers["Authorization"] = headers["Authorization"]

        # Merge other headers, overriding any default ones except Authorization
        return {**default_headers, **headers}

    def rerank(
        self,
        model: str,
        api_key: str,
        api_base: str,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        headers: Optional[dict],
        litellm_logging_obj: LiteLLMLoggingObj,
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: Optional[bool] = True,
        max_chunks_per_doc: Optional[int] = None,
        _is_async: Optional[bool] = False,  # New parameter
    ) -> RerankResponse:
        headers = self.validate_environment(api_key=api_key, headers=headers)

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
        ## LOGGING
        litellm_logging_obj.pre_call(
            input=request_data_dict,
            api_key=api_key,
            additional_args={
                "complete_input_dict": request_data_dict,
                "api_base": api_base,
                "headers": headers,
            },
        )

        if _is_async:
            return self.async_rerank(request_data=request_data, api_key=api_key, api_base=api_base, headers=headers)  # type: ignore # Call async method

        client = _get_httpx_client()
        response = client.post(
            api_base,
            headers=headers,
            json=request_data_dict,
        )

        returned_response = RerankResponse(**response.json())

        _response_headers = response.headers

        llm_response_headers = {
            "{}-{}".format("llm_provider", k): v for k, v in _response_headers.items()
        }
        returned_response._hidden_params["additional_headers"] = llm_response_headers

        return returned_response

    async def async_rerank(
        self,
        request_data: RerankRequest,
        api_key: str,
        api_base: str,
        headers: dict,
    ) -> RerankResponse:
        request_data_dict = request_data.dict(exclude_none=True)

        client = get_async_httpx_client(llm_provider=litellm.LlmProviders.COHERE)

        response = await client.post(
            api_base,
            headers=headers,
            json=request_data_dict,
        )

        returned_response = RerankResponse(**response.json())

        _response_headers = dict(response.headers)

        llm_response_headers = {
            "{}-{}".format("llm_provider", k): v for k, v in _response_headers.items()
        }
        returned_response._hidden_params["additional_headers"] = llm_response_headers
        returned_response._hidden_params["model"] = request_data.model

        return returned_response
