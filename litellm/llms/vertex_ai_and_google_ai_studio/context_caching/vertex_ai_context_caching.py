import types
from typing import Callable, List, Literal, Optional, Tuple, Union

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.openai import AllMessageValues
from litellm.types.llms.vertex_ai import (
    RequestBody,
    VertexAICachedContentResponseObject,
)
from litellm.utils import ModelResponse

from ..common_utils import VertexAIError
from .transformation import (
    separate_cached_messages,
    transform_openai_messages_to_gemini_context_caching,
)


class ContextCachingEndpoints:
    """
    Covers context caching endpoints for Vertex AI + Google AI Studio

    v0: covers Google AI Studio
    """

    def __init__(self) -> None:
        pass

    def _get_token_and_url(
        self,
        model: str,
        gemini_api_key: Optional[str],
        custom_llm_provider: Literal["gemini"],
        api_base: Optional[str],
    ) -> Tuple[Optional[str], str]:
        """
        Internal function. Returns the token and url for the call.

        Handles logic if it's google ai studio vs. vertex ai.

        Returns
            token, url
        """
        if custom_llm_provider == "gemini":
            _gemini_model_name = "models/{}".format(model)
            auth_header = None
            endpoint = "cachedContents"
            url = "https://generativelanguage.googleapis.com/v1beta/{}?key={}".format(
                endpoint, gemini_api_key
            )

        else:
            raise NotImplementedError
        if (
            api_base is not None
        ):  # for cloudflare ai gateway - https://github.com/BerriAI/litellm/issues/4317
            if custom_llm_provider == "gemini":
                url = "{}/{}".format(api_base, endpoint)
                auth_header = (
                    gemini_api_key  # cloudflare expects api key as bearer token
                )
            else:
                url = "{}:{}".format(api_base, endpoint)

        return auth_header, url

    def create_cache(
        self,
        messages: List[AllMessageValues],  # receives openai format messages
        api_key: str,
        api_base: Optional[str],
        model: str,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]],
        timeout: Optional[Union[float, httpx.Timeout]],
        logging_obj: Logging,
        extra_headers: Optional[dict] = None,
        cached_content: Optional[str] = None,
    ) -> Tuple[List[AllMessageValues], Optional[str]]:
        """
        Receives
        - messages: List of dict - messages in the openai format

        Returns
        - messages - List[dict] - filtered list of messages in the openai format.
        - cached_content - str - the cache content id, to be passed in the gemini request body

        Follows - https://ai.google.dev/api/caching#request-body
        """
        if cached_content is not None:
            return messages, cached_content

        ## AUTHORIZATION ##
        token, url = self._get_token_and_url(
            model=model,
            gemini_api_key=api_key,
            custom_llm_provider="gemini",
            api_base=api_base,
        )

        headers = {
            "Content-Type": "application/json",
        }
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        if extra_headers is not None:
            headers.update(extra_headers)

        if client is None or not isinstance(client, HTTPHandler):
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            client = HTTPHandler(**_params)  # type: ignore
        else:
            client = client

        cached_messages, non_cached_messages = separate_cached_messages(
            messages=messages
        )

        if len(cached_messages) == 0:
            return messages, None

        cached_content_request_body = (
            transform_openai_messages_to_gemini_context_caching(
                model=model, messages=cached_messages
            )
        )

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": cached_content_request_body,
                "api_base": url,
                "headers": headers,
            },
        )

        try:
            response = client.post(
                url=url, headers=headers, json=cached_content_request_body  # type: ignore
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise VertexAIError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
            raise VertexAIError(status_code=408, message="Timeout error occurred.")

        raw_response_cached = response.json()
        cached_content_response_obj = VertexAICachedContentResponseObject(
            name=raw_response_cached.get("name"), model=raw_response_cached.get("model")
        )
        return (non_cached_messages, cached_content_response_obj["name"])

    def async_create_cache(self):
        pass

    def get_cache(self):
        pass

    async def async_get_cache(self):
        pass
