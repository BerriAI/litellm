from typing import List, Literal, Optional, Tuple, Union, Dict, Any

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.llms.openai.openai import AllMessageValues
from litellm.types.llms.vertex_ai import (
    CachedContentListAllResponseBody,
    VertexAICachedContentResponseObject,
)

from ..common_utils import VertexAIError
from ..vertex_llm_base import VertexBase
from .transformation import (
    CacheSplitResult,
    extract_cache_configuration,
)

class ContextCachingEndpoints(VertexBase):
    """
    Covers context caching endpoints for Vertex AI + Google AI Studio

    v0: covers Google AI Studio
    """

    def __init__(self) -> None:
        pass

    def _get_token_and_url_context_caching(
        self,
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
            auth_header = None
            endpoint = "cachedContents"
            url = "https://generativelanguage.googleapis.com/v1beta/{}?key={}".format(
                endpoint, gemini_api_key
            )

        else:
            raise NotImplementedError

        return self._check_custom_proxy(
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            gemini_api_key=gemini_api_key,
            endpoint=endpoint,
            stream=None,
            auth_header=auth_header,
            url=url,
        )

    def check_cache(
        self,
        cache_key: str,
        client: HTTPHandler,
        headers: dict,
        api_key: str,
        api_base: Optional[str],
        logging_obj: Logging,
    ) -> Optional[str]:
        """
        Checks if content already cached.

        Currently, checks cache list, for cache key == displayName, since Google doesn't let us set the name of the cache (their API docs are out of sync with actual implementation).

        Returns
        - cached_content_name - str - cached content name stored on google. (if found.)
        OR
        - None
        """

        _, url = self._get_token_and_url_context_caching(
            gemini_api_key=api_key,
            custom_llm_provider="gemini",
            api_base=api_base,
        )
        try:
            ## LOGGING
            logging_obj.pre_call(
                input="",
                api_key="",
                additional_args={
                    "complete_input_dict": {},
                    "api_base": url,
                    "headers": headers,
                },
            )

            resp = client.get(url=url, headers=headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                return None
            raise VertexAIError(
                status_code=e.response.status_code, message=e.response.text
            )
        except Exception as e:
            raise VertexAIError(status_code=500, message=str(e))
        raw_response = resp.json()
        logging_obj.post_call(original_response=raw_response)

        if "cachedContents" not in raw_response:
            return None

        all_cached_items = CachedContentListAllResponseBody(**raw_response)

        if "cachedContents" not in all_cached_items:
            return None

        for cached_item in all_cached_items["cachedContents"]:
            display_name = cached_item.get("displayName")
            if display_name is not None and display_name == cache_key:
                return cached_item.get("name")

        return None

    async def async_check_cache(
        self,
        cache_key: str,
        client: AsyncHTTPHandler,
        headers: dict,
        api_key: str,
        api_base: Optional[str],
        logging_obj: Logging,
    ) -> Optional[str]:
        """
        Checks if content already cached.

        Currently, checks cache list, for cache key == displayName, since Google doesn't let us set the name of the cache (their API docs are out of sync with actual implementation).

        Returns
        - cached_content_name - str - cached content name stored on google. (if found.)
        OR
        - None
        """

        _, url = self._get_token_and_url_context_caching(
            gemini_api_key=api_key,
            custom_llm_provider="gemini",
            api_base=api_base,
        )
        try:
            ## LOGGING
            logging_obj.pre_call(
                input="",
                api_key="",
                additional_args={
                    "complete_input_dict": {},
                    "api_base": url,
                    "headers": headers,
                },
            )

            resp = await client.get(url=url, headers=headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                return None
            raise VertexAIError(
                status_code=e.response.status_code, message=e.response.text
            )
        except Exception as e:
            raise VertexAIError(status_code=500, message=str(e))
        raw_response = resp.json()
        logging_obj.post_call(original_response=raw_response)

        if "cachedContents" not in raw_response:
            return None

        all_cached_items = CachedContentListAllResponseBody(**raw_response)

        if "cachedContents" not in all_cached_items:
            return None

        for cached_item in all_cached_items["cachedContents"]:
            display_name = cached_item.get("displayName")
            if display_name is not None and display_name == cache_key:
                return cached_item.get("name")

        return None

    def check_and_create_cache(
        self,
        messages: List[AllMessageValues],  # receives openai format messages
        optional_params: Dict[str, Any],
        api_key: str,
        api_base: Optional[str],
        model: str,
        client: Optional[HTTPHandler],
        timeout: Optional[Union[float, httpx.Timeout]],
        logging_obj: Logging,
        extra_headers: Optional[dict] = None,
    ) -> CacheSplitResult:
        """
        Receives
        - messages: List of dict - messages in the openai format

        Returns
        - messages - List[dict] - filtered list of messages in the openai format.
        - cached_content - str - the cache content id, to be passed in the gemini request body

        Follows - https://ai.google.dev/api/caching#request-body
        """

        cache_split_result = extract_cache_configuration(
            model=model,
            messages=messages,
            optional_params=optional_params,
        )

        if (
            cache_split_result.cache_request_body is None
            or cache_split_result.cached_content is not None
            or cache_split_result.cache_key is None
        ):
            return cache_split_result

        ## AUTHORIZATION ##
        token, url = self._get_token_and_url_context_caching(
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

        ## CHECK IF CACHED ALREADY
        google_cache_name = self.check_cache(
            cache_key=cache_split_result.cache_key,
            client=client,
            headers=headers,
            api_key=api_key,
            api_base=api_base,
            logging_obj=logging_obj,
        )
        if google_cache_name:
            return cache_split_result.with_cached_content(cached_content=google_cache_name)

        ## TRANSFORM REQUEST

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": cache_split_result.cache_request_body,
                "api_base": url,
                "headers": headers,
            },
        )

        try:
            response = client.post(
                url=url, headers=headers, json=cache_split_result.cache_request_body  # type: ignore
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
        return cache_split_result.with_cached_content(cached_content=cached_content_response_obj["name"])

    async def async_check_and_create_cache(
        self,
        messages: List[AllMessageValues],  # receives openai format messages
        optional_params: Dict[str, Any],
        api_key: str,
        api_base: Optional[str],
        model: str,
        client: Optional[AsyncHTTPHandler],
        timeout: Optional[Union[float, httpx.Timeout]],
        logging_obj: Logging,
        extra_headers: Optional[dict] = None,
    ) -> CacheSplitResult:
        """
        Receives
        - messages: List of dict - messages in the openai format

        Returns
        - messages - List[dict] - filtered list of messages in the openai format.
        - cached_content - str - the cache content id, to be passed in the gemini request body

        Follows - https://ai.google.dev/api/caching#request-body
        """

        cache_split_result = extract_cache_configuration(
            model=model,
            messages=messages,
            optional_params=optional_params,
        )

        if (
            cache_split_result.cache_request_body is None
            or cache_split_result.cached_content is not None
            or cache_split_result.cache_key is None
        ):
            return cache_split_result

        ## AUTHORIZATION ##
        token, url = self._get_token_and_url_context_caching(
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

        if client is None or not isinstance(client, AsyncHTTPHandler):
            client = get_async_httpx_client(
                params={"timeout": timeout}, llm_provider=litellm.LlmProviders.VERTEX_AI
            )
        else:
            client = client

        ## CHECK IF CACHED ALREADY
        google_cache_name = await self.async_check_cache(
            cache_key=cache_split_result.cache_key,
            client=client,
            headers=headers,
            api_key=api_key,
            api_base=api_base,
            logging_obj=logging_obj,
        )
        if google_cache_name:
            return cache_split_result.with_cached_content(cached_content=google_cache_name)

        ## TRANSFORM REQUEST

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": cache_split_result.cache_request_body,
                "api_base": url,
                "headers": headers,
            },
        )

        try:
            response = await client.post(
                url=url, headers=headers, json=cache_split_result.cache_request_body  # type: ignore
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
        return cache_split_result.with_cached_content(cached_content=cached_content_response_obj["name"])

    def get_cache(self):
        pass

    async def async_get_cache(self):
        pass
