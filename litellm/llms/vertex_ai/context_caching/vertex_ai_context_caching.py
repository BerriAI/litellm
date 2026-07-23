from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import httpx

import litellm
from litellm.caching.caching import Cache, LiteLLMCacheType
from litellm.constants import MINIMUM_PROMPT_CACHE_TOKEN_COUNT
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm._logging import verbose_logger
from litellm.llms.openai.openai import AllMessageValues
from litellm.utils import is_prompt_caching_valid_prompt
from litellm.types.llms.vertex_ai import (
    CachedContent,
    CachedContentListAllResponseBody,
    VertexAICachedContentResponseObject,
)

from ..common_utils import VertexAIError, get_vertex_base_url
from ..vertex_llm_base import VertexBase
from .id_cache import (
    ResolvedCacheId,
    invalidate_cache_id,
    lookup_cache_id,
    make_cache_id_key,
    store_cache_id,
)
from .transformation import (
    separate_cached_messages,
    transform_openai_messages_to_gemini_context_caching,
)

local_cache_obj = Cache(type=LiteLLMCacheType.LOCAL)  # only used for calling 'get_cache_key' function

MAX_PAGINATION_PAGES = 100  # Reasonable upper bound for pagination


@dataclass(frozen=True, slots=True)
class _PageMatch:
    # A displayName match was found on this page; stop paginating. ``resolved`` is None
    # when the matched item had no usable ``name`` (treat as miss, but still stop early,
    # since displayName is unique so no later page can match).
    resolved: Optional[ResolvedCacheId]


def _find_matching_cache(cached_items: list[CachedContent], cache_key: str) -> Optional[_PageMatch]:
    for cached_item in cached_items:
        display_name = cached_item.get("displayName")
        if display_name is not None and display_name == cache_key:
            name = cached_item.get("name")
            resolved = None if name is None else ResolvedCacheId(name=name, expire_time=cached_item.get("expireTime"))
            return _PageMatch(resolved=resolved)
    return None


_CONTEXT_CACHE_ID_KEY_FIELD = "_vertex_context_cache_id_key"


def _stash_context_cache_id_key(logging_obj: Logging, key: str) -> None:
    """Record the resolving id-cache key on the logging object so a downstream NOT_FOUND
    (server-side deleted/revoked cachedContent) can invalidate it via
    `maybe_invalidate_stale_context_cache`. No-op when the feature is off or
    model_call_details is not a dict."""
    if not litellm.enable_vertex_context_cache_id_caching:
        return
    model_call_details = getattr(logging_obj, "model_call_details", None)
    if isinstance(model_call_details, dict):
        model_call_details[_CONTEXT_CACHE_ID_KEY_FIELD] = key


def maybe_invalidate_stale_context_cache(logging_obj: Logging, status_code: int, response_text: str) -> None:
    """If a request's resolved cachedContent is rejected downstream because it was
    deleted/revoked server-side, drop the cached id so the next request re-resolves it.

    Vertex reports a deleted cache as HTTP 400 INVALID_ARGUMENT ("Invalid resource state
    for cache content <id>"), not 404, so we key off the *served id* appearing in the error
    body rather than the status text: it names our exact id, so unrelated 4xx never evict a
    live entry, and it doesn't matter which 4xx wording Vertex uses."""
    if status_code not in (400, 403, 404):
        return
    model_call_details = getattr(logging_obj, "model_call_details", None)
    if not isinstance(model_call_details, dict):
        return
    key = model_call_details.get(_CONTEXT_CACHE_ID_KEY_FIELD)
    if not isinstance(key, str):
        return
    served_name = lookup_cache_id(key)
    if served_name is None:
        return
    served_id = served_name.rsplit("/", 1)[-1]
    if served_id not in response_text and served_name not in response_text:
        return
    invalidate_cache_id(key)


class ContextCachingEndpoints(VertexBase):
    """
    Covers context caching endpoints for Vertex AI + Google AI Studio

    v0: covers Google AI Studio
    """

    def __init__(self) -> None:
        super().__init__()

    def _get_token_and_url_context_caching(
        self,
        gemini_api_key: Optional[str],
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
        api_base: Optional[str],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_auth_header: Optional[str],
        model: Optional[str] = None,
    ) -> Tuple[Optional[str], str]:
        """
        Internal function. Returns the token and url for the call.

        Handles logic if it's google ai studio vs. vertex ai.

        Returns
            token, url
        """
        auth_header: Optional[str]
        if custom_llm_provider == "gemini":
            auth_header = {"x-goog-api-key": gemini_api_key}  # type: ignore[assignment]
            endpoint = "cachedContents"
            url = "https://generativelanguage.googleapis.com/v1beta/{}".format(endpoint)
        elif custom_llm_provider == "vertex_ai":
            auth_header = vertex_auth_header
            endpoint = "cachedContents"
            base_url = get_vertex_base_url(vertex_location)
            url = f"{base_url}/v1/projects/{vertex_project}/locations/{vertex_location}/{endpoint}"
        else:
            auth_header = vertex_auth_header
            endpoint = "cachedContents"
            base_url = get_vertex_base_url(vertex_location)
            url = f"{base_url}/v1beta1/projects/{vertex_project}/locations/{vertex_location}/{endpoint}"

        return self._check_custom_proxy(
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            gemini_api_key=gemini_api_key,
            endpoint=endpoint,
            stream=None,
            auth_header=auth_header,
            url=url,
            model=model,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_api_version=("v1beta1" if custom_llm_provider == "vertex_ai_beta" else "v1"),
        )

    def check_cache(
        self,
        cache_key: str,
        client: HTTPHandler,
        headers: dict,
        api_key: str,
        api_base: Optional[str],
        logging_obj: Logging,
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_auth_header: Optional[str],
        model: Optional[str] = None,
    ) -> Optional[ResolvedCacheId]:
        """
        Checks if content already cached.

        Currently, checks cache list, for cache key == displayName, since Google doesn't let us set the name of the cache (their API docs are out of sync with actual implementation).

        Returns
        - cached_content_name - str - cached content name stored on google. (if found.)
        OR
        - None
        """

        _, base_url = self._get_token_and_url_context_caching(
            gemini_api_key=api_key,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_auth_header=vertex_auth_header,
            model=model,
        )

        page_token: Optional[str] = None

        # Iterate through all pages
        for _ in range(MAX_PAGINATION_PAGES):
            # Build URL with pagination token if present
            if page_token:
                separator = "&" if "?" in base_url else "?"
                url = f"{base_url}{separator}pageToken={page_token}"
            else:
                url = base_url

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
                raise VertexAIError(status_code=e.response.status_code, message=e.response.text)
            except Exception as e:
                raise VertexAIError(status_code=500, message=str(e))

            raw_response = resp.json()
            logging_obj.post_call(original_response=raw_response)

            if "cachedContents" not in raw_response:
                return None

            all_cached_items = CachedContentListAllResponseBody(**raw_response)

            if "cachedContents" not in all_cached_items:
                return None

            match = _find_matching_cache(all_cached_items["cachedContents"], cache_key)
            if match is not None:
                return match.resolved

            # Check if there are more pages
            page_token = all_cached_items.get("nextPageToken")
            if not page_token:
                # No more pages, cache not found
                break

        return None

    async def async_check_cache(
        self,
        cache_key: str,
        client: AsyncHTTPHandler,
        headers: dict,
        api_key: str,
        api_base: Optional[str],
        logging_obj: Logging,
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_auth_header: Optional[str],
        model: Optional[str] = None,
    ) -> Optional[ResolvedCacheId]:
        """
        Checks if content already cached.

        Currently, checks cache list, for cache key == displayName, since Google doesn't let us set the name of the cache (their API docs are out of sync with actual implementation).

        Returns
        - cached_content_name - str - cached content name stored on google. (if found.)
        OR
        - None
        """

        _, base_url = self._get_token_and_url_context_caching(
            gemini_api_key=api_key,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_auth_header=vertex_auth_header,
            model=model,
        )

        page_token: Optional[str] = None

        # Iterate through all pages
        for _ in range(MAX_PAGINATION_PAGES):
            # Build URL with pagination token if present
            if page_token:
                separator = "&" if "?" in base_url else "?"
                url = f"{base_url}{separator}pageToken={page_token}"
            else:
                url = base_url

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
                raise VertexAIError(status_code=e.response.status_code, message=e.response.text)
            except Exception as e:
                raise VertexAIError(status_code=500, message=str(e))

            raw_response = resp.json()
            logging_obj.post_call(original_response=raw_response)

            if "cachedContents" not in raw_response:
                return None

            all_cached_items = CachedContentListAllResponseBody(**raw_response)

            if "cachedContents" not in all_cached_items:
                return None

            match = _find_matching_cache(all_cached_items["cachedContents"], cache_key)
            if match is not None:
                return match.resolved

            # Check if there are more pages
            page_token = all_cached_items.get("nextPageToken")
            if not page_token:
                # No more pages, cache not found
                break

        return None

    def check_and_create_cache(
        self,
        messages: List[AllMessageValues],  # receives openai format messages
        optional_params: dict,  # cache the tools if present, in case cache content exists in messages
        api_key: str,
        api_base: Optional[str],
        model: str,
        client: Optional[HTTPHandler],
        timeout: Optional[Union[float, httpx.Timeout]],
        logging_obj: Logging,
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_auth_header: Optional[str],
        extra_headers: Optional[dict] = None,
        cached_content: Optional[str] = None,
    ) -> Tuple[List[AllMessageValues], dict, Optional[str]]:
        """
        Receives
        - messages: List of dict - messages in the openai format

        Returns
        - messages - List[dict] - filtered list of messages in the openai format.
        - cached_content - str - the cache content id, to be passed in the gemini request body

        Follows - https://ai.google.dev/api/caching#request-body
        """
        if cached_content is not None:
            return messages, optional_params, cached_content

        cached_messages, non_cached_messages = separate_cached_messages(messages=messages)

        if len(cached_messages) == 0:
            return messages, optional_params, None

        # Gemini requires a minimum of 1024 tokens for context caching.
        # Skip caching if the cached content is too small to avoid API errors.
        if not is_prompt_caching_valid_prompt(
            model=model,
            messages=cached_messages,
            custom_llm_provider=custom_llm_provider,
        ):
            verbose_logger.debug(
                "Vertex AI context caching: cached content is below minimum token "
                "count (%d). Skipping context caching.",
                MINIMUM_PROMPT_CACHE_TOKEN_COUNT,
            )
            return messages, optional_params, None

        tools = optional_params.pop("tools", None)
        tool_choice = optional_params.pop("tool_choice", None)

        ## AUTHORIZATION ##
        token, url = self._get_token_and_url_context_caching(
            gemini_api_key=api_key,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_auth_header=vertex_auth_header,
            model=model,
        )

        headers = {
            "Content-Type": "application/json",
        }
        if isinstance(token, dict):
            headers.update(token)
        elif token is not None:
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
        generated_cache_key = local_cache_obj.get_cache_key(
            messages=cached_messages, tools=tools, tool_choice=tool_choice, model=model
        )
        id_cache_key = make_cache_id_key(
            content_key=generated_cache_key,
            custom_llm_provider=custom_llm_provider,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            api_base=api_base,
            api_key=api_key,
        )
        _stash_context_cache_id_key(logging_obj, id_cache_key)
        cached_id = lookup_cache_id(id_cache_key)
        if cached_id is not None:
            return non_cached_messages, optional_params, cached_id
        resolved_cache = self.check_cache(
            cache_key=generated_cache_key,
            client=client,
            headers=headers,
            api_key=api_key,
            api_base=api_base,
            logging_obj=logging_obj,
            custom_llm_provider=custom_llm_provider,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_auth_header=vertex_auth_header,
            model=model,
        )
        if resolved_cache is not None:
            store_cache_id(id_cache_key, resolved_cache.name, resolved_cache.expire_time)
            return non_cached_messages, optional_params, resolved_cache.name

        ## TRANSFORM REQUEST
        cached_content_request_body = transform_openai_messages_to_gemini_context_caching(
            model=model,
            messages=cached_messages,
            cache_key=generated_cache_key,
            custom_llm_provider=custom_llm_provider,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
        )

        cached_content_request_body["tools"] = tools
        if tool_choice is not None:
            cached_content_request_body["toolConfig"] = tool_choice

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
                url=url,
                headers=headers,
                json=cached_content_request_body,  # type: ignore
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
        created_name = cached_content_response_obj["name"]
        store_cache_id(id_cache_key, created_name, raw_response_cached.get("expireTime"))
        return (
            non_cached_messages,
            optional_params,
            created_name,
        )

    async def async_check_and_create_cache(
        self,
        messages: List[AllMessageValues],  # receives openai format messages
        optional_params: dict,  # cache the tools if present, in case cache content exists in messages
        api_key: str,
        api_base: Optional[str],
        model: str,
        client: Optional[AsyncHTTPHandler],
        timeout: Optional[Union[float, httpx.Timeout]],
        logging_obj: Logging,
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_auth_header: Optional[str],
        extra_headers: Optional[dict] = None,
        cached_content: Optional[str] = None,
    ) -> Tuple[List[AllMessageValues], dict, Optional[str]]:
        """
        Receives
        - messages: List of dict - messages in the openai format

        Returns
        - messages - List[dict] - filtered list of messages in the openai format.
        - cached_content - str - the cache content id, to be passed in the gemini request body

        Follows - https://ai.google.dev/api/caching#request-body
        """
        if cached_content is not None:
            return messages, optional_params, cached_content

        cached_messages, non_cached_messages = separate_cached_messages(messages=messages)

        if len(cached_messages) == 0:
            return messages, optional_params, None

        # Gemini requires a minimum of 1024 tokens for context caching.
        # Skip caching if the cached content is too small to avoid API errors.
        if not is_prompt_caching_valid_prompt(
            model=model,
            messages=cached_messages,
            custom_llm_provider=custom_llm_provider,
        ):
            verbose_logger.debug(
                "Vertex AI context caching: cached content is below minimum token "
                "count (%d). Skipping context caching.",
                MINIMUM_PROMPT_CACHE_TOKEN_COUNT,
            )
            return messages, optional_params, None

        tools = optional_params.pop("tools", None)
        tool_choice = optional_params.pop("tool_choice", None)

        ## AUTHORIZATION ##
        token, url = self._get_token_and_url_context_caching(
            gemini_api_key=api_key,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_auth_header=vertex_auth_header,
            model=model,
        )

        headers = {
            "Content-Type": "application/json",
        }
        if isinstance(token, dict):
            headers.update(token)
        elif token is not None:
            headers["Authorization"] = f"Bearer {token}"
        if extra_headers is not None:
            headers.update(extra_headers)

        if client is None or not isinstance(client, AsyncHTTPHandler):
            client = get_async_httpx_client(params={"timeout": timeout}, llm_provider=litellm.LlmProviders.VERTEX_AI)
        else:
            client = client

        ## CHECK IF CACHED ALREADY
        generated_cache_key = local_cache_obj.get_cache_key(
            messages=cached_messages, tools=tools, tool_choice=tool_choice, model=model
        )
        id_cache_key = make_cache_id_key(
            content_key=generated_cache_key,
            custom_llm_provider=custom_llm_provider,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            api_base=api_base,
            api_key=api_key,
        )
        _stash_context_cache_id_key(logging_obj, id_cache_key)
        cached_id = lookup_cache_id(id_cache_key)
        if cached_id is not None:
            return non_cached_messages, optional_params, cached_id
        resolved_cache = await self.async_check_cache(
            cache_key=generated_cache_key,
            client=client,
            headers=headers,
            api_key=api_key,
            api_base=api_base,
            logging_obj=logging_obj,
            custom_llm_provider=custom_llm_provider,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_auth_header=vertex_auth_header,
            model=model,
        )

        if resolved_cache is not None:
            store_cache_id(id_cache_key, resolved_cache.name, resolved_cache.expire_time)
            return non_cached_messages, optional_params, resolved_cache.name

        ## TRANSFORM REQUEST
        cached_content_request_body = transform_openai_messages_to_gemini_context_caching(
            model=model,
            messages=cached_messages,
            cache_key=generated_cache_key,
            custom_llm_provider=custom_llm_provider,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
        )

        cached_content_request_body["tools"] = tools
        if tool_choice is not None:
            cached_content_request_body["toolConfig"] = tool_choice

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
            response = await client.post(
                url=url,
                headers=headers,
                json=cached_content_request_body,  # type: ignore
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
        created_name = cached_content_response_obj["name"]
        store_cache_id(id_cache_key, created_name, raw_response_cached.get("expireTime"))
        return (
            non_cached_messages,
            optional_params,
            created_name,
        )

    def get_cache(self):
        pass

    async def async_get_cache(self):
        pass
