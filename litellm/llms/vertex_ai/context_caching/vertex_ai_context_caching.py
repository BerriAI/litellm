from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import httpx
from pydantic import ConfigDict, RootModel, with_config
from typing_extensions import NotRequired, ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_logger
from litellm.caching.caching import Cache, LiteLLMCacheType
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.llms.openai.openai import AllMessageValues
from litellm.types.llms.openai import ChatCompletionToolParam
from litellm.utils import get_prompt_cache_min_tokens, is_prompt_caching_valid_prompt

from ..common_utils import VertexAIError, get_vertex_base_url
from ..vertex_llm_base import VertexBase, publisher_model_location
from .known_cached_contents import KnownCachedContents, known_cached_contents
from .transformation import (
    cached_messages_end_on_supported_turn,
    separate_cached_messages,
    transform_openai_messages_to_gemini_context_caching,
)

local_cache_obj: Final = Cache(type=LiteLLMCacheType.LOCAL)  # only used for calling 'get_cache_key' function

MAX_PAGINATION_PAGES: Final = 100  # Reasonable upper bound for pagination


@with_config(ConfigDict(extra="allow", strict=True, hide_input_in_errors=True))
class _CachedContentItem(TypedDict):
    displayName: ReadOnly[NotRequired[object]]
    name: ReadOnly[NotRequired[str | None]]
    expireTime: ReadOnly[NotRequired[str | None]]


@with_config(ConfigDict(extra="allow", strict=True, hide_input_in_errors=True))
class _CachedContentListBody(TypedDict):
    cachedContents: ReadOnly[NotRequired[Sequence[_CachedContentItem]]]
    nextPageToken: ReadOnly[NotRequired[str | None]]


@with_config(ConfigDict(extra="allow", strict=True, hide_input_in_errors=True))
class _CachedContentUsageMetadata(TypedDict):
    totalTokenCount: ReadOnly[NotRequired[int | None]]


@with_config(ConfigDict(extra="allow", strict=True, hide_input_in_errors=True))
class _CreatedCachedContent(TypedDict):
    name: ReadOnly[NotRequired[str | None]]
    expireTime: ReadOnly[NotRequired[str | None]]
    usageMetadata: ReadOnly[NotRequired[_CachedContentUsageMetadata | None]]


class _CachedContentListBodyRoot(RootModel[_CachedContentListBody]):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)


class _CreatedCachedContentRoot(RootModel[_CreatedCachedContent]):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)


_TOOL_PARAMS: Final = ("tools", "tool_choice")


def _sync_cache_client(timeout: float | httpx.Timeout | None) -> HTTPHandler:
    if timeout is None:
        return HTTPHandler()
    return HTTPHandler(timeout=httpx.Timeout(timeout) if isinstance(timeout, (float, int)) else timeout)


class _CachedContentsNotServedError(VertexAIError):
    pass


def _record_cache_creation_tokens(logging_obj: Logging, created: _CreatedCachedContent) -> None:
    tokens: Final = (created.get("usageMetadata") or {}).get("totalTokenCount") or 0
    logging_obj.vertex_ai_cache_creation_input_tokens = max(tokens, 0)  # rebind-ok: set provider count


def _wait_seconds(timeout: float | httpx.Timeout | None) -> float | None:
    match timeout:
        case httpx.Timeout():
            return timeout.read
        case _:
            return timeout


@dataclass(frozen=True, slots=True)
class _CacheTarget:
    model: str
    project: str | None
    location: str | None


def _cache_target(
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    api_base: str | None,
    model: str,
    vertex_project: str | None,
    vertex_location: str | None,
) -> _CacheTarget:
    model_location: Final = (
        None if custom_llm_provider == "gemini" or api_base is None else publisher_model_location(api_base)
    )
    if model_location is None:
        return _CacheTarget(model=model, project=vertex_project, location=vertex_location)
    return _CacheTarget(model=model_location.model, project=model_location.project, location=model_location.location)


def _prompt_cache_min_tokens(models: Sequence[str], custom_llm_provider: str) -> int:
    return max(
        get_prompt_cache_min_tokens(
            model if model.startswith(custom_llm_provider) else f"{custom_llm_provider}/{model}"
        )
        for model in models
    )


def _context_caching_skip_reason(
    cached_messages: Sequence[AllMessageValues],
    tools: Sequence[ChatCompletionToolParam] | None,
    deployment_model: str,
    cache_model: str,
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
) -> str | None:
    if not cached_messages_end_on_supported_turn(cached_messages):
        return (
            "Vertex AI context caching: cached message block ends on a model turn once "
            "system messages are extracted, which the cachedContents API rejects."
        )
    min_tokens: Final = _prompt_cache_min_tokens((deployment_model, cache_model), custom_llm_provider)
    if not is_prompt_caching_valid_prompt(
        model=cache_model,
        messages=list(cached_messages),
        tools=None if tools is None else list(tools),
        custom_llm_provider=custom_llm_provider,
        min_token_count=min_tokens,
    ):
        return f"Vertex AI context caching: cached content is below minimum token count ({min_tokens})."
    return None


class ContextCachingEndpoints(VertexBase):
    """
    Covers context caching endpoints for Vertex AI + Google AI Studio

    v0: covers Google AI Studio
    """

    def __init__(self, known_caches: KnownCachedContents = known_cached_contents) -> None:
        super().__init__()
        self._known_caches: Final = known_caches

    def _get_token_and_url_context_caching(
        self,
        gemini_api_key: str | None,
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
        api_base: str | None,
        vertex_project: str | None,
        vertex_location: str | None,
        vertex_auth_header: str | None,
        model: str | None = None,
    ) -> tuple[str | None, str]:
        """
        Internal function. Returns the token and url for the call.

        Handles logic if it's google ai studio vs. vertex ai.

        Returns
            token, url
        """
        auth_header: str | None
        if custom_llm_provider == "gemini":
            auth_header = {"x-goog-api-key": gemini_api_key}
            endpoint = "cachedContents"
            url = f"https://generativelanguage.googleapis.com/v1beta/{endpoint}"
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
            collection_endpoint=True,
        )

    def check_cache(
        self,
        cache_key: str,
        client: HTTPHandler,
        headers: dict[str, str],  # mutable-ok: passed through to the HTTP client, which takes a dict
        api_key: str,
        api_base: str | None,
        logging_obj: Logging,
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
        vertex_project: str | None,
        vertex_location: str | None,
        vertex_auth_header: str | None,
        model: str | None = None,
    ) -> str | None:
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
        scope: Final = KnownCachedContents.scope(
            base_url, custom_llm_provider, headers, vertex_project=vertex_project, vertex_location=vertex_location
        )
        known_name: Final = self._known_caches.get(scope, cache_key)
        if known_name is not None:
            return known_name

        page_token: str | None = None

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
                if e.response.status_code == 404 and custom_llm_provider == "gemini":
                    raise _CachedContentsNotServedError(status_code=404, message=e.response.text)
                raise VertexAIError(status_code=e.response.status_code, message=e.response.text)
            except Exception as e:
                raise VertexAIError(status_code=500, message=str(e))

            raw_response = _CachedContentListBodyRoot.model_validate(resp.json()).root
            logging_obj.post_call(original_response=raw_response)

            if "cachedContents" not in raw_response:
                return None

            # Check current page for matching cache_key
            for cached_item in raw_response["cachedContents"]:
                display_name = cached_item.get("displayName")
                if display_name is None or display_name != cache_key:
                    continue
                expire_time = cached_item.get("expireTime")
                if not self._known_caches.has_not_expired(expire_time):
                    continue
                name = cached_item.get("name")
                self._known_caches.remember(scope, cache_key, name, expire_time)
                return name

            # Check if there are more pages
            page_token = raw_response.get("nextPageToken")
            if not page_token:
                # No more pages, cache not found
                break

        return None

    async def async_check_cache(
        self,
        cache_key: str,
        client: AsyncHTTPHandler,
        headers: dict[str, str],  # mutable-ok: passed through to the HTTP client, which takes a dict
        api_key: str,
        api_base: str | None,
        logging_obj: Logging,
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
        vertex_project: str | None,
        vertex_location: str | None,
        vertex_auth_header: str | None,
        model: str | None = None,
    ) -> str | None:
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
        scope: Final = KnownCachedContents.scope(
            base_url, custom_llm_provider, headers, vertex_project=vertex_project, vertex_location=vertex_location
        )
        known_name: Final = self._known_caches.get(scope, cache_key)
        if known_name is not None:
            return known_name

        page_token: str | None = None

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
                if e.response.status_code == 404 and custom_llm_provider == "gemini":
                    raise _CachedContentsNotServedError(status_code=404, message=e.response.text)
                raise VertexAIError(status_code=e.response.status_code, message=e.response.text)
            except Exception as e:
                raise VertexAIError(status_code=500, message=str(e))

            raw_response = _CachedContentListBodyRoot.model_validate(resp.json()).root
            logging_obj.post_call(original_response=raw_response)

            if "cachedContents" not in raw_response:
                return None

            # Check current page for matching cache_key
            for cached_item in raw_response["cachedContents"]:
                display_name = cached_item.get("displayName")
                if display_name is None or display_name != cache_key:
                    continue
                expire_time = cached_item.get("expireTime")
                if not self._known_caches.has_not_expired(expire_time):
                    continue
                name = cached_item.get("name")
                self._known_caches.remember(scope, cache_key, name, expire_time)
                return name

            # Check if there are more pages
            page_token = raw_response.get("nextPageToken")
            if not page_token:
                # No more pages, cache not found
                break

        return None

    def check_and_create_cache(
        self,
        messages: list[AllMessageValues],  # receives openai format messages
        optional_params: dict,  # cache the tools if present, in case cache content exists in messages
        api_key: str,
        api_base: str | None,
        model: str,
        client: HTTPHandler | None,
        timeout: float | httpx.Timeout | None,
        logging_obj: Logging,
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
        vertex_project: str | None,
        vertex_location: str | None,
        vertex_auth_header: str | None,
        extra_headers: dict | None = None,
        cached_content: str | None = None,
    ) -> tuple[list[AllMessageValues], dict, str | None]:
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

        target: Final = _cache_target(custom_llm_provider, api_base, model, vertex_project, vertex_location)
        tools: Final = optional_params.get("tools")
        tool_choice: Final = optional_params.get("tool_choice")
        skip_reason: Final = _context_caching_skip_reason(
            cached_messages, tools, model, target.model, custom_llm_provider
        )
        if skip_reason is not None:
            verbose_logger.debug("%s Skipping context caching.", skip_reason)
            return messages, optional_params, None

        ## AUTHORIZATION ##
        token, url = self._get_token_and_url_context_caching(
            gemini_api_key=api_key,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            vertex_project=target.project,
            vertex_location=target.location,
            vertex_auth_header=vertex_auth_header,
            model=target.model,
        )

        headers: Final = {
            "Content-Type": "application/json",
        }
        if isinstance(token, dict):
            headers.update(token)
        elif token is not None:
            headers["Authorization"] = f"Bearer {token}"
        if extra_headers is not None:
            headers.update(extra_headers)

        http_client: Final = client if isinstance(client, HTTPHandler) else _sync_cache_client(timeout)

        ## CHECK IF CACHED ALREADY
        generated_cache_key: Final = local_cache_obj.get_cache_key(
            messages=cached_messages, tools=tools, tool_choice=tool_choice, model=target.model
        )
        try:
            google_cache_name: Final = self.check_cache(
                cache_key=generated_cache_key,
                client=http_client,
                headers=headers,
                api_key=api_key,
                api_base=api_base,
                logging_obj=logging_obj,
                custom_llm_provider=custom_llm_provider,
                vertex_project=target.project,
                vertex_location=target.location,
                vertex_auth_header=vertex_auth_header,
                model=target.model,
            )
        except _CachedContentsNotServedError:
            verbose_logger.debug("Gemini context caching: %s answers 404. Skipping context caching.", url)
            return messages, optional_params, None
        for tool_param in _TOOL_PARAMS:
            optional_params.pop(tool_param, None)
        if google_cache_name:
            return non_cached_messages, optional_params, google_cache_name

        ## TRANSFORM REQUEST
        cached_content_request_body: Final = transform_openai_messages_to_gemini_context_caching(
            model=target.model,
            messages=cached_messages,
            cache_key=generated_cache_key,
            custom_llm_provider=custom_llm_provider,
            vertex_project=target.project,
            vertex_location=target.location,
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
            response: Final = http_client.post(
                url=url,
                headers=headers,
                json=cached_content_request_body,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code: Final = err.response.status_code
            raise VertexAIError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
            raise VertexAIError(status_code=408, message="Timeout error occurred.")

        raw_response_cached: Final = _CreatedCachedContentRoot.model_validate(response.json()).root
        _record_cache_creation_tokens(logging_obj, raw_response_cached)
        self._known_caches.remember(
            KnownCachedContents.scope(
                url, custom_llm_provider, headers, vertex_project=target.project, vertex_location=target.location
            ),
            generated_cache_key,
            raw_response_cached.get("name"),
            raw_response_cached.get("expireTime"),
        )
        return (
            non_cached_messages,
            optional_params,
            raw_response_cached.get("name"),
        )

    async def async_check_and_create_cache(
        self,
        messages: list[AllMessageValues],  # receives openai format messages
        optional_params: dict,  # cache the tools if present, in case cache content exists in messages
        api_key: str,
        api_base: str | None,
        model: str,
        client: AsyncHTTPHandler | None,
        timeout: float | httpx.Timeout | None,
        logging_obj: Logging,
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
        vertex_project: str | None,
        vertex_location: str | None,
        vertex_auth_header: str | None,
        extra_headers: dict | None = None,
        cached_content: str | None = None,
    ) -> tuple[list[AllMessageValues], dict, str | None]:
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

        target: Final = _cache_target(custom_llm_provider, api_base, model, vertex_project, vertex_location)
        tools: Final = optional_params.get("tools")
        tool_choice: Final = optional_params.get("tool_choice")
        skip_reason: Final = _context_caching_skip_reason(
            cached_messages, tools, model, target.model, custom_llm_provider
        )
        if skip_reason is not None:
            verbose_logger.debug("%s Skipping context caching.", skip_reason)
            return messages, optional_params, None

        ## AUTHORIZATION ##
        token, url = self._get_token_and_url_context_caching(
            gemini_api_key=api_key,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            vertex_project=target.project,
            vertex_location=target.location,
            vertex_auth_header=vertex_auth_header,
            model=target.model,
        )

        headers: Final = {
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
        generated_cache_key: Final = local_cache_obj.get_cache_key(
            messages=cached_messages, tools=tools, tool_choice=tool_choice, model=target.model
        )
        scope: Final = KnownCachedContents.scope(
            url, custom_llm_provider, headers, vertex_project=target.project, vertex_location=target.location
        )

        async def lookup_or_create() -> str | None:
            google_cache_name: Final = await self.async_check_cache(
                cache_key=generated_cache_key,
                client=client,
                headers=headers,
                api_key=api_key,
                api_base=api_base,
                logging_obj=logging_obj,
                custom_llm_provider=custom_llm_provider,
                vertex_project=target.project,
                vertex_location=target.location,
                vertex_auth_header=vertex_auth_header,
                model=target.model,
            )

            if google_cache_name:
                return google_cache_name

            ## TRANSFORM REQUEST
            cached_content_request_body: Final = transform_openai_messages_to_gemini_context_caching(
                model=target.model,
                messages=cached_messages,
                cache_key=generated_cache_key,
                custom_llm_provider=custom_llm_provider,
                vertex_project=target.project,
                vertex_location=target.location,
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
                response: Final = await client.post(
                    url=url,
                    headers=headers,
                    json=cached_content_request_body,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as err:
                error_code: Final = err.response.status_code
                raise VertexAIError(status_code=error_code, message=err.response.text)
            except httpx.TimeoutException:
                raise VertexAIError(status_code=408, message="Timeout error occurred.")

            raw_response_cached: Final = _CreatedCachedContentRoot.model_validate(response.json()).root
            _record_cache_creation_tokens(logging_obj, raw_response_cached)
            self._known_caches.remember(
                scope, generated_cache_key, raw_response_cached.get("name"), raw_response_cached.get("expireTime")
            )
            return raw_response_cached.get("name")

        try:
            cache_name: Final = await self._known_caches.single_flight(
                scope, generated_cache_key, lookup_or_create, _wait_seconds(timeout)
            )
        except _CachedContentsNotServedError:
            verbose_logger.debug("Gemini context caching: %s answers 404. Skipping context caching.", url)
            return messages, optional_params, None
        for tool_param in _TOOL_PARAMS:
            optional_params.pop(tool_param, None)
        return non_cached_messages, optional_params, cache_name

    def get_cache(self):
        pass

    async def async_get_cache(self):
        pass
