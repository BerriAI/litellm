from collections.abc import Mapping, Sequence
from functools import partial
from typing import Final, Literal

import httpx
from pydantic import ConfigDict, RootModel, with_config
from typing_extensions import NotRequired, ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_logger
from litellm.caching.caching import Cache, LiteLLMCacheType
from litellm.constants import MINIMUM_PROMPT_CACHE_TOKEN_COUNT
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.prompt_templates.image_handling import (
    async_inline_remote_media,
    has_remote_media,
    inline_remote_media,
)
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import (
    split_leading_system_run,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.llms.openai.openai import AllMessageValues
from litellm.utils import is_prompt_caching_valid_prompt

from ..common_utils import (
    GEMINI_FILES_API_URI_PREFIXES,
    VertexAIError,
    get_supports_system_message,
    get_vertex_base_url,
)
from ..gemini.transformation import (
    openai_messages_may_need_sync_gcs_metadata_fetch,
    reject_system_message_videos,
    should_inline_remote_media,
)
from ..vertex_llm_base import VertexBase
from .transformation import (
    cached_block_ttl,
    cached_messages_end_on_supported_turn,
    extract_ttl_from_cached_messages,
    reject_unusable_tool_call_arguments,
    separate_cached_messages,
    transform_openai_messages_to_gemini_context_caching,
)

local_cache_obj: Final = Cache(type=LiteLLMCacheType.LOCAL)  # only used for calling 'get_cache_key' function

MAX_PAGINATION_PAGES: Final = 100  # Reasonable upper bound for pagination


def _cache_key(
    cached_messages: Sequence[AllMessageValues], ttl: str | None, tools: object, tool_choice: object, model: str
) -> str:
    key: Final = local_cache_obj.get_cache_key(
        messages=cached_messages, tools=tools, tool_choice=tool_choice, model=model
    )
    return key if ttl == extract_ttl_from_cached_messages(cached_messages) else f"{key}-{ttl}"


@with_config(ConfigDict(extra="allow", strict=True, hide_input_in_errors=True))
class _CachedContentItem(TypedDict):
    displayName: ReadOnly[NotRequired[object]]
    name: ReadOnly[NotRequired[str | None]]


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
    usageMetadata: ReadOnly[NotRequired[_CachedContentUsageMetadata | None]]


class _CachedContentListBodyRoot(RootModel[_CachedContentListBody]):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)


class _CreatedCachedContentRoot(RootModel[_CreatedCachedContent]):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)


def _record_cache_creation_tokens(logging_obj: Logging, created: _CreatedCachedContent) -> None:
    tokens: Final = (created.get("usageMetadata") or {}).get("totalTokenCount") or 0
    logging_obj.vertex_ai_cache_creation_input_tokens = max(tokens, 0)  # rebind-ok: set provider count


def _cached_prefix_covers_whole_request(uncached_messages: Sequence[AllMessageValues]) -> bool:
    if uncached_messages:
        return False
    verbose_logger.debug(
        "Vertex AI context caching: the cached prefix covers the whole request, and an explicit cache only "
        "serves that exact prefix, so implicit caching serves it without a storage fee. Skipping context caching."
    )
    return True


def _conversation_turns_start(
    cached_messages: Sequence[AllMessageValues],
    model: str,
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
) -> int:
    if not get_supports_system_message(model=model, custom_llm_provider=custom_llm_provider):
        return 0
    leading_system_run, _ = split_leading_system_run(cached_messages)
    return len(leading_system_run)


def _cached_turns_link_web_media(
    cached_messages: Sequence[AllMessageValues],
    model: str,
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
) -> bool:
    should_inline: Final = should_inline_remote_media(custom_llm_provider)
    turns_start: Final = _conversation_turns_start(cached_messages, model, custom_llm_provider)
    if not has_remote_media(
        cached_messages[turns_start:],
        lambda media: not should_inline(media) and not media.url.startswith(GEMINI_FILES_API_URI_PREFIXES),
    ):
        return False
    verbose_logger.debug(
        "Vertex AI context caching: a cached turn links web media that Gemini fetches itself, which the "
        "cachedContents API does not take. Skipping context caching."
    )
    return True


def _inline_cached_turns(
    cached_messages: Sequence[AllMessageValues],
    model: str,
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
) -> tuple[AllMessageValues, ...]:
    turns_start: Final = _conversation_turns_start(cached_messages, model, custom_llm_provider)
    turns: Final = inline_remote_media(
        list(cached_messages[turns_start:]), should_inline=should_inline_remote_media(custom_llm_provider)
    )
    return (*cached_messages[:turns_start], *turns)


async def _async_inline_cached_turns(
    cached_messages: Sequence[AllMessageValues],
    model: str,
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
) -> tuple[AllMessageValues, ...]:
    turns_start: Final = _conversation_turns_start(cached_messages, model, custom_llm_provider)
    turns: Final = await async_inline_remote_media(
        list(cached_messages[turns_start:]), should_inline=should_inline_remote_media(custom_llm_provider)
    )
    return (*cached_messages[:turns_start], *turns)


def _reject_system_message_videos_gemini_would_drop(
    messages: Sequence[AllMessageValues],
    supports_system_message: bool,
    drop_params: bool,
    model: str,
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
) -> None:
    if supports_system_message:
        reject_system_message_videos(messages, drop_params, model, custom_llm_provider)


class ContextCachingEndpoints(VertexBase):
    """
    Covers context caching endpoints for Vertex AI + Google AI Studio

    v0: covers Google AI Studio
    """

    def __init__(self) -> None:
        super().__init__()

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
        )

    def check_cache(
        self,
        cache_key: str,
        client: HTTPHandler,
        headers: dict,
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
                if display_name is not None and display_name == cache_key:
                    return cached_item.get("name")

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
        headers: dict,
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
                if display_name is not None and display_name == cache_key:
                    return cached_item.get("name")

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
        drop_params: bool = False,
        litellm_params: Mapping[str, object] | None = None,
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

        supports_system_message: Final = get_supports_system_message(
            model=model, custom_llm_provider=custom_llm_provider
        )
        cached_messages, non_cached_messages = separate_cached_messages(
            messages=messages, supports_system_message=supports_system_message
        )

        if (
            len(cached_messages) == 0
            or _cached_prefix_covers_whole_request(messages[len(cached_messages) :])
            or _cached_turns_link_web_media(cached_messages, model, custom_llm_provider)
        ):
            return messages, optional_params, None

        if not cached_messages_end_on_supported_turn(cached_messages):
            verbose_logger.debug(
                "Vertex AI context caching: cached message block ends on a model turn once "
                "system messages are extracted, which the cachedContents API rejects. "
                "Skipping context caching."
            )
            return messages, optional_params, None

        # Gemini requires a minimum of 1024 tokens for context caching.
        # Skip caching if the cached content is too small to avoid API errors.
        if not is_prompt_caching_valid_prompt(
            model=model,
            messages=cached_messages,
            tools=optional_params.get("tools"),
            custom_llm_provider=custom_llm_provider,
        ):
            verbose_logger.debug(
                "Vertex AI context caching: cached content is below minimum token "
                "count (%d). Skipping context caching.",
                MINIMUM_PROMPT_CACHE_TOKEN_COUNT,
            )
            return messages, optional_params, None
        reject_unusable_tool_call_arguments(cached_messages, model, custom_llm_provider, drop_params)
        _reject_system_message_videos_gemini_would_drop(
            messages, supports_system_message, drop_params, model, custom_llm_provider
        )

        tools: Final = optional_params.pop("tools", None)
        tool_choice: Final = optional_params.pop("tool_choice", None)

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

        headers: Final = {
            "Content-Type": "application/json",
        }
        if isinstance(token, dict):
            headers.update(token)
        elif token is not None:
            headers["Authorization"] = f"Bearer {token}"
        if extra_headers is not None:
            headers.update(extra_headers)

        if client is None or not isinstance(client, HTTPHandler):
            _params: Final = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            client = HTTPHandler(**_params)
        else:
            client = client

        ## CHECK IF CACHED ALREADY
        ttl: Final = cached_block_ttl(messages)
        generated_cache_key: Final = _cache_key(cached_messages, ttl, tools, tool_choice, model)
        google_cache_name: Final = self.check_cache(
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
        if google_cache_name:
            return non_cached_messages, optional_params, google_cache_name

        ## TRANSFORM REQUEST
        cached_content_request_body: Final = transform_openai_messages_to_gemini_context_caching(
            model=model,
            messages=list(_inline_cached_turns(cached_messages, model, custom_llm_provider)),
            cache_key=generated_cache_key,
            ttl=ttl,
            custom_llm_provider=custom_llm_provider,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            drop_params=drop_params,
            litellm_params=litellm_params,
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
            response: Final = client.post(
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
        drop_params: bool = False,
        litellm_params: Mapping[str, object] | None = None,
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

        supports_system_message: Final = get_supports_system_message(
            model=model, custom_llm_provider=custom_llm_provider
        )
        cached_messages, non_cached_messages = separate_cached_messages(
            messages=messages, supports_system_message=supports_system_message
        )

        if (
            len(cached_messages) == 0
            or _cached_prefix_covers_whole_request(messages[len(cached_messages) :])
            or _cached_turns_link_web_media(cached_messages, model, custom_llm_provider)
        ):
            return messages, optional_params, None

        if not cached_messages_end_on_supported_turn(cached_messages):
            verbose_logger.debug(
                "Vertex AI context caching: cached message block ends on a model turn once "
                "system messages are extracted, which the cachedContents API rejects. "
                "Skipping context caching."
            )
            return messages, optional_params, None

        # Gemini requires a minimum of 1024 tokens for context caching.
        # Skip caching if the cached content is too small to avoid API errors.
        if not is_prompt_caching_valid_prompt(
            model=model,
            messages=cached_messages,
            tools=optional_params.get("tools"),
            custom_llm_provider=custom_llm_provider,
        ):
            verbose_logger.debug(
                "Vertex AI context caching: cached content is below minimum token "
                "count (%d). Skipping context caching.",
                MINIMUM_PROMPT_CACHE_TOKEN_COUNT,
            )
            return messages, optional_params, None
        reject_unusable_tool_call_arguments(cached_messages, model, custom_llm_provider, drop_params)
        _reject_system_message_videos_gemini_would_drop(
            messages, supports_system_message, drop_params, model, custom_llm_provider
        )

        tools: Final = optional_params.pop("tools", None)
        tool_choice: Final = optional_params.pop("tool_choice", None)

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
        ttl: Final = cached_block_ttl(messages)
        generated_cache_key: Final = _cache_key(cached_messages, ttl, tools, tool_choice, model)
        google_cache_name: Final = await self.async_check_cache(
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

        if google_cache_name:
            return non_cached_messages, optional_params, google_cache_name

        ## TRANSFORM REQUEST
        inlined_turns: Final = list(await _async_inline_cached_turns(cached_messages, model, custom_llm_provider))
        transform: Final = partial(
            transform_openai_messages_to_gemini_context_caching,
            model=model,
            messages=inlined_turns,
            cache_key=generated_cache_key,
            ttl=ttl,
            custom_llm_provider=custom_llm_provider,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            drop_params=drop_params,
            litellm_params=litellm_params,
        )
        cached_content_request_body: Final = (
            await asyncify(transform)()
            if openai_messages_may_need_sync_gcs_metadata_fetch(inlined_turns)
            else transform()
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
        return (
            non_cached_messages,
            optional_params,
            raw_response_cached.get("name"),
        )

    def get_cache(self):
        pass

    async def async_get_cache(self):
        pass
