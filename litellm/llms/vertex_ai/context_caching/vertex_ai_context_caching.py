import asyncio
import re
import threading
import time
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Final, Literal

import httpx
from pydantic import ConfigDict, RootModel, with_config
from typing_extensions import NotRequired, ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_logger
from litellm.caching.caching import Cache, LiteLLMCacheType
from litellm.constants import MINIMUM_PROMPT_CACHE_TOKEN_COUNT
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.llms.openai.openai import AllMessageValues
from litellm.utils import is_prompt_caching_valid_prompt

from ..common_utils import VertexAIError, get_vertex_base_url
from ..vertex_llm_base import VertexBase
from .transformation import (
    cached_messages_end_on_supported_turn,
    separate_cached_messages,
    transform_openai_messages_to_gemini_context_caching,
)

local_cache_obj: Final = Cache(type=LiteLLMCacheType.LOCAL)  # only used for calling 'get_cache_key' function

MAX_PAGINATION_PAGES: Final = 100  # Reasonable upper bound for pagination

# A remembered cache stops being reused this long before Vertex expires it, so a request
# resolved just before expiry does not reach generateContent with an expired cache.
KNOWN_CACHE_EXPIRY_MARGIN_SECONDS: Final = 10.0
MAX_KNOWN_CACHED_CONTENTS: Final = 10_000

_RFC3339_FRACTION: Final = re.compile(r"\.(\d{1,6})\d*")


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


def _record_cache_creation_tokens(logging_obj: Logging, created: _CreatedCachedContent) -> None:
    tokens: Final = (created.get("usageMetadata") or {}).get("totalTokenCount") or 0
    logging_obj.vertex_ai_cache_creation_input_tokens = max(tokens, 0)  # rebind-ok: set provider count


def _parse_expire_time(expire_time: object) -> float | None:
    """Parse a Vertex RFC 3339 ``expireTime`` (nanosecond precision, ``Z`` suffix) to epoch seconds."""
    if not isinstance(expire_time, str) or not expire_time:
        return None
    normalized: Final = _RFC3339_FRACTION.sub(lambda m: "." + m.group(1), expire_time.replace("Z", "+00:00"), count=1)
    try:
        parsed: Final = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc).timestamp()
    return parsed.timestamp()


class _KnownCachedContents:
    """Process-local record of the cachedContents this process created or found.

    Vertex meters every cachedContents list page against a per-minute, per-location quota,
    so listing on every cache-marked request exhausts it under load. A cache is only
    remembered with the ``expireTime`` Vertex reported, and is never reused within
    ``KNOWN_CACHE_EXPIRY_MARGIN_SECONDS`` of it.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[tuple[str, str], str], tuple[str, float]] = {}  # mutable-ok: guarded by _lock
        self._lock: Final = threading.Lock()
        self._inflight: weakref.WeakKeyDictionary[  # mutable-ok: private, guarded by _lock
            asyncio.AbstractEventLoop, dict[tuple[tuple[str, str], str], asyncio.Future[str | None]]
        ] = weakref.WeakKeyDictionary()

    @staticmethod
    def scope(
        base_url: str,
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
        headers: Mapping[str, str],
    ) -> tuple[str, str]:
        """Where a cache name is valid: the Vertex URL names project and location, a Gemini key names the project.

        The Gemini key is read from the headers actually sent, so an ``extra_headers`` override scopes the
        cache to the caller's project. It is kept as-is in this in-memory scope and never logged or persisted.
        """
        if custom_llm_provider != "gemini":
            return (base_url, "")
        sent_keys: Final = [value for name, value in headers.items() if name.lower() == "x-goog-api-key"]
        return (base_url, sent_keys[-1] if sent_keys else "")

    @staticmethod
    def is_usable(expire_time: object) -> bool:
        expires_at: Final = _parse_expire_time(expire_time)
        return expires_at is None or expires_at - KNOWN_CACHE_EXPIRY_MARGIN_SECONDS > time.time()

    def get(self, scope: tuple[str, str], cache_key: str) -> str | None:
        with self._lock:
            entry: Final = self._entries.get((scope, cache_key))
            if entry is None:
                return None
            name, reuse_until = entry
            if reuse_until > time.time():
                return name
            del self._entries[(scope, cache_key)]
            return None

    def remember(self, scope: tuple[str, str], cache_key: str, name: str | None, expire_time: object) -> None:
        expires_at: Final = _parse_expire_time(expire_time)
        if not name or expires_at is None:
            return
        reuse_until: Final = expires_at - KNOWN_CACHE_EXPIRY_MARGIN_SECONDS
        now: Final = time.time()
        if reuse_until <= now:
            return
        with self._lock:
            if len(self._entries) >= MAX_KNOWN_CACHED_CONTENTS:
                self._entries = {k: v for k, v in self._entries.items() if v[1] > now}
                while len(self._entries) >= MAX_KNOWN_CACHED_CONTENTS:
                    del self._entries[next(iter(self._entries))]
            self._entries[(scope, cache_key)] = (name, reuse_until)

    async def single_flight(
        self, scope: tuple[str, str], cache_key: str, resolve: Callable[[], Awaitable[str | None]]
    ) -> str | None:
        """Run ``resolve`` once per (scope, cache_key) at a time within this event loop.

        Concurrent requests for the same prefix wait for the first one's lookup-or-create
        instead of each listing (and creating a duplicate cache). If that first attempt
        fails, each waiter resolves on its own.
        """
        loop: Final = asyncio.get_running_loop()
        own: Final[asyncio.Future[str | None]] = loop.create_future()
        with self._lock:
            inflight: Final = self._inflight.setdefault(loop, {})
            leader: Final = inflight.setdefault((scope, cache_key), own)
        if leader is not own:
            name: Final = await asyncio.shield(leader)
            if name:
                return name
            return await resolve()
        try:
            resolved: Final = await resolve()
        except BaseException:
            own.set_result(None)
            raise
        else:
            own.set_result(resolved)
            return resolved
        finally:
            with self._lock:
                inflight.pop((scope, cache_key), None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._inflight.clear()


known_cached_contents: Final = _KnownCachedContents()


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
        scope: Final = _KnownCachedContents.scope(base_url, custom_llm_provider, headers)
        known_name: Final = known_cached_contents.get(scope, cache_key)
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
                if not _KnownCachedContents.is_usable(expire_time):
                    continue
                name = cached_item.get("name")
                known_cached_contents.remember(scope, cache_key, name, expire_time)
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
        scope: Final = _KnownCachedContents.scope(base_url, custom_llm_provider, headers)
        known_name: Final = known_cached_contents.get(scope, cache_key)
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
                if not _KnownCachedContents.is_usable(expire_time):
                    continue
                name = cached_item.get("name")
                known_cached_contents.remember(scope, cache_key, name, expire_time)
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
        generated_cache_key: Final = local_cache_obj.get_cache_key(
            messages=cached_messages, tools=tools, tool_choice=tool_choice, model=model
        )
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
        known_cached_contents.remember(
            _KnownCachedContents.scope(url, custom_llm_provider, headers),
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
        generated_cache_key: Final = local_cache_obj.get_cache_key(
            messages=cached_messages, tools=tools, tool_choice=tool_choice, model=model
        )
        scope: Final = _KnownCachedContents.scope(url, custom_llm_provider, headers)

        async def lookup_or_create() -> str | None:
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
                return google_cache_name

            ## TRANSFORM REQUEST
            cached_content_request_body: Final = transform_openai_messages_to_gemini_context_caching(
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
            known_cached_contents.remember(
                scope, generated_cache_key, raw_response_cached.get("name"), raw_response_cached.get("expireTime")
            )
            return raw_response_cached.get("name")

        cache_name: Final = await known_cached_contents.single_flight(scope, generated_cache_key, lookup_or_create)
        return non_cached_messages, optional_params, cache_name

    def get_cache(self):
        pass

    async def async_get_cache(self):
        pass
