"""
This contains LLMCachingHandler

This exposes two methods:
    - async_get_cache
    - async_set_cache

This file is a wrapper around caching.py

This class is used to handle caching logic specific for LLM API requests (completion / embedding / text_completion / transcription etc)

It utilizes the (RedisCache, s3Cache, RedisSemanticCache, QdrantSemanticCache, InMemoryCache, DiskCache) based on what the user has setup

In each method it will call the appropriate method from caching.py
"""

import asyncio
import datetime
import inspect
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Generator, Mapping
from typing import TYPE_CHECKING, Any, Final, Optional, TypeVar

from pydantic import ConfigDict, SkipValidation, TypeAdapter, ValidationError

import litellm
from litellm._internal_context import is_internal_call, post_response_phase
from litellm._logging import print_verbose, verbose_logger
from litellm.caching import InMemoryCache
from litellm.caching.caching import S3Cache, response_cache_phase
from litellm.constants import CACHE_WRITE_SHUTDOWN_FLUSH_TIMEOUT_SECONDS, RESPONSE_CACHE_EXCLUDED_PROVIDERS
from litellm.litellm_core_utils.hidden_params import get_hidden_params
from litellm.litellm_core_utils.llm_response_utils.response_metadata import (
    update_response_metadata,
)
from litellm.litellm_core_utils.logging_utils import (
    assemble_complete_response_from_streaming_chunks,
)
from litellm.types.caching import CACHED_STREAM_EVENTS_KEY, EMBEDDING_CACHE_FORMAT_VERSION, CachedEmbedding
from litellm.types.integrations.custom_logger import converted_stream_requested
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.openai import ChatCompletionFileObject, ResponsesAPIResponse
from litellm.types.rerank import RerankResponse
from litellm.types.utils import (
    CachingDetails,
    CallTypes,
    Choices,
    Embedding,
    EmbeddingResponse,
    ModelResponse,
    TextChoices,
    TextCompletionResponse,
    TranscriptionResponse,
    Usage,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.anthropic.pass_through.messages.response_cache import (
        AnthropicMessagesStreamCacheWriter,
    )
    from litellm.types.utils import PromptTokensDetailsWrapper
else:
    LiteLLMLoggingObj = Any

_StreamResultT = TypeVar("_StreamResultT")


from litellm.litellm_core_utils.core_helpers import (
    get_parent_otel_span_from_kwargs,
)
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

EmbeddingCacheInputElement = str | list[int] | ChatCompletionFileObject


class CachingHandlerResponse(LiteLLMBaseModel):
    """
    This is the response object for the caching handler. We need to separate embedding cached responses and (completion / text_completion / transcription) cached responses

    For embeddings there can be a cache hit for some of the inputs in the list and a cache miss for others
    """

    cached_result: object | None = None
    final_embedding_cached_response: EmbeddingResponse | None = None
    embedding_all_elements_cache_hit: bool = False  # this is set to True when all elements in the list have a cache hit in the embedding cache, if true return the final_embedding_cached_response no need to make an API call
    embedding_uncached_input: SkipValidation[list[EmbeddingCacheInputElement]] | None = None


in_memory_cache_obj: Final = InMemoryCache()


def _drop_logging_obj_from_kwargs(request_kwargs: dict[str, object]) -> dict[str, object]:
    """
    The caching handler is stored on the Logging object
    (``logging_obj._llm_caching_handler``), so keeping ``litellm_logging_obj``
    inside ``request_kwargs`` closes a reference cycle
    (Logging -> LLMCachingHandler -> kwargs -> Logging) that keeps the full
    request payload (messages included) alive until a generational GC pass
    instead of being freed by refcount when the request ends. Nothing in the
    caching layer reads the logging object from these kwargs; cache-key
    generation ignores litellm-internal params.
    """
    if "litellm_logging_obj" not in request_kwargs:
        return request_kwargs
    return {k: v for k, v in request_kwargs.items() if k != "litellm_logging_obj"}


def _is_response_cache_excluded(model: str | None, kwargs: Mapping[str, object]) -> bool:
    from litellm.llms.github_copilot.per_user_auth import is_github_copilot_per_user_request

    custom_llm_provider: Final = kwargs.get("custom_llm_provider")
    model_provider: Final = model.split("/", maxsplit=1)[0] if model is not None else None
    return (
        (isinstance(custom_llm_provider, str) and custom_llm_provider in RESPONSE_CACHE_EXCLUDED_PROVIDERS)
        or model_provider in RESPONSE_CACHE_EXCLUDED_PROVIDERS
        or is_github_copilot_per_user_request(kwargs)
    )


def _is_chat_completion_cached_dict(cached_result: dict) -> bool:
    cached_id: Final = cached_result.get("id")
    if isinstance(cached_id, str) and cached_id.startswith("chatcmpl"):
        return True
    obj: Final = cached_result.get("object")
    if isinstance(obj, str):
        return obj.startswith("chat.completion")
    return "choices" in cached_result


def _starts_content_block(event: object) -> bool:
    text: Final = event.decode("utf-8", errors="replace") if isinstance(event, bytes) else event
    return isinstance(text, str) and "event: content_block_start" in text.splitlines()


def is_response_without_output(result: object) -> bool:
    if isinstance(result, (ModelResponse, TextCompletionResponse)):
        return not result.choices
    if isinstance(result, ResponsesAPIResponse):
        return not getattr(result, "output", None)
    if not isinstance(result, dict):
        return False
    cached_stream_events: Final = result.get(CACHED_STREAM_EVENTS_KEY)
    if isinstance(cached_stream_events, (list, tuple)):
        return not any(_starts_content_block(event) for event in cached_stream_events)
    if "choices" in result:
        return not result["choices"]
    if result.get("object") == "response":
        return not result.get("output")
    if result.get("type") == "message":
        return not result.get("content")
    return False


def _choice_carries_output(choice: Choices | TextChoices) -> bool:
    payload: Final[Mapping[str, object]] = (
        choice.message.model_dump(exclude_none=True) if isinstance(choice, Choices) else {"text": choice.text}
    )
    return any(key != "role" and value not in ("", [], {}) for key, value in payload.items())


def _assembled_stream_without_output(response: ModelResponse | TextCompletionResponse) -> bool:
    """The stream wrapper closes a stream whose chunks carried no choice with an empty choice of its own, so the
    assembled response has choices and the no-output check on it alone would store that empty answer."""
    return not any(_choice_carries_output(choice) for choice in response.choices)


def _stream_replay_requested(kwargs: Mapping[str, object]) -> bool:
    if kwargs.get("stream", False) is True:
        return True
    return converted_stream_requested(kwargs) and not kwargs.get("_agentic_loop_depth")


def _should_defer_streaming_cache_hit_callbacks(*, cached_result: object) -> bool:
    """
    When the cache hit is replayed as a stream, do not run success callbacks at cache-hit time.

    Cached chat/text completion replay uses CustomStreamWrapper; cached Responses
    replay uses CachedResponsesAPIStreamingIterator; cached Anthropic Messages
    replay uses CachedAnthropicMessagesStreamIterator. All invoke logging success
    handlers when the stream finishes; firing them here too would double-count
    spend and callback records. A plain (non-stream) replay logs here, since nothing
    else will.
    """
    from litellm.llms.anthropic.pass_through.messages.response_cache import (
        CachedAnthropicMessagesStreamIterator,
    )
    from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator

    return isinstance(
        cached_result, (CustomStreamWrapper, BaseResponsesAPIStreamingIterator, CachedAnthropicMessagesStreamIterator)
    )


def _prompt_tokens_details_as_mapping(details: "PromptTokensDetailsWrapper") -> Mapping[str, object]:
    """Dump prompt token details to an opaque field mapping, tolerating non-pydantic stand-ins."""
    return details.model_dump(exclude_none=True) if hasattr(details, "model_dump") else {}


_PENDING_CACHE_WRITES: Final[set["asyncio.Task[None]"]] = set()  # mutable-ok: strong refs to pending write tasks


async def _complete_cache_write_despite_cancellation(write_factory: Callable[[], Awaitable[None]]) -> None:
    with response_cache_phase("set"):
        try:
            await write_factory()
        except asyncio.CancelledError:
            try:
                await asyncio.wait_for(write_factory(), timeout=CACHE_WRITE_SHUTDOWN_FLUSH_TIMEOUT_SECONDS)
            except Exception as flush_error:  # noqa: BLE001  # shutdown flush failures are logged, never raised
                verbose_logger.warning(
                    "LiteLLM Cache: pending cache write failed during event loop shutdown: %s", flush_error
                )
            raise


def create_cache_write_task(write_factory: Callable[[], Awaitable[None]]) -> "asyncio.Task[None]":
    with post_response_phase():
        task: Final = asyncio.create_task(_complete_cache_write_despite_cancellation(write_factory))
    _PENDING_CACHE_WRITES.add(task)
    task.add_done_callback(_PENDING_CACHE_WRITES.discard)
    return task


def _request_cache_key(request_kwargs: Mapping[str, Any]) -> str | None:
    """Read the caller-supplied ``cache_key`` off the request kwargs."""
    return request_kwargs.get("cache_key", None)


def _set_cached_hidden_param(response: object, key: str, value: object) -> None:
    if isinstance(response, TextCompletionResponse):
        setattr(response.hidden_params, key, value)
        return
    hidden_params: Final = get_hidden_params(response)
    if hidden_params is not None:
        hidden_params[key] = value


class _CachedEmbeddingRecord(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    embedding: list[float] | str | None
    index: int | None
    object: str | None
    model: str | None
    prompt_tokens: int | None
    prompt_tokens_details: dict | None
    format_version: int


def _current_format_embedding_entry(entry: object) -> CachedEmbedding | None:
    try:
        record: Final = _CachedEmbeddingRecord.model_validate(entry)
    except ValidationError:
        return None
    if record.format_version != EMBEDDING_CACHE_FORMAT_VERSION:
        return None
    cached: Final[CachedEmbedding] = {
        "embedding": record.embedding,
        "index": record.index,
        "object": record.object,
        "model": record.model,
        "prompt_tokens": record.prompt_tokens,
        "prompt_tokens_details": record.prompt_tokens_details,
        "format_version": record.format_version,
    }
    return cached


class LLMCachingHandler:
    def __init__(
        self,
        original_function: Callable[..., object],
        request_kwargs: dict[str, object],
        start_time: datetime.datetime,
    ):
        from litellm.caching import DualCache, RedisCache

        self.async_streaming_chunks: list[object] = []
        self.sync_streaming_chunks: list[object] = []
        self.request_kwargs = _drop_logging_obj_from_kwargs(request_kwargs)
        self.preset_cache_key: str | None = None
        self.original_function = original_function
        self.start_time = start_time
        if litellm.cache is not None and isinstance(litellm.cache.cache, RedisCache):
            self.dual_cache: DualCache | None = DualCache(
                redis_cache=litellm.cache.cache,
                in_memory_cache=in_memory_cache_obj,
            )
        else:
            self.dual_cache = None

    async def async_get_cache(
        self,
        model: str,
        original_function: Callable[..., object],
        logging_obj: LiteLLMLoggingObj,
        start_time: datetime.datetime,
        call_type: str,
        kwargs: dict[str, Any],
        args: tuple[object, ...] | None = None,
    ) -> CachingHandlerResponse | None:
        """
        Internal method to get from the cache.
        Handles different call types (embeddings, chat/completions, text_completion, transcription)
        and accordingly returns the cached response

        Args:
            model: str:
            original_function: Callable:
            logging_obj: LiteLLMLoggingObj:
            start_time: datetime.datetime:
            call_type: str:
            kwargs: Dict[str, Any]:
            args: Optional[Tuple[Any, ...]] = None:


        Returns:
            CachingHandlerResponse:
        Raises:
            None
        """
        if _is_response_cache_excluded(model=model, kwargs=kwargs):
            self.request_kwargs = _drop_logging_obj_from_kwargs(kwargs)
            return None

        # Check if caching should be performed BEFORE doing expensive operations
        if (
            (kwargs.get("caching", None) is None and litellm.cache is not None) or kwargs.get("caching", False) is True
        ) and (
            kwargs.get("cache", {}).get("no-cache", False) is not True
        ):  # allow users to control returning cached responses from the completion function
            args = args or ()
            final_embedding_cached_response: EmbeddingResponse | None = None
            embedding_all_elements_cache_hit: bool = False
            cached_result: object | None = None
            kwargs = kwargs.copy()
            #########################################################
            # Init cache timing metrics
            #########################################################
            cache_check_start_time: Final = time.perf_counter()
            cache_check_end_time: float | None = None
            #########################################################
            parent_otel_span: Final = get_parent_otel_span_from_kwargs(kwargs)
            kwargs["parent_otel_span"] = parent_otel_span

            if litellm.cache is not None and self._is_call_type_supported_by_cache(original_function=original_function):
                verbose_logger.debug("Checking Async Cache")
                cached_result = await self._retrieve_from_cache(
                    call_type=call_type,
                    kwargs=kwargs,
                    args=args,
                )
                cache_check_end_time = time.perf_counter()

                if cached_result is not None and not isinstance(cached_result, list):
                    verbose_logger.debug("Cache Hit!")
                    cache_hit: Final = True
                    end_time: Final = datetime.datetime.now()
                    model, custom_llm_provider, _, _ = litellm.get_llm_provider(
                        model=model,
                        custom_llm_provider=kwargs.get("custom_llm_provider", None),
                        api_base=kwargs.get("api_base", None),
                        api_key=kwargs.get("api_key", None),
                    )
                    cache_duration_ms: Final = (cache_check_end_time - cache_check_start_time) * 1000
                    self._update_litellm_logging_obj_environment(
                        logging_obj=logging_obj,
                        model=model,
                        kwargs=kwargs,
                        cached_result=cached_result,
                        is_async=True,
                        custom_llm_provider=custom_llm_provider,
                        cache_duration_ms=cache_duration_ms,
                    )

                    call_type = original_function.__name__

                    cached_result = self._convert_cached_result_to_model_response(
                        cached_result=cached_result,
                        call_type=call_type,
                        kwargs=kwargs,
                        logging_obj=logging_obj,
                        model=model,
                        custom_llm_provider=kwargs.get("custom_llm_provider", None),
                        args=args,
                    )
                    if not _should_defer_streaming_cache_hit_callbacks(cached_result=cached_result):
                        # LOG SUCCESS
                        self._async_log_cache_hit_on_callbacks(
                            logging_obj=logging_obj,
                            cached_result=cached_result,
                            start_time=start_time,
                            end_time=end_time,
                            cache_hit=cache_hit,
                        )
                    cache_key: Final = (
                        self.preset_cache_key
                        or self.request_kwargs.get("cache_key")
                        or litellm.cache.get_cache_key(**self.request_kwargs)
                    )
                    _set_cached_hidden_param(cached_result, "cache_key", cache_key)
                    return CachingHandlerResponse(cached_result=cached_result)
                elif (
                    call_type == CallTypes.aembedding.value
                    and cached_result is not None
                    and isinstance(cached_result, list)
                    and litellm.cache is not None
                    and not isinstance(litellm.cache.cache, S3Cache)  # s3 doesn't support bulk writing. Exclude.
                ):
                    (
                        final_embedding_cached_response,
                        embedding_all_elements_cache_hit,
                    ) = self._process_async_embedding_cached_response(
                        final_embedding_cached_response=final_embedding_cached_response,
                        cached_result=cached_result,
                        kwargs=kwargs,
                        logging_obj=logging_obj,
                        start_time=start_time,
                        model=model,
                    )
                    return CachingHandlerResponse(
                        final_embedding_cached_response=final_embedding_cached_response,
                        embedding_all_elements_cache_hit=embedding_all_elements_cache_hit,
                        embedding_uncached_input=self.handle_kwargs_input_list_or_str(kwargs),
                    )

            verbose_logger.debug("CACHE RESULT: %s", cached_result)
            return CachingHandlerResponse(
                cached_result=cached_result,
                final_embedding_cached_response=final_embedding_cached_response,
            )
        # Caching disabled - return None to indicate no caching attempted
        return None

    _async_get_cache = async_get_cache

    def sync_get_cache(
        self,
        model: str,
        original_function: Callable[..., object],
        logging_obj: LiteLLMLoggingObj,
        start_time: datetime.datetime,
        call_type: str,
        kwargs: dict[str, Any],
        args: tuple[object, ...] | None = None,
    ) -> CachingHandlerResponse:
        cached_result: Any | None = None

        # Check if caching should be performed BEFORE doing expensive kwargs copy
        if _is_response_cache_excluded(model=model, kwargs=kwargs):
            self.request_kwargs = _drop_logging_obj_from_kwargs(kwargs)
            return CachingHandlerResponse(cached_result=None)
        if litellm.cache is not None and self._is_call_type_supported_by_cache(original_function=original_function):
            args = args or ()
            # Now that we confirmed caching will happen, prepare kwargs
            new_kwargs: Final = kwargs.copy()
            new_kwargs.update(
                convert_args_to_kwargs(
                    self.original_function,
                    args,
                )
            )
            if new_kwargs.get("metadata") is None:
                new_kwargs.pop("metadata", None)
            if new_kwargs.get("stream") is True and "cache_key" not in new_kwargs:
                new_kwargs["cache_key"] = litellm.cache.get_cache_key(**new_kwargs)
            self.request_kwargs = _drop_logging_obj_from_kwargs(new_kwargs)
            print_verbose("Checking Sync Cache")
            with response_cache_phase("get"):
                cached_result = litellm.cache.get_cache(**new_kwargs)
            if is_response_without_output(cached_result):
                verbose_logger.debug("LiteLLM Cache: cached response has no output, treating it as a miss")
                return CachingHandlerResponse(cached_result=None)
            if cached_result is not None:
                if "detail" in cached_result:
                    # implies an error occurred
                    pass
                else:
                    call_type = original_function.__name__
                    cached_result = self._convert_cached_result_to_model_response(
                        cached_result=cached_result,
                        call_type=call_type,
                        kwargs=kwargs,
                        logging_obj=logging_obj,
                        model=model,
                        custom_llm_provider=kwargs.get("custom_llm_provider", None),
                        args=args,
                    )

                    # LOG SUCCESS
                    cache_hit: Final = True
                    end_time: Final = datetime.datetime.now()
                    (
                        model,
                        custom_llm_provider,
                        dynamic_api_key,
                        api_base,
                    ) = litellm.get_llm_provider(
                        model=model or "",
                        custom_llm_provider=kwargs.get("custom_llm_provider", None),
                        api_base=kwargs.get("api_base", None),
                        api_key=kwargs.get("api_key", None),
                    )
                    self._update_litellm_logging_obj_environment(
                        logging_obj=logging_obj,
                        model=f"{custom_llm_provider}/{model}",
                        kwargs=kwargs,
                        cached_result=cached_result,
                        is_async=False,
                        custom_llm_provider=custom_llm_provider,
                    )

                    if not _should_defer_streaming_cache_hit_callbacks(cached_result=cached_result):
                        if is_internal_call.get():
                            _set_cached_hidden_param(cached_result, "response_cost", 0.0)
                        else:
                            logging_obj.handle_sync_success_callbacks_for_async_calls(
                                result=cached_result,
                                start_time=start_time,
                                end_time=end_time,
                                cache_hit=cache_hit,
                            )
                    cache_key: Final = (
                        self.preset_cache_key
                        or self.request_kwargs.get("cache_key")
                        or litellm.cache.get_cache_key(**self.request_kwargs)
                    )
                    _set_cached_hidden_param(cached_result, "cache_key", cache_key)
                    return CachingHandlerResponse(cached_result=cached_result)
        return CachingHandlerResponse(cached_result=cached_result)

    _sync_get_cache = sync_get_cache

    def handle_kwargs_input_list_or_str(self, kwargs: dict[str, object]) -> list[EmbeddingCacheInputElement]:
        """
        Handles the input of kwargs['input'] being a list or a string
        """
        if isinstance(kwargs["input"], str):
            return [kwargs["input"]]
        elif isinstance(kwargs["input"], list):
            return kwargs["input"]
        else:
            raise litellm.BadRequestError(
                message="input must be a string or a list of strings and content blocks",
                model=str(kwargs.get("model")),
                llm_provider=str(kwargs.get("custom_llm_provider")),
            )

    def _extract_model_from_cached_results(self, non_null_list: list[tuple[int, CachedEmbedding]]) -> str | None:
        """
        Helper method to extract the model name from cached results.

        Args:
            non_null_list: List of (idx, cr) tuples where cr is the cached result dict

        Returns:
            Optional[str]: The model name if found, None otherwise
        """
        for _, cr in non_null_list:
            if isinstance(cr, dict) and cr.get("model"):
                return cr["model"]
        return None

    def _process_async_embedding_cached_response(
        self,
        final_embedding_cached_response: EmbeddingResponse | None,
        cached_result: list[CachedEmbedding | None],
        kwargs: dict[str, Any],
        logging_obj: LiteLLMLoggingObj,
        start_time: datetime.datetime,
        model: str,
    ) -> tuple[EmbeddingResponse | None, bool]:
        """
        Returns the final embedding cached response and a boolean indicating if all elements in the list have a cache hit

        For embedding responses, there can be a cache hit for some of the inputs in the list and a cache miss for others
        This function processes the cached embedding responses and returns the final embedding cached response and a boolean indicating if all elements in the list have a cache hit

        Args:
            final_embedding_cached_response: Optional[EmbeddingResponse]:
            cached_result: List[Optional[Dict[str, Any]]]:
            kwargs: Dict[str, Any]:
            logging_obj: LiteLLMLoggingObj:
            start_time: datetime.datetime:
            model: str:

        Returns:
            Tuple[Optional[EmbeddingResponse], bool]:
            Returns the final embedding cached response and a boolean indicating if all elements in the list have a cache hit


        """
        embedding_all_elements_cache_hit: bool = False
        remaining_list: Final = []
        non_null_list: Final = []
        kwargs_input_as_list: Final = self.handle_kwargs_input_list_or_str(kwargs)
        for idx, cr in enumerate(cached_result):
            if cr is None:
                remaining_list.append(kwargs_input_as_list[idx])
            else:
                non_null_list.append((idx, cr))
        kwargs["input"] = remaining_list
        if len(non_null_list) > 0:
            # Use the model from the first non-null cached result, fallback to kwargs if not present
            model_name = self._extract_model_from_cached_results(non_null_list)
            if not model_name:
                model_name = kwargs.get("model")
            final_embedding_cached_response = EmbeddingResponse(
                model=model_name,
                data=[None] * len(kwargs_input_as_list),
            )
            final_embedding_cached_response.hidden_params["cache_hit"] = True

            prompt_tokens = 0
            aggregated_details: dict | None = None
            for val in non_null_list:
                idx, cr = val  # (idx, cr) tuple
                if cr is not None:
                    embedding_data = cr.get("embedding")
                    if embedding_data is not None:
                        final_embedding_cached_response.data[idx] = Embedding(
                            embedding=embedding_data,
                            index=idx,
                            object="embedding",
                        )
                    cached_prompt_tokens = cr.get("prompt_tokens")
                    if cached_prompt_tokens is not None:
                        prompt_tokens += cached_prompt_tokens
                    elif isinstance(kwargs_input_as_list[idx], str):
                        from litellm.utils import token_counter

                        prompt_tokens += token_counter(text=kwargs_input_as_list[idx], count_response_tokens=True)
                    # Aggregate prompt_tokens_details from cached items
                    item_details = cr.get("prompt_tokens_details")
                    if item_details:
                        if aggregated_details is None:
                            aggregated_details = {}
                        for key, value in item_details.items():
                            if isinstance(value, (int, float)):
                                aggregated_details[key] = aggregated_details.get(key, 0) + value
                            else:
                                aggregated_details[key] = value

            ## USAGE
            from litellm.types.utils import PromptTokensDetailsWrapper

            prompt_tokens_details: PromptTokensDetailsWrapper | None = None
            if aggregated_details:
                try:
                    prompt_tokens_details = PromptTokensDetailsWrapper(**aggregated_details)
                except Exception:
                    prompt_tokens_details = None
            usage: Final = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=prompt_tokens,
                prompt_tokens_details=prompt_tokens_details,
            )
            final_embedding_cached_response.usage = usage
        if len(remaining_list) == 0:
            # LOG SUCCESS
            cache_hit: Final = True
            embedding_all_elements_cache_hit = True
            end_time: Final = datetime.datetime.now()
            (
                model,
                custom_llm_provider,
                dynamic_api_key,
                api_base,
            ) = litellm.get_llm_provider(
                model=model,
                custom_llm_provider=kwargs.get("custom_llm_provider", None),
                api_base=kwargs.get("api_base", None),
                api_key=kwargs.get("api_key", None),
            )

            self._update_litellm_logging_obj_environment(
                logging_obj=logging_obj,
                model=model,
                kwargs=kwargs,
                cached_result=final_embedding_cached_response,
                is_async=True,
                is_embedding=True,
                custom_llm_provider=custom_llm_provider,
            )
            self._async_log_cache_hit_on_callbacks(
                logging_obj=logging_obj,
                cached_result=final_embedding_cached_response,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
            )
            return final_embedding_cached_response, embedding_all_elements_cache_hit
        return final_embedding_cached_response, embedding_all_elements_cache_hit

    def combine_usage(self, usage1: Usage, usage2: Usage) -> Usage:
        return Usage(
            prompt_tokens=usage1.prompt_tokens + usage2.prompt_tokens,
            completion_tokens=usage1.completion_tokens + usage2.completion_tokens,
            total_tokens=usage1.total_tokens + usage2.total_tokens,
            prompt_tokens_details=self._merge_prompt_tokens_details(
                usage1.prompt_tokens_details,
                usage2.prompt_tokens_details,
            ),
        )

    def _merge_prompt_tokens_details(
        self,
        details1: Optional["PromptTokensDetailsWrapper"],
        details2: Optional["PromptTokensDetailsWrapper"],
    ) -> Optional["PromptTokensDetailsWrapper"]:
        """Merge two PromptTokensDetailsWrapper objects by summing numeric fields."""
        if details1 is None and details2 is None:
            return None
        if details1 is None:
            return details2
        if details2 is None:
            return details1

        dict1: Final = _prompt_tokens_details_as_mapping(details1)
        dict2: Final = _prompt_tokens_details_as_mapping(details2)

        merged: Final[dict] = {}
        for key in set(dict1.keys()) | set(dict2.keys()):
            v1 = dict1.get(key, 0)
            v2 = dict2.get(key, 0)
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                merged[key] = v1 + v2
            elif isinstance(v1, dict) and isinstance(v2, dict):
                # Recursively merge nested dicts (e.g. cache_creation_token_details)
                nested: dict = {}
                for nk in set(v1.keys()) | set(v2.keys()):
                    nv1 = v1.get(nk, 0)
                    nv2 = v2.get(nk, 0)
                    if isinstance(nv1, (int, float)) and isinstance(nv2, (int, float)):
                        nested[nk] = nv1 + nv2
                    elif nv1:
                        nested[nk] = nv1
                    else:
                        nested[nk] = nv2
                merged[key] = nested
            elif v1:
                merged[key] = v1
            else:
                merged[key] = v2

        if not merged:
            return None

        from litellm.types.utils import PromptTokensDetailsWrapper

        try:
            return PromptTokensDetailsWrapper(**merged)
        except Exception:
            return None

    def combine_cached_embedding_response_with_api_result(
        self,
        _caching_handler_response: CachingHandlerResponse,
        embedding_response: EmbeddingResponse,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> EmbeddingResponse:
        """
        Combines the cached embedding response with the API EmbeddingResponse

        For caching there can be a cache hit for some of the inputs in the list and a cache miss for others
        This function combines the cached embedding response with the API EmbeddingResponse

        Args:
            caching_handler_response: CachingHandlerResponse:
            embedding_response: EmbeddingResponse:

        Returns:
            EmbeddingResponse:
        """
        if _caching_handler_response.final_embedding_cached_response is None:
            return embedding_response

        cached: Final = _caching_handler_response.final_embedding_cached_response
        fresh_items: Final = iter(embedding_response.data or ())
        merged_usage: Final = (
            self.combine_usage(usage1=cached.usage, usage2=embedding_response.usage)
            if cached.usage is not None and embedding_response.usage is not None
            else cached.usage
        )
        cached_hidden_params: Final = cached.hidden_params
        merged: Final = EmbeddingResponse(
            model=cached.model,
            data=[
                item
                if item is not None
                else Embedding(embedding=next(fresh_items)["embedding"], index=position, object="embedding")
                for position, item in enumerate(cached.data)
            ],
            usage=merged_usage,
            hidden_params={
                **cached_hidden_params,
                "cache_hit": True,
            },
            _response_headers=cached._response_headers,
        )
        merged._response_ms = (end_time - start_time).total_seconds() * 1000
        return merged

    _combine_cached_embedding_response_with_api_result = combine_cached_embedding_response_with_api_result

    def _async_log_cache_hit_on_callbacks(
        self,
        logging_obj: LiteLLMLoggingObj,
        cached_result: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        cache_hit: bool,
    ):
        """
        Helper function to log the success of a cached result on callbacks

        Args:
            logging_obj (LiteLLMLoggingObj): The logging object.
            cached_result: The cached result.
            start_time (datetime): The start time of the operation.
            end_time (datetime): The end time of the operation.
            cache_hit (bool): Whether it was a cache hit.

        An internal sub-call's hit fires nothing and costs nothing, since the parent call that folds it in logs it.
        """
        if is_internal_call.get():
            _set_cached_hidden_param(cached_result, "response_cost", 0.0)
            return

        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(
            async_coroutine=logging_obj.async_success_handler(
                result=cached_result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
            )
        )

        logging_obj.handle_sync_success_callbacks_for_async_calls(
            result=cached_result,
            start_time=start_time,
            end_time=end_time,
            cache_hit=cache_hit,
        )

    async def _retrieve_from_cache(
        self, call_type: str, kwargs: dict[str, object], args: tuple[object, ...]
    ) -> object | None:
        """
        Internal method to
        - get cache key
        - check what type of cache is used - Redis, RedisSemantic, Qdrant, S3
        - async get cache value
        - return the cached value

        Args:
            call_type: str:
            kwargs: Dict[str, Any]:
            args: Optional[Tuple[Any, ...]] = None:

        Returns:
            Optional[Any]:
        Raises:
            None
        """
        if litellm.cache is None:
            return None

        new_kwargs: Final = kwargs.copy()
        new_kwargs.update(
            convert_args_to_kwargs(
                self.original_function,
                args,
            )
        )
        if new_kwargs.get("metadata") is None:
            new_kwargs.pop("metadata", None)
        model_value: Final = new_kwargs.get("model")
        model: Final = model_value if isinstance(model_value, str) else None
        self.request_kwargs = _drop_logging_obj_from_kwargs(new_kwargs)
        if _is_response_cache_excluded(model=model, kwargs=new_kwargs):
            return None
        if new_kwargs.get("stream") is True and "cache_key" not in new_kwargs:
            new_kwargs["cache_key"] = litellm.cache.get_cache_key(**new_kwargs)
        cached_result: object | None = None
        if call_type == CallTypes.aembedding.value:
            new_kwargs["input"] = self.handle_kwargs_input_list_or_str(new_kwargs)
            tasks: Final[list[Awaitable[object]]] = []
            for idx, i in enumerate(new_kwargs["input"]):
                preset_cache_key = litellm.cache.get_cache_key(**{**new_kwargs, "input": i})
                tasks.append(
                    litellm.cache.async_get_cache(
                        cache_key=preset_cache_key,
                        dynamic_cache_object=self.dual_cache,
                    )
                )
            with response_cache_phase("get"):
                entries: Final = await asyncio.gather(*tasks)
            cached_result = [_current_format_embedding_entry(entry) for entry in entries]
            ## check if cached result is None ##
            if cached_result is not None and isinstance(cached_result, list):
                # set cached_result to None if all elements are None
                if all(result is None for result in cached_result):
                    cached_result = None
        else:
            request_kwargs: Final = new_kwargs.copy()
            request_cache_key: Final = _request_cache_key(request_kwargs)
            request_kwargs.pop("cache_key", None)
            if litellm.cache.supports_async() is True:
                ## check if dual cache is supported ##
                self.preset_cache_key = request_cache_key or litellm.cache.get_cache_key(**request_kwargs)
                with response_cache_phase("get"):
                    cached_result = await litellm.cache.async_get_cache(
                        dynamic_cache_object=self.dual_cache,
                        cache_key=self.preset_cache_key,
                        **request_kwargs,
                    )
            else:  # fallback for caches that don't support async
                self.preset_cache_key = request_cache_key or litellm.cache.get_cache_key(**request_kwargs)
                with response_cache_phase("get"):
                    cached_result = litellm.cache.get_cache(
                        dynamic_cache_object=self.dual_cache,
                        cache_key=self.preset_cache_key,
                        **request_kwargs,
                    )
        if is_response_without_output(cached_result):
            verbose_logger.debug("LiteLLM Cache: cached response has no output, treating it as a miss")
            self._forget_worker_copy()
            return None
        return cached_result

    def _forget_worker_copy(self) -> None:
        """Drop this worker's memory copy of a stored response without output, so the next read reaches Redis,
        where the refill lands: the text completion and messages writers skip the memory tier."""
        if self.dual_cache is None or self.preset_cache_key is None:
            return
        self.dual_cache.in_memory_cache.delete_cache(self.preset_cache_key)

    def _convert_cached_result_to_model_response(
        self,
        cached_result: Any,
        call_type: str,
        kwargs: dict[str, object],
        logging_obj: LiteLLMLoggingObj,
        model: str,
        args: tuple[object, ...],
        custom_llm_provider: str | None = None,
    ) -> (
        ModelResponse
        | TextCompletionResponse
        | EmbeddingResponse
        | RerankResponse
        | TranscriptionResponse
        | CustomStreamWrapper
        | None
    ):
        """
        Internal method to process the cached result

        Checks the call type and converts the cached result to the appropriate model response object
        example if call type is text_completion -> returns TextCompletionResponse object

        Args:
            cached_result: Any:
            call_type: str:
            kwargs: Dict[str, Any]:
            logging_obj: LiteLLMLoggingObj:
            model: str:
            custom_llm_provider: Optional[str] = None:
            args: Optional[Tuple[Any, ...]] = None:

        Returns:
            Optional[Any]:
        """
        from litellm.utils import convert_to_model_response_object

        if (call_type == CallTypes.acompletion.value or call_type == CallTypes.completion.value) and isinstance(
            cached_result, dict
        ):
            if _stream_replay_requested(kwargs):
                cached_result = self._convert_cached_stream_response(
                    cached_result=cached_result,
                    call_type=call_type,
                    logging_obj=logging_obj,
                    model=model,
                )
            else:
                cached_result = convert_to_model_response_object(
                    response_object=cached_result,
                    model_response_object=ModelResponse(),
                )
        if (
            call_type == CallTypes.atext_completion.value or call_type == CallTypes.text_completion.value
        ) and isinstance(cached_result, dict):
            if _stream_replay_requested(kwargs):
                cached_result = self._convert_cached_stream_response(
                    cached_result=cached_result,
                    call_type=call_type,
                    logging_obj=logging_obj,
                    model=model,
                )
            else:
                cached_result = TextCompletionResponse(**cached_result)
        elif (call_type == CallTypes.aembedding.value or call_type == CallTypes.embedding.value) and isinstance(
            cached_result, dict
        ):
            cached_result = convert_to_model_response_object(
                response_object=cached_result,
                model_response_object=EmbeddingResponse(),
                response_type="embedding",
            )

        elif (call_type == CallTypes.arerank.value or call_type == CallTypes.rerank.value) and isinstance(
            cached_result, dict
        ):
            cached_result = convert_to_model_response_object(
                response_object=cached_result,
                model_response_object=None,
                response_type="rerank",
            )
        elif (call_type == CallTypes.atranscription.value or call_type == CallTypes.transcription.value) and isinstance(
            cached_result, dict
        ):
            hidden_params: Final = {
                "model": "whisper-1",
                "custom_llm_provider": custom_llm_provider,
                "cache_hit": True,
            }
            cached_result = convert_to_model_response_object(
                response_object=cached_result,
                model_response_object=TranscriptionResponse(),
                response_type="audio_transcription",
                hidden_params=hidden_params,
            )
        elif (
            call_type == CallTypes.anthropic_messages.value or call_type == CallTypes.aanthropic_messages.value
        ) and isinstance(cached_result, dict):
            from litellm.llms.anthropic.pass_through.messages.response_cache import (
                convert_cached_anthropic_messages_result,
            )

            cached_result = convert_cached_anthropic_messages_result(
                cached_result=cached_result,
                logging_obj=logging_obj,
                kwargs=kwargs,
            )
        elif (call_type == "aresponses" or call_type == "responses") and isinstance(cached_result, dict):
            use_chat_completion_cache: Final = _is_chat_completion_cached_dict(cached_result)
            if use_chat_completion_cache:
                if _stream_replay_requested(kwargs):
                    bridge_call_type: Final = (
                        CallTypes.acompletion.value if call_type == "aresponses" else CallTypes.completion.value
                    )
                    cached_result = self._convert_cached_stream_response(
                        cached_result=cached_result,
                        call_type=bridge_call_type,
                        logging_obj=logging_obj,
                        model=model,
                    )
                else:
                    cached_result = convert_to_model_response_object(
                        response_object=cached_result,
                        model_response_object=ModelResponse(),
                    )
            else:
                from litellm.responses.streaming_iterator import (
                    CachedResponsesAPIStreamingIterator,
                )

                response_obj: Final = ResponsesAPIResponse(**cached_result)
                _set_cached_hidden_param(response_obj, "cache_hit", True)

                if _stream_replay_requested(kwargs):
                    cached_result = CachedResponsesAPIStreamingIterator(
                        response=response_obj,
                        logging_obj=logging_obj,
                        request_data=kwargs,
                        call_type=call_type,
                    )
                else:
                    cached_result = response_obj

        _set_cached_hidden_param(cached_result, "cache_hit", True)

        #########################################################
        # Add final timing metrics to the cached result
        #########################################################
        update_response_metadata(
            result=cached_result,
            logging_obj=logging_obj,
            model=model,
            kwargs=kwargs,
            start_time=self.start_time,
            end_time=datetime.datetime.now(),
        )
        return cached_result

    def _convert_cached_stream_response(
        self,
        cached_result: dict[str, object],
        call_type: str,
        logging_obj: LiteLLMLoggingObj,
        model: str,
    ) -> CustomStreamWrapper:
        from litellm.utils import (
            CustomStreamWrapper,
            convert_to_streaming_response,
            convert_to_streaming_response_async,
        )

        _stream_cached_result: AsyncGenerator | Generator
        if call_type == CallTypes.acompletion.value or call_type == CallTypes.atext_completion.value:
            _stream_cached_result = convert_to_streaming_response_async(
                response_object=cached_result,
            )
        else:
            _stream_cached_result = convert_to_streaming_response(
                response_object=cached_result,
            )
        return CustomStreamWrapper(
            completion_stream=_stream_cached_result,
            model=model,
            custom_llm_provider="cached_response",
            logging_obj=logging_obj,
        )

    async def async_set_cache(
        self,
        result: object,
        original_function: Callable,
        kwargs: dict[str, Any],
        args: tuple[object, ...] | None = None,
    ):
        """
        Internal method to check the type of the result & cache used and adds the result to the cache accordingly

        Args:
            result: Any:
            original_function: Callable:
            kwargs: Dict[str, Any]:
            args: Optional[Tuple[Any, ...]] = None:

        Returns:
            None
        Raises:
            None
        """
        from litellm.litellm_core_utils.core_helpers import (
            get_parent_otel_span_from_kwargs,
        )

        if litellm.cache is None:
            return
        if is_response_without_output(result):
            verbose_logger.debug("LiteLLM Cache: not caching a response with no output")
            return
        cache: Final = litellm.cache

        new_kwargs: Final = kwargs.copy()
        new_kwargs.update(
            convert_args_to_kwargs(
                original_function,
                args,
            )
        )
        parent_otel_span: Final = get_parent_otel_span_from_kwargs(new_kwargs)
        new_kwargs["parent_otel_span"] = parent_otel_span
        # [OPTIONAL] ADD TO CACHE
        if self.should_store_result_in_cache(original_function=original_function, kwargs=new_kwargs):
            if (
                isinstance(result, litellm.ModelResponse)
                or isinstance(result, litellm.EmbeddingResponse)
                or isinstance(result, TranscriptionResponse)
                or isinstance(result, RerankResponse)
                or isinstance(result, ResponsesAPIResponse)
            ):
                if (
                    isinstance(result, EmbeddingResponse)
                    and not isinstance(cache.cache, S3Cache)  # s3 doesn't support bulk writing. Exclude.
                ):
                    create_cache_write_task(
                        lambda: cache.async_add_cache_pipeline(
                            result, dynamic_cache_object=self.dual_cache, **new_kwargs
                        )
                    )
                else:
                    result_json: Final = result.model_dump_json()
                    create_cache_write_task(
                        lambda: cache.async_add_cache(
                            result_json,
                            dynamic_cache_object=self.dual_cache,
                            **new_kwargs,
                        )
                    )
            else:
                create_cache_write_task(lambda: cache.async_add_cache(result, **new_kwargs))

    def sync_set_cache(
        self,
        result: object,
        kwargs: dict[str, object],
        args: tuple[object, ...] | None = None,
    ):
        """
        Sync internal method to add the result to the cache
        """
        if litellm.cache is None:
            return
        if is_response_without_output(result):
            verbose_logger.debug("LiteLLM Cache: not caching a response with no output")
            return

        new_kwargs: Final = kwargs.copy()
        new_kwargs.update(
            convert_args_to_kwargs(
                self.original_function,
                args,
            )
        )

        if self.should_store_result_in_cache(original_function=self.original_function, kwargs=new_kwargs):
            with response_cache_phase("set"):
                litellm.cache.add_cache(result, **new_kwargs)

        return

    def should_store_result_in_cache(
        self, original_function: Callable[..., object], kwargs: Mapping[str, object]
    ) -> bool:
        """
        Helper function to determine if the result should be stored in the cache.

        Returns:
            bool: True if the result should be stored in the cache, False otherwise.
        """
        model_value: Final = kwargs.get("model")
        model: Final = model_value if isinstance(model_value, str) else None
        cache_options_value: Final = kwargs.get("cache")
        cache_options: Final[Mapping[str, object]] = (
            TypeAdapter(Mapping[str, object]).validate_python(cache_options_value)
            if isinstance(cache_options_value, Mapping)
            else {}
        )
        no_store: Final = cache_options.get("no-store", False)
        return (
            not _is_response_cache_excluded(model=model, kwargs=kwargs)
            and self._is_call_type_supported_by_cache(original_function=original_function)
            and no_store is not True
        )

    _should_store_result_in_cache = should_store_result_in_cache

    def wrap_streaming_result_for_cache(
        self, result: _StreamResultT, call_type: str
    ) -> "_StreamResultT | AnthropicMessagesStreamCacheWriter":
        if call_type not in (
            CallTypes.anthropic_messages.value,
            CallTypes.aanthropic_messages.value,
        ):
            return result
        if litellm.cache is None or not self.should_store_result_in_cache(
            original_function=self.original_function, kwargs=self.request_kwargs
        ):
            return result
        if not isinstance(result, AsyncIterator):
            return result
        from litellm.llms.anthropic.pass_through.messages.response_cache import (
            AnthropicMessagesStreamCacheWriter,
        )

        return AnthropicMessagesStreamCacheWriter(stream=result, caching_handler=self)

    def _is_call_type_supported_by_cache(
        self,
        original_function: Callable[..., object],
    ) -> bool:
        """
        Helper function to determine if the call type is supported by the cache.

        call types are acompletion, aembedding, atext_completion, atranscription, arerank

        Defined on `litellm.types.utils.CallTypes`

        Returns:
            bool: True if the call type is supported by the cache, False otherwise.
        """
        if litellm.cache is None or litellm.cache.supported_call_types is None:
            return False
        call_type: Final = str(original_function.__name__)
        covering_call_types: Final = ("aresponses", "responses") if call_type == "aresponses" else (call_type,)
        return any(name in litellm.cache.supported_call_types for name in covering_call_types)

    async def add_streaming_response_to_cache(self, processed_chunk: ModelResponse) -> None:
        """
        Internal method to add the streaming response to the cache


        - If 'streaming_chunk' has a 'finish_reason' then assemble a litellm.ModelResponse object
        - Else append the chunk to self.async_streaming_chunks

        """

        if not self.should_store_result_in_cache(
            original_function=self.original_function,
            kwargs=self.request_kwargs,
        ):
            return
        complete_streaming_response: Final[ModelResponse | TextCompletionResponse | None] = (
            assemble_complete_response_from_streaming_chunks(
                result=processed_chunk,
                start_time=self.start_time,
                end_time=datetime.datetime.now(),
                request_kwargs=self.request_kwargs,
                streaming_chunks=self.async_streaming_chunks,
                is_async=True,
            )
        )
        if complete_streaming_response is None or _assembled_stream_without_output(complete_streaming_response):
            return
        await self.async_set_cache(
            result=complete_streaming_response,
            original_function=self.original_function,
            kwargs=self.request_kwargs,
        )

    _add_streaming_response_to_cache = add_streaming_response_to_cache

    def sync_add_streaming_response_to_cache(self, processed_chunk: ModelResponse) -> None:
        """
        Sync internal method to add the streaming response to the cache
        """
        if not self.should_store_result_in_cache(
            original_function=self.original_function,
            kwargs=self.request_kwargs,
        ):
            return
        complete_streaming_response: Final[ModelResponse | TextCompletionResponse | None] = (
            assemble_complete_response_from_streaming_chunks(
                result=processed_chunk,
                start_time=self.start_time,
                end_time=datetime.datetime.now(),
                request_kwargs=self.request_kwargs,
                streaming_chunks=self.sync_streaming_chunks,
                is_async=False,
            )
        )

        if complete_streaming_response is None or _assembled_stream_without_output(complete_streaming_response):
            return
        self.sync_set_cache(
            result=complete_streaming_response,
            kwargs=self.request_kwargs,
        )

    _sync_add_streaming_response_to_cache = sync_add_streaming_response_to_cache

    def _update_litellm_logging_obj_environment(
        self,
        logging_obj: LiteLLMLoggingObj,
        model: str,
        kwargs: dict[str, Any],
        cached_result: object,
        is_async: bool,
        is_embedding: bool = False,
        custom_llm_provider: str | None = None,
        cache_duration_ms: float | None = None,
    ):
        """
        Helper function to update the LiteLLMLoggingObj environment variables.

        Args:
            logging_obj (LiteLLMLoggingObj): The logging object to update.
            model (str): The model being used.
            kwargs (Dict[str, Any]): The keyword arguments from the original function call.
            cached_result (Any): The cached result to log.
            is_async (bool): Whether the call is asynchronous or not.
            is_embedding (bool): Whether the call is for embeddings or not.
            custom_llm_provider (Optional[str]): The custom llm provider being used.

        Returns:
            None
        """
        litellm_params: Final = {
            "logger_fn": kwargs.get("logger_fn", None),
            "acompletion": is_async,
            "api_base": kwargs.get("api_base", ""),
            "metadata": kwargs.get("metadata", {}),
            "model_info": kwargs.get("model_info", {}),
            "proxy_server_request": kwargs.get("proxy_server_request", None),
            "stream_response": kwargs.get("stream_response", {}),
            "custom_llm_provider": custom_llm_provider,
        }

        if litellm.cache is not None:
            litellm_params["preset_cache_key"] = (
                self.preset_cache_key or litellm.cache.get_preset_cache_key_from_kwargs(**kwargs)
            )
        else:
            litellm_params["preset_cache_key"] = None

        logging_obj.update_environment_variables(
            model=model,
            user=kwargs.get("user", None),
            optional_params={},
            litellm_params=litellm_params,
            input=(kwargs.get("messages", "") if not is_embedding else kwargs.get("input", "")),
            api_key=kwargs.get("api_key", None),
            original_response=str(cached_result),
            additional_args=None,
            stream=kwargs.get("stream", False),
            custom_llm_provider=custom_llm_provider,
        )

        logging_obj.caching_details = CachingDetails(
            cache_hit=True,
            cache_duration_ms=cache_duration_ms,
        )


def convert_args_to_kwargs(
    original_function: Callable,
    args: tuple[object, ...] | None = None,
) -> dict[str, object]:
    # Get the signature of the original function
    signature: Final = inspect.signature(original_function)

    # Get parameter names in the order they appear in the original function
    param_names: Final = list(signature.parameters.keys())

    # Create a mapping of positional arguments to parameter names
    args_to_kwargs: Final = {}
    if args:
        for index, arg in enumerate(args):
            if index < len(param_names):
                param_name = param_names[index]
                args_to_kwargs[param_name] = arg

    return args_to_kwargs
