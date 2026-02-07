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
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

from pydantic import BaseModel

import litellm
from litellm._logging import print_verbose, verbose_logger
from litellm.caching import InMemoryCache
from litellm.caching.caching import S3Cache
from litellm.litellm_core_utils.llm_response_utils.response_metadata import (
    update_response_metadata,
)
from litellm.litellm_core_utils.logging_utils import (
    _assemble_complete_response_from_streaming_chunks,
)
from litellm.types.caching import CachedEmbedding
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.rerank import RerankResponse
from litellm.types.utils import (
    CachingDetails,
    CallTypes,
    Embedding,
    EmbeddingResponse,
    ModelResponse,
    TextCompletionResponse,
    TranscriptionResponse,
    Usage,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


from litellm.litellm_core_utils.core_helpers import (
    _get_parent_otel_span_from_kwargs,
)
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper


class CachingHandlerResponse(BaseModel):
    """
    This is the response object for the caching handler. We need to separate embedding cached responses and (completion / text_completion / transcription) cached responses

    For embeddings there can be a cache hit for some of the inputs in the list and a cache miss for others
    """

    cached_result: Optional[Any] = None
    final_embedding_cached_response: Optional[EmbeddingResponse] = None
    embedding_all_elements_cache_hit: bool = (
        False  # this is set to True when all elements in the list have a cache hit in the embedding cache, if true return the final_embedding_cached_response no need to make an API call
    )


in_memory_cache_obj = InMemoryCache()


class LLMCachingHandler:
    def __init__(
        self,
        original_function: Callable,
        request_kwargs: Dict[str, Any],
        start_time: datetime.datetime,
    ):
        from litellm.caching import DualCache, RedisCache

        self.async_streaming_chunks: List[ModelResponse] = []
        self.sync_streaming_chunks: List[ModelResponse] = []
        self.request_kwargs = request_kwargs
        self.original_function = original_function
        self.start_time = start_time
        if litellm.cache is not None and isinstance(litellm.cache.cache, RedisCache):
            self.dual_cache: Optional[DualCache] = DualCache(
                redis_cache=litellm.cache.cache,
                in_memory_cache=in_memory_cache_obj,
            )
        else:
            self.dual_cache = None
        pass

    async def _async_get_cache(
        self,
        model: str,
        original_function: Callable,
        logging_obj: LiteLLMLoggingObj,
        start_time: datetime.datetime,
        call_type: str,
        kwargs: Dict[str, Any],
        args: Optional[Tuple[Any, ...]] = None,
    ) -> Optional[CachingHandlerResponse]:
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
        # Check if caching should be performed BEFORE doing expensive operations
        if (
            (kwargs.get("caching", None) is None and litellm.cache is not None)
            or kwargs.get("caching", False) is True
        ) and (
            kwargs.get("cache", {}).get("no-cache", False) is not True
        ):  # allow users to control returning cached responses from the completion function
            args = args or ()
            final_embedding_cached_response: Optional[EmbeddingResponse] = None
            embedding_all_elements_cache_hit: bool = False
            cached_result: Optional[Any] = None
            kwargs = kwargs.copy()
            #########################################################
            # Init cache timing metrics
            #########################################################
            cache_check_start_time = time.perf_counter()
            cache_check_end_time: Optional[float] = None
            #########################################################
            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
            kwargs["parent_otel_span"] = parent_otel_span
            
            if litellm.cache is not None and self._is_call_type_supported_by_cache(
                original_function=original_function
            ):
                verbose_logger.debug("Checking Async Cache")
                cached_result = await self._retrieve_from_cache(
                    call_type=call_type,
                    kwargs=kwargs,
                    args=args,
                )
                cache_check_end_time = time.perf_counter()

                if cached_result is not None and not isinstance(cached_result, list):
                    verbose_logger.debug("Cache Hit!")
                    cache_hit = True
                    end_time = datetime.datetime.now()
                    model, custom_llm_provider, _, _ = litellm.get_llm_provider(
                        model=model,
                        custom_llm_provider=kwargs.get("custom_llm_provider", None),
                        api_base=kwargs.get("api_base", None),
                        api_key=kwargs.get("api_key", None),
                    )
                    cache_duration_ms = (cache_check_end_time - cache_check_start_time) * 1000
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
                    if kwargs.get("stream", False) is False:
                        # LOG SUCCESS
                        self._async_log_cache_hit_on_callbacks(
                            logging_obj=logging_obj,
                            cached_result=cached_result,
                            start_time=start_time,
                            end_time=end_time,
                            cache_hit=cache_hit,
                        )
                    cache_key = litellm.cache.get_cache_key(**kwargs)
                    if (
                        isinstance(cached_result, BaseModel)
                        or isinstance(cached_result, CustomStreamWrapper)
                    ) and hasattr(cached_result, "_hidden_params"):
                        cached_result._hidden_params["cache_key"] = cache_key  # type: ignore
                    return CachingHandlerResponse(cached_result=cached_result)
                elif (
                    call_type == CallTypes.aembedding.value
                    and cached_result is not None
                    and isinstance(cached_result, list)
                    and litellm.cache is not None
                    and not isinstance(
                        litellm.cache.cache, S3Cache
                    )  # s3 doesn't support bulk writing. Exclude.
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
                    )
        
            verbose_logger.debug(f"CACHE RESULT: {cached_result}")
            return CachingHandlerResponse(
                cached_result=cached_result,
                final_embedding_cached_response=final_embedding_cached_response,
            )
        # Caching disabled - return None to indicate no caching attempted
        return None

    def _sync_get_cache(
        self,
        model: str,
        original_function: Callable,
        logging_obj: LiteLLMLoggingObj,
        start_time: datetime.datetime,
        call_type: str,
        kwargs: Dict[str, Any],
        args: Optional[Tuple[Any, ...]] = None,
    ) -> CachingHandlerResponse:
        from litellm.utils import CustomStreamWrapper


        cached_result: Optional[Any] = None
        
        # Check if caching should be performed BEFORE doing expensive kwargs copy
        if litellm.cache is not None and self._is_call_type_supported_by_cache(
            original_function=original_function
        ):
            args = args or ()
            # Now that we confirmed caching will happen, prepare kwargs
            new_kwargs = kwargs.copy()
            new_kwargs.update(
                convert_args_to_kwargs(
                    self.original_function,
                    args,
                )
            )
            print_verbose("Checking Sync Cache")
            cached_result = litellm.cache.get_cache(**new_kwargs)
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
                    cache_hit = True
                    end_time = datetime.datetime.now()
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
                    )

                    logging_obj.handle_sync_success_callbacks_for_async_calls(
                        result=cached_result,
                        start_time=start_time,
                        end_time=end_time,
                        cache_hit=cache_hit
                    )
                    cache_key = litellm.cache.get_cache_key(**kwargs)
                    if (
                        isinstance(cached_result, BaseModel)
                        or isinstance(cached_result, CustomStreamWrapper)
                    ) and hasattr(cached_result, "_hidden_params"):
                        cached_result._hidden_params["cache_key"] = cache_key  # type: ignore
                    return CachingHandlerResponse(cached_result=cached_result)
        return CachingHandlerResponse(cached_result=cached_result)

    def handle_kwargs_input_list_or_str(self, kwargs: Dict[str, Any]) -> List[str]:
        """
        Handles the input of kwargs['input'] being a list or a string
        """
        if isinstance(kwargs["input"], str):
            return [kwargs["input"]]
        elif isinstance(kwargs["input"], list):
            return kwargs["input"]
        else:
            raise ValueError("input must be a string or a list")

    def _extract_model_from_cached_results(
        self, non_null_list: List[Tuple[int, CachedEmbedding]]
    ) -> Optional[str]:
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
        final_embedding_cached_response: Optional[EmbeddingResponse],
        cached_result: List[Optional[CachedEmbedding]],
        kwargs: Dict[str, Any],
        logging_obj: LiteLLMLoggingObj,
        start_time: datetime.datetime,
        model: str,
    ) -> Tuple[Optional[EmbeddingResponse], bool]:
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
        remaining_list = []
        non_null_list = []
        kwargs_input_as_list = self.handle_kwargs_input_list_or_str(kwargs)
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
            final_embedding_cached_response._hidden_params["cache_hit"] = True

            prompt_tokens = 0
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
                    if isinstance(kwargs_input_as_list[idx], str):
                        from litellm.utils import token_counter

                        prompt_tokens += token_counter(
                            text=kwargs_input_as_list[idx], count_response_tokens=True
                        )
            ## USAGE
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=prompt_tokens,
            )
            final_embedding_cached_response.usage = usage
        if len(remaining_list) == 0:
            # LOG SUCCESS
            cache_hit = True
            embedding_all_elements_cache_hit = True
            end_time = datetime.datetime.now()
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
        )

    def _combine_cached_embedding_response_with_api_result(
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

        idx = 0
        final_data_list = []
        for item in _caching_handler_response.final_embedding_cached_response.data:
            if item is None and embedding_response.data is not None:
                final_data_list.append(embedding_response.data[idx])
                idx += 1
            else:
                final_data_list.append(item)

        _caching_handler_response.final_embedding_cached_response.data = final_data_list
        _caching_handler_response.final_embedding_cached_response._hidden_params[
            "cache_hit"
        ] = True
        _caching_handler_response.final_embedding_cached_response._response_ms = (
            end_time - start_time
        ).total_seconds() * 1000

        ## USAGE
        if (
            _caching_handler_response.final_embedding_cached_response.usage is not None
            and embedding_response.usage is not None
        ):
            _caching_handler_response.final_embedding_cached_response.usage = self.combine_usage(
                usage1=_caching_handler_response.final_embedding_cached_response.usage,
                usage2=embedding_response.usage,
            )

        return _caching_handler_response.final_embedding_cached_response

    def _async_log_cache_hit_on_callbacks(
        self,
        logging_obj: LiteLLMLoggingObj,
        cached_result: Any,
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
        """
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(
            async_coroutine=logging_obj.async_success_handler(
                result=cached_result, start_time=start_time, end_time=end_time, cache_hit=cache_hit
            )
        )

        logging_obj.handle_sync_success_callbacks_for_async_calls(
            result=cached_result, start_time=start_time, end_time=end_time, cache_hit=cache_hit
        )

    async def _retrieve_from_cache(
        self, call_type: str, kwargs: Dict[str, Any], args: Tuple[Any, ...]
    ) -> Optional[Any]:
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

        new_kwargs = kwargs.copy()
        new_kwargs.update(
            convert_args_to_kwargs(
                self.original_function,
                args,
            )
        )
        cached_result: Optional[Any] = None
        if call_type == CallTypes.aembedding.value:
            if isinstance(new_kwargs["input"], str):
                new_kwargs["input"] = [new_kwargs["input"]]
            elif not isinstance(new_kwargs["input"], list):
                raise ValueError("input must be a string or a list")
            tasks = []
            for idx, i in enumerate(new_kwargs["input"]):
                preset_cache_key = litellm.cache.get_cache_key(
                    **{**new_kwargs, "input": i}
                )
                tasks.append(
                    litellm.cache.async_get_cache(
                        cache_key=preset_cache_key,
                        dynamic_cache_object=self.dual_cache,
                    )
                )
            cached_result = await asyncio.gather(*tasks)
            ## check if cached result is None ##
            if cached_result is not None and isinstance(cached_result, list):
                # set cached_result to None if all elements are None
                if all(result is None for result in cached_result):
                    cached_result = None
        else:
            if litellm.cache._supports_async() is True:
                ## check if dual cache is supported ##
                cached_result = await litellm.cache.async_get_cache(
                    dynamic_cache_object=self.dual_cache, **new_kwargs
                )
            else:  # fallback for caches that don't support async
                cached_result = litellm.cache.get_cache(
                    dynamic_cache_object=self.dual_cache, **new_kwargs
                )
        return cached_result

    def _convert_cached_result_to_model_response(
        self,
        cached_result: Any,
        call_type: str,
        kwargs: Dict[str, Any],
        logging_obj: LiteLLMLoggingObj,
        model: str,
        args: Tuple[Any, ...],
        custom_llm_provider: Optional[str] = None,
    ) -> Optional[
        Union[
            ModelResponse,
            TextCompletionResponse,
            EmbeddingResponse,
            RerankResponse,
            TranscriptionResponse,
            CustomStreamWrapper,
        ]
    ]:
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

        if (
            call_type == CallTypes.acompletion.value
            or call_type == CallTypes.completion.value
        ) and isinstance(cached_result, dict):
            if kwargs.get("stream", False) is True:
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
            call_type == CallTypes.atext_completion.value
            or call_type == CallTypes.text_completion.value
        ) and isinstance(cached_result, dict):
            if kwargs.get("stream", False) is True:
                cached_result = self._convert_cached_stream_response(
                    cached_result=cached_result,
                    call_type=call_type,
                    logging_obj=logging_obj,
                    model=model,
                )
            else:
                cached_result = TextCompletionResponse(**cached_result)
        elif (
            call_type == CallTypes.aembedding.value
            or call_type == CallTypes.embedding.value
        ) and isinstance(cached_result, dict):
            cached_result = convert_to_model_response_object(
                response_object=cached_result,
                model_response_object=EmbeddingResponse(),
                response_type="embedding",
            )

        elif (
            call_type == CallTypes.arerank.value or call_type == CallTypes.rerank.value
        ) and isinstance(cached_result, dict):
            cached_result = convert_to_model_response_object(
                response_object=cached_result,
                model_response_object=None,
                response_type="rerank",
            )
        elif (
            call_type == CallTypes.atranscription.value
            or call_type == CallTypes.transcription.value
        ) and isinstance(cached_result, dict):
            hidden_params = {
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
            call_type == "aresponses"
            or call_type == "responses"
        ) and isinstance(cached_result, dict):
            # Convert cached dict back to ResponsesAPIResponse object
            cached_result = ResponsesAPIResponse(**cached_result)

        if (
            hasattr(cached_result, "_hidden_params")
            and cached_result._hidden_params is not None
            and isinstance(cached_result._hidden_params, dict)
        ):
            cached_result._hidden_params["cache_hit"] = True
        
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
        cached_result: Any,
        call_type: str,
        logging_obj: LiteLLMLoggingObj,
        model: str,
    ) -> CustomStreamWrapper:
        from litellm.utils import (
            CustomStreamWrapper,
            convert_to_streaming_response,
            convert_to_streaming_response_async,
        )

        _stream_cached_result: Union[AsyncGenerator, Generator]
        if (
            call_type == CallTypes.acompletion.value
            or call_type == CallTypes.atext_completion.value
        ):
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
        result: Any,
        original_function: Callable,
        kwargs: Dict[str, Any],
        args: Optional[Tuple[Any, ...]] = None,
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
            _get_parent_otel_span_from_kwargs,
        )

        if litellm.cache is None:
            return

        new_kwargs = kwargs.copy()
        new_kwargs.update(
            convert_args_to_kwargs(
                original_function,
                args,
            )
        )
        parent_otel_span = _get_parent_otel_span_from_kwargs(new_kwargs)
        new_kwargs["parent_otel_span"] = parent_otel_span
        # [OPTIONAL] ADD TO CACHE
        if self._should_store_result_in_cache(
            original_function=original_function, kwargs=new_kwargs
        ):
            if (
                isinstance(result, litellm.ModelResponse)
                or isinstance(result, litellm.EmbeddingResponse)
                or isinstance(result, TranscriptionResponse)
                or isinstance(result, RerankResponse)
                or isinstance(result, ResponsesAPIResponse)
            ):
                if (
                    isinstance(result, EmbeddingResponse)
                    and litellm.cache is not None
                    and not isinstance(
                        litellm.cache.cache, S3Cache
                    )  # s3 doesn't support bulk writing. Exclude.
                ):
                    asyncio.create_task(
                        litellm.cache.async_add_cache_pipeline(
                            result, dynamic_cache_object=self.dual_cache, **new_kwargs
                        )
                    )
                else:
                    asyncio.create_task(
                        litellm.cache.async_add_cache(
                            result.model_dump_json(),
                            dynamic_cache_object=self.dual_cache,
                            **new_kwargs,
                        )
                    )
            else:
                asyncio.create_task(litellm.cache.async_add_cache(result, **new_kwargs))

    def sync_set_cache(
        self,
        result: Any,
        kwargs: Dict[str, Any],
        args: Optional[Tuple[Any, ...]] = None,
    ):
        """
        Sync internal method to add the result to the cache
        """

        new_kwargs = kwargs.copy()
        new_kwargs.update(
            convert_args_to_kwargs(
                self.original_function,
                args,
            )
        )
        if litellm.cache is None:
            return

        if self._should_store_result_in_cache(
            original_function=self.original_function, kwargs=new_kwargs
        ):
            litellm.cache.add_cache(result, **new_kwargs)

        return

    def _should_store_result_in_cache(
        self, original_function: Callable, kwargs: Dict[str, Any]
    ) -> bool:
        """
        Helper function to determine if the result should be stored in the cache.

        Returns:
            bool: True if the result should be stored in the cache, False otherwise.
        """
        return (
            (litellm.cache is not None)
            and litellm.cache.supported_call_types is not None
            and (str(original_function.__name__) in litellm.cache.supported_call_types)
            and (kwargs.get("cache", {}).get("no-store", False) is not True)
        )

    def _is_call_type_supported_by_cache(
        self,
        original_function: Callable,
    ) -> bool:
        """
        Helper function to determine if the call type is supported by the cache.

        call types are acompletion, aembedding, atext_completion, atranscription, arerank

        Defined on `litellm.types.utils.CallTypes`

        Returns:
            bool: True if the call type is supported by the cache, False otherwise.
        """
        if (
            litellm.cache is not None
            and litellm.cache.supported_call_types is not None
            and str(original_function.__name__) in litellm.cache.supported_call_types
        ):
            return True
        return False

    async def _add_streaming_response_to_cache(self, processed_chunk: ModelResponse):
        """
        Internal method to add the streaming response to the cache


        - If 'streaming_chunk' has a 'finish_reason' then assemble a litellm.ModelResponse object
        - Else append the chunk to self.async_streaming_chunks

        """

        complete_streaming_response: Optional[
            Union[ModelResponse, TextCompletionResponse]
        ] = _assemble_complete_response_from_streaming_chunks(
            result=processed_chunk,
            start_time=self.start_time,
            end_time=datetime.datetime.now(),
            request_kwargs=self.request_kwargs,
            streaming_chunks=self.async_streaming_chunks,
            is_async=True,
        )
        # if a complete_streaming_response is assembled, add it to the cache
        if complete_streaming_response is not None:
            await self.async_set_cache(
                result=complete_streaming_response,
                original_function=self.original_function,
                kwargs=self.request_kwargs,
            )

    def _sync_add_streaming_response_to_cache(self, processed_chunk: ModelResponse):
        """
        Sync internal method to add the streaming response to the cache
        """
        complete_streaming_response: Optional[
            Union[ModelResponse, TextCompletionResponse]
        ] = _assemble_complete_response_from_streaming_chunks(
            result=processed_chunk,
            start_time=self.start_time,
            end_time=datetime.datetime.now(),
            request_kwargs=self.request_kwargs,
            streaming_chunks=self.sync_streaming_chunks,
            is_async=False,
        )

        # if a complete_streaming_response is assembled, add it to the cache
        if complete_streaming_response is not None:
            self.sync_set_cache(
                result=complete_streaming_response,
                kwargs=self.request_kwargs,
            )

    def _update_litellm_logging_obj_environment(
        self,
        logging_obj: LiteLLMLoggingObj,
        model: str,
        kwargs: Dict[str, Any],
        cached_result: Any,
        is_async: bool,
        is_embedding: bool = False,
        custom_llm_provider: Optional[str] = None,
        cache_duration_ms: Optional[float] = None,
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
        litellm_params = {
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
                litellm.cache._get_preset_cache_key_from_kwargs(**kwargs)
            )
        else:
            litellm_params["preset_cache_key"] = None

        logging_obj.update_environment_variables(
            model=model,
            user=kwargs.get("user", None),
            optional_params={},
            litellm_params=litellm_params,
            input=(
                kwargs.get("messages", "")
                if not is_embedding
                else kwargs.get("input", "")
            ),
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
    args: Optional[Tuple[Any, ...]] = None,
) -> Dict[str, Any]:
    # Get the signature of the original function
    signature = inspect.signature(original_function)

    # Get parameter names in the order they appear in the original function
    param_names = list(signature.parameters.keys())

    # Create a mapping of positional arguments to parameter names
    args_to_kwargs = {}
    if args:
        for index, arg in enumerate(args):
            if index < len(param_names):
                param_name = param_names[index]
                args_to_kwargs[param_name] = arg

    return args_to_kwargs
