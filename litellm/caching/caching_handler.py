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
import threading
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
from litellm.caching.caching import (
    Cache,
    QdrantSemanticCache,
    RedisCache,
    RedisSemanticCache,
    S3Cache,
)
from litellm.litellm_core_utils.logging_utils import (
    _assemble_complete_response_from_streaming_chunks,
)
from litellm.types.rerank import RerankResponse
from litellm.types.utils import (
    CallTypes,
    Embedding,
    EmbeddingResponse,
    ModelResponse,
    TextCompletionResponse,
    TranscriptionResponse,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.utils import CustomStreamWrapper
else:
    LiteLLMLoggingObj = Any
    CustomStreamWrapper = Any


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


class LLMCachingHandler:
    def __init__(
        self,
        original_function: Callable,
        request_kwargs: Dict[str, Any],
        start_time: datetime.datetime,
    ):
        self.async_streaming_chunks: List[ModelResponse] = []
        self.sync_streaming_chunks: List[ModelResponse] = []
        self.request_kwargs = request_kwargs
        self.original_function = original_function
        self.start_time = start_time
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
    ) -> CachingHandlerResponse:
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
        from litellm.utils import CustomStreamWrapper

        args = args or ()

        final_embedding_cached_response: Optional[EmbeddingResponse] = None
        embedding_all_elements_cache_hit: bool = False
        cached_result: Optional[Any] = None
        if (
            (kwargs.get("caching", None) is None and litellm.cache is not None)
            or kwargs.get("caching", False) is True
        ) and (
            kwargs.get("cache", {}).get("no-cache", False) is not True
        ):  # allow users to control returning cached responses from the completion function
            if litellm.cache is not None and self._is_call_type_supported_by_cache(
                original_function=original_function
            ):
                print_verbose("Checking Cache")
                cached_result = await self._retrieve_from_cache(
                    call_type=call_type,
                    kwargs=kwargs,
                    args=args,
                )

                if cached_result is not None and not isinstance(cached_result, list):
                    print_verbose("Cache Hit!")
                    cache_hit = True
                    end_time = datetime.datetime.now()
                    model, _, _, _ = litellm.get_llm_provider(
                        model=model,
                        custom_llm_provider=kwargs.get("custom_llm_provider", None),
                        api_base=kwargs.get("api_base", None),
                        api_key=kwargs.get("api_key", None),
                    )
                    self._update_litellm_logging_obj_environment(
                        logging_obj=logging_obj,
                        model=model,
                        kwargs=kwargs,
                        cached_result=cached_result,
                        is_async=True,
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
                    cache_key = litellm.cache._get_preset_cache_key_from_kwargs(
                        **kwargs
                    )
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
        return CachingHandlerResponse(
            cached_result=cached_result,
            final_embedding_cached_response=final_embedding_cached_response,
        )

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

        args = args or ()
        cached_result: Optional[Any] = None
        if litellm.cache is not None and self._is_call_type_supported_by_cache(
            original_function=original_function
        ):
            print_verbose("Checking Cache")
            cached_result = litellm.cache.get_cache(*args, **kwargs)
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
                        model=model,
                        kwargs=kwargs,
                        cached_result=cached_result,
                        is_async=False,
                    )

                    threading.Thread(
                        target=logging_obj.success_handler,
                        args=(cached_result, start_time, end_time, cache_hit),
                    ).start()
                    cache_key = litellm.cache._get_preset_cache_key_from_kwargs(
                        **kwargs
                    )
                    if (
                        isinstance(cached_result, BaseModel)
                        or isinstance(cached_result, CustomStreamWrapper)
                    ) and hasattr(cached_result, "_hidden_params"):
                        cached_result._hidden_params["cache_key"] = cache_key  # type: ignore
                    return CachingHandlerResponse(cached_result=cached_result)
        return CachingHandlerResponse(cached_result=cached_result)

    def _process_async_embedding_cached_response(
        self,
        final_embedding_cached_response: Optional[EmbeddingResponse],
        cached_result: List[Optional[Dict[str, Any]]],
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
        for idx, cr in enumerate(cached_result):
            if cr is None:
                remaining_list.append(kwargs["input"][idx])
            else:
                non_null_list.append((idx, cr))
        original_kwargs_input = kwargs["input"]
        kwargs["input"] = remaining_list
        if len(non_null_list) > 0:
            print_verbose(f"EMBEDDING CACHE HIT! - {len(non_null_list)}")
            final_embedding_cached_response = EmbeddingResponse(
                model=kwargs.get("model"),
                data=[None] * len(original_kwargs_input),
            )
            final_embedding_cached_response._hidden_params["cache_hit"] = True

            for val in non_null_list:
                idx, cr = val  # (idx, cr) tuple
                if cr is not None:
                    final_embedding_cached_response.data[idx] = Embedding(
                        embedding=cr["embedding"],
                        index=idx,
                        object="embedding",
                    )
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
        asyncio.create_task(
            logging_obj.async_success_handler(
                cached_result, start_time, end_time, cache_hit
            )
        )
        threading.Thread(
            target=logging_obj.success_handler,
            args=(cached_result, start_time, end_time, cache_hit),
        ).start()

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

        cached_result: Optional[Any] = None
        if call_type == CallTypes.aembedding.value and isinstance(
            kwargs["input"], list
        ):
            tasks = []
            for idx, i in enumerate(kwargs["input"]):
                preset_cache_key = litellm.cache.get_cache_key(
                    *args, **{**kwargs, "input": i}
                )
                tasks.append(litellm.cache.async_get_cache(cache_key=preset_cache_key))
            cached_result = await asyncio.gather(*tasks)
            ## check if cached result is None ##
            if cached_result is not None and isinstance(cached_result, list):
                # set cached_result to None if all elements are None
                if all(result is None for result in cached_result):
                    cached_result = None
        else:
            if litellm.cache._supports_async() is True:
                cached_result = await litellm.cache.async_get_cache(*args, **kwargs)
            else:  # for s3 caching. [NOT RECOMMENDED IN PROD - this will slow down responses since boto3 is sync]
                cached_result = litellm.cache.get_cache(*args, **kwargs)
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
        from litellm.utils import (
            CustomStreamWrapper,
            convert_to_model_response_object,
            convert_to_streaming_response,
            convert_to_streaming_response_async,
        )

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
        kwargs.update(convert_args_to_kwargs(result, original_function, kwargs, args))
        if litellm.cache is None:
            return
        # [OPTIONAL] ADD TO CACHE
        if self._should_store_result_in_cache(
            original_function=original_function, kwargs=kwargs
        ):
            if (
                isinstance(result, litellm.ModelResponse)
                or isinstance(result, litellm.EmbeddingResponse)
                or isinstance(result, TranscriptionResponse)
                or isinstance(result, RerankResponse)
            ):
                if (
                    isinstance(result, EmbeddingResponse)
                    and isinstance(kwargs["input"], list)
                    and litellm.cache is not None
                    and not isinstance(
                        litellm.cache.cache, S3Cache
                    )  # s3 doesn't support bulk writing. Exclude.
                ):
                    asyncio.create_task(
                        litellm.cache.async_add_cache_pipeline(result, **kwargs)
                    )
                elif isinstance(litellm.cache.cache, S3Cache):
                    threading.Thread(
                        target=litellm.cache.add_cache,
                        args=(result,),
                        kwargs=kwargs,
                    ).start()
                else:
                    asyncio.create_task(
                        litellm.cache.async_add_cache(
                            result.model_dump_json(), **kwargs
                        )
                    )
            else:
                asyncio.create_task(litellm.cache.async_add_cache(result, **kwargs))

    def sync_set_cache(
        self,
        result: Any,
        kwargs: Dict[str, Any],
        args: Optional[Tuple[Any, ...]] = None,
    ):
        """
        Sync internal method to add the result to the cache
        """
        kwargs.update(
            convert_args_to_kwargs(result, self.original_function, kwargs, args)
        )
        if litellm.cache is None:
            return

        if self._should_store_result_in_cache(
            original_function=self.original_function, kwargs=kwargs
        ):
            litellm.cache.add_cache(result, **kwargs)

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
        )


def convert_args_to_kwargs(
    result: Any,
    original_function: Callable,
    kwargs: Dict[str, Any],
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
