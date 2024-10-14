"""
This contains LLMCachingHandler 

This exposes two methods:
    - async_get_cache
    - async_set_cache

This file is a wrapper around caching.py

In each method it will call the appropriate method from caching.py
"""

import asyncio
import datetime
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel

import litellm
from litellm._logging import print_verbose
from litellm.caching.caching import (
    Cache,
    QdrantSemanticCache,
    RedisCache,
    RedisSemanticCache,
    S3Cache,
)
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
else:
    LiteLLMLoggingObj = Any


class CachingHandlerResponse(BaseModel):
    """
    This is the response object for the caching handler. We need to separate embedding cached responses and (completion / text_completion / transcription) cached responses

    For embeddings there can be a cache hit for some of the inputs in the list and a cache miss for others
    """

    cached_result: Optional[Any] = None
    final_embedding_cached_response: Optional[EmbeddingResponse] = None


class LLMCachingHandler:
    def __init__(self):
        pass

    async def async_get_cache(
        self,
        model: str,
        original_function: Callable,
        logging_obj: LiteLLMLoggingObj,
        start_time: datetime.datetime,
        call_type: str,
        kwargs: Dict[str, Any],
        args: Optional[Tuple[Any, ...]] = None,
    ) -> CachingHandlerResponse:
        from litellm.utils import (
            CustomStreamWrapper,
            convert_to_model_response_object,
            convert_to_streaming_response_async,
        )

        final_embedding_cached_response: Optional[EmbeddingResponse] = None
        cached_result: Optional[Any] = None
        if (
            (kwargs.get("caching", None) is None and litellm.cache is not None)
            or kwargs.get("caching", False) is True
        ) and (
            kwargs.get("cache", {}).get("no-cache", False) is not True
        ):  # allow users to control returning cached responses from the completion function
            # checking cache
            print_verbose("INSIDE CHECKING CACHE")
            if (
                litellm.cache is not None
                and litellm.cache.supported_call_types is not None
                and str(original_function.__name__)
                in litellm.cache.supported_call_types
            ):
                print_verbose("Checking Cache")
                if call_type == CallTypes.aembedding.value and isinstance(
                    kwargs["input"], list
                ):
                    tasks = []
                    for idx, i in enumerate(kwargs["input"]):
                        preset_cache_key = litellm.cache.get_cache_key(
                            *args, **{**kwargs, "input": i}
                        )
                        tasks.append(
                            litellm.cache.async_get_cache(cache_key=preset_cache_key)
                        )
                    cached_result = await asyncio.gather(*tasks)
                    ## check if cached result is None ##
                    if cached_result is not None and isinstance(cached_result, list):
                        if len(cached_result) == 1 and cached_result[0] is None:
                            cached_result = None
                elif isinstance(litellm.cache.cache, RedisSemanticCache) or isinstance(
                    litellm.cache.cache, RedisCache
                ):
                    preset_cache_key = litellm.cache.get_cache_key(*args, **kwargs)
                    kwargs["preset_cache_key"] = (
                        preset_cache_key  # for streaming calls, we need to pass the preset_cache_key
                    )
                    cached_result = await litellm.cache.async_get_cache(*args, **kwargs)
                elif isinstance(litellm.cache.cache, QdrantSemanticCache):
                    preset_cache_key = litellm.cache.get_cache_key(*args, **kwargs)
                    kwargs["preset_cache_key"] = (
                        preset_cache_key  # for streaming calls, we need to pass the preset_cache_key
                    )
                    cached_result = await litellm.cache.async_get_cache(*args, **kwargs)
                else:  # for s3 caching. [NOT RECOMMENDED IN PROD - this will slow down responses since boto3 is sync]
                    preset_cache_key = litellm.cache.get_cache_key(*args, **kwargs)
                    kwargs["preset_cache_key"] = (
                        preset_cache_key  # for streaming calls, we need to pass the preset_cache_key
                    )
                    cached_result = litellm.cache.get_cache(*args, **kwargs)
                if cached_result is not None and not isinstance(cached_result, list):
                    print_verbose("Cache Hit!")
                    cache_hit = True
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
                    print_verbose(
                        f"Async Wrapper: Completed Call, calling async_success_handler: {logging_obj.async_success_handler}"
                    )
                    logging_obj.update_environment_variables(
                        model=model,
                        user=kwargs.get("user", None),
                        optional_params={},
                        litellm_params={
                            "logger_fn": kwargs.get("logger_fn", None),
                            "acompletion": True,
                            "metadata": kwargs.get("metadata", {}),
                            "model_info": kwargs.get("model_info", {}),
                            "proxy_server_request": kwargs.get(
                                "proxy_server_request", None
                            ),
                            "preset_cache_key": kwargs.get("preset_cache_key", None),
                            "stream_response": kwargs.get("stream_response", {}),
                            "api_base": kwargs.get("api_base", ""),
                        },
                        input=kwargs.get("messages", ""),
                        api_key=kwargs.get("api_key", None),
                        original_response=str(cached_result),
                        additional_args=None,
                        stream=kwargs.get("stream", False),
                    )
                    call_type = original_function.__name__
                    if call_type == CallTypes.acompletion.value and isinstance(
                        cached_result, dict
                    ):
                        if kwargs.get("stream", False) is True:
                            cached_result = convert_to_streaming_response_async(
                                response_object=cached_result,
                            )
                            cached_result = CustomStreamWrapper(
                                completion_stream=cached_result,
                                model=model,
                                custom_llm_provider="cached_response",
                                logging_obj=logging_obj,
                            )
                        else:
                            cached_result = convert_to_model_response_object(
                                response_object=cached_result,
                                model_response_object=ModelResponse(),
                            )
                    if call_type == CallTypes.atext_completion.value and isinstance(
                        cached_result, dict
                    ):
                        if kwargs.get("stream", False) is True:
                            cached_result = convert_to_streaming_response_async(
                                response_object=cached_result,
                            )
                            cached_result = CustomStreamWrapper(
                                completion_stream=cached_result,
                                model=model,
                                custom_llm_provider="cached_response",
                                logging_obj=logging_obj,
                            )
                        else:
                            cached_result = TextCompletionResponse(**cached_result)
                    elif call_type == CallTypes.aembedding.value and isinstance(
                        cached_result, dict
                    ):
                        cached_result = convert_to_model_response_object(
                            response_object=cached_result,
                            model_response_object=EmbeddingResponse(),
                            response_type="embedding",
                        )
                    elif call_type == CallTypes.arerank.value and isinstance(
                        cached_result, dict
                    ):
                        cached_result = convert_to_model_response_object(
                            response_object=cached_result,
                            model_response_object=None,
                            response_type="rerank",
                        )
                    elif call_type == CallTypes.atranscription.value and isinstance(
                        cached_result, dict
                    ):
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
                    if kwargs.get("stream", False) is False:
                        # LOG SUCCESS
                        asyncio.create_task(
                            logging_obj.async_success_handler(
                                cached_result, start_time, end_time, cache_hit
                            )
                        )
                        threading.Thread(
                            target=logging_obj.success_handler,
                            args=(cached_result, start_time, end_time, cache_hit),
                        ).start()
                    cache_key = kwargs.get("preset_cache_key", None)
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
                        final_embedding_cached_response._hidden_params["cache_hit"] = (
                            True
                        )

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
                        print_verbose(
                            f"Async Wrapper: Completed Call, calling async_success_handler: {logging_obj.async_success_handler}"
                        )
                        logging_obj.update_environment_variables(
                            model=model,
                            user=kwargs.get("user", None),
                            optional_params={},
                            litellm_params={
                                "logger_fn": kwargs.get("logger_fn", None),
                                "acompletion": True,
                                "metadata": kwargs.get("metadata", {}),
                                "model_info": kwargs.get("model_info", {}),
                                "proxy_server_request": kwargs.get(
                                    "proxy_server_request", None
                                ),
                                "preset_cache_key": kwargs.get(
                                    "preset_cache_key", None
                                ),
                                "stream_response": kwargs.get("stream_response", {}),
                                "api_base": "",
                            },
                            input=kwargs.get("messages", ""),
                            api_key=kwargs.get("api_key", None),
                            original_response=str(final_embedding_cached_response),
                            additional_args=None,
                            stream=kwargs.get("stream", False),
                        )
                        asyncio.create_task(
                            logging_obj.async_success_handler(
                                final_embedding_cached_response,
                                start_time,
                                end_time,
                                cache_hit,
                            )
                        )
                        threading.Thread(
                            target=logging_obj.success_handler,
                            args=(
                                final_embedding_cached_response,
                                start_time,
                                end_time,
                                cache_hit,
                            ),
                        ).start()
                        return CachingHandlerResponse(
                            final_embedding_cached_response=final_embedding_cached_response
                        )
        return CachingHandlerResponse(
            cached_result=cached_result,
            final_embedding_cached_response=final_embedding_cached_response,
        )
