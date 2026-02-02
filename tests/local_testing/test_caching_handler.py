import os
import sys
import time
import traceback
from litellm._uuid import uuid

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import hashlib
import random

import pytest

import litellm
from litellm import aembedding, completion, embedding, aresponses, responses
from litellm.caching.caching import Cache

from unittest.mock import AsyncMock, patch, MagicMock
from litellm.caching.caching_handler import LLMCachingHandler, CachingHandlerResponse
from litellm.caching.caching import LiteLLMCacheType
from litellm.types.utils import CallTypes
from litellm.types.rerank import RerankResponse
from litellm.types.utils import (
    ModelResponse,
    EmbeddingResponse,
    TextCompletionResponse,
    TranscriptionResponse,
    Embedding,
)
from litellm.types.llms.openai import ResponsesAPIResponse
from datetime import timedelta, datetime
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm._logging import verbose_logger
import logging


def setup_cache():
    # Set up the cache
    cache = Cache(type=LiteLLMCacheType.LOCAL)
    litellm.cache = cache
    return cache


chat_completion_response = litellm.ModelResponse(
    id=str(uuid.uuid4()),
    choices=[
        litellm.Choices(
            message=litellm.Message(
                role="assistant", content="Hello, how can I help you today?"
            )
        )
    ],
)

text_completion_response = litellm.TextCompletionResponse(
    id=str(uuid.uuid4()),
    choices=[litellm.utils.TextChoices(text="Hello, how can I help you today?")],
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response", [chat_completion_response, text_completion_response]
)
async def test_async_set_get_cache(response):
    litellm.set_verbose = True
    setup_cache()
    verbose_logger.setLevel(logging.DEBUG)
    caching_handler = LLMCachingHandler(
        original_function=completion, request_kwargs={}, start_time=datetime.now()
    )

    messages = [{"role": "user", "content": f"Unique message {datetime.now()}"}]

    logging_obj = LiteLLMLogging(
        litellm_call_id=str(datetime.now()),
        call_type=CallTypes.completion.value,
        model="gpt-3.5-turbo",
        messages=messages,
        function_id=str(uuid.uuid4()),
        stream=False,
        start_time=datetime.now(),
    )

    result = response
    print("result", result)

    original_function = (
        litellm.acompletion
        if isinstance(response, litellm.ModelResponse)
        else litellm.atext_completion
    )
    if isinstance(response, litellm.ModelResponse):
        kwargs = {"messages": messages}
        call_type = CallTypes.acompletion.value
    else:
        kwargs = {"prompt": f"Hello, how can I help you today? {datetime.now()}"}
        call_type = CallTypes.atext_completion.value

    await caching_handler.async_set_cache(
        result=result, original_function=original_function, kwargs=kwargs
    )

    await asyncio.sleep(2)

    # Verify the result was cached
    cached_response = await caching_handler._async_get_cache(
        model="gpt-3.5-turbo",
        original_function=original_function,
        logging_obj=logging_obj,
        start_time=datetime.now(),
        call_type=call_type,
        kwargs=kwargs,
    )

    assert cached_response.cached_result is not None
    assert cached_response.cached_result.id == result.id


@pytest.mark.asyncio
async def test_async_log_cache_hit_on_callbacks():
    """
    Assert logging callbacks are called after a cache hit
    """
    # Setup
    caching_handler = LLMCachingHandler(
        original_function=completion, request_kwargs={}, start_time=datetime.now()
    )

    mock_logging_obj = MagicMock()
    mock_logging_obj.async_success_handler = AsyncMock()
    mock_logging_obj.success_handler = MagicMock()
    mock_logging_obj.handle_sync_success_callbacks_for_async_calls = MagicMock()

    cached_result = "Mocked cached result"
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=1)
    cache_hit = True

    # Call the method
    caching_handler._async_log_cache_hit_on_callbacks(
        logging_obj=mock_logging_obj,
        cached_result=cached_result,
        start_time=start_time,
        end_time=end_time,
        cache_hit=cache_hit,
    )

    # Wait for the async task to complete
    await asyncio.sleep(0.5)

    print("mock logging obj methods called", mock_logging_obj.mock_calls)

    # Assertions
    mock_logging_obj.async_success_handler.assert_called_once_with(
        result=cached_result, start_time=start_time, end_time=end_time, cache_hit=cache_hit
    )

    # Wait for the thread to complete
    await asyncio.sleep(0.5)

    mock_logging_obj.handle_sync_success_callbacks_for_async_calls.assert_called_once_with(
        result=cached_result, start_time=start_time, end_time=end_time, cache_hit=cache_hit
    )


@pytest.mark.parametrize(
    "call_type, cached_result, expected_type",
    [
        (
            CallTypes.completion.value,
            {
                "id": "test",
                "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
            },
            ModelResponse,
        ),
        (
            CallTypes.text_completion.value,
            {"id": "test", "choices": [{"text": "Hello"}]},
            TextCompletionResponse,
        ),
        (
            CallTypes.embedding.value,
            {"data": [{"embedding": [0.1, 0.2, 0.3]}]},
            EmbeddingResponse,
        ),
        (
            CallTypes.rerank.value,
            {"id": "test", "results": [{"index": 0, "relevance_score": 0.9}]},
            RerankResponse,
        ),
        (
            CallTypes.transcription.value,
            {"text": "Hello, world!"},
            TranscriptionResponse,
        ),
    ],
)
def test_convert_cached_result_to_model_response(
    call_type, cached_result, expected_type
):
    """
    Assert that the cached result is converted to the correct type
    """
    caching_handler = LLMCachingHandler(
        original_function=lambda: None, request_kwargs={}, start_time=datetime.now()
    )
    logging_obj = LiteLLMLogging(
        litellm_call_id=str(datetime.now()),
        call_type=call_type,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, how can I help you today?"}],
        function_id=str(uuid.uuid4()),
        stream=False,
        start_time=datetime.now(),
    )

    result = caching_handler._convert_cached_result_to_model_response(
        cached_result=cached_result,
        call_type=call_type,
        kwargs={},
        logging_obj=logging_obj,
        model="test-model",
        args=(),
    )

    assert isinstance(result, expected_type)
    assert result is not None


def test_combine_cached_embedding_response_with_api_result():
    """
    If the cached response has [cache_hit, None, cache_hit]
    result should be [cache_hit, api_result, cache_hit]
    """
    # Setup
    caching_handler = LLMCachingHandler(
        original_function=lambda: None, request_kwargs={}, start_time=datetime.now()
    )

    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=1)

    # Create a CachingHandlerResponse with some cached and some None values
    cached_response = EmbeddingResponse(
        data=[
            Embedding(embedding=[0.1, 0.2, 0.3], index=0, object="embedding"),
            None,
            Embedding(embedding=[0.7, 0.8, 0.9], index=2, object="embedding"),
        ]
    )
    caching_handler_response = CachingHandlerResponse(
        final_embedding_cached_response=cached_response
    )

    # Create an API EmbeddingResponse for the missing value
    api_response = EmbeddingResponse(
        data=[Embedding(embedding=[0.4, 0.5, 0.6], index=1, object="embedding")]
    )

    # Call the method
    result = caching_handler._combine_cached_embedding_response_with_api_result(
        _caching_handler_response=caching_handler_response,
        embedding_response=api_response,
        start_time=start_time,
        end_time=end_time,
    )

    # Assertions
    assert isinstance(result, EmbeddingResponse)
    assert len(result.data) == 3
    assert result.data[0].embedding == [0.1, 0.2, 0.3]
    assert result.data[1].embedding == [0.4, 0.5, 0.6]
    assert result.data[2].embedding == [0.7, 0.8, 0.9]
    assert result._hidden_params["cache_hit"] == True
    assert isinstance(result._response_ms, float)
    assert result._response_ms > 0


def test_combine_cached_embedding_response_multiple_missing_values():
    """
    If the cached response has [cache_hit, None, None, cache_hit, None]
    result should be            [cache_hit, api_result, api_result, cache_hit, api_result]
    """

    # Setup
    caching_handler = LLMCachingHandler(
        original_function=lambda: None, request_kwargs={}, start_time=datetime.now()
    )

    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=1)

    # Create a CachingHandlerResponse with some cached and some None values
    cached_response = EmbeddingResponse(
        data=[
            Embedding(embedding=[0.1, 0.2, 0.3], index=0, object="embedding"),
            None,
            None,
            Embedding(embedding=[0.7, 0.8, 0.9], index=3, object="embedding"),
            None,
        ]
    )

    caching_handler_response = CachingHandlerResponse(
        final_embedding_cached_response=cached_response
    )

    # Create an API EmbeddingResponse for the missing values
    api_response = EmbeddingResponse(
        data=[
            Embedding(embedding=[0.4, 0.5, 0.6], index=1, object="embedding"),
            Embedding(embedding=[0.4, 0.5, 0.6], index=2, object="embedding"),
            Embedding(embedding=[0.4, 0.5, 0.6], index=4, object="embedding"),
        ]
    )

    # Call the method
    result = caching_handler._combine_cached_embedding_response_with_api_result(
        _caching_handler_response=caching_handler_response,
        embedding_response=api_response,
        start_time=start_time,
        end_time=end_time,
    )

    # Assertions
    assert isinstance(result, EmbeddingResponse)
    assert len(result.data) == 5
    assert result.data[0].embedding == [0.1, 0.2, 0.3]
    assert result.data[1].embedding == [0.4, 0.5, 0.6]
    assert result.data[2].embedding == [0.4, 0.5, 0.6]
    assert result.data[3].embedding == [0.7, 0.8, 0.9]


@pytest.mark.asyncio
async def test_embedding_cache_model_field_consistency():
    """
    Test that the model field is consistently preserved in cached embedding responses.
    This ensures that cache hits return the same model field as the original API response.
    """
    # Setup cache
    setup_cache()
    
    caching_handler = LLMCachingHandler(
        original_function=aembedding, request_kwargs={}, start_time=datetime.now()
    )

    # Create a mock embedding response with a specific model
    original_model = "text-embedding-005"
    embedding_response = EmbeddingResponse(
        model=original_model,
        data=[
            Embedding(embedding=[0.1, 0.2, 0.3], index=0, object="embedding"),
            Embedding(embedding=[0.4, 0.5, 0.6], index=1, object="embedding"),
        ]
    )

    # Mock logging object
    logging_obj = LiteLLMLogging(
        litellm_call_id=str(datetime.now()),
        call_type=CallTypes.aembedding.value,
        model=original_model,
        messages=[],  # Not used for embeddings
        function_id=str(uuid.uuid4()),
        stream=False,
        start_time=datetime.now(),
    )

    # Test parameters
    kwargs = {
        "model": original_model,
        "input": ["test input 1", "test input 2"],
        "caching": True
    }

    # Step 1: Cache the embedding response
    await caching_handler.async_set_cache(
        result=embedding_response,
        original_function=aembedding,
        kwargs=kwargs
    )

    # Step 2: Retrieve from cache
    cached_response = await caching_handler._async_get_cache(
        model=original_model,
        original_function=aembedding,
        logging_obj=logging_obj,
        start_time=datetime.now(),
        call_type=CallTypes.aembedding.value,
        kwargs=kwargs,
    )

    # Step 3: Verify the model field is preserved
    assert cached_response.final_embedding_cached_response is not None
    assert cached_response.final_embedding_cached_response.model == original_model
    assert len(cached_response.final_embedding_cached_response.data) == 2
    assert cached_response.final_embedding_cached_response.data[0].embedding == [0.1, 0.2, 0.3]
    assert cached_response.final_embedding_cached_response.data[0].index == 0
    assert cached_response.final_embedding_cached_response.data[1].embedding == [0.4, 0.5, 0.6]
    assert cached_response.final_embedding_cached_response.data[1].index == 1
    
    # Verify cache hit flag is set
    assert cached_response.final_embedding_cached_response._hidden_params["cache_hit"] == True


@pytest.mark.asyncio
async def test_embedding_cache_model_field_with_vendor_prefix():
    """
    Test that the model field is preserved even when using vendor-prefixed model names.
    This simulates the real-world scenario where models might be prefixed with vendor names.
    """
    # Setup cache
    setup_cache()
    
    caching_handler = LLMCachingHandler(
        original_function=aembedding, request_kwargs={}, start_time=datetime.now()
    )

    # Test with vendor-prefixed model name (like vertex_ai/text-embedding-005)
    vendor_model = "vertex_ai/text-embedding-005"
    actual_model = "text-embedding-005"  # What the provider actually returns
    
    # Create embedding response with the actual model name (as returned by provider)
    embedding_response = EmbeddingResponse(
        model=actual_model,  # Provider returns this
        data=[
            Embedding(embedding=[0.1, 0.2, 0.3], index=0, object="embedding"),
        ]
    )

    # Mock logging object
    logging_obj = LiteLLMLogging(
        litellm_call_id=str(datetime.now()),
        call_type=CallTypes.aembedding.value,
        model=vendor_model,
        messages=[],
        function_id=str(uuid.uuid4()),
        stream=False,
        start_time=datetime.now(),
    )

    # Test parameters with vendor-prefixed model
    kwargs = {
        "model": vendor_model,  # Request uses vendor prefix
        "input": ["test input"],
        "caching": True
    }

    # Cache the response
    await caching_handler.async_set_cache(
        result=embedding_response,
        original_function=aembedding,
        kwargs=kwargs
    )

    # Retrieve from cache
    cached_response = await caching_handler._async_get_cache(
        model=vendor_model,
        original_function=aembedding,
        logging_obj=logging_obj,
        start_time=datetime.now(),
        call_type=CallTypes.aembedding.value,
        kwargs=kwargs,
    )

    # Verify the model field matches the original provider response, not the request
    assert cached_response.final_embedding_cached_response is not None
    assert cached_response.final_embedding_cached_response.model == actual_model  # Should be the provider's model name
    assert cached_response.final_embedding_cached_response.model != vendor_model  # Should NOT be the vendor-prefixed name


def test_extract_model_from_cached_results():
    """
    Test the helper method that extracts model names from cached results.
    """
    caching_handler = LLMCachingHandler(
        original_function=aembedding, request_kwargs={}, start_time=datetime.now()
    )

    # Test with valid cached results
    non_null_list = [
        (0, {"embedding": [0.1, 0.2], "index": 0, "object": "embedding", "model": "text-embedding-005"}),
        (1, {"embedding": [0.3, 0.4], "index": 1, "object": "embedding", "model": "text-embedding-005"}),
    ]
    
    model_name = caching_handler._extract_model_from_cached_results(non_null_list)
    assert model_name == "text-embedding-005"

    # Test with missing model field
    non_null_list_no_model = [
        (0, {"embedding": [0.1, 0.2], "index": 0, "object": "embedding"}),
        (1, {"embedding": [0.3, 0.4], "index": 1, "object": "embedding"}),
    ]
    
    model_name = caching_handler._extract_model_from_cached_results(non_null_list_no_model)
    assert model_name is None

    # Test with empty list
    model_name = caching_handler._extract_model_from_cached_results([])
    assert model_name is None


@pytest.mark.asyncio
async def test_async_responses_api_caching():
    """
    Test that responses API calls are properly cached and retrieved.
    This verifies the full cache lifecycle for ResponsesAPIResponse objects.
    """
    # Setup cache
    setup_cache()
    
    caching_handler = LLMCachingHandler(
        original_function=aresponses, request_kwargs={}, start_time=datetime.now()
    )

    # Create a mock ResponsesAPIResponse
    original_model = "gpt-4o"
    responses_api_response = ResponsesAPIResponse(
        id="resp_test123",
        created_at=int(time.time()),
        status="completed",
        model=original_model,
        object="response",
        output=[
            {
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "This is a test response from the responses API.",
                        "annotations": []
                    }
                ]
            }
        ]
    )

    # Mock logging object
    logging_obj = LiteLLMLogging(
        litellm_call_id=str(datetime.now()),
        call_type=CallTypes.aresponses.value,
        model=original_model,
        messages=[],  # Responses API uses input, not messages
        function_id=str(uuid.uuid4()),
        stream=False,
        start_time=datetime.now(),
    )

    # Test parameters
    kwargs = {
        "model": original_model,
        "input": "Tell me a short story",
        "max_output_tokens": 100,
        "caching": True
    }

    # Step 1: Cache the responses API response
    await caching_handler.async_set_cache(
        result=responses_api_response,
        original_function=aresponses,
        kwargs=kwargs
    )

    await asyncio.sleep(0.5)

    # Step 2: Retrieve from cache
    cached_response = await caching_handler._async_get_cache(
        model=original_model,
        original_function=aresponses,
        logging_obj=logging_obj,
        start_time=datetime.now(),
        call_type=CallTypes.aresponses.value,
        kwargs=kwargs,
    )

    # Step 3: Verify the response is properly cached and retrieved
    assert cached_response.cached_result is not None
    assert isinstance(cached_response.cached_result, ResponsesAPIResponse)
    assert cached_response.cached_result.id == responses_api_response.id
    assert cached_response.cached_result.model == original_model
    assert cached_response.cached_result.status == "completed"
    assert len(cached_response.cached_result.output) == 1
    
    # Verify cache hit flag is set
    assert cached_response.cached_result._hidden_params["cache_hit"] == True


def test_sync_responses_api_caching():
    """
    Test that synchronous responses API calls are properly cached and retrieved.
    """
    # Setup cache
    setup_cache()
    
    caching_handler = LLMCachingHandler(
        original_function=responses, request_kwargs={}, start_time=datetime.now()
    )

    # Create a mock ResponsesAPIResponse
    original_model = "gpt-4o"
    responses_api_response = ResponsesAPIResponse(
        id="resp_sync_test456",
        created_at=int(time.time()),
        status="completed",
        model=original_model,
        object="response",
        output=[
            {
                "type": "message",
                "id": "msg_456",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Sync response test.",
                        "annotations": []
                    }
                ]
            }
        ]
    )

    # Mock logging object
    logging_obj = LiteLLMLogging(
        litellm_call_id=str(datetime.now()),
        call_type=CallTypes.responses.value,
        model=original_model,
        messages=[],
        function_id=str(uuid.uuid4()),
        stream=False,
        start_time=datetime.now(),
    )

    # Test parameters
    kwargs = {
        "model": original_model,
        "input": "Tell me another story",
        "max_output_tokens": 100,
        "caching": True
    }

    # Step 1: Cache the responses API response
    caching_handler.sync_set_cache(
        result=responses_api_response,
        kwargs=kwargs
    )

    time.sleep(0.5)

    # Step 2: Retrieve from cache
    cached_response = caching_handler._sync_get_cache(
        model=original_model,
        original_function=responses,
        logging_obj=logging_obj,
        start_time=datetime.now(),
        call_type=CallTypes.responses.value,
        kwargs=kwargs,
    )

    # Step 3: Verify the response is properly cached and retrieved
    assert cached_response.cached_result is not None
    assert isinstance(cached_response.cached_result, ResponsesAPIResponse)
    assert cached_response.cached_result.id == responses_api_response.id
    assert cached_response.cached_result.model == original_model
    assert cached_response.cached_result.status == "completed"
    
    # Verify cache hit flag is set
    assert cached_response.cached_result._hidden_params["cache_hit"] == True


def test_convert_cached_responses_api_result_to_model_response():
    """
    Test that cached ResponsesAPIResponse results are properly converted back
    to ResponsesAPIResponse objects with correct structure.
    """
    caching_handler = LLMCachingHandler(
        original_function=responses, request_kwargs={}, start_time=datetime.now()
    )
    
    logging_obj = LiteLLMLogging(
        litellm_call_id=str(datetime.now()),
        call_type=CallTypes.responses.value,
        model="gpt-4o",
        messages=[],
        function_id=str(uuid.uuid4()),
        stream=False,
        start_time=datetime.now(),
    )

    # Simulate cached result as a dictionary
    cached_result = {
        "id": "resp_convert_test789",
        "created_at": int(time.time()),
        "status": "completed",
        "model": "gpt-4o",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": "msg_789",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Conversion test response.",
                        "annotations": []
                    }
                ]
            }
        ]
    }

    # Convert cached result to ResponsesAPIResponse
    result = caching_handler._convert_cached_result_to_model_response(
        cached_result=cached_result,
        call_type=CallTypes.responses.value,
        kwargs={"model": "gpt-4o", "input": "test"},
        logging_obj=logging_obj,
        model="gpt-4o",
        args=(),
    )

    # Verify conversion
    assert isinstance(result, ResponsesAPIResponse)
    assert result.id == "resp_convert_test789"
    assert result.model == "gpt-4o"
    assert result.status == "completed"
    assert len(result.output) == 1


@pytest.mark.asyncio
async def test_responses_api_cache_with_different_inputs():
    """
    Test that different inputs to the responses API result in different cache keys.
    This ensures cache isolation between different requests.
    """
    # Setup cache
    setup_cache()
    
    caching_handler = LLMCachingHandler(
        original_function=aresponses, request_kwargs={}, start_time=datetime.now()
    )

    original_model = "gpt-4o"

    # First request
    response_1 = ResponsesAPIResponse(
        id="resp_1",
        created_at=int(time.time()),
        status="completed",
        model=original_model,
        object="response",
        output=[
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Response 1", "annotations": []}]
            }
        ]
    )

    kwargs_1 = {
        "model": original_model,
        "input": "First unique input",
        "caching": True
    }

    await caching_handler.async_set_cache(
        result=response_1,
        original_function=aresponses,
        kwargs=kwargs_1
    )

    # Second request with different input
    response_2 = ResponsesAPIResponse(
        id="resp_2",
        created_at=int(time.time()),
        status="completed",
        model=original_model,
        object="response",
        output=[
            {
                "type": "message",
                "id": "msg_2",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Response 2", "annotations": []}]
            }
        ]
    )

    kwargs_2 = {
        "model": original_model,
        "input": "Second unique input",
        "caching": True
    }

    await caching_handler.async_set_cache(
        result=response_2,
        original_function=aresponses,
        kwargs=kwargs_2
    )

    await asyncio.sleep(0.5)

    # Retrieve both from cache
    logging_obj_1 = LiteLLMLogging(
        litellm_call_id=str(datetime.now()),
        call_type=CallTypes.aresponses.value,
        model=original_model,
        messages=[],
        function_id=str(uuid.uuid4()),
        stream=False,
        start_time=datetime.now(),
    )

    logging_obj_2 = LiteLLMLogging(
        litellm_call_id=str(datetime.now()),
        call_type=CallTypes.aresponses.value,
        model=original_model,
        messages=[],
        function_id=str(uuid.uuid4()),
        stream=False,
        start_time=datetime.now(),
    )

    cached_1 = await caching_handler._async_get_cache(
        model=original_model,
        original_function=aresponses,
        logging_obj=logging_obj_1,
        start_time=datetime.now(),
        call_type=CallTypes.aresponses.value,
        kwargs=kwargs_1,
    )

    cached_2 = await caching_handler._async_get_cache(
        model=original_model,
        original_function=aresponses,
        logging_obj=logging_obj_2,
        start_time=datetime.now(),
        call_type=CallTypes.aresponses.value,
        kwargs=kwargs_2,
    )

    # Verify each input gets its own cached response
    assert cached_1.cached_result is not None
    assert cached_2.cached_result is not None
    assert cached_1.cached_result.id == "resp_1"
    assert cached_2.cached_result.id == "resp_2"
    
    # Access output content properly (could be dict or object)
    output_1 = cached_1.cached_result.output[0]
    if isinstance(output_1, dict):
        text_1 = output_1["content"][0]["text"]
    else:
        text_1 = output_1.content[0].text if hasattr(output_1.content[0], 'text') else output_1.content[0]["text"]
    
    output_2 = cached_2.cached_result.output[0]
    if isinstance(output_2, dict):
        text_2 = output_2["content"][0]["text"]
    else:
        text_2 = output_2.content[0].text if hasattr(output_2.content[0], 'text') else output_2.content[0]["text"]
    
    assert text_1 == "Response 1"
    assert text_2 == "Response 2"


@pytest.mark.parametrize(
    "call_type, cached_result, expected_type",
    [
        (
            CallTypes.responses.value,
            {
                "id": "resp_param_test",
                "created_at": 1234567890,
                "status": "completed",
                "model": "gpt-4o",
                "object": "response",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_param",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Test", "annotations": []}
                        ]
                    }
                ]
            },
            ResponsesAPIResponse,
        ),
        (
            CallTypes.aresponses.value,
            {
                "id": "resp_async_param_test",
                "created_at": 1234567890,
                "status": "completed",
                "model": "gpt-4o",
                "object": "response",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_async_param",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Async Test", "annotations": []}
                        ]
                    }
                ]
            },
            ResponsesAPIResponse,
        ),
    ],
)
def test_convert_cached_responses_result_parameterized(
    call_type, cached_result, expected_type
):
    """
    Parameterized test to verify both sync and async responses API cached results
    are converted to the correct ResponsesAPIResponse type.
    """
    caching_handler = LLMCachingHandler(
        original_function=lambda: None, request_kwargs={}, start_time=datetime.now()
    )
    logging_obj = LiteLLMLogging(
        litellm_call_id=str(datetime.now()),
        call_type=call_type,
        model="gpt-4o",
        messages=[],
        function_id=str(uuid.uuid4()),
        stream=False,
        start_time=datetime.now(),
    )

    result = caching_handler._convert_cached_result_to_model_response(
        cached_result=cached_result,
        call_type=call_type,
        kwargs={},
        logging_obj=logging_obj,
        model="gpt-4o",
        args=(),
    )

    assert isinstance(result, expected_type)
    assert result is not None
    assert result.id == cached_result["id"]
    assert result.status == cached_result["status"]
