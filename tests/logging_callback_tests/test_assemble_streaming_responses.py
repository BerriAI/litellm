"""
Testing for _assemble_complete_response_from_streaming_chunks

- Test 1 - ModelResponse with 1 list of streaming chunks. Assert chunks are added to the streaming_chunks, after final chunk sent assert complete_streaming_response is not None
- Test 2 - TextCompletionResponse with 1 list of streaming chunks. Assert chunks are added to the streaming_chunks, after final chunk sent assert complete_streaming_response is not None
- Test 3 - Have multiple lists of streaming chunks, Assert that chunks are added to the correct list and that complete_streaming_response is None. After final chunk sent assert complete_streaming_response is not None


"""

import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

from pydantic.main import Model

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import httpx
import pytest
from respx import MockRouter

import litellm
from litellm import Choices, Message, ModelResponse

from litellm.litellm_core_utils.litellm_logging import (
    _assemble_complete_response_from_streaming_chunks,
)


@pytest.mark.parametrize("is_async", [True, False])
def test_assemble_complete_response_from_streaming_chunks_1(is_async):
    """
    Test 1 - ModelResponse with 1 list of streaming chunks. Assert chunks are added to the streaming_chunks, after final chunk sent assert complete_streaming_response is not None
    """

    request_kwargs = {
        "model": "test_model",
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    list_streaming_chunks = []
    chunk = {
        "id": "chatcmpl-9mWtyDnikZZoB75DyfUzWUxiiE2Pi",
        "choices": [
            litellm.utils.StreamingChoices(
                delta=litellm.utils.Delta(
                    content="hello in response",
                    function_call=None,
                    role=None,
                    tool_calls=None,
                ),
                index=0,
                logprobs=None,
            )
        ],
        "created": 1721353246,
        "model": "gpt-3.5-turbo",
        "object": "chat.completion.chunk",
        "system_fingerprint": None,
        "usage": None,
    }
    chunk = litellm.ModelResponse(**chunk, stream=True)
    complete_streaming_response = _assemble_complete_response_from_streaming_chunks(
        result=chunk,
        start_time=datetime.now(),
        end_time=datetime.now(),
        request_kwargs=request_kwargs,
        streaming_chunks=list_streaming_chunks,
        is_async=is_async,
    )

    # this is the 1st chunk - complete_streaming_response should be None

    print("list_streaming_chunks", list_streaming_chunks)
    print("complete_streaming_response", complete_streaming_response)
    assert complete_streaming_response is None
    assert len(list_streaming_chunks) == 1
    assert list_streaming_chunks[0] == chunk

    # Add final chunk
    chunk = {
        "id": "chatcmpl-9mWtyDnikZZoB75DyfUzWUxiiE2Pi",
        "choices": [
            litellm.utils.StreamingChoices(
                finish_reason="stop",
                delta=litellm.utils.Delta(
                    content="end of response",
                    function_call=None,
                    role=None,
                    tool_calls=None,
                ),
                index=0,
                logprobs=None,
            )
        ],
        "created": 1721353246,
        "model": "gpt-3.5-turbo",
        "object": "chat.completion.chunk",
        "system_fingerprint": None,
        "usage": None,
    }
    chunk = litellm.ModelResponse(**chunk, stream=True)
    complete_streaming_response = _assemble_complete_response_from_streaming_chunks(
        result=chunk,
        start_time=datetime.now(),
        end_time=datetime.now(),
        request_kwargs=request_kwargs,
        streaming_chunks=list_streaming_chunks,
        is_async=is_async,
    )

    print("list_streaming_chunks", list_streaming_chunks)
    print("complete_streaming_response", complete_streaming_response)

    # this is the 2nd chunk - complete_streaming_response should not be None
    assert complete_streaming_response is not None
    assert len(list_streaming_chunks) == 2

    assert isinstance(complete_streaming_response, ModelResponse)
    assert isinstance(complete_streaming_response.choices[0], Choices)

    pass


def test_assemble_complete_response_from_streaming_chunks_2():
    pass


@pytest.mark.parametrize("is_async", [True, False])
def test_assemble_complete_response_from_streaming_chunks_3(is_async):

    request_kwargs = {
        "model": "test_model",
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    list_streaming_chunks_1 = []
    list_streaming_chunks_2 = []

    chunk = {
        "id": "chatcmpl-9mWtyDnikZZoB75DyfUzWUxiiE2Pi",
        "choices": [
            litellm.utils.StreamingChoices(
                delta=litellm.utils.Delta(
                    content="hello in response",
                    function_call=None,
                    role=None,
                    tool_calls=None,
                ),
                index=0,
                logprobs=None,
            )
        ],
        "created": 1721353246,
        "model": "gpt-3.5-turbo",
        "object": "chat.completion.chunk",
        "system_fingerprint": None,
        "usage": None,
    }
    chunk = litellm.ModelResponse(**chunk, stream=True)
    complete_streaming_response = _assemble_complete_response_from_streaming_chunks(
        result=chunk,
        start_time=datetime.now(),
        end_time=datetime.now(),
        request_kwargs=request_kwargs,
        streaming_chunks=list_streaming_chunks_1,
        is_async=is_async,
    )

    # this is the 1st chunk - complete_streaming_response should be None

    print("list_streaming_chunks_1", list_streaming_chunks_1)
    print("complete_streaming_response", complete_streaming_response)
    assert complete_streaming_response is None
    assert len(list_streaming_chunks_1) == 1
    assert list_streaming_chunks_1[0] == chunk
    assert len(list_streaming_chunks_2) == 0

    # now add a chunk to the 2nd list

    complete_streaming_response = _assemble_complete_response_from_streaming_chunks(
        result=chunk,
        start_time=datetime.now(),
        end_time=datetime.now(),
        request_kwargs=request_kwargs,
        streaming_chunks=list_streaming_chunks_2,
        is_async=is_async,
    )

    print("list_streaming_chunks_2", list_streaming_chunks_2)
    print("complete_streaming_response", complete_streaming_response)
    assert complete_streaming_response is None
    assert len(list_streaming_chunks_2) == 1
    assert list_streaming_chunks_2[0] == chunk
    assert len(list_streaming_chunks_1) == 1

    # now add a chunk to the 1st list
