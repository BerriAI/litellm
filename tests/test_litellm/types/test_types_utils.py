import asyncio
import os
import sys
from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))
import json

from litellm.types.utils import HiddenParams


def test_hidden_params_response_ms():
    hidden_params = HiddenParams()
    setattr(hidden_params, "_response_ms", 100)
    hidden_params_dict = hidden_params.model_dump()
    assert hidden_params_dict.get("_response_ms") == 100


def test_chat_completion_delta_tool_call():
    from litellm.types.utils import ChatCompletionDeltaToolCall, Function

    tool = ChatCompletionDeltaToolCall(
        id="call_m87w",
        function=Function(
            arguments='{"location": "San Francisco", "unit": "imperial"}',
            name="get_current_weather",
        ),
        type="function",
        index=0,
    )

    assert "function" in tool


def test_empty_choices():
    from litellm.types.utils import Choices

    Choices()


def test_usage_dump():
    from litellm.types.utils import (
        CompletionTokensDetailsWrapper,
        PromptTokensDetailsWrapper,
        Usage,
    )

    current_usage = Usage(
        completion_tokens=37,
        prompt_tokens=7,
        total_tokens=44,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=0,
            rejected_prediction_tokens=None,
            text_tokens=None,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None,
            cached_tokens=None,
            text_tokens=7,
            image_tokens=None,
            web_search_requests=1,
        ),
        web_search_requests=None,
    )

    assert current_usage.prompt_tokens_details.web_search_requests == 1

    new_usage = Usage(**current_usage.model_dump())
    assert new_usage.prompt_tokens_details.web_search_requests == 1


def test_usage_completion_tokens_details_text_tokens():
    from litellm.types.utils import Usage

    # Test data from the reported issue
    usage_data = {
        'completion_tokens': 77,
        'prompt_tokens': 11937,
        'total_tokens': 12014,
        'completion_tokens_details': {
            'accepted_prediction_tokens': None,
            'audio_tokens': None,
            'reasoning_tokens': 65,
            'rejected_prediction_tokens': None,
            'text_tokens': 12
        },
        'prompt_tokens_details': {
            'audio_tokens': None,
            'cached_tokens': None,
            'text_tokens': 11937,
            'image_tokens': None
        }
    }

    # Create Usage object
    u = Usage(**usage_data)
    
    # Verify the object has the text_tokens field
    assert hasattr(u.completion_tokens_details, 'text_tokens')
    assert u.completion_tokens_details.text_tokens == 12
    
    # Get model_dump output
    dump_result = u.model_dump()
    
    # Verify text_tokens is present in the model_dump output
    assert 'completion_tokens_details' in dump_result
    assert 'text_tokens' in dump_result['completion_tokens_details']
    assert dump_result['completion_tokens_details']['text_tokens'] == 12
    
    # Verify the full completion_tokens_details structure
    expected_completion_details = {
        'accepted_prediction_tokens': None,
        'audio_tokens': None,
        'reasoning_tokens': 65,
        'rejected_prediction_tokens': None,
        'text_tokens': 12,
        'image_tokens': None
    }
    assert dump_result['completion_tokens_details'] == expected_completion_details
    
    # Verify round-trip serialization works
    new_usage = Usage(**dump_result)
    assert new_usage.completion_tokens_details.text_tokens == 12

def test_streaming_choices_logprobs_model_dump_json():
    """
    Verify that StreamingChoices with logprobs can be serialized via
    model_dump_json() without PydanticSerializationError.

    Regression test for https://github.com/BerriAI/litellm/issues/18801.
    vLLM streaming responses with logprobs=True crashed the proxy with:
      TypeError: 'MockValSer' object cannot be converted to 'SchemaSerializer'
    because logprobs was not declared as a Pydantic field.
    """
    from litellm.types.utils import (
        ChatCompletionTokenLogprob,
        ChoiceLogprobs,
        Delta,
        ModelResponseStream,
        StreamingChoices,
    )

    logprobs_data = ChoiceLogprobs(
        content=[
            ChatCompletionTokenLogprob(
                token="Hello",
                logprob=-0.5,
                bytes=[72, 101, 108, 108, 111],
            )
        ]
    )

    choice = StreamingChoices(
        delta=Delta(content="Hello", role="assistant"),
        finish_reason=None,
        logprobs=logprobs_data,
    )
    chunk = ModelResponseStream(
        choices=[choice],
    )

    # This used to raise PydanticSerializationError with MockValSer
    json_str = chunk.model_dump_json(exclude_none=True, exclude_unset=True)
    assert "Hello" in json_str
    assert "logprob" in json_str


def test_streaming_choices_logprobs_from_dict():
    """
    Verify that StreamingChoices correctly converts a logprobs dict
    into a ChoiceLogprobs instance.
    """
    from litellm.types.utils import ChoiceLogprobs, Delta, StreamingChoices

    logprobs_dict = {
        "content": [
            {
                "token": "test",
                "logprob": -1.0,
                "bytes": [116, 101, 115, 116],
            }
        ]
    }

    choice = StreamingChoices(
        delta=Delta(content="test"),
        logprobs=logprobs_dict,
    )

    assert isinstance(choice.logprobs, ChoiceLogprobs)
    assert choice.logprobs.content[0].token == "test"


def test_streaming_choices_logprobs_none():
    """
    Verify that StreamingChoices handles logprobs=None correctly
    (the common case when logprobs is not requested).
    """
    from litellm.types.utils import Delta, StreamingChoices

    choice = StreamingChoices(
        delta=Delta(content="hello"),
    )
    assert choice.logprobs is None

    # Should serialize without error
    chunk_dict = choice.model_dump(exclude_none=True)
    assert "logprobs" not in chunk_dict

