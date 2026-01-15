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


def test_get_message_role_with_pydantic_message():
    """Test get_message_role with Pydantic Message (output from LLM)."""
    from litellm.types.utils import Message, get_message_role

    msg = Message(content="Hello", role="assistant")
    assert get_message_role(msg) == "assistant"


def test_get_message_role_with_typeddict():
    """Test get_message_role with TypedDict (input to LLM)."""
    from litellm.types.utils import get_message_role

    msg = {"role": "user", "content": "Hello"}
    assert get_message_role(msg) == "user"


def test_get_message_content_with_pydantic_message():
    """Test get_message_content with Pydantic Message (output from LLM)."""
    from litellm.types.utils import Message, get_message_content

    msg = Message(content="Hello world", role="assistant")
    assert get_message_content(msg) == "Hello world"


def test_get_message_content_with_typeddict():
    """Test get_message_content with TypedDict (input to LLM)."""
    from litellm.types.utils import get_message_content

    msg = {"role": "user", "content": "Hello world"}
    assert get_message_content(msg) == "Hello world"


def test_get_message_attr_with_pydantic_message():
    """Test get_message_attr with Pydantic Message (output from LLM)."""
    from litellm.types.utils import Message, get_message_attr

    msg = Message(content="Hello", role="assistant")
    assert get_message_attr(msg, "role") == "assistant"
    assert get_message_attr(msg, "content") == "Hello"
    assert get_message_attr(msg, "nonexistent", "default") == "default"


def test_get_message_attr_with_typeddict():
    """Test get_message_attr with TypedDict (input to LLM)."""
    from litellm.types.utils import get_message_attr

    msg = {"role": "user", "content": "Hello"}
    assert get_message_attr(msg, "role") == "user"
    assert get_message_attr(msg, "content") == "Hello"
    assert get_message_attr(msg, "nonexistent", "default") == "default"


def test_mixed_message_list():
    """Test helpers work with mixed list of TypedDict and Pydantic messages."""
    from litellm.types.utils import (
        LiteLLMAnyMessage,
        Message,
        get_message_role,
        get_message_content,
    )

    messages: list[LiteLLMAnyMessage] = [
        {"role": "user", "content": "What is 2+2?"},
        Message(content="4", role="assistant"),
        {"role": "user", "content": "Thanks!"},
    ]

    roles = [get_message_role(m) for m in messages]
    assert roles == ["user", "assistant", "user"]

    contents = [get_message_content(m) for m in messages]
    assert contents == ["What is 2+2?", "4", "Thanks!"]
