"""
Tests for LangfuseLLMObsOTELAttributes, specifically verifying that pydantic
Message objects in the messages list are correctly serialized without raising
TypeError. Regression test for:
  https://github.com/BerriAI/litellm/issues/26977
"""

import json
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.langfuse.langfuse_otel_attributes import (
    LangfuseLLMObsOTELAttributes,
)
from litellm.types.utils import Message


def test_set_messages_with_pydantic_message_objects():
    """
    Regression test for issue #26977:
    set_messages should not raise 'Object of type Message is not JSON serializable'
    when messages is a list of litellm.Message (pydantic) objects.
    """
    span = MagicMock()
    kwargs = {
        "messages": [Message(role="user", content="hello")],
        "optional_params": {},
    }

    # Should not raise TypeError
    LangfuseLLMObsOTELAttributes.set_messages(span, kwargs)

    # Verify that set_attribute was called with a valid JSON string
    span.set_attribute.assert_called_once()
    key, value = span.set_attribute.call_args[0]
    assert key == "langfuse.observation.input"

    # The value must be parseable JSON
    parsed = json.loads(value)
    assert "messages" in parsed
    assert parsed["messages"][0]["role"] == "user"
    assert parsed["messages"][0]["content"] == "hello"


def test_set_messages_with_dict_messages():
    """
    Ensure that plain dict messages still work correctly after the fix.
    """
    span = MagicMock()
    kwargs = {
        "messages": [{"role": "user", "content": "hello"}],
        "optional_params": {},
    }

    LangfuseLLMObsOTELAttributes.set_messages(span, kwargs)

    span.set_attribute.assert_called_once()
    key, value = span.set_attribute.call_args[0]
    assert key == "langfuse.observation.input"

    parsed = json.loads(value)
    assert parsed["messages"][0]["role"] == "user"


def test_set_messages_with_tools():
    """
    Ensure tools are included in the serialized input and pydantic messages
    are handled correctly together.
    """
    span = MagicMock()
    kwargs = {
        "messages": [Message(role="user", content="what's the weather?")],
        "optional_params": {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                }
            ]
        },
    }

    LangfuseLLMObsOTELAttributes.set_messages(span, kwargs)

    span.set_attribute.assert_called_once()
    key, value = span.set_attribute.call_args[0]
    parsed = json.loads(value)
    assert "tools" in parsed
    assert parsed["tools"][0]["function"]["name"] == "get_weather"
    assert parsed["messages"][0]["role"] == "user"
