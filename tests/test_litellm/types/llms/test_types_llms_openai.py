import asyncio
import os
import sys
from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))
import json

import litellm


def test_generic_event():
    from litellm.types.llms.openai import GenericEvent

    event = {"type": "test", "test": "test"}
    event = GenericEvent(**event)
    assert event.type == "test"
    assert event.test == "test"


def test_output_item_added_event():
    from litellm.types.llms.openai import OutputItemAddedEvent

    event = {
        "type": "response.output_item.added",
        "sequence_number": 4,
        "output_index": 1,
        "item": None,
    }
    event = OutputItemAddedEvent(**event)
    assert event.type == "response.output_item.added"
    assert event.sequence_number == 4
    assert event.output_index == 1
    assert event.item is None


def test_responses_api_response_output_text_from_output_text_block():
    from litellm.types.llms.openai import ResponsesAPIResponse

    response = ResponsesAPIResponse(
        id="resp_1",
        created_at=0,
        output=[
            {"type": "message", "content": [{"type": "output_text", "text": "Hello"}]}
        ],
    )

    assert response.output_text == "Hello"


def test_responses_api_response_output_text_from_text_value_dict_output():
    from litellm.types.llms.openai import ResponsesAPIResponse

    output = {"type": "message", "content": {"type": "text", "value": "Hi"}}
    response = ResponsesAPIResponse.model_construct(
        id="resp_2", created_at=0, output=output
    )

    assert response.output_text == "Hi"


def test_responses_api_response_output_text_ignores_non_message_and_blank():
    from litellm.types.llms.openai import ResponsesAPIResponse
    from litellm.types.responses.main import (
        GenericResponseOutputItem,
        OutputFunctionToolCall,
        OutputText,
    )

    tool_call_item = OutputFunctionToolCall(
        type="function_call",
        status="completed",
        arguments="{}",
        call_id="call_1",
        name="test_func",
        id="fc_1",
    )
    message_item = GenericResponseOutputItem(
        type="message",
        id="item_1",
        status="completed",
        role="assistant",
        content=[
            OutputText(type="text", text="A", annotations=None),
            OutputText(type="output_text", text=" ", annotations=None),
            OutputText(type="text", text="B", annotations=None),
        ],
    )
    response = ResponsesAPIResponse(
        id="resp_3",
        created_at=0,
        output=[
            tool_call_item,
            message_item,
        ],
    )

    assert response.output_text == "A B"
