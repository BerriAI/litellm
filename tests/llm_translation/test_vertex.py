import json
import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm import get_optional_params


def test_completion_pydantic_obj_2():
    from pydantic import BaseModel
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    litellm.set_verbose = True

    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    class EventsList(BaseModel):
        events: list[CalendarEvent]

    messages = [
        {"role": "user", "content": "List important events from the 20th century."}
    ]
    expected_request_body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "List important events from the 20th century."}],
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "response_schema": {
                "properties": {
                    "events": {
                        "items": {
                            "properties": {
                                "name": {"type": "string"},
                                "date": {"type": "string"},
                                "participants": {
                                    "items": {"type": "string"},
                                    "type": "array",
                                },
                            },
                            "required": [
                                "name",
                                "date",
                                "participants",
                            ],
                            "type": "object",
                        },
                        "type": "array",
                    }
                },
                "required": [
                    "events",
                ],
                "type": "object",
            },
        },
    }
    client = HTTPHandler()
    with patch.object(client, "post", new=MagicMock()) as mock_post:
        mock_post.return_value = expected_request_body
        try:
            litellm.completion(
                model="gemini/gemini-1.5-pro",
                messages=messages,
                response_format=EventsList,
                client=client,
            )
        except Exception as e:
            print(e)

        mock_post.assert_called_once()

        print(mock_post.call_args.kwargs)

        assert mock_post.call_args.kwargs["json"] == expected_request_body


def test_build_vertex_schema():
    from litellm.llms.vertex_ai_and_google_ai_studio.common_utils import (
        _build_vertex_schema,
    )
    import json

    schema = {
        "type": "object",
        "properties": {
            "recipes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"recipe_name": {"type": "string"}},
                    "required": ["recipe_name"],
                },
            }
        },
        "required": ["recipes"],
    }

    new_schema = _build_vertex_schema(schema)
    print(f"new_schema: {new_schema}")
    assert new_schema["type"] == schema["type"]
    assert new_schema["properties"] == schema["properties"]
    assert "required" in new_schema and new_schema["required"] == schema["required"]


@pytest.mark.parametrize(
    "tools, key",
    [
        ([{"googleSearchRetrieval": {}}], "googleSearchRetrieval"),
        ([{"code_execution": {}}], "code_execution"),
    ],
)
def test_vertex_tool_params(tools, key):

    optional_params = get_optional_params(
        model="gemini-1.5-pro",
        custom_llm_provider="vertex_ai",
        tools=tools,
    )
    print(optional_params)
    assert optional_params["tools"][0][key] == {}


@pytest.mark.parametrize(
    "tool, expect_parameters",
    [
        (
            {
                "name": "test_function",
                "description": "test_function_description",
                "parameters": {
                    "type": "object",
                    "properties": {"test_param": {"type": "string"}},
                },
            },
            True,
        ),
        (
            {
                "name": "test_function",
            },
            False,
        ),
    ],
)
def test_vertex_function_translation(tool, expect_parameters):
    """
    If param not set, don't set it in the request
    """

    tools = [tool]
    optional_params = get_optional_params(
        model="gemini-1.5-pro",
        custom_llm_provider="vertex_ai",
        tools=tools,
    )
    print(optional_params)
    if expect_parameters:
        assert "parameters" in optional_params["tools"][0]["function_declarations"][0]
    else:
        assert (
            "parameters" not in optional_params["tools"][0]["function_declarations"][0]
        )


def test_function_calling_with_gemini():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    litellm.set_verbose = True
    client = HTTPHandler()
    with patch.object(client, "post", new=MagicMock()) as mock_post:
        try:
            litellm.completion(
                model="gemini/gemini-1.5-pro-002",
                messages=[
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": "You are a helpful assistant that can interact with a computer to solve tasks.\n<IMPORTANT>\n* If user provides a path, you should NOT assume it's relative to the current working directory. Instead, you should explore the file system to find the file before working on it.\n</IMPORTANT>\n",
                            }
                        ],
                        "role": "system",
                    },
                    {
                        "content": [{"type": "text", "text": "Hey, how's it going?"}],
                        "role": "user",
                    },
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "finish",
                            "description": "Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.",
                        },
                    },
                ],
                client=client,
            )
        except Exception as e:
            print(e)
        mock_post.assert_called_once()
        print(mock_post.call_args.kwargs)

        assert mock_post.call_args.kwargs["json"]["tools"] == [
            {
                "function_declarations": [
                    {
                        "name": "finish",
                        "description": "Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.",
                    }
                ]
            }
        ]
