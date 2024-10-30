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


def test_multiple_function_call():
    litellm.set_verbose = True
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "do test"}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "test"}],
            "tool_calls": [
                {
                    "index": 0,
                    "function": {"arguments": '{"arg": "test"}', "name": "test"},
                    "id": "call_597e00e6-11d4-4ed2-94b2-27edee250aec",
                    "type": "function",
                },
                {
                    "index": 1,
                    "function": {"arguments": '{"arg": "test2"}', "name": "test2"},
                    "id": "call_2414e8f9-283a-002b-182a-1290ab912c02",
                    "type": "function",
                },
            ],
        },
        {
            "tool_call_id": "call_597e00e6-11d4-4ed2-94b2-27edee250aec",
            "role": "tool",
            "name": "test",
            "content": [{"type": "text", "text": "42"}],
        },
        {
            "tool_call_id": "call_2414e8f9-283a-002b-182a-1290ab912c02",
            "role": "tool",
            "name": "test2",
            "content": [{"type": "text", "text": "15"}],
        },
        {"role": "user", "content": [{"type": "text", "text": "tell me the results."}]},
    ]

    response_body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": 'The `default_api.test` function call returned a JSON object indicating a successful execution.  The `fields` key contains a nested dictionary with a `key` of "content" and a `value` with a `string_value` of "42".\n\nSimilarly, the `default_api.test2` function call also returned a JSON object showing successful execution.  The `fields` key contains a nested dictionary with a `key` of "content" and a `value` with a `string_value` of "15".\n\nIn short, both test functions executed successfully and returned different numerical string values ("42" and "15").  The significance of these numbers depends on the internal logic of the `test` and `test2` functions within the `default_api`.\n'
                        }
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
                "avgLogprobs": -0.20577410289219447,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 128,
            "candidatesTokenCount": 168,
            "totalTokenCount": 296,
        },
        "modelVersion": "gemini-1.5-flash-002",
    }

    mock_response = MagicMock()
    mock_response.json.return_value = response_body

    with patch.object(client, "post", return_value=mock_response) as mock_post:
        r = litellm.completion(
            messages=messages, model="gemini/gemini-1.5-flash-002", client=client
        )
        assert len(r.choices) > 0

        assert mock_post.call_args.kwargs["json"] == {
            "contents": [
                {"role": "user", "parts": [{"text": "do test"}]},
                {
                    "role": "model",
                    "parts": [
                        {"text": "test"},
                        {
                            "function_call": {
                                "name": "test",
                                "args": {
                                    "fields": {
                                        "key": "arg",
                                        "value": {"string_value": "test"},
                                    }
                                },
                            }
                        },
                        {
                            "function_call": {
                                "name": "test2",
                                "args": {
                                    "fields": {
                                        "key": "arg",
                                        "value": {"string_value": "test2"},
                                    }
                                },
                            }
                        },
                    ],
                },
                {
                    "parts": [
                        {
                            "function_response": {
                                "name": "test",
                                "response": {
                                    "fields": {
                                        "key": "content",
                                        "value": {"string_value": "42"},
                                    }
                                },
                            }
                        },
                        {
                            "function_response": {
                                "name": "test2",
                                "response": {
                                    "fields": {
                                        "key": "content",
                                        "value": {"string_value": "15"},
                                    }
                                },
                            }
                        },
                    ]
                },
                {"role": "user", "parts": [{"text": "tell me the results."}]},
            ],
            "generationConfig": {},
        }


def test_multiple_function_call_changed_text_pos():
    litellm.set_verbose = True
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "do test"}]},
        {
            "tool_calls": [
                {
                    "index": 0,
                    "function": {"arguments": '{"arg": "test"}', "name": "test"},
                    "id": "call_597e00e6-11d4-4ed2-94b2-27edee250aec",
                    "type": "function",
                },
                {
                    "index": 1,
                    "function": {"arguments": '{"arg": "test2"}', "name": "test2"},
                    "id": "call_2414e8f9-283a-002b-182a-1290ab912c02",
                    "type": "function",
                },
            ],
            "role": "assistant",
            "content": [{"type": "text", "text": "test"}],
        },
        {
            "tool_call_id": "call_2414e8f9-283a-002b-182a-1290ab912c02",
            "role": "tool",
            "name": "test2",
            "content": [{"type": "text", "text": "15"}],
        },
        {
            "tool_call_id": "call_597e00e6-11d4-4ed2-94b2-27edee250aec",
            "role": "tool",
            "name": "test",
            "content": [{"type": "text", "text": "42"}],
        },
        {"role": "user", "content": [{"type": "text", "text": "tell me the results."}]},
    ]

    response_body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": 'The code executed two functions, `test` and `test2`.\n\n* **`test`**:  Returned a dictionary indicating that the "key" field has a "value" field containing a string value of "42".  This is likely a response from a function that processed the input "test" and returned a calculated or pre-defined value.\n\n* **`test2`**: Returned a dictionary indicating that the "key" field has a "value" field containing a string value of "15". Similar to `test`, this suggests a function that processes the input "test2" and returns a specific result.\n\nIn short, both functions appear to be simple tests that return different hardcoded or calculated values based on their input arguments.\n'
                        }
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
                "avgLogprobs": -0.32848488592332409,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 128,
            "candidatesTokenCount": 155,
            "totalTokenCount": 283,
        },
        "modelVersion": "gemini-1.5-flash-002",
    }
    mock_response = MagicMock()
    mock_response.json.return_value = response_body

    with patch.object(client, "post", return_value=mock_response) as mock_post:
        resp = litellm.completion(
            messages=messages, model="gemini/gemini-1.5-flash-002", client=client
        )
        assert len(resp.choices) > 0
        mock_post.assert_called_once()

        assert mock_post.call_args.kwargs["json"]["contents"] == [
            {"role": "user", "parts": [{"text": "do test"}]},
            {
                "role": "model",
                "parts": [
                    {"text": "test"},
                    {
                        "function_call": {
                            "name": "test",
                            "args": {
                                "fields": {
                                    "key": "arg",
                                    "value": {"string_value": "test"},
                                }
                            },
                        }
                    },
                    {
                        "function_call": {
                            "name": "test2",
                            "args": {
                                "fields": {
                                    "key": "arg",
                                    "value": {"string_value": "test2"},
                                }
                            },
                        }
                    },
                ],
            },
            {
                "parts": [
                    {
                        "function_response": {
                            "name": "test2",
                            "response": {
                                "fields": {
                                    "key": "content",
                                    "value": {"string_value": "15"},
                                }
                            },
                        }
                    },
                    {
                        "function_response": {
                            "name": "test",
                            "response": {
                                "fields": {
                                    "key": "content",
                                    "value": {"string_value": "42"},
                                }
                            },
                        }
                    },
                ]
            },
            {"role": "user", "parts": [{"text": "tell me the results."}]},
        ]
