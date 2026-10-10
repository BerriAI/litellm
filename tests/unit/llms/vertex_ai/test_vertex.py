import base64
import functools
import json
import re
import sys
from types import SimpleNamespace
from typing import Final

from dotenv import load_dotenv
import httpx
import respx
from google.oauth2.credentials import Credentials

import litellm.litellm_core_utils
import litellm.litellm_core_utils.prompt_templates
import litellm.litellm_core_utils.prompt_templates.factory

load_dotenv()
from unittest.mock import MagicMock

import pytest

import litellm
from litellm import Router, get_optional_params
from litellm.llms.vertex_ai.gemini.transformation import _process_gemini_media
from litellm.types.llms.vertex_ai import BlobType


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def test_completion_pydantic_obj_2():
    from pydantic import BaseModel

    from litellm.llms.custom_httpx.http_handler import HTTPHandler


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
            "response_json_schema": {
                "$defs": {
                    "CalendarEvent": {
                        "properties": {
                            "name": {"title": "Name", "type": "string"},
                            "date": {"title": "Date", "type": "string"},
                            "participants": {
                                "items": {"type": "string"},
                                "title": "Participants",
                                "type": "array",
                            },
                        },
                        "required": ["name", "date", "participants"],
                        "title": "CalendarEvent",
                        "type": "object",
                    }
                },
                "properties": {
                    "events": {
                        "items": {"$ref": "#/$defs/CalendarEvent"},
                        "title": "Events",
                        "type": "array",
                    }
                },
                "required": ["events"],
                "title": "EventsList",
                "type": "object",
            },
        },
    }
    client = HTTPHandler()
    with patch.object(client, "post", new=MagicMock()) as mock_post:
        mock_post.return_value = expected_request_body
        try:
            response = litellm.completion(
                model="gemini/gemini-2.5-flash",
                messages=messages,
                response_format=EventsList,
                api_key="test-api-key",
                client=client,
            )
            # print(response)
        except Exception as e:
            print(e)

        mock_post.assert_called_once()

        print(mock_post.call_args.kwargs)

        assert mock_post.call_args.kwargs["json"] == expected_request_body


def test_build_vertex_schema():
    import json

    from litellm.llms.vertex_ai.common_utils import build_vertex_schema

    schema = {
        "type": "object",
        "my-random-key": "my-random-value",
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

    new_schema = build_vertex_schema(schema)
    print(f"new_schema: {new_schema}")
    assert new_schema["type"] == schema["type"]
    assert new_schema["properties"] == schema["properties"]
    assert "required" in new_schema and new_schema["required"] == schema["required"]
    assert "my-random-key" not in new_schema


@pytest.mark.parametrize(
    "tools, key",
    [
        ([{"googleSearch": {}}], "googleSearch"),
        ([{"googleSearchRetrieval": {}}], "googleSearchRetrieval"),
        ([{"enterpriseWebSearch": {}}], "enterpriseWebSearch"),
        ([{"code_execution": {}}], "code_execution"),
        ([{"googleMaps": {}}], "googleMaps"),
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


def test_vertex_tool_type_field_removal():
    """
    Test that the 'type' field is removed from tools during processing
    to avoid issues with Vertex AI API while maintaining functionality.
    """
    # Test with Google Search tool that has 'type' field
    tools_with_type = [{"type": "google_search", "googleSearch": {}}]

    optional_params = get_optional_params(
        model="gemini-1.5-pro",
        custom_llm_provider="vertex_ai",
        tools=tools_with_type,
    )

    # Verify the tool is processed correctly
    assert "tools" in optional_params
    assert len(optional_params["tools"]) == 1
    assert "googleSearch" in optional_params["tools"][0]
    assert optional_params["tools"][0]["googleSearch"] == {}

    # Verify the 'type' field is not present in the final result
    assert "type" not in optional_params["tools"][0]

    # Test with function tool that has 'type' field
    function_tools_with_type = [
        {
            "type": "function",
            "function": {
                "name": "test_function",
                "description": "A test function",
                "parameters": {
                    "type": "object",
                    "properties": {"param": {"type": "string"}},
                },
            },
        }
    ]

    optional_params_function = get_optional_params(
        model="gemini-1.5-pro",
        custom_llm_provider="vertex_ai",
        tools=function_tools_with_type,
    )

    # Verify function tool is processed correctly
    assert "tools" in optional_params_function
    assert len(optional_params_function["tools"]) == 1
    assert "function_declarations" in optional_params_function["tools"][0]
    assert len(optional_params_function["tools"][0]["function_declarations"]) == 1
    assert (
        optional_params_function["tools"][0]["function_declarations"][0]["name"]
        == "test_function"
    )

    # Verify the 'type' field is not present in the final result
    assert "type" not in optional_params_function["tools"][0]


def test_function_calling_with_gemini():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

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
                api_key="test-api-key",
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
            messages=messages,
            model="gemini/gemini-1.5-flash-002",
            api_key="test-api-key",
            client=client,
        )
        assert len(r.choices) > 0

        print(mock_post.call_args.kwargs["json"])

        assert mock_post.call_args.kwargs["json"] == {
            "contents": [
                {"role": "user", "parts": [{"text": "do test"}]},
                {
                    "role": "model",
                    "parts": [
                        {"text": "test"},
                        {"function_call": {"name": "test", "args": {"arg": "test"}}},
                        {"function_call": {"name": "test2", "args": {"arg": "test2"}}},
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "name": "test",
                                "response": {"content": "42"},
                            }
                        },
                        {
                            "function_response": {
                                "name": "test2",
                                "response": {"content": "15"},
                            }
                        },
                    ],
                },
                {"role": "user", "parts": [{"text": "tell me the results."}]},
            ],
        }


def test_multiple_function_call_changed_text_pos():
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
            messages=messages,
            model="gemini/gemini-1.5-flash-002",
            api_key="test-api-key",
            client=client,
        )
        assert len(resp.choices) > 0
        mock_post.assert_called_once()

        print(mock_post.call_args.kwargs["json"]["contents"])

        assert mock_post.call_args.kwargs["json"]["contents"] == [
            {"role": "user", "parts": [{"text": "do test"}]},
            {
                "role": "model",
                "parts": [
                    {"text": "test"},
                    {"function_call": {"name": "test", "args": {"arg": "test"}}},
                    {"function_call": {"name": "test2", "args": {"arg": "test2"}}},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "name": "test2",
                            "response": {"content": "15"},
                        }
                    },
                    {
                        "function_response": {
                            "name": "test",
                            "response": {"content": "42"},
                        }
                    },
                ],
            },
            {"role": "user", "parts": [{"text": "tell me the results."}]},
        ]


def test_function_calling_with_gemini_multiple_results():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    # Step 1: send the conversation and available functions to the model
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco, Tokyo, and Paris? - give me 3 responses",
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    response_body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_current_weather",
                                "args": {"location": "San Francisco"},
                            }
                        },
                        {
                            "functionCall": {
                                "name": "get_current_weather",
                                "args": {"location": "Tokyo"},
                            }
                        },
                        {
                            "functionCall": {
                                "name": "get_current_weather",
                                "args": {"location": "Paris"},
                            }
                        },
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
                "avgLogprobs": -0.0040788948535919189,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 90,
            "candidatesTokenCount": 22,
            "totalTokenCount": 112,
        },
        "modelVersion": "gemini-1.5-flash-002",
    }

    mock_response = MagicMock()
    mock_response.json.return_value = response_body

    with patch.object(client, "post", return_value=mock_response):
        response = litellm.completion(
            model="gemini/gemini-1.5-flash-002",
            messages=messages,
            tools=tools,
            tool_choice="required",
            api_key="test-api-key",
            client=client,
        )
        print("Response\n", response)

        assert len(response.choices[0].message.tool_calls) == 3

        expected_locations = ["San Francisco", "Tokyo", "Paris"]
        for idx, tool_call in enumerate(response.choices[0].message.tool_calls):
            json_args = json.loads(tool_call.function.arguments)
            assert json_args["location"] == expected_locations[idx]


def test_logprobs_unit_test():
    from litellm import VertexGeminiConfig

    result = VertexGeminiConfig()._transform_logprobs(
        logprobs_result={
            "topCandidates": [
                {
                    "candidates": [
                        {"token": "```", "logProbability": -1.5496514e-06},
                        {"token": "`", "logProbability": -13.375002},
                        {"token": "``", "logProbability": -21.875002},
                    ]
                },
                {
                    "candidates": [
                        {"token": "tool", "logProbability": 0},
                        {"token": "too", "logProbability": -29.031433},
                        {"token": "to", "logProbability": -34.11199},
                    ]
                },
                {
                    "candidates": [
                        {"token": "_", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "code", "logProbability": 0},
                        {"token": "co", "logProbability": -28.114716},
                        {"token": "c", "logProbability": -29.283161},
                    ]
                },
                {
                    "candidates": [
                        {"token": "\n", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "print", "logProbability": 0},
                        {"token": "p", "logProbability": -19.7494},
                        {"token": "prin", "logProbability": -21.117342},
                    ]
                },
                {
                    "candidates": [
                        {"token": "(", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "default", "logProbability": 0},
                        {"token": "get", "logProbability": -16.811178},
                        {"token": "ge", "logProbability": -19.031078},
                    ]
                },
                {
                    "candidates": [
                        {"token": "_", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "api", "logProbability": 0},
                        {"token": "ap", "logProbability": -26.501019},
                        {"token": "a", "logProbability": -30.905857},
                    ]
                },
                {
                    "candidates": [
                        {"token": ".", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "get", "logProbability": 0},
                        {"token": "ge", "logProbability": -19.984676},
                        {"token": "g", "logProbability": -20.527714},
                    ]
                },
                {
                    "candidates": [
                        {"token": "_", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "current", "logProbability": 0},
                        {"token": "cur", "logProbability": -28.193565},
                        {"token": "cu", "logProbability": -29.636738},
                    ]
                },
                {
                    "candidates": [
                        {"token": "_", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "weather", "logProbability": 0},
                        {"token": "we", "logProbability": -27.887215},
                        {"token": "wea", "logProbability": -31.851082},
                    ]
                },
                {
                    "candidates": [
                        {"token": "(", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "location", "logProbability": 0},
                        {"token": "loc", "logProbability": -19.152641},
                        {"token": " location", "logProbability": -21.981709},
                    ]
                },
                {
                    "candidates": [
                        {"token": '="', "logProbability": -0.034490786},
                        {"token": "='", "logProbability": -3.398928},
                        {"token": "=", "logProbability": -7.6194153},
                    ]
                },
                {
                    "candidates": [
                        {"token": "San", "logProbability": -6.5561944e-06},
                        {"token": '\\"', "logProbability": -12.015556},
                        {"token": "Paris", "logProbability": -14.647776},
                    ]
                },
                {
                    "candidates": [
                        {"token": " Francisco", "logProbability": -3.5760596e-07},
                        {"token": " Frans", "logProbability": -14.83527},
                        {"token": " francisco", "logProbability": -19.796852},
                    ]
                },
                {
                    "candidates": [
                        {"token": '"))', "logProbability": -6.079254e-06},
                        {"token": ",", "logProbability": -12.106029},
                        {"token": '",', "logProbability": -14.56927},
                    ]
                },
                {
                    "candidates": [
                        {"token": "\n", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "print", "logProbability": -0.04140338},
                        {"token": "```", "logProbability": -3.2049975},
                        {"token": "p", "logProbability": -22.087523},
                    ]
                },
                {
                    "candidates": [
                        {"token": "(", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "default", "logProbability": 0},
                        {"token": "get", "logProbability": -20.266342},
                        {"token": "de", "logProbability": -20.906395},
                    ]
                },
                {
                    "candidates": [
                        {"token": "_", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "api", "logProbability": 0},
                        {"token": "ap", "logProbability": -27.712265},
                        {"token": "a", "logProbability": -31.986958},
                    ]
                },
                {
                    "candidates": [
                        {"token": ".", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "get", "logProbability": 0},
                        {"token": "g", "logProbability": -23.569286},
                        {"token": "ge", "logProbability": -23.829632},
                    ]
                },
                {
                    "candidates": [
                        {"token": "_", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "current", "logProbability": 0},
                        {"token": "cur", "logProbability": -30.125153},
                        {"token": "curr", "logProbability": -31.756569},
                    ]
                },
                {
                    "candidates": [
                        {"token": "_", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "weather", "logProbability": 0},
                        {"token": "we", "logProbability": -27.743786},
                        {"token": "w", "logProbability": -30.594503},
                    ]
                },
                {
                    "candidates": [
                        {"token": "(", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "location", "logProbability": 0},
                        {"token": "loc", "logProbability": -21.177715},
                        {"token": " location", "logProbability": -22.166002},
                    ]
                },
                {
                    "candidates": [
                        {"token": '="', "logProbability": -1.5617967e-05},
                        {"token": "='", "logProbability": -11.080961},
                        {"token": "=", "logProbability": -15.164277},
                    ]
                },
                {
                    "candidates": [
                        {"token": "Tokyo", "logProbability": -3.0041514e-05},
                        {"token": "tokyo", "logProbability": -10.650261},
                        {"token": "Paris", "logProbability": -12.096886},
                    ]
                },
                {
                    "candidates": [
                        {"token": '"))', "logProbability": -1.1922384e-07},
                        {"token": '",', "logProbability": -16.61921},
                        {"token": ",", "logProbability": -17.911102},
                    ]
                },
                {
                    "candidates": [
                        {"token": "\n", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "print", "logProbability": -3.5760596e-07},
                        {"token": "```", "logProbability": -14.949171},
                        {"token": "p", "logProbability": -24.321035},
                    ]
                },
                {
                    "candidates": [
                        {"token": "(", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "default", "logProbability": 0},
                        {"token": "de", "logProbability": -27.885206},
                        {"token": "def", "logProbability": -28.40597},
                    ]
                },
                {
                    "candidates": [
                        {"token": "_", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "api", "logProbability": 0},
                        {"token": "ap", "logProbability": -25.905933},
                        {"token": "a", "logProbability": -30.408901},
                    ]
                },
                {
                    "candidates": [
                        {"token": ".", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "get", "logProbability": 0},
                        {"token": "g", "logProbability": -22.274963},
                        {"token": "ge", "logProbability": -23.285828},
                    ]
                },
                {
                    "candidates": [
                        {"token": "_", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "current", "logProbability": 0},
                        {"token": "cur", "logProbability": -28.442535},
                        {"token": "curr", "logProbability": -29.95087},
                    ]
                },
                {
                    "candidates": [
                        {"token": "_", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "weather", "logProbability": 0},
                        {"token": "we", "logProbability": -27.307909},
                        {"token": "w", "logProbability": -31.076736},
                    ]
                },
                {
                    "candidates": [
                        {"token": "(", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "location", "logProbability": 0},
                        {"token": "loc", "logProbability": -21.535915},
                        {"token": "lo", "logProbability": -23.028284},
                    ]
                },
                {
                    "candidates": [
                        {"token": '="', "logProbability": -8.821511e-06},
                        {"token": "='", "logProbability": -11.700986},
                        {"token": "=", "logProbability": -14.50358},
                    ]
                },
                {
                    "candidates": [
                        {"token": "Paris", "logProbability": 0},
                        {"token": "paris", "logProbability": -18.07075},
                        {"token": "Par", "logProbability": -21.911625},
                    ]
                },
                {
                    "candidates": [
                        {"token": '"))', "logProbability": 0},
                        {"token": '")', "logProbability": -17.916853},
                        {"token": ",", "logProbability": -18.318272},
                    ]
                },
                {
                    "candidates": [
                        {"token": "\n", "logProbability": 0},
                        {"token": "ont", "logProbability": -1.2676506e30},
                        {"token": " п", "logProbability": -1.2676506e30},
                    ]
                },
                {
                    "candidates": [
                        {"token": "```", "logProbability": -3.5763796e-06},
                        {"token": "print", "logProbability": -12.535343},
                        {"token": "``", "logProbability": -19.670813},
                    ]
                },
            ],
            "chosenCandidates": [
                {"token": "```", "logProbability": -1.5496514e-06},
                {"token": "tool", "logProbability": 0},
                {"token": "_", "logProbability": 0},
                {"token": "code", "logProbability": 0},
                {"token": "\n", "logProbability": 0},
                {"token": "print", "logProbability": 0},
                {"token": "(", "logProbability": 0},
                {"token": "default", "logProbability": 0},
                {"token": "_", "logProbability": 0},
                {"token": "api", "logProbability": 0},
                {"token": ".", "logProbability": 0},
                {"token": "get", "logProbability": 0},
                {"token": "_", "logProbability": 0},
                {"token": "current", "logProbability": 0},
                {"token": "_", "logProbability": 0},
                {"token": "weather", "logProbability": 0},
                {"token": "(", "logProbability": 0},
                {"token": "location", "logProbability": 0},
                {"token": '="', "logProbability": -0.034490786},
                {"token": "San", "logProbability": -6.5561944e-06},
                {"token": " Francisco", "logProbability": -3.5760596e-07},
                {"token": '"))', "logProbability": -6.079254e-06},
                {"token": "\n", "logProbability": 0},
                {"token": "print", "logProbability": -0.04140338},
                {"token": "(", "logProbability": 0},
                {"token": "default", "logProbability": 0},
                {"token": "_", "logProbability": 0},
                {"token": "api", "logProbability": 0},
                {"token": ".", "logProbability": 0},
                {"token": "get", "logProbability": 0},
                {"token": "_", "logProbability": 0},
                {"token": "current", "logProbability": 0},
                {"token": "_", "logProbability": 0},
                {"token": "weather", "logProbability": 0},
                {"token": "(", "logProbability": 0},
                {"token": "location", "logProbability": 0},
                {"token": '="', "logProbability": -1.5617967e-05},
                {"token": "Tokyo", "logProbability": -3.0041514e-05},
                {"token": '"))', "logProbability": -1.1922384e-07},
                {"token": "\n", "logProbability": 0},
                {"token": "print", "logProbability": -3.5760596e-07},
                {"token": "(", "logProbability": 0},
                {"token": "default", "logProbability": 0},
                {"token": "_", "logProbability": 0},
                {"token": "api", "logProbability": 0},
                {"token": ".", "logProbability": 0},
                {"token": "get", "logProbability": 0},
                {"token": "_", "logProbability": 0},
                {"token": "current", "logProbability": 0},
                {"token": "_", "logProbability": 0},
                {"token": "weather", "logProbability": 0},
                {"token": "(", "logProbability": 0},
                {"token": "location", "logProbability": 0},
                {"token": '="', "logProbability": -8.821511e-06},
                {"token": "Paris", "logProbability": 0},
                {"token": '"))', "logProbability": 0},
                {"token": "\n", "logProbability": 0},
                {"token": "```", "logProbability": -3.5763796e-06},
            ],
        }
    )

    print(result)


def test_logprobs():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    response_body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "I do not have access to real-time information, including current weather conditions.  To get the current weather in San Francisco, I recommend checking a reliable weather website or app such as Google Weather, AccuWeather, or the National Weather Service.\n"
                        }
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
                "avgLogprobs": -0.04666396617889404,
                "logprobsResult": {
                    "chosenCandidates": [
                        {"token": "I", "logProbability": -1.08472495e-05},
                        {"token": " do", "logProbability": -0.00012611414},
                        {"token": " not", "logProbability": 0},
                        {"token": " have", "logProbability": 0},
                        {"token": " access", "logProbability": -0.0008849616},
                        {"token": " to", "logProbability": 0},
                        {"token": " real", "logProbability": -1.1922384e-07},
                        {"token": "-", "logProbability": 0},
                        {"token": "time", "logProbability": 0},
                        {"token": " information", "logProbability": -2.2409657e-05},
                        {"token": ",", "logProbability": 0},
                        {"token": " including", "logProbability": 0},
                        {"token": " current", "logProbability": -0.14274147},
                        {"token": " weather", "logProbability": 0},
                        {"token": " conditions", "logProbability": -0.0056300927},
                        {"token": ".", "logProbability": -3.5760596e-07},
                        {"token": "  ", "logProbability": -0.06392521},
                        {"token": "To", "logProbability": -2.3844768e-07},
                        {"token": " get", "logProbability": -0.058974747},
                        {"token": " the", "logProbability": 0},
                        {"token": " current", "logProbability": 0},
                        {"token": " weather", "logProbability": -2.3844768e-07},
                        {"token": " in", "logProbability": -2.3844768e-07},
                        {"token": " San", "logProbability": 0},
                        {"token": " Francisco", "logProbability": 0},
                        {"token": ",", "logProbability": 0},
                        {"token": " I", "logProbability": -0.6188003},
                        {"token": " recommend", "logProbability": -1.0370523e-05},
                        {"token": " checking", "logProbability": -0.00014005086},
                        {"token": " a", "logProbability": 0},
                        {"token": " reliable", "logProbability": -1.5496514e-06},
                        {"token": " weather", "logProbability": -8.344534e-07},
                        {"token": " website", "logProbability": -0.0078000566},
                        {"token": " or", "logProbability": -1.1922384e-07},
                        {"token": " app", "logProbability": 0},
                        {"token": " such", "logProbability": -0.9289338},
                        {"token": " as", "logProbability": 0},
                        {"token": " Google", "logProbability": -0.0046935496},
                        {"token": " Weather", "logProbability": 0},
                        {"token": ",", "logProbability": 0},
                        {"token": " Accu", "logProbability": 0},
                        {"token": "Weather", "logProbability": -0.00013909786},
                        {"token": ",", "logProbability": 0},
                        {"token": " or", "logProbability": -0.31303275},
                        {"token": " the", "logProbability": -0.17583296},
                        {"token": " National", "logProbability": -0.010806266},
                        {"token": " Weather", "logProbability": 0},
                        {"token": " Service", "logProbability": 0},
                        {"token": ".", "logProbability": -0.00068947335},
                        {"token": "\n", "logProbability": 0},
                    ]
                },
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 11,
            "candidatesTokenCount": 50,
            "totalTokenCount": 61,
        },
        "modelVersion": "gemini-1.5-flash-002",
    }
    mock_response = MagicMock()
    mock_response.json.return_value = response_body

    with patch.object(client, "post", return_value=mock_response):
        resp = litellm.completion(
            model="gemini/gemini-1.5-flash-002",
            messages=[
                {"role": "user", "content": "What's the weather like in San Francisco?"}
            ],
            logprobs=True,
            api_key="test-api-key",
            client=client,
        )
        print(resp)

        assert resp.choices[0].logprobs is not None


def test_process_gemini_media():
    """Test the _process_gemini_media function for different image sources"""
    from litellm.types.llms.vertex_ai import FileDataType

    # Test GCS URI
    gcs_result = _process_gemini_media("gs://bucket/image.png")
    assert gcs_result["file_data"] == FileDataType(
        mime_type="image/png", file_uri="gs://bucket/image.png"
    )

    # Test gs url with format specified
    gcs_result = _process_gemini_media("gs://bucket/image", format="image/jpeg")
    assert gcs_result["file_data"] == FileDataType(
        mime_type="image/jpeg", file_uri="gs://bucket/image"
    )

    # Test gs url without extension using mime_type from image_url object
    image_message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "gs://bucket/image-without-extension",
                        "mime_type": "image/png",
                    },
                }
            ],
        }
    ]
    from litellm.llms.vertex_ai.gemini.transformation import (
        gemini_convert_messages_with_history,
    )

    converted = gemini_convert_messages_with_history(messages=image_message, model="gemini-2.5-flash")
    assert converted[0]["parts"][0]["file_data"] == FileDataType(
        mime_type="image/png", file_uri="gs://bucket/image-without-extension"
    )

    # Test HTTPS JPG URL
    https_result = _process_gemini_media("https://example.com/image.jpg")
    print("https_result JPG", https_result)
    assert https_result["file_data"] == FileDataType(
        mime_type="image/jpeg", file_uri="https://example.com/image.jpg"
    )

    # Test HTTPS PNG URL
    https_result = _process_gemini_media("https://example.com/image.png")
    print("https_result PNG", https_result)
    assert https_result["file_data"] == FileDataType(
        mime_type="image/png", file_uri="https://example.com/image.png"
    )

    # Test HTTPS VIDEO URL
    https_result = _process_gemini_media("https://cloud-samples-data/video/animals.mp4")
    print("https_result PNG", https_result)
    assert https_result["file_data"] == FileDataType(
        mime_type="video/mp4", file_uri="https://cloud-samples-data/video/animals.mp4"
    )

    # Test HTTPS PDF URL
    https_result = _process_gemini_media("https://cloud-samples-data/pdf/animals.pdf")
    print("https_result PDF", https_result)
    assert https_result["file_data"] == FileDataType(
        mime_type="application/pdf",
        file_uri="https://cloud-samples-data/pdf/animals.pdf",
    )

    # Test base64 image
    base64_image = "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    base64_result = _process_gemini_media(base64_image)
    print("base64_result", base64_result)
    assert base64_result["inline_data"]["mime_type"] == "image/jpeg"
    assert base64_result["inline_data"]["data"] == "/9j/4AAQSkZJRg..."


def test_get_image_mime_type_from_url():
    """Test MIME type inference for remote media URLs"""
    from litellm.litellm_core_utils.prompt_templates.common_utils import get_image_mime_type_from_url

    # Test JPEG images
    assert (
        get_image_mime_type_from_url("https://example.com/image.jpg") == "image/jpeg"
    )
    assert (
        get_image_mime_type_from_url("https://example.com/image.jpeg") == "image/jpeg"
    )
    assert (
        get_image_mime_type_from_url("https://example.com/IMAGE.JPG") == "image/jpeg"
    )

    # Test PNG images
    assert get_image_mime_type_from_url("https://example.com/image.png") == "image/png"
    assert get_image_mime_type_from_url("https://example.com/IMAGE.PNG") == "image/png"

    # Test WebP images
    assert (
        get_image_mime_type_from_url("https://example.com/image.webp") == "image/webp"
    )
    assert (
        get_image_mime_type_from_url("https://example.com/IMAGE.WEBP") == "image/webp"
    )

    # Test audio formats
    assert get_image_mime_type_from_url("https://example.com/audio.ogg") == "audio/ogg"
    assert get_image_mime_type_from_url("https://example.com/track.OGG") == "audio/ogg"

    # Test unsupported formats
    assert get_image_mime_type_from_url("https://example.com/image.gif") is None
    assert get_image_mime_type_from_url("https://example.com/image.bmp") is None
    assert get_image_mime_type_from_url("https://example.com/image") is None
    assert get_image_mime_type_from_url("invalid_url") is None


@pytest.mark.parametrize(
    "model, expected_url",
    [
        (
            "textembedding-gecko@001",
            "https://us-central1-aiplatform.googleapis.com/v1/projects/project-id/locations/us-central1/publishers/google/models/textembedding-gecko@001:predict",
        ),
        (
            "123456789",
            "https://us-central1-aiplatform.googleapis.com/v1/projects/project-id/locations/us-central1/endpoints/123456789:predict",
        ),
    ],
)
def test_vertex_embedding_url(model, expected_url):
    """
    Test URL generation for embedding models, including numeric model IDs (fine-tuned models

    Relevant issue: https://github.com/BerriAI/litellm/issues/6482

    When a fine-tuned embedding model is used, the URL is different from the standard one.
    """
    from litellm.llms.vertex_ai.common_utils import get_vertex_url

    url, endpoint = get_vertex_url(
        mode="embedding",
        model=model,
        stream=False,
        vertex_project="project-id",
        vertex_location="us-central1",
        vertex_api_version="v1",
    )

    assert url == expected_url
    assert endpoint == "predict"


from unittest.mock import Mock, patch

import pytest


# Add these fixtures below existing fixtures
@pytest.fixture
def vertex_client():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    return HTTPHandler()


@pytest.fixture
def encoded_images():
    image_paths = [
        "./tests/llm_translation/duck.png",
        # "./duck.png",
        "./tests/llm_translation/guinea.png",
        # "./guinea.png",
    ]
    return [encode_image_to_base64(path) for path in image_paths]


    # assert result["file_data"]["file_uri"] == http_url


@pytest.mark.parametrize(
    "input_string, expected_closer_index",
    [
        ("Duck", 0),  # Duck closer to duck image
        ("Guinea", 1),  # Guinea closer to guinea image
    ],
)
def test_aaavertex_embeddings_distances(
    vertex_client, encoded_images, input_string, expected_closer_index
):
    """
    Test cosine distances between image and text embeddings using Vertex AI multimodalembedding@001
    """
    from unittest.mock import patch

    # Mock different embedding values to simulate realistic distances
    mock_image_embeddings = [
        [0.9] + [0.1] * 767,  # Duck embedding - closer to "Duck"
        [0.1] * 767 + [0.9],  # Guinea embedding - closer to "Guinea"
    ]

    image_embeddings = []
    mock_response = MagicMock()

    def mock_auth_token(*args, **kwargs):
        return "my-fake-token", "pathrise-project"

    with (
        patch.object(vertex_client, "post", return_value=mock_response),
        patch.object(
            litellm.main.vertex_multimodal_embedding,
            "_ensure_access_token",
            side_effect=mock_auth_token,
        ),
    ):
        for idx, encoded_image in enumerate(encoded_images):
            mock_response.json.return_value = {
                "predictions": [{"imageEmbedding": mock_image_embeddings[idx]}]
            }
            mock_response.status_code = 200
            response = litellm.embedding(
                model="vertex_ai/multimodalembedding@001",
                input=[f"data:image/png;base64,{encoded_image}"],
                client=vertex_client,
            )
            print("response: ", response)
            image_embeddings.append(response.data[0].embedding)

    # Mock text embedding based on input string
    mock_text_embedding = (
        [0.9] + [0.1] * 767 if input_string == "Duck" else [0.1] * 767 + [0.9]
    )
    text_mock_response = MagicMock()
    text_mock_response.json.return_value = {
        "predictions": [{"imageEmbedding": mock_text_embedding}]
    }
    text_mock_response.status_code = 200
    with (
        patch.object(vertex_client, "post", return_value=text_mock_response),
        patch.object(
            litellm.main.vertex_multimodal_embedding,
            "_ensure_access_token",
            side_effect=mock_auth_token,
        ),
    ):
        text_response = litellm.embedding(
            model="vertex_ai/multimodalembedding@001",
            input=[input_string],
            client=vertex_client,
        )
        print("text_response: ", text_response)
        text_embedding = text_response.data[0].embedding


def test_vertex_parallel_tool_calls_true():
    """
    Test that parallel_tool_calls = True sets the correct tool_config.
    """
    tools = [
        {"type": "function", "function": {"name": "get_weather"}},
        {"type": "function", "function": {"name": "get_time"}},
    ]
    optional_params = get_optional_params(
        model="gemini-1.5-pro",
        custom_llm_provider="vertex_ai",
        tools=tools,
        parallel_tool_calls=True,
    )
    assert "tools" in optional_params


def test_vertex_parallel_tool_calls_false_multiple_tools_dropped():
    """
    parallel_tool_calls=False with multiple tools is dropped for Gemini
  (unsupported upstream). Request should succeed without the param.
    """
    tools = [
        {"type": "function", "function": {"name": "get_weather"}},
        {"type": "function", "function": {"name": "get_time"}},
    ]
    optional_params = get_optional_params(
        model="gemini-1.5-pro",
        custom_llm_provider="vertex_ai",
        tools=tools,
        parallel_tool_calls=False,
    )
    assert "parallel_tool_calls" not in optional_params
    assert "tools" in optional_params

    optional_params = get_optional_params(
        model="gemini-1.5-pro",
        custom_llm_provider="vertex_ai",
        functions=tools,
        parallel_tool_calls=False,
    )
    assert "parallel_tool_calls" not in optional_params


def test_vertex_parallel_tool_calls_false_single_tool():
    """
    Test that parallel_tool_calls = False with a single tool does not raise an error
    and does not add 'tool_config' if not otherwise specified.
    """
    tools = [
        {"type": "function", "function": {"name": "get_weather"}},
    ]
    optional_params = get_optional_params(
        model="gemini-1.5-pro",
        custom_llm_provider="vertex_ai",
        tools=tools,
        parallel_tool_calls=False,
    )
    assert "tools" in optional_params


from litellm.llms.vertex_ai.gemini.transformation import transform_request_body


def test_system_prompt_only_adds_blank_user_message():
    """
    Test that the system prompt only adds a blank user message when a system message is passed in.

    Relevant Issue - https://github.com/BerriAI/litellm/issues/13769
    """
    SYSTEM_INSTRUCTION = "System instructions for the model"
    data = transform_request_body(
        messages=[{"role": "system", "content": SYSTEM_INSTRUCTION}],
        model="gemini-2.5-flash",
        optional_params={},
        custom_llm_provider="vertex_ai",
        litellm_params={},
        cached_content=None,
    )
    print("Final data: ", data)

    # validate that a blank user message is added when a system message is passed in
    assert len(data["contents"]) == 1
    first_content = data["contents"][0]
    assert first_content["role"] == "user"
    assert len(first_content["parts"]) == 1

    #########################################################
    # system message was passed in
    #########################################################
    assert len(data["system_instruction"]) == 1
    assert data["system_instruction"]["parts"][0]["text"] == SYSTEM_INSTRUCTION


@pytest.fixture
def _cached_vertex_credentials(monkeypatch: pytest.MonkeyPatch) -> str:
    from litellm.main import (
        vertex_chat_completion,
        vertex_embedding,
        vertex_model_garden_chat_completion,
        vertex_partner_models_chat_completion,
    )

    credential_key: Final = "test-vertex-gemini-credentials"
    credentials: Final = Credentials(token="test-vertex-token")
    for vertex_configuration in (
        vertex_chat_completion,
        vertex_embedding,
        vertex_model_garden_chat_completion,
        vertex_partner_models_chat_completion,
    ):
        monkeypatch.setitem(
            vertex_configuration._credentials_project_mapping,
            (credential_key, "test-project"),
            (credentials, "test-project"),
        )
    return credential_key


def test_litellm_completion_vertex_exception(
    respx_mock: respx.MockRouter, _cached_vertex_credentials: str
) -> None:
    url: Final = (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/"
        "locations/us-central1/publishers/google/models/gemini-3.5-flash-lite:generateContent"
    )
    route: Final = respx_mock.post(url).mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "code": 400,
                    "message": "Gemini rejected the request.",
                    "status": "INVALID_ARGUMENT",
                }
            },
        )
    )

    with pytest.raises(litellm.BadRequestError) as exc_info:
        litellm.completion(
            model="vertex_ai/gemini-3.5-flash-lite",
            messages=[{"role": "user", "content": "hello"}],
            vertex_project="test-project",
            vertex_location="us-central1",
            vertex_credentials=_cached_vertex_credentials,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.llm_provider == "vertex_ai"
    assert "Gemini rejected the request." in exc_info.value.message
    assert route.call_count == 1


def test_router_completion_vertex_exception(
    respx_mock: respx.MockRouter, _cached_vertex_credentials: str
) -> None:
    url: Final = (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/"
        "locations/us-central1/publishers/google/models/gemini-3.5-flash-lite:generateContent"
    )
    route: Final = respx_mock.post(url).mock(
        return_value=httpx.Response(
            429,
            json={
                "error": {
                    "code": 429,
                    "message": "Vertex Gemini quota exceeded.",
                    "status": "RESOURCE_EXHAUSTED",
                }
            },
        )
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": "vertex-gemini-test",
                "litellm_params": {
                    "model": "vertex_ai/gemini-3.5-flash-lite",
                    "vertex_project": "test-project",
                    "vertex_location": "us-central1",
                    "vertex_credentials": _cached_vertex_credentials,
                },
            }
        ],
        num_retries=0,
    )

    with pytest.raises(litellm.RateLimitError) as exc_info:
        router.completion(
            model="vertex-gemini-test",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert exc_info.value.status_code == 429
    assert exc_info.value.llm_provider == "vertex_ai"
    assert "Vertex Gemini quota exceeded." in exc_info.value.message
    assert route.call_count == 1


def _vertex_gemini_response(text: str = "hello") -> dict[str, object]:
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": text}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 1,
            "totalTokenCount": 2,
        },
    }


@pytest.mark.asyncio
async def test_completion_fine_tuned_model(
    respx_mock: respx.MockRouter,
    _cached_vertex_credentials: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    url: Final = (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/"
        "locations/us-central1/endpoints/4965075652664360960:generateContent"
    )
    route: Final = respx_mock.post(url).mock(
        return_value=httpx.Response(200, json=_vertex_gemini_response("A canvas vast"))
    )

    response: Final = await litellm.acompletion(
        model="vertex_ai_beta/4965075652664360960",
        messages=[{"role": "user", "content": "Write a short poem about the sky"}],
        vertex_project="test-project",
        vertex_location="us-central1",
        vertex_credentials=_cached_vertex_credentials,
        max_retries=0,
    )

    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content) == {
        "contents": [
            {"role": "user", "parts": [{"text": "Write a short poem about the sky"}]}
        ]
    }
    assert response.choices[0].message.content.startswith("A canvas vast")
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.total_tokens == 2


@pytest.mark.parametrize(
    ("base_model", "metadata"),
    [
        (None, {"model_info": {"base_model": "vertex_ai/gemini-1.5-pro"}}),
        ("vertex_ai/gemini-1.5-pro", None),
    ],
)
def test_gemini_finetuned_endpoint(
    respx_mock: respx.MockRouter,
    _cached_vertex_credentials: str,
    base_model: str | None,
    metadata: dict[str, object] | None,
) -> None:
    url: Final = (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/"
        "locations/us-central1/endpoints/4965075652664360960:generateContent"
    )
    route: Final = respx_mock.post(url).mock(
        return_value=httpx.Response(200, json=_vertex_gemini_response())
    )

    litellm.completion(
        model="vertex_ai/4965075652664360960",
        messages=[{"role": "user", "content": "search for weather in boston"}],
        vertex_project="test-project",
        vertex_location="us-central1",
        vertex_credentials=_cached_vertex_credentials,
        base_model=base_model,
        metadata=metadata,
        max_retries=0,
    )

    assert route.call_count == 1
    assert str(route.calls[0].request.url).endswith(
        "endpoints/4965075652664360960:generateContent"
    )


def test_gemini_nullable_object_tool_schema_httpx(
    respx_mock: respx.MockRouter, _cached_vertex_credentials: str
) -> None:
    url: Final = (
        "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/"
        "publishers/google/models/gemini-3.5-flash:generateContent"
    )
    route: Final = respx_mock.post(url).mock(
        return_value=httpx.Response(200, json=_vertex_gemini_response())
    )

    litellm.completion(
        model="vertex_ai/gemini-3.5-flash",
        messages=[{"role": "user", "content": "call the tool"}],
        tools=[
            {
                "type": "function",
                "strict": True,
                "function": {
                    "name": "create_support_ticket",
                    "description": "Create a support ticket",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["ticket_id", "customer_context"],
                        "properties": {
                            "ticket_id": {"type": "string"},
                            "optional_context": {
                                "type": ["object", "null"],
                                "additionalProperties": False,
                                "required": ["source"],
                                "properties": {"source": {"type": "string"}},
                            },
                            "customer_context": {
                                "anyOf": [
                                    {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["user_id", "plan"],
                                        "properties": {
                                            "user_id": {"type": "string"},
                                            "plan": {"type": "string"},
                                        },
                                    },
                                    {"type": "null"},
                                ],
                            },
                        },
                    },
                },
            }
        ],
        tool_choice="required",
        vertex_project="test-project",
        vertex_location="global",
        vertex_credentials=_cached_vertex_credentials,
        max_retries=0,
    )

    assert route.call_count == 1
    request_body: Final = json.loads(route.calls[0].request.content)
    parameters_schema: Final = request_body["tools"][0]["function_declarations"][0][
        "parameters"
    ]
    context_schema: Final = parameters_schema["properties"]["customer_context"]
    nullable_object_schema: Final = context_schema["anyOf"][0]
    assert nullable_object_schema["nullable"] is True
    assert nullable_object_schema["properties"] == {
        "user_id": {"type": "string"},
        "plan": {"type": "string"},
    }
    assert parameters_schema["properties"]["optional_context"]["anyOf"] == [
        {
            "type": "object",
            "required": ["source"],
            "properties": {"source": {"type": "string"}},
        },
        {"type": "null"},
    ]


@pytest.mark.parametrize("value_in_dict", [{}, {"disable_attribution": False}])
def test_gemini_pro_grounding(
    respx_mock: respx.MockRouter,
    _cached_vertex_credentials: str,
    value_in_dict: dict[str, bool],
) -> None:
    url: Final = (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/"
        "locations/us-central1/publishers/google/models/gemini-1.0-pro-001:generateContent"
    )
    route: Final = respx_mock.post(url).mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Argentina won the World Cup"}],
                        },
                        "finishReason": "STOP",
                        "groundingMetadata": {
                            "webSearchQueries": ["latest world cup winner"],
                            "groundingChunks": [
                                {
                                    "web": {
                                        "uri": "https://example.com",
                                        "title": "Example",
                                    }
                                }
                            ],
                        },
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 3,
                },
            },
        )
    )

    response: Final = litellm.completion(
        model="vertex_ai_beta/gemini-1.0-pro-001",
        messages=[{"role": "user", "content": "Who won the world cup?"}],
        tools=[{"googleSearchRetrieval": value_in_dict}],
        vertex_project="test-project",
        vertex_location="us-central1",
        vertex_credentials=_cached_vertex_credentials,
        max_retries=0,
    )

    assert route.call_count == 1
    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body["tools"] == [{"googleSearchRetrieval": value_in_dict}]
    assert response._hidden_params["vertex_ai_grounding_metadata"] == [
        {
            "webSearchQueries": ["latest world cup winner"],
            "groundingChunks": [
                {"web": {"uri": "https://example.com", "title": "Example"}}
            ],
        }
    ]


_CUSTOM_API_BASE: Final = "https://vertex-proxy.example.com/v1/custom-route"


@pytest.mark.parametrize(
    ("model", "endpoint", "response_body"),
    [
        ("gemini-2.5-flash-lite", "generateContent", _vertex_gemini_response()),
        (
            "claude-3-5-sonnet@20240620",
            "rawPredict",
            {
                "id": "msg-custom-base",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20240620",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        ),
    ],
)
def test_gemini_pro_httpx_custom_api_base(
    respx_mock: respx.MockRouter,
    _cached_vertex_credentials: str,
    model: str,
    endpoint: str,
    response_body: dict[str, object],
) -> None:
    route: Final = respx_mock.post(f"{_CUSTOM_API_BASE}:{endpoint}").mock(
        return_value=httpx.Response(200, json=response_body)
    )

    response: Final = litellm.completion(
        model=f"vertex_ai/{model}",
        messages=[{"role": "user", "content": "Hello world"}],
        api_base=_CUSTOM_API_BASE,
        extra_headers={"hello": "world"},
        vertex_project="test-project",
        vertex_location="us-central1",
        vertex_credentials=_cached_vertex_credentials,
        max_retries=0,
    )

    assert route.call_count == 1
    assert str(route.calls[0].request.url) == f"{_CUSTOM_API_BASE}:{endpoint}"
    assert route.calls[0].request.headers["hello"] == "world"
    assert response.choices[0].message.content == "hello"


def test_prompt_factory() -> None:
    from litellm.llms.vertex_ai.gemini.transformation import (
        gemini_convert_messages_with_history,
    )

    messages: Final = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {"role": "user", "content": "What is the weather in San Francisco?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location":"San Francisco"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_weather",
            "content": "Sunny, 68 degrees",
        },
    ]

    translated: Final = gemini_convert_messages_with_history(messages=messages)

    assert translated == [
        {
            "role": "user",
            "parts": [
                {"text": "You are a helpful assistant."},
                {"text": "What is the weather in San Francisco?"},
            ],
        },
        {
            "role": "model",
            "parts": [
                {"text": ""},
                {
                    "function_call": {
                        "name": "get_weather",
                        "args": {"location": "San Francisco"},
                    }
                },
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "get_weather",
                        "response": {"content": "Sunny, 68 degrees"},
                    }
                }
            ],
        },
    ]


def test_signed_s3_url_with_format(
    respx_mock: respx.MockRouter, _cached_vertex_credentials: str
) -> None:
    url: Final = (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/"
        "locations/us-central1/publishers/google/models/gemini-2.0-flash-001:generateContent"
    )
    route: Final = respx_mock.post(url).mock(
        return_value=httpx.Response(200, json=_vertex_gemini_response())
    )

    litellm.completion(
        model="vertex_ai/gemini-2.0-flash-001",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://images.example.com/logo.png?signature=test",
                            "format": "image/jpeg",
                        },
                    },
                    {"type": "text", "text": "Describe this image"},
                ],
            }
        ],
        vertex_project="test-project",
        vertex_location="us-central1",
        vertex_credentials=_cached_vertex_credentials,
        max_retries=0,
    )

    assert route.call_count == 1
    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body["contents"][0]["parts"] == [
        {
            "file_data": {
                "mime_type": "image/jpeg",
                "file_uri": "https://images.example.com/logo.png?signature=test",
            }
        },
        {"text": "Describe this image"},
    ]


@pytest.mark.asyncio
async def test_vertex_ai_deepseek(
    respx_mock: respx.MockRouter,
    _cached_vertex_credentials: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post(
        "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/"
        "locations/us-central1/endpoints/openapi/chat/completions"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello"},
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
                "model": "deepseek-ai/deepseek-r1-0528-maas",
            },
        )
    )

    await litellm.acompletion(
        model="vertex_ai/deepseek-ai/deepseek-r1-0528-maas",
        messages=[{"role": "user", "content": "Hi!"}],
        vertex_project="test-project",
        vertex_location="us-central1",
        vertex_credentials=_cached_vertex_credentials,
        max_retries=0,
    )

    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content) == {
        "model": "deepseek-ai/deepseek-r1-0528-maas",
        "messages": [{"role": "user", "content": "Hi!"}],
        "stream": False,
    }


def test_vertex_ai_response_id(
    respx_mock: respx.MockRouter, _cached_vertex_credentials: str
) -> None:
    url: Final = (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/"
        "locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent"
    )
    route: Final = respx_mock.post(url).mock(
        return_value=httpx.Response(
            200,
            json={
                "responseId": "vertex_ai_response_123",
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Hello! How can I help you today?"}],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 8,
                    "totalTokenCount": 18,
                },
            },
        )
    )

    response: Final = litellm.completion(
        model="vertex_ai/gemini-1.5-pro",
        messages=[{"role": "user", "content": "Hi!"}],
        vertex_project="test-project",
        vertex_location="us-central1",
        vertex_credentials=_cached_vertex_credentials,
        max_retries=0,
    )

    assert route.call_count == 1
    assert response.id == "vertex_ai_response_123"
    assert response.choices[0].message.content == "Hello! How can I help you today?"


@pytest.mark.asyncio
async def test_vertexai_embedding_finetuned(
    respx_mock: respx.MockRouter,
    _cached_vertex_credentials: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    url: Final = (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/"
        "locations/us-central1/endpoints/1004708436694269952:predict"
    )
    route: Final = respx_mock.post(url).mock(
        return_value=httpx.Response(
            200,
            json={
                "predictions": [
                    [[-0.000431762, -0.04416759, -0.03443353]],
                    [[-0.000431762, -0.04416759, -0.03443353]],
                ]
            },
        )
    )
    inputs: Final = ["good morning from litellm", "this is another item"]

    response: Final = await litellm.aembedding(
        model="vertex_ai/1004708436694269952",
        input=inputs,
        vertex_project="test-project",
        vertex_location="us-central1",
        vertex_credentials=_cached_vertex_credentials,
        max_retries=0,
    )

    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content) == {
        "instances": [{"inputs": item} for item in inputs],
        "parameters": {},
    }
    assert [item["embedding"] for item in response.data] == [
        [-0.000431762, -0.04416759, -0.03443353],
        [-0.000431762, -0.04416759, -0.03443353],
    ]


@pytest.mark.parametrize("max_retries", [None, 3])
@pytest.mark.asyncio
async def test_vertexai_model_garden_model_completion(
    respx_mock: respx.MockRouter,
    _cached_vertex_credentials: str,
    monkeypatch: pytest.MonkeyPatch,
    max_retries: int | None,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setitem(
        sys.modules,
        "vertexai",
        SimpleNamespace(preview=SimpleNamespace(language_models=SimpleNamespace())),
    )
    url: Final = (
        "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/test-project/"
        "locations/us-central1/endpoints/5464397967697903616/chat/completions"
    )
    messages: Final = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is your name?"},
    ]
    route: Final = respx_mock.post(url).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chat-vertex-model-garden",
                "object": "chat.completion",
                "created": 1731702782,
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Litellm Bot"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
        )
    )

    response: Final = await litellm.acompletion(
        model="vertex_ai/openai/5464397967697903616",
        messages=messages,
        vertex_project="test-project",
        vertex_location="us-central1",
        vertex_credentials=_cached_vertex_credentials,
        max_retries=max_retries,
    )

    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content) == {
        "model": "",
        "messages": messages,
        "stream": False,
    }
    assert response.id == "chat-vertex-model-garden"
    assert response.created == 1731702782
    assert response.model == "vertex_ai/meta-llama/Llama-3.1-8B-Instruct"
    assert len(response.choices) == 1
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "Litellm Bot"
    assert response.choices[0].finish_reason == "stop"
    assert (
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        response.usage.total_tokens,
    ) == (3, 2, 5)


@pytest.mark.asyncio
async def test_partner_models_httpx_ai21(
    respx_mock: respx.MockRouter,
    _cached_vertex_credentials: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    url: Final = (
        "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/test-project/"
        "locations/us-central1/publishers/ai21/models/jamba-1.5-mini@001:rawPredict"
    )
    messages: Final = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in San Francisco?"},
    ]
    tools: Final = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]
    route: Final = respx_mock.post(url).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chat-ai21",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Checking the weather.",
                            "tool_calls": [
                                {
                                    "id": "call-weather",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location":"San Francisco"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 4,
                    "total_tokens": 7,
                },
                "model": "jamba-1.5-mini@001",
            },
        )
    )

    response: Final = await litellm.acompletion(
        model="vertex_ai/jamba-1.5-mini@001",
        messages=messages,
        tools=tools,
        top_p=0.5,
        vertex_project="test-project",
        vertex_location="us-central1",
        vertex_credentials=_cached_vertex_credentials,
        max_retries=0,
    )

    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content) == {
        "model": "jamba-1.5-mini@001",
        "messages": messages,
        "top_p": 0.5,
        "tools": tools,
        "stream": False,
    }
    assert response.id == "chat-ai21"
    assert len(response.choices) == 1
    assert response.choices[0].message.content == "Checking the weather."
    assert response.choices[0].message.tool_calls[0].function.name == "get_weather"
    assert (
        response.choices[0].message.tool_calls[0].function.arguments
        == '{"location":"San Francisco"}'
    )
    assert (
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        response.usage.total_tokens,
    ) == (3, 4, 7)


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_gemini_context_caching_anthropic_format(
    respx_mock: respx.MockRouter,
    fake_provider_credentials: None,
    sync_mode: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    cache_list_route: Final = respx_mock.get(
        url__regex=r".*cachedContents.*"
    ).mock(return_value=httpx.Response(200, json={"cachedContents": []}))
    cache_route: Final = respx_mock.post(url__regex=r".*cachedContents.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "name": "cachedContents/test-cache",
                "model": "models/gemini-2.5-flash-lite-001",
                "expireTime": "2024-08-26T22:36:15Z",
                "usageMetadata": {"totalTokenCount": 323383},
            },
        )
    )
    generation_route: Final = respx_mock.post(
        url__regex=r".*generateContent.*"
    ).mock(return_value=httpx.Response(200, json=_vertex_gemini_response()))
    messages: Final = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Legal agreement " * 4000,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Summarize the agreement",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "assistant", "content": "The agreement lasts one year."},
        {"role": "user", "content": "What is its duration?"},
    ]

    if sync_mode:
        litellm.completion(
            model="gemini/gemini-2.5-flash-lite-001",
            messages=messages,
            temperature=0.2,
            max_tokens=10,
        )
    else:
        await litellm.acompletion(
            model="gemini/gemini-2.5-flash-lite-001",
            messages=messages,
            temperature=0.2,
            max_tokens=10,
        )

    assert cache_route.call_count == 1
    assert generation_route.call_count == 1
    assert cache_list_route.call_count == 1
    assert [(call.request.method, str(call.request.url)) for call in respx_mock.calls] == [
        ("GET", "https://generativelanguage.googleapis.com/v1beta/cachedContents"),
        ("POST", "https://generativelanguage.googleapis.com/v1beta/cachedContents"),
        (
            "POST",
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite-001:generateContent",
        ),
    ]
    cache_request: Final = json.loads(cache_route.calls[0].request.content)
    assert cache_request["model"] == "models/gemini-2.5-flash-lite-001"
    assert cache_request["contents"] == [
        {"role": "user", "parts": [{"text": "Summarize the agreement"}]}
    ]
    assert cache_request["system_instruction"] == {
        "parts": [{"text": "Legal agreement " * 4000}]
    }
    assert json.loads(generation_route.calls[0].request.content) == {
        "contents": [
            {"role": "model", "parts": [{"text": "The agreement lasts one year."}]},
            {"role": "user", "parts": [{"text": "What is its duration?"}]},
        ],
        "generationConfig": {"temperature": 0.2, "max_output_tokens": 10},
        "cachedContent": "cachedContents/test-cache",
    }


@pytest.mark.asyncio
async def test_anthropic_message_via_anthropic_messages(
    respx_mock: respx.MockRouter,
    _cached_vertex_credentials: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.experimental_pass_through.transformation import (
        VertexAIPartnerModelsAnthropicMessagesConfig,
    )
    from litellm.utils import ProviderConfigManager

    provider_configs: Final = (
        ProviderConfigManager.get_provider_anthropic_messages_config(
            model="vertex_ai/claude-3-5-sonnet@20240620",
            provider=litellm.LlmProviders.VERTEX_AI,
        ),
        ProviderConfigManager.get_provider_anthropic_messages_config(
            model="claude-3-5-sonnet@20240620",
            provider=litellm.LlmProviders.VERTEX_AI,
        ),
    )
    for provider_config in provider_configs:
        assert isinstance(provider_config, VertexAIPartnerModelsAnthropicMessagesConfig)
        monkeypatch.setitem(
            provider_config._credentials_project_mapping,
            (_cached_vertex_credentials, "test-project"),
            (Credentials(token="test-vertex-token"), "test-project"),
        )
    route: Final = respx_mock.post(
        url__regex=r".*claude-3-5-sonnet@20240620.*"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg-vertex",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20240620",
                "content": [{"type": "text", "text": "Here are the recipes."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 4, "output_tokens": 5},
            },
        )
    )
    messages: Final = [{"role": "user", "content": "List 5 cookie recipes"}]
    call_params: Final = {
        "model": "vertex_ai/claude-3-5-sonnet@20240620",
        "messages": messages,
        "max_tokens": 100,
        "vertex_project": "test-project",
        "vertex_location": "us-east5",
        "vertex_credentials": _cached_vertex_credentials,
        "max_retries": 0,
    }

    response: Final = await litellm.anthropic_messages(**call_params)
    completion_response: Final = await litellm.acompletion(**call_params)

    assert response["stop_reason"] == "end_turn"
    assert completion_response.choices[0].message.content == "Here are the recipes."
    assert route.call_count == 2
    messages_request: Final = route.calls[0].request
    completion_request: Final = route.calls[1].request
    assert str(messages_request.url) == (
        "https://us-east5-aiplatform.googleapis.com/v1/projects/test-project/locations/us-east5/"
        "publishers/anthropic/models/claude-3-5-sonnet@20240620:rawPredict"
    )
    assert str(completion_request.url) == str(messages_request.url)
    for request in (messages_request, completion_request):
        assert request.headers["authorization"] == "Bearer test-vertex-token"
        assert request.headers["content-type"] == "application/json"
    messages_body: Final = json.loads(messages_request.content)
    assert messages_body == {
        "anthropic_version": "vertex-2023-10-16",
        "max_tokens": 100,
        "messages": messages,
        "stream": False,
    }
    assert set(json.loads(completion_request.content)) <= set(messages_body)


def test_gemini_fine_tuned_model_request_consistency(
    respx_mock: respx.MockRouter, _cached_vertex_credentials: str
) -> None:
    route: Final = respx_mock.post(
        url__regex=r".*publishers/google/models/.*:generateContent"
    ).mock(return_value=httpx.Response(200, json=_vertex_gemini_response()))
    messages: Final = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather?"},
    ]
    tools: Final = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]
    common_params: Final = {
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "vertex_project": "test-project",
        "vertex_location": "us-central1",
        "vertex_credentials": _cached_vertex_credentials,
        "max_retries": 0,
    }

    litellm.completion(model="vertex_ai/gemini/ft-uuid", **common_params)
    litellm.completion(model="vertex_ai/gemini-2.5-flash", **common_params)

    assert route.call_count == 2
    first_request: Final = json.loads(route.calls[0].request.content)
    second_request: Final = json.loads(route.calls[1].request.content)
    assert str(route.calls[0].request.url) == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/"
        "publishers/google/models/ft-uuid:generateContent"
    )
    assert first_request["system_instruction"] == {
        "parts": [{"text": "You are a helpful assistant."}]
    }
    assert first_request["tools"][0]["function_declarations"][0]["name"] == "get_weather"
    assert first_request == second_request


_RECIPES_SCHEMA: Final = {
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
    "additionalProperties": False,
}


def _recipes_response(model: str, invalid_response: bool) -> dict[str, object]:
    if "claude" in model and invalid_response:
        return {
            "id": "msg-recipes",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20240620",
            "content": [{"type": "text", "text": "Hi! My name is Claude."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 4},
        }
    if "claude" in model:
        return {
            "id": "msg-recipes",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20240620",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu-recipes",
                    "name": "json_tool_call",
                    "input": {"recipes": [{"recipe_name": "Sugar Cookies"}]},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 3, "output_tokens": 4},
        }
    if invalid_response:
        return _vertex_gemini_response('[{"recipe_world": "Sugar Cookies"}]')
    return _vertex_gemini_response('{"recipes": [{"recipe_name": "Sugar Cookies"}]}')


_JSON_SCHEMA_MODELS: Final = pytest.mark.parametrize(
    ("model", "vertex_location", "endpoint"),
    [
        ("vertex_ai_beta/gemini-2.0-flash-001", "us-central1", "gemini-2.0-flash-001:generateContent"),
        ("vertex_ai_beta/gemini-2.5-flash-lite", "us-central1", "gemini-2.5-flash-lite:generateContent"),
        ("vertex_ai/claude-3-5-sonnet@20240620", "us-east5", "claude-3-5-sonnet@20240620:rawPredict"),
    ],
)


@_JSON_SCHEMA_MODELS
@pytest.mark.parametrize("invalid_response", [True, False])
@pytest.mark.parametrize("enforce_validation", [True, False])
def test_gemini_pro_json_schema_args_sent_httpx(
    respx_mock: respx.MockRouter,
    _cached_vertex_credentials: str,
    model: str,
    vertex_location: str,
    endpoint: str,
    invalid_response: bool,
    enforce_validation: bool,
) -> None:
    route: Final = respx_mock.post(url__regex=rf".*/{re.escape(endpoint)}$").mock(
        return_value=httpx.Response(200, json=_recipes_response(model, invalid_response))
    )
    call: Final = functools.partial(
        litellm.completion,
        model=model,
        messages=[{"role": "user", "content": "List 5 cookie recipes"}],
        response_format={
            "type": "json_object",
            "response_schema": _RECIPES_SCHEMA,
            "enforce_validation": enforce_validation,
        },
        vertex_project="test-project",
        vertex_location=vertex_location,
        vertex_credentials=_cached_vertex_credentials,
        max_retries=0,
    )

    if invalid_response and enforce_validation:
        with pytest.raises(litellm.JSONSchemaValidationError):
            call()
    else:
        response = call()
        assert response.model == model.split("/")[1]
        if not invalid_response:
            assert json.loads(response.choices[0].message.content) == {
                "recipes": [{"recipe_name": "Sugar Cookies"}]
            }

    assert route.call_count == 1
    request_body: Final = json.loads(route.calls[0].request.content)
    if "claude" in model:
        assert request_body["tool_choice"] == {"type": "tool", "name": "json_tool_call"}
        assert request_body["tools"] == [{"name": "json_tool_call", "input_schema": _RECIPES_SCHEMA}]
    else:
        assert request_body["generationConfig"]["response_mime_type"] == "application/json"
        assert request_body["generationConfig"]["response_json_schema"] == _RECIPES_SCHEMA


@_JSON_SCHEMA_MODELS
@pytest.mark.parametrize("invalid_response", [True, False])
@pytest.mark.parametrize("enforce_validation", [True, False])
def test_gemini_pro_json_schema_args_sent_httpx_openai_schema(
    respx_mock: respx.MockRouter,
    _cached_vertex_credentials: str,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    vertex_location: str,
    endpoint: str,
    invalid_response: bool,
    enforce_validation: bool,
) -> None:
    from pydantic import BaseModel

    class Recipe(BaseModel):
        recipe_name: str

    class ResponseSchema(BaseModel):
        recipes: list[Recipe]

    monkeypatch.setattr(litellm, "enable_json_schema_validation", enforce_validation)
    route: Final = respx_mock.post(url__regex=rf".*/{re.escape(endpoint)}$").mock(
        return_value=httpx.Response(200, json=_recipes_response(model, invalid_response))
    )
    call: Final = functools.partial(
        litellm.completion,
        model=model,
        messages=[{"role": "user", "content": "List 5 cookie recipes"}],
        response_format=ResponseSchema,
        vertex_project="test-project",
        vertex_location=vertex_location,
        vertex_credentials=_cached_vertex_credentials,
        max_retries=0,
    )

    if invalid_response and enforce_validation:
        with pytest.raises(litellm.JSONSchemaValidationError):
            call()
    else:
        response = call()
        assert response.model == model.split("/")[1]
        if not invalid_response:
            assert json.loads(response.choices[0].message.content) == {
                "recipes": [{"recipe_name": "Sugar Cookies"}]
            }

    assert route.call_count == 1
    request_body: Final = json.loads(route.calls[0].request.content)
    if "claude" in model:
        assert request_body["tool_choice"] == {"type": "tool", "name": "json_tool_call"}
        assert set(request_body["tools"][0]["input_schema"]["properties"]) == {"recipes"}
    else:
        generation_config: Final = request_body["generationConfig"]
        assert generation_config["response_mime_type"] == "application/json"
        assert set(generation_config["response_json_schema"]["properties"]) == {"recipes"}


@pytest.mark.parametrize("content_filter_type", ["prompt", "response"])
@pytest.mark.asyncio
async def test_gemini_pro_json_schema_httpx_content_policy_error(
    respx_mock: respx.MockRouter,
    _cached_vertex_credentials: str,
    content_filter_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    response_body: Final = (
        {
            "promptFeedback": {"blockReason": "OTHER"},
            "usageMetadata": {"promptTokenCount": 1, "totalTokenCount": 1},
        }
        if content_filter_type == "prompt"
        else {
            "candidates": [
                {
                    "content": {"role": "model", "parts": []},
                    "finishReason": "SAFETY",
                }
            ],
            "usageMetadata": {"promptTokenCount": 1, "totalTokenCount": 1},
        }
    )
    route: Final = respx_mock.post(
        url__regex=r".*gemini-2.5-flash-lite:generateContent"
    ).mock(return_value=httpx.Response(200, json=response_body))

    response: Final = await litellm.acompletion(
        model="vertex_ai_beta/gemini-2.5-flash-lite",
        messages=[{"role": "user", "content": "List recipes"}],
        response_format={"type": "json_object"},
        vertex_project="test-project",
        vertex_location="us-central1",
        vertex_credentials=_cached_vertex_credentials,
        max_retries=0,
    )

    assert route.call_count == 1
    assert response.choices[0].finish_reason == "content_filter"


@pytest.mark.parametrize("route_type", ["completion", "embedding"])
def test_litellm_api_base(
    respx_mock: respx.MockRouter,
    _cached_vertex_credentials: str,
    monkeypatch: pytest.MonkeyPatch,
    route_type: str,
) -> None:
    monkeypatch.setattr(litellm, "api_base", "https://litellm.com")
    route: Final = respx_mock.post(url__regex=r"https://litellm\.com.*").mock(
        return_value=httpx.Response(
            200,
            json=(
                _vertex_gemini_response()
                if route_type == "completion"
                else {
                    "predictions": [
                        {"embeddings": {"values": [0.1, 0.2], "statistics": {"token_count": 1}}}
                    ]
                }
            ),
        )
    )

    if route_type == "completion":
        litellm.completion(
            model="vertex_ai/gemini-2.0-flash-001",
            messages=[{"role": "user", "content": "Hello"}],
            vertex_project="test-project",
            vertex_location="us-central1",
            vertex_credentials=_cached_vertex_credentials,
            max_retries=0,
        )
    else:
        litellm.embedding(
            model="vertex_ai/gemini-2.0-flash-001",
            input=["Hello"],
            vertex_project="test-project",
            vertex_location="us-central1",
            vertex_credentials=_cached_vertex_credentials,
            max_retries=0,
        )

    assert route.call_count == 1
    assert str(route.calls[0].request.url) == (
        "https://litellm.com/v1/projects/test-project/locations/us-central1/publishers/google/models/"
        f"gemini-2.0-flash-001:{'generateContent' if route_type == 'completion' else 'predict'}"
    )


def test_vertex_anthropic_completion(
    respx_mock: respx.MockRouter, _cached_vertex_credentials: str
) -> None:
    route: Final = respx_mock.post(
        url__regex=r".*claude-sonnet-4-6@default.*"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg-vertex",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6@default",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "I will greet the user.",
                        "signature": "test-signature",
                    },
                    {"type": "text", "text": "Hello, world!"},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 2, "output_tokens": 3},
            },
        )
    )

    response: Final = litellm.completion(
        model="vertex_ai/claude-sonnet-4-6@default",
        messages=[{"role": "user", "content": "Hello, world!"}],
        vertex_project="test-project",
        vertex_location="global",
        vertex_credentials=_cached_vertex_credentials,
        thinking={"type": "enabled", "budget_tokens": 1024},
        input_cost_per_token=0.001,
        output_cost_per_token=0.002,
        max_retries=0,
    )

    assert route.call_count == 1
    assert response.model == "claude-sonnet-4-6@default"
    assert response._hidden_params["response_cost"] == pytest.approx(
        2 * 0.001 + 3 * 0.002
    )
    assert response.choices[0].message.reasoning_content == "I will greet the user."
    assert response.choices[0].message.thinking_blocks == [
        {
            "type": "thinking",
            "thinking": "I will greet the user.",
            "signature": "test-signature",
        }
    ]
    assert response.choices[0].message.content == "Hello, world!"


def test_vertex_gemini_streaming_returns_text_and_finish_reason(
    respx_mock: respx.MockRouter, _cached_vertex_credentials: str
) -> None:
    route: Final = respx_mock.post(
        url__regex=(
            r"https://us-central1-aiplatform\.googleapis\.com/v1(?:beta1)?/projects/"
            r"test-project/locations/us-central1/publishers/google/models/"
            r"gemini-2\.5-flash-lite:streamGenerateContent.*"
        )
    ).mock(
        return_value=httpx.Response(
            200,
            text=(
                'data: {"candidates":[{"content":{"parts":[{"text":"Vertex streams "}],'
                '"role":"model"},"index":0}]}\n\n'
                'data: {"candidates":[{"content":{"parts":[{"text":"without a live API."}],'
                '"role":"model"},"index":0,"finishReason":"STOP"}]}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )
    )

    stream: Final = litellm.completion(
        model="vertex_ai_beta/gemini-2.5-flash-lite",
        messages=[{"role": "user", "content": "Say a sentence."}],
        stream=True,
        vertex_project="test-project",
        vertex_location="us-central1",
        vertex_credentials=_cached_vertex_credentials,
        max_retries=0,
    )
    chunks: Final = tuple(stream)

    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == (
        "Vertex streams without a live API."
    )
    assert tuple(
        chunk.choices[0].finish_reason
        for chunk in chunks
        if chunk.choices[0].finish_reason is not None
    ) == ("stop",)
    assert route.call_count == 1
