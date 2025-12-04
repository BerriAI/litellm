import base64
import json
import os
import sys

from dotenv import load_dotenv

import litellm.litellm_core_utils
import litellm.litellm_core_utils.prompt_templates
import litellm.litellm_core_utils.prompt_templates.factory

load_dotenv()
from unittest.mock import MagicMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest

import litellm
from litellm import get_optional_params
from litellm.llms.vertex_ai.gemini.transformation import _process_gemini_image
from litellm.types.llms.vertex_ai import BlobType


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


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
                                "name": {"title": "Name", "type": "string"},
                                "date": {"title": "Date", "type": "string"},
                                "participants": {
                                    "items": {"type": "string"},
                                    "title": "Participants",
                                    "type": "array",
                                },
                            },
                            "propertyOrdering": [
                                "name",
                                "date",
                                "participants",
                            ],
                            "required": ["name", "date", "participants"],
                            "title": "CalendarEvent",
                            "type": "object",
                        },
                        "title": "Events",
                        "type": "array",
                    }
                },
                "propertyOrdering": ["events"],
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
                model="gemini/gemini-1.5-pro",
                messages=messages,
                response_format=EventsList,
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

    from litellm.llms.vertex_ai.common_utils import _build_vertex_schema

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

    new_schema = _build_vertex_schema(schema)
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
                    "properties": {"param": {"type": "string"}}
                }
            }
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
    assert optional_params_function["tools"][0]["function_declarations"][0]["name"] == "test_function"
    
    # Verify the 'type' field is not present in the final result
    assert "type" not in optional_params_function["tools"][0]


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
                ]
            },
            {"role": "user", "parts": [{"text": "tell me the results."}]},
        ]


def test_function_calling_with_gemini_multiple_results():
    litellm.set_verbose = True
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
    litellm.set_verbose = True
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
            client=client,
        )
        print(resp)

        assert resp.choices[0].logprobs is not None


def test_process_gemini_image():
    """Test the _process_gemini_image function for different image sources"""
    from litellm.llms.vertex_ai.gemini.transformation import _process_gemini_image
    from litellm.types.llms.vertex_ai import FileDataType

    # Test GCS URI
    gcs_result = _process_gemini_image("gs://bucket/image.png")
    assert gcs_result["file_data"] == FileDataType(
        mime_type="image/png", file_uri="gs://bucket/image.png"
    )

    # Test gs url with format specified
    gcs_result = _process_gemini_image("gs://bucket/image", format="image/jpeg")
    assert gcs_result["file_data"] == FileDataType(
        mime_type="image/jpeg", file_uri="gs://bucket/image"
    )

    # Test HTTPS JPG URL
    https_result = _process_gemini_image("https://example.com/image.jpg")
    print("https_result JPG", https_result)
    assert https_result["file_data"] == FileDataType(
        mime_type="image/jpeg", file_uri="https://example.com/image.jpg"
    )

    # Test HTTPS PNG URL
    https_result = _process_gemini_image("https://example.com/image.png")
    print("https_result PNG", https_result)
    assert https_result["file_data"] == FileDataType(
        mime_type="image/png", file_uri="https://example.com/image.png"
    )

    # Test HTTPS VIDEO URL
    https_result = _process_gemini_image("https://cloud-samples-data/video/animals.mp4")
    print("https_result PNG", https_result)
    assert https_result["file_data"] == FileDataType(
        mime_type="video/mp4", file_uri="https://cloud-samples-data/video/animals.mp4"
    )

    # Test HTTPS PDF URL
    https_result = _process_gemini_image("https://cloud-samples-data/pdf/animals.pdf")
    print("https_result PDF", https_result)
    assert https_result["file_data"] == FileDataType(
        mime_type="application/pdf",
        file_uri="https://cloud-samples-data/pdf/animals.pdf",
    )

    # Test base64 image
    base64_image = "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    base64_result = _process_gemini_image(base64_image)
    print("base64_result", base64_result)
    assert base64_result["inline_data"]["mimeType"] == "image/jpeg"
    assert base64_result["inline_data"]["data"] == "/9j/4AAQSkZJRg..."


def test_get_image_mime_type_from_url():
    """Test the _get_image_mime_type_from_url function for different image URLs"""
    from litellm.llms.vertex_ai.gemini.transformation import (
        _get_image_mime_type_from_url,
    )

    # Test JPEG images
    assert (
        _get_image_mime_type_from_url("https://example.com/image.jpg") == "image/jpeg"
    )
    assert (
        _get_image_mime_type_from_url("https://example.com/image.jpeg") == "image/jpeg"
    )
    assert (
        _get_image_mime_type_from_url("https://example.com/IMAGE.JPG") == "image/jpeg"
    )

    # Test PNG images
    assert _get_image_mime_type_from_url("https://example.com/image.png") == "image/png"
    assert _get_image_mime_type_from_url("https://example.com/IMAGE.PNG") == "image/png"

    # Test WebP images
    assert (
        _get_image_mime_type_from_url("https://example.com/image.webp") == "image/webp"
    )
    assert (
        _get_image_mime_type_from_url("https://example.com/IMAGE.WEBP") == "image/webp"
    )

    # Test audio formats
    assert _get_image_mime_type_from_url("https://example.com/audio.ogg") == "audio/ogg"
    assert _get_image_mime_type_from_url("https://example.com/track.OGG") == "audio/ogg"

    # Test unsupported formats
    assert _get_image_mime_type_from_url("https://example.com/image.gif") is None
    assert _get_image_mime_type_from_url("https://example.com/image.bmp") is None
    assert _get_image_mime_type_from_url("https://example.com/image") is None
    assert _get_image_mime_type_from_url("invalid_url") is None


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
    from litellm.llms.vertex_ai.common_utils import _get_vertex_url

    url, endpoint = _get_vertex_url(
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


@pytest.fixture
def mock_convert_url_to_base64():
    with patch(
        "litellm.litellm_core_utils.prompt_templates.factory.convert_url_to_base64",
    ) as mock:
        # Setup the mock to return a valid image object
        mock.return_value = "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
        yield mock


@pytest.fixture
def mock_blob():
    return Mock(spec=BlobType)


@pytest.mark.parametrize(
    "http_url",
    [
        "http://img1.etsystatic.com/260/0/7813604/il_fullxfull.4226713999_q86e.jpg",
        "http://example.com/image.jpg",
        "http://subdomain.domain.com/path/to/image.png",
    ],
)
def test_process_gemini_image_http_url(
    http_url: str, mock_convert_url_to_base64: Mock, mock_blob: Mock
) -> None:
    """
    Test that _process_gemini_image correctly handles HTTP URLs.

    Args:
        http_url: Test HTTP URL
        mock_convert_to_anthropic: Mocked convert_to_anthropic_image_obj function
        mock_blob: Mocked BlobType instance

    Vertex AI supports image urls. Ensure no network requests are made.
    """
    expected_image_data = "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    mock_convert_url_to_base64.return_value = expected_image_data
    # Act
    result = _process_gemini_image(http_url)
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

    with patch.object(vertex_client, "post", return_value=mock_response), patch.object(
        litellm.main.vertex_multimodal_embedding,
        "_ensure_access_token",
        side_effect=mock_auth_token,
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
    with patch.object(
        vertex_client, "post", return_value=text_mock_response
    ), patch.object(
        litellm.main.vertex_multimodal_embedding,
        "_ensure_access_token",
        side_effect=mock_auth_token,
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


def test_vertex_parallel_tool_calls_false_multiple_tools_error():
    """
    Test that parallel_tool_calls = False with multiple tools raises UnsupportedParamsError
    when drop_params is False.
    """
    tools = [
        {"type": "function", "function": {"name": "get_weather"}},
        {"type": "function", "function": {"name": "get_time"}},
    ]
    with pytest.raises(litellm.utils.UnsupportedParamsError) as excinfo:
        get_optional_params(
            model="gemini-1.5-pro",
            custom_llm_provider="vertex_ai",
            tools=tools,
            parallel_tool_calls=False,
        )
    assert (
        "`parallel_tool_calls=False` is not supported by Gemini when multiple tools are"
        in str(excinfo.value)
    )

    # works when specified as "functions"
    with pytest.raises(litellm.utils.UnsupportedParamsError) as excinfo:
        get_optional_params(
            model="gemini-1.5-pro",
            custom_llm_provider="vertex_ai",
            functions=tools,
            parallel_tool_calls=False,
        )
    assert (
        "`parallel_tool_calls=False` is not supported by Gemini when multiple tools are"
        in str(excinfo.value)
    )


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


from litellm.llms.vertex_ai.gemini.transformation import _transform_request_body


def test_system_prompt_only_adds_blank_user_message():
    """
    Test that the system prompt only adds a blank user message when a system message is passed in.

    Relevant Issue - https://github.com/BerriAI/litellm/issues/13769
    """
    SYSTEM_INSTRUCTION = "System instructions for the model"
    data = _transform_request_body(
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
