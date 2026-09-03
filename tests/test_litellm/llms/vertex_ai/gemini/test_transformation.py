
from typing import Literal

import pytest

from litellm.llms.vertex_ai.gemini import transformation
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
from litellm.types.llms import openai
from litellm.types import completion
from litellm.types.llms.vertex_ai import ContentType, PartType, RequestBody


def _transform_vertex_messages(
    messages: list[openai.AllMessageValues],
    *,
    model: str = "gemini-3.7-flash",
    provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"] = "vertex_ai",
) -> RequestBody:
    return transformation._transform_request_body(
        messages=messages,
        model=model,
        optional_params={},
        custom_llm_provider=provider,
        litellm_params={},
        cached_content=None,
    )


@pytest.mark.asyncio
async def test__transform_request_body_labels():
    """
    Test that Vertex AI requests use the optional Vertex AI
    "labels" parameters sent by client.
    """

    # Set up the test parameters
    model = "vertex_ai/gemini-1.5-pro"
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"},
        {"role": "user", "content": "hi"},
    ]
    optional_params = {"labels": {"lparam1": "lvalue1", "lparam2": "lvalue2"}}
    litellm_params = {}
    transform_request_params = {
        "messages": messages,
        "model": model,
        "optional_params": optional_params,
        "custom_llm_provider": "vertex_ai",
        "litellm_params": litellm_params,
        "cached_content": None,
    }

    rb: RequestBody = transformation._transform_request_body(**transform_request_params)

    # Check URL
    assert rb["contents"] == [
        {"parts": [{"text": "hi"}], "role": "user"},
        {"parts": [{"text": "Hello! How can I assist you today?"}], "role": "model"},
        {"parts": [{"text": "hi"}], "role": "user"},
    ]
    assert "labels" in rb and rb["labels"] == {
        "lparam1": "lvalue1",
        "lparam2": "lvalue2",
    }


@pytest.mark.asyncio
async def test__transform_request_body_metadata():
    """
    Test that Vertex AI requests use the optional Open AI
    "metadata" parameters sent by client.
    """

    # Set up the test parameters
    model = "vertex_ai/gemini-1.5-pro"
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"},
        {"role": "user", "content": "hi"},
    ]
    optional_params = {}
    litellm_params = {
        "metadata": {"requester_metadata": {"rparam1": "rvalue1", "rparam2": "rvalue2"}}
    }
    transform_request_params = {
        "messages": messages,
        "model": model,
        "optional_params": optional_params,
        "custom_llm_provider": "vertex_ai",
        "litellm_params": litellm_params,
        "cached_content": None,
    }

    rb: RequestBody = transformation._transform_request_body(**transform_request_params)

    # Check URL
    assert rb["contents"] == [
        {"parts": [{"text": "hi"}], "role": "user"},
        {"parts": [{"text": "Hello! How can I assist you today?"}], "role": "model"},
        {"parts": [{"text": "hi"}], "role": "user"},
    ]
    assert "labels" in rb and rb["labels"] == {
        "rparam1": "rvalue1",
        "rparam2": "rvalue2",
    }


@pytest.mark.asyncio
async def test__transform_request_body_labels_and_metadata():
    """
    Test that Vertex AI requests use the optional Vertex AI
    "labels" parameters sent by client and that the "metadata"
    optional Open AI parameters are ignored if the client uses
    "labels" parameters.
    """

    # Set up the test parameters
    model = "vertex_ai/gemini-1.5-pro"
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"},
        {"role": "user", "content": "hi"},
    ]
    optional_params = {"labels": {"lparam1": "lvalue1", "lparam2": "lvalue2"}}
    litellm_params = {
        "metadata": {"requester_metadata": {"rparam1": "rvalue1", "rparam2": "rvalue2"}}
    }
    transform_request_params = {
        "messages": messages,
        "model": model,
        "optional_params": optional_params,
        "custom_llm_provider": "vertex_ai",
        "litellm_params": litellm_params,
        "cached_content": None,
    }

    rb: RequestBody = transformation._transform_request_body(**transform_request_params)

    # Check URL
    assert rb["contents"] == [
        {"parts": [{"text": "hi"}], "role": "user"},
        {"parts": [{"text": "Hello! How can I assist you today?"}], "role": "model"},
        {"parts": [{"text": "hi"}], "role": "user"},
    ]
    assert "labels" in rb and rb["labels"] == {
        "lparam1": "lvalue1",
        "lparam2": "lvalue2",
    }


@pytest.mark.asyncio
async def test__transform_request_body_image_config():
    """
    Test that Vertex AI Gemini supports the imageConfig parameter for gemini-2.5-flash-image model.
    """
    model = "gemini-2.5-flash-image"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Create a picture of a nano banana dish in a fancy restaurant with a Gemini theme",
                }
            ],
        }
    ]
    optional_params = {
        "imageConfig": {"aspectRatio": "16:9"},
        "responseModalities": ["Image"],
    }
    litellm_params = {}
    transform_request_params = {
        "messages": messages,
        "model": model,
        "optional_params": optional_params,
        "custom_llm_provider": "gemini",
        "litellm_params": litellm_params,
        "cached_content": None,
    }

    rb: RequestBody = transformation._transform_request_body(**transform_request_params)

    assert "generationConfig" in rb
    assert "imageConfig" in rb["generationConfig"]
    assert rb["generationConfig"]["imageConfig"] == {"aspectRatio": "16:9"}


@pytest.mark.asyncio
async def test__transform_request_body_image_config_snake_case():
    """
    Test that Vertex AI Gemini supports the image_config parameter (snake_case) for gemini-2.5-flash-image model.
    This should be transformed to imageConfig with aspectRatio.
    """
    model = "gemini-2.5-flash-image"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Create a picture of a nano banana dish in a fancy restaurant with a Gemini theme",
                }
            ],
        }
    ]
    optional_params = {"image_config": {"aspect_ratio": "16:9"}}
    litellm_params = {}
    transform_request_params = {
        "messages": messages,
        "model": model,
        "optional_params": optional_params,
        "custom_llm_provider": "gemini",
        "litellm_params": litellm_params,
        "cached_content": None,
    }

    rb: RequestBody = transformation._transform_request_body(**transform_request_params)

    assert "generationConfig" in rb
    assert "image_config" in rb["generationConfig"]
    assert rb["generationConfig"]["image_config"] == {"aspect_ratio": "16:9"}


@pytest.mark.asyncio
async def test__transform_request_body_image_config_with_image_size():
    """Test imageSize parameter support in imageConfig"""
    model = "gemini-3-pro-image-preview"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Generate a 4K image of Tokyo skyline"}
            ],
        }
    ]
    optional_params = {
        "imageConfig": {"aspectRatio": "16:9", "imageSize": "4K"},
        "responseModalities": ["Image"],
    }
    litellm_params = {}
    transform_request_params = {
        "messages": messages,
        "model": model,
        "optional_params": optional_params,
        "custom_llm_provider": "gemini",
        "litellm_params": litellm_params,
        "cached_content": None,
    }

    rb: RequestBody = transformation._transform_request_body(**transform_request_params)

    assert "generationConfig" in rb
    assert "imageConfig" in rb["generationConfig"]
    assert rb["generationConfig"]["imageConfig"]["aspectRatio"] == "16:9"
    assert rb["generationConfig"]["imageConfig"]["imageSize"] == "4K"


def test__transform_request_body_google_maps_json_schema_uses_response_format():
    """googleMaps + JSON schema must use responseFormat, not response_mime_type."""
    messages = [{"role": "user", "content": "Find restaurants in Mumbai"}]
    schema = {
        "type": "object",
        "properties": {"places": {"type": "array"}},
        "required": ["places"],
    }
    optional_params = {
        "tools": [{"googleMaps": {}}],
        "response_mime_type": "application/json",
        "response_json_schema": schema,
    }
    transform_request_params = {
        "messages": messages,
        "model": "gemini/gemini-3.1-flash-lite",
        "optional_params": optional_params,
        "custom_llm_provider": "gemini",
        "litellm_params": {},
        "cached_content": None,
    }

    rb: RequestBody = transformation._transform_request_body(**transform_request_params)

    gen = rb["generationConfig"]
    assert "responseFormat" in gen
    assert gen["responseFormat"]["text"]["mimeType"] == "APPLICATION_JSON"
    assert gen["responseFormat"]["text"]["schema"] == schema
    assert "response_mime_type" not in gen
    assert "response_json_schema" not in gen


def test_map_function_google_search_snake_case():
    """
    Test that google_search tool (snake_case) is properly mapped to googleSearch.
    Fixes issue where tools=[{"google_search": {}}] was being stripped.
    """
    config = VertexGeminiConfig()
    optional_params = {}

    # Test snake_case google_search
    tools = [{"google_search": {}}]
    result = config._map_function(tools, optional_params)

    assert len(result) == 1
    assert "googleSearch" in result[0]
    assert result[0]["googleSearch"] == {}


def test_map_function_google_search_camel_case():
    """
    Test that googleSearch tool (camelCase) still works.
    """
    config = VertexGeminiConfig()
    optional_params = {}

    # Test camelCase googleSearch
    tools = [{"googleSearch": {}}]
    result = config._map_function(tools, optional_params)

    assert len(result) == 1
    assert "googleSearch" in result[0]
    assert result[0]["googleSearch"] == {}


def test_map_function_google_search_retrieval_snake_case():
    """
    Test that google_search_retrieval tool (snake_case) is properly mapped.
    """
    config = VertexGeminiConfig()
    optional_params = {}

    tools = [
        {
            "google_search_retrieval": {
                "dynamic_retrieval_config": {"mode": "MODE_DYNAMIC"}
            }
        }
    ]
    result = config._map_function(tools, optional_params)

    assert len(result) == 1
    assert "googleSearchRetrieval" in result[0]


def test_map_function_enterprise_web_search_snake_case():
    """
    Test that enterprise_web_search tool (snake_case) is properly mapped.
    """
    config = VertexGeminiConfig()
    optional_params = {}

    tools = [{"enterprise_web_search": {}}]
    result = config._map_function(tools, optional_params)

    assert len(result) == 1
    assert "enterpriseWebSearch" in result[0]


@pytest.mark.parametrize("provider", ["vertex_ai", "vertex_ai_beta"])
def test_text_only_model_tail_appends_user_placeholder(
    provider: Literal["vertex_ai", "vertex_ai_beta"],
) -> None:
    body: RequestBody = _transform_vertex_messages(
        [
            {"role": "user", "content": "Remember cobalt"},
            {"role": "assistant", "content": "stored"},
        ],
        provider=provider,
    )

    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "Remember cobalt"}]},
        {"role": "model", "parts": [{"text": "stored"}]},
        {"role": "user", "parts": [{"text": "."}]},
    ]


def test_text_only_model_tail_before_system_message_appends_user_placeholder() -> None:
    body: RequestBody = _transform_vertex_messages(
        [
            {"role": "user", "content": "Remember cobalt"},
            {"role": "assistant", "content": "stored"},
            {"role": "system", "content": "Keep replies short"},
        ]
    )

    assert body["contents"][-1] == {
        "role": "user",
        "parts": [{"text": "."}],
    }
    assert body["system_instruction"] == {
        "parts": [{"text": "Keep replies short"}]
    }


@pytest.mark.parametrize(
    "assistant_message",
    [
        {
            "role": "assistant",
            "content": "stored",
            "provider_specific_fields": {"thought_signatures": ["sig-1"]},
        },
        {
            "role": "assistant",
            "content": "stored",
            "reasoning_content": "Remembering the requested word",
        },
    ],
)
def test_text_model_tail_with_reasoning_metadata_appends_user_placeholder(
    assistant_message: openai.ChatCompletionAssistantMessage,
) -> None:
    body: RequestBody = _transform_vertex_messages(
        [
            {"role": "user", "content": "Remember cobalt"},
            assistant_message,
        ]
    )

    assert body["contents"][-1] == {
        "role": "user",
        "parts": [{"text": "."}],
    }


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("gemini-2.5-flash", "vertex_ai"),
        ("gemini-3.7-flash", "gemini"),
    ],
)
def test_text_model_tail_outside_vertex_gemini_3_scope_is_unchanged(
    model: str,
    provider: Literal["vertex_ai", "gemini"],
) -> None:
    body: RequestBody = _transform_vertex_messages(
        [
            {"role": "user", "content": "Remember cobalt"},
            {"role": "assistant", "content": "stored"},
        ],
        model=model,
        provider=provider,
    )

    assert body["contents"][-1] == {
        "role": "model",
        "parts": [{"text": "stored"}],
    }


def test_user_tail_is_unchanged() -> None:
    body: RequestBody = _transform_vertex_messages(
        [
            {"role": "user", "content": "Remember cobalt"},
            {"role": "assistant", "content": "stored"},
            {"role": "user", "content": "What was it?"},
        ]
    )

    assert body["contents"][-1] == {
        "role": "user",
        "parts": [{"text": "What was it?"}],
    }


def test_empty_assistant_tail_does_not_add_a_second_user_turn() -> None:
    body: RequestBody = _transform_vertex_messages(
        [
            {"role": "user", "content": "Remember cobalt"},
            {"role": "assistant", "content": None},
        ]
    )

    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "Remember cobalt"}]}
    ]


def test_function_call_model_tail_is_unchanged() -> None:
    body: RequestBody = _transform_vertex_messages(
        [
            {"role": "user", "content": "Look up cobalt"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
        ]
    )

    assert body["contents"][-1]["role"] == "model"
    assert "function_call" in body["contents"][-1]["parts"][0]


def test_server_side_tool_model_tail_is_unchanged() -> None:
    body: RequestBody = _transform_vertex_messages(
        [
            {"role": "user", "content": "Search for cobalt"},
            {
                "role": "assistant",
                "content": "stored",
                "provider_specific_fields": {
                    "server_side_tool_invocations": [
                        {
                            "tool_type": "googleSearch",
                            "id": "tool-1",
                            "args": {"query": "cobalt"},
                            "response": {"result": "stored"},
                        }
                    ]
                },
            },
        ]
    )

    assert body["contents"][-1]["role"] == "model"
    assert any(
        "toolCall" in part or "toolResponse" in part
        for part in body["contents"][-1]["parts"]
    )


def test_assistant_media_model_tail_is_unchanged() -> None:
    image_data_uri = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    body: RequestBody = _transform_vertex_messages(
        [
            {"role": "user", "content": "Generate an image"},
            {
                "role": "assistant",
                "content": "generated",
                "images": [
                    {
                        "image_url": {"url": image_data_uri, "detail": "auto"},
                        "index": 0,
                        "type": "image_url",
                    }
                ],
            },
        ]
    )

    assert body["contents"][-1]["role"] == "model"
    assert any(
        "inline_data" in part for part in body["contents"][-1]["parts"]
    )


def test_text_part_with_disallowed_key_model_tail_is_unchanged() -> None:
    contents: tuple[ContentType, ...] = (
        ContentType(role="user", parts=[PartType(text="Remember cobalt")]),
        ContentType(
            role="model",
            parts=[PartType(text="stored", media_resolution="low")],
        ),
    )

    assert transformation._append_user_after_text_only_model_tail(contents) == (
        {"role": "user", "parts": [{"text": "Remember cobalt"}]},
        {"role": "model", "parts": [{"text": "stored", "media_resolution": "low"}]},
    )


def test_empty_parts_model_tail_is_unchanged() -> None:
    contents: tuple[ContentType, ...] = (
        ContentType(role="user", parts=[PartType(text="Remember cobalt")]),
        ContentType(role="model", parts=[]),
    )

    assert transformation._append_user_after_text_only_model_tail(contents) == contents
