
import json
import uuid
from itertools import chain
from typing import Final
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import CONVERTED_SYSTEM_NOTE
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.vertex_ai.gemini import transformation
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
from litellm.types.llms import openai
from litellm.types import completion
from litellm.types.llms.vertex_ai import RequestBody


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

    rb: RequestBody = transformation.transform_request_body(**transform_request_params)

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

    rb: RequestBody = transformation.transform_request_body(**transform_request_params)

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

    rb: RequestBody = transformation.transform_request_body(**transform_request_params)

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

    rb: RequestBody = transformation.transform_request_body(**transform_request_params)

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

    rb: RequestBody = transformation.transform_request_body(**transform_request_params)

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

    rb: RequestBody = transformation.transform_request_body(**transform_request_params)

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

    rb: RequestBody = transformation.transform_request_body(**transform_request_params)

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
    result = config.map_function(tools, optional_params)

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
    result = config.map_function(tools, optional_params)

    assert len(result) == 1
    assert "googleSearch" in result[0]
    assert result[0]["googleSearch"] == {}


def test_map_function_google_search_retrieval_snake_case():
    """
    Test that google_search_retrieval tool (snake_case) is properly mapped.
    """
    config = VertexGeminiConfig()
    optional_params = {}

    tools = [{"google_search_retrieval": {"dynamic_retrieval_config": {"mode": "MODE_DYNAMIC"}}}]
    result = config.map_function(tools, optional_params)

    assert len(result) == 1
    assert "googleSearchRetrieval" in result[0]


def test_map_function_enterprise_web_search_snake_case():
    """
    Test that enterprise_web_search tool (snake_case) is properly mapped.
    """
    config = VertexGeminiConfig()
    optional_params = {}

    tools = [{"enterprise_web_search": {}}]
    result = config.map_function(tools, optional_params)

    assert len(result) == 1
    assert "enterpriseWebSearch" in result[0]


async def test_gemini_ai_studio_async_completion_inlines_remote_images_off_the_event_loop(async_only_image_fetch):
    image_url = f"http://img.example/{uuid.uuid4()}.png"
    captured = {}

    def handle(request):
        captured["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "Green"}], "role": "model"}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
            },
        )

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))

    response = await litellm.acompletion(
        model="gemini/gemini-3.8-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What colour is this?"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        api_key="fake-gemini-key",
        client=client,
    )

    assert response.choices[0].message.content == "Green"
    assert async_only_image_fetch.fetched == [image_url]
    assert image_url not in captured["body"]
    assert async_only_image_fetch.base64_png in captured["body"]


async def test_gemini_ai_studio_async_completion_passes_files_api_uris_through_unfetched(async_only_image_fetch):
    files_api_pdf = f"https://generativelanguage.googleapis.com/v1beta/files/{uuid.uuid4().hex}"
    files_api_image = f"https://generativelanguage.googleapis.com/v1beta/files/{uuid.uuid4().hex}"
    captured = {}

    def handle(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "A report"}], "role": "model"}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
            },
        )

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))

    response = await litellm.acompletion(
        model="gemini/gemini-3.8-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize these"},
                    {"type": "file", "file": {"file_id": files_api_pdf, "format": "application/pdf"}},
                    {"type": "image_url", "image_url": {"url": files_api_image, "format": "image/png"}},
                ],
            }
        ],
        api_key="fake-gemini-key",
        client=client,
    )

    assert response.choices[0].message.content == "A report"
    assert async_only_image_fetch.fetched == []
    file_parts = [part["file_data"] for part in captured["body"]["contents"][0]["parts"] if "file_data" in part]
    assert file_parts == [
        {"mime_type": "application/pdf", "file_uri": files_api_pdf},
        {"mime_type": "image/png", "file_uri": files_api_image},
    ]


async def test_vertex_ai_async_transform_inlines_only_the_urls_gemini_cannot_fetch_itself(async_only_image_fetch):
    plain_http_png = f"http://img.example/{uuid.uuid4()}.png"
    extensionless_https = f"https://cdn.example/files/{uuid.uuid4().hex}"
    https_png = f"https://img.example/{uuid.uuid4()}.png"
    hinted_extensionless = f"https://cdn.example/files/{uuid.uuid4().hex}"
    files_api_pdf = f"https://generativelanguage.googleapis.com/v1beta/files/{uuid.uuid4().hex}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe these"},
                {"type": "image_url", "image_url": {"url": plain_http_png}},
                {"type": "image_url", "image_url": {"url": extensionless_https}},
                {"type": "image_url", "image_url": {"url": https_png}},
                {"type": "image_url", "image_url": {"url": hinted_extensionless, "mime_type": "image/webp"}},
                {"type": "file", "file": {"file_id": files_api_pdf, "format": "application/pdf"}},
            ],
        }
    ]

    body = await transformation.async_transform_request_body(
        gemini_api_key=None,
        messages=messages,
        api_base=None,
        model="gemini-3.8-flash",
        client=None,
        timeout=None,
        extra_headers=None,
        optional_params={},
        logging_obj=Mock(),
        custom_llm_provider="vertex_ai",
        litellm_params={},
        vertex_project="qa-project",
        vertex_location="us-central1",
        vertex_auth_header=None,
    )

    inlined = {"inline_data": {"mime_type": "image/png", "data": async_only_image_fetch.base64_png}}
    assert body["contents"][0]["parts"] == [
        {"text": "Describe these"},
        inlined,
        inlined,
        {"file_data": {"mime_type": "image/png", "file_uri": https_png}},
        {"file_data": {"mime_type": "image/webp", "file_uri": hinted_extensionless}},
        {"file_data": {"mime_type": "application/pdf", "file_uri": files_api_pdf}},
    ]
    assert sorted(async_only_image_fetch.fetched) == sorted([plain_http_png, extensionless_https])


@pytest.mark.parametrize("custom_llm_provider, expected_provider", [("gemini", "gemini"), (None, "vertex_ai")])
def test_tool_call_conversion_error_names_the_calling_provider(
    custom_llm_provider: str | None, expected_provider: str
) -> None:
    messages: Final = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": ["Boston"]}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]

    with pytest.raises(litellm.BadRequestError) as exc_info:
        transformation.gemini_convert_messages_with_history(messages, custom_llm_provider=custom_llm_provider)

    assert exc_info.value.llm_provider == expected_provider


def test_tool_call_arguments_gemini_cannot_use_are_dropped_when_drop_params_is_set() -> None:
    messages: Final = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": ["Boston"]}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]

    contents: Final = transformation.gemini_convert_messages_with_history(
        messages, litellm_params={"drop_params": True}, custom_llm_provider="gemini"
    )

    assert contents[1]["parts"] == [{"function_call": {"name": "get_weather", "args": {}}}]


def test_gemini_completion_reports_a_tool_call_it_cannot_convert_as_a_gemini_bad_request() -> None:
    with pytest.raises(litellm.BadRequestError) as exc_info:
        litellm.completion(
            model="gemini/gemini-flash-latest",
            api_key="test-key",
            messages=[
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": {"location": "Boston"}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ],
        )

    assert exc_info.value.llm_provider == "gemini"
    assert "must be a JSON string, got dict" in str(exc_info.value)


def _gemini_request_body(messages: list) -> RequestBody:
    return transformation.transform_request_body(
        messages=messages,
        model="gemini-3.8-flash",
        optional_params={},
        custom_llm_provider="vertex_ai",
        litellm_params={},
        cached_content=None,
    )


def _conversation_with_a_reminder_after_each_user_turn(user_turns: int) -> list:
    exchanges: Final = tuple(
        chain.from_iterable(
            (
                {"role": "user", "content": f"question {turn}"},
                {"role": "system", "content": f"{turn} of 8 turns used"},
                {"role": "assistant", "content": f"answer {turn}"},
            )
            for turn in range(1, user_turns + 1)
        )
    )
    return [{"role": "system", "content": "You are terse."}, *exchanges[:-1]]


def test_a_reminder_after_each_user_turn_keeps_the_gemini_prompt_prefix_stable_across_turns():
    turn_three: Final = _gemini_request_body(_conversation_with_a_reminder_after_each_user_turn(3))
    turn_four: Final = _gemini_request_body(_conversation_with_a_reminder_after_each_user_turn(4))

    assert turn_three["system_instruction"] == {"parts": [{"text": "You are terse."}]}
    assert turn_four["system_instruction"] == turn_three["system_instruction"]
    assert len(turn_four["contents"]) > len(turn_three["contents"])
    assert turn_four["contents"][: len(turn_three["contents"])] == turn_three["contents"]


def test_a_mid_conversation_system_message_reaches_gemini_as_a_user_part_after_the_operator_note():
    messages: Final = [
        {"role": "system", "content": "You are terse."},
        {"role": "system", "content": [{"type": "text", "text": "Answer in "}, {"type": "text", "text": "French."}]},
        {"role": "user", "content": "question 1"},
        {"role": "system", "content": "1 of 8 turns used"},
    ]
    sent: Final = json.loads(json.dumps(messages))

    body: Final = _gemini_request_body(messages)

    assert body["system_instruction"] == {"parts": [{"text": "You are terse."}, {"text": "Answer in French."}]}
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "question 1"}, {"text": CONVERTED_SYSTEM_NOTE}, {"text": "1 of 8 turns used"}]}
    ]
    assert messages == sent


def test_a_system_message_between_a_tool_call_and_its_result_goes_after_the_function_response():
    body: Final = _gemini_request_body(
        [
            {"role": "user", "content": "weather in Boston?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Boston"}'},
                    }
                ],
            },
            {"role": "system", "content": "Tool results are authoritative."},
            {"role": "tool", "tool_call_id": "call_1", "content": "72F"},
        ]
    )

    contents: Final = body["contents"]
    assert "system_instruction" not in body
    assert [content["role"] for content in contents] == ["user", "model", "user", "user"]
    assert contents[1]["parts"][0]["function_call"]["name"] == "get_weather"
    assert contents[2]["parts"][0]["function_response"]["response"] == {"content": "72F"}
    assert contents[3]["parts"] == [{"text": CONVERTED_SYSTEM_NOTE}, {"text": "Tool results are authoritative."}]


def test_reminders_between_parallel_tool_results_go_after_the_last_function_response():
    def call(call_id: str, name: str) -> dict:
        return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}

    body: Final = _gemini_request_body(
        [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "weather and time in Boston?"},
            {"role": "assistant", "content": None, "tool_calls": [call("a", "get_weather"), call("b", "get_time")]},
            {"role": "system", "content": "reminder 1"},
            {"role": "tool", "tool_call_id": "a", "content": "72F"},
            {"role": "system", "content": "reminder 2"},
            {"role": "tool", "tool_call_id": "b", "content": "3pm"},
            {"role": "user", "content": "thanks"},
        ]
    )

    contents: Final = body["contents"]
    assert [content["role"] for content in contents] == ["user", "model", "user", "user"]
    assert [part["function_response"]["name"] for part in contents[2]["parts"]] == ["get_weather", "get_time"]
    assert contents[3]["parts"] == [
        {"text": CONVERTED_SYSTEM_NOTE},
        {"text": "reminder 1"},
        {"text": CONVERTED_SYSTEM_NOTE},
        {"text": "reminder 2"},
        {"text": "thanks"},
    ]
