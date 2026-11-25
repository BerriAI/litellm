
import asyncio
import gzip
import itertools
import json
import threading
import traceback
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from itertools import chain
from typing import Final, cast
from unittest.mock import Mock

import httpx
import pytest
import respx

import litellm
from litellm.constants import MAX_IMAGE_URL_DOWNLOAD_SIZE_MB
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import CONVERTED_SYSTEM_NOTE
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.gemini import transformation
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
from litellm.types import completion
from litellm.types.llms import openai
from litellm.types.llms.vertex_ai import ContentType, RequestBody
from tests.unit.conftest import AsyncOnlyImageFetch


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


_EMPTY_USER_MESSAGES: Final = (
    pytest.param({"role": "user", "content": ""}, id="empty_string"),
    pytest.param({"role": "user", "content": []}, id="empty_list"),
    pytest.param({"role": "user", "content": None}, id="null"),
    pytest.param({"role": "user"}, id="missing"),
    pytest.param({"role": "user", "content": [{"type": "text", "text": ""}]}, id="empty_text_block"),
)


@pytest.mark.parametrize("empty_user_message", _EMPTY_USER_MESSAGES)
def test_an_empty_trailing_user_message_still_ends_the_request_on_a_user_turn(
    empty_user_message: openai.AllMessageValues,
) -> None:
    contents: Final = transformation.gemini_convert_messages_with_history(
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
            empty_user_message,
        ]
    )

    assert contents == [
        {"role": "user", "parts": [{"text": "Hi"}]},
        {"role": "model", "parts": [{"text": "Hello! How can I help?"}]},
        {"role": "user", "parts": [{"text": " "}]},
    ]


@pytest.mark.parametrize("malformed_content", [pytest.param(0, id="int"), pytest.param({}, id="object")])
def test_a_trailing_user_message_with_malformed_content_gets_no_blank_user_turn(malformed_content: object) -> None:
    contents: Final = transformation.gemini_convert_messages_with_history(
        cast(
            list[openai.AllMessageValues],
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": malformed_content},
            ],
        )
    )

    assert contents == [
        {"role": "user", "parts": [{"text": "Hi"}]},
        {"role": "model", "parts": [{"text": "Hello!"}]},
    ]


_WEATHER_TOOL_CALL: Final = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_weather",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }
    ],
}
_WEATHER_TOOL_RESULT: Final = {"role": "tool", "tool_call_id": "call_weather", "content": "Sunny"}


def _all_parts(contents: list[ContentType]) -> list[object]:
    return list(chain.from_iterable(content["parts"] for content in contents))


@pytest.mark.parametrize(
    "later_turns",
    [
        pytest.param(
            [{"role": "assistant", "content": "Still there?"}, {"role": "user", "content": "Yes"}],
            id="assistant_reply",
        ),
        pytest.param([_WEATHER_TOOL_CALL, _WEATHER_TOOL_RESULT], id="function_call"),
    ],
)
@pytest.mark.parametrize("empty_user_message", _EMPTY_USER_MESSAGES)
def test_an_empty_user_message_before_later_turns_gets_no_blank_user_turn(
    empty_user_message: openai.AllMessageValues, later_turns: list[openai.AllMessageValues]
) -> None:
    contents: Final = transformation.gemini_convert_messages_with_history(
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}, empty_user_message, *later_turns]
    )

    assert {"text": " "} not in _all_parts(contents)


@pytest.mark.parametrize("empty_user_message", _EMPTY_USER_MESSAGES)
def test_an_empty_user_message_after_tool_results_gets_no_blank_user_turn(
    empty_user_message: openai.AllMessageValues,
) -> None:
    contents: Final = transformation.gemini_convert_messages_with_history(
        [{"role": "user", "content": "Weather in Paris?"}, _WEATHER_TOOL_CALL, _WEATHER_TOOL_RESULT, empty_user_message]
    )

    assert [tuple(part) for part in contents[2]["parts"]] == [("function_response",)]
    assert {"text": " "} not in _all_parts(contents)


def _echo_request_parts(request: httpx.Request) -> httpx.Response:
    parts: Final = json.loads(request.content)["contents"][0]["parts"]
    return httpx.Response(
        200,
        json={
            "candidates": [{"content": {"parts": [{"text": json.dumps(parts)}], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
        },
    )


async def test_gemini_ai_studio_async_completion_sends_a_video_url_part_as_video(
    async_only_image_fetch: AsyncOnlyImageFetch,
) -> None:
    video_url: Final = f"https://videos.example/{uuid.uuid4()}.mp4"
    response: Final = await litellm.acompletion(
        model="gemini/gemini-3.8-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the clip"},
                    {"type": "video_url", "video_url": {"url": video_url}},
                ],
            }
        ],
        api_key="fake-gemini-key",
        client=AsyncHTTPHandler(transport=httpx.MockTransport(_echo_request_parts)),
    )

    assert json.loads(response.choices[0].message.content) == [
        {"text": "Describe the clip"},
        {"file_data": {"mime_type": "video/mp4", "file_uri": video_url}},
    ]
    assert async_only_image_fetch.fetched == []


@pytest.mark.parametrize(
    "video_part, expected_part",
    [
        pytest.param(
            {"type": "video_url", "video_url": "https://videos.example/clip.mov"},
            {"file_data": {"mime_type": "video/mov", "file_uri": "https://videos.example/clip.mov"}},
            id="bare-url-string",
        ),
        pytest.param(
            {"type": "video_url", "video_url": {"url": "gs://qa-bucket/clip.mp4"}},
            {"file_data": {"mime_type": "video/mp4", "file_uri": "gs://qa-bucket/clip.mp4"}},
            id="gcs-uri",
        ),
        pytest.param(
            {"type": "video_url", "video_url": {"url": "data:video/webm;base64,GkXfow=="}},
            {"inline_data": {"mime_type": "video/webm", "data": "GkXfow=="}},
            id="data-url",
        ),
        pytest.param(
            {
                "type": "video_url",
                "video_url": {
                    "url": "https://cdn.example/v/abc",
                    "format": "video/mp4",
                    "detail": "low",
                    "video_metadata": {"fps": 2},
                },
            },
            {
                "file_data": {"mime_type": "video/mp4", "file_uri": "https://cdn.example/v/abc"},
                "media_resolution": {"level": "MEDIA_RESOLUTION_LOW"},
                "video_metadata": {"fps": 2},
            },
            id="format-detail-and-metadata",
        ),
        pytest.param(
            {"type": "video_url", "video_url": {"url": "https://www.youtube.com/watch?v=aqz-KE-bpKQ"}},
            {"file_data": {"mime_type": "video/mp4", "file_uri": "https://www.youtube.com/watch?v=aqz-KE-bpKQ"}},
            id="youtube-link",
        ),
        pytest.param(
            {"type": "video_url", "video_url": {"url": "HTTPS://WWW.YouTube.com/watch?v=aqz-KE-bpKQ"}},
            {"file_data": {"mime_type": "video/mp4", "file_uri": "https://www.youtube.com/watch?v=aqz-KE-bpKQ"}},
            id="youtube-link-in-capitals",
        ),
        pytest.param(
            {"type": "video_url", "video_url": {"url": "https://youtu.be:443/aqz-KE-bpKQ"}},
            {"file_data": {"mime_type": "video/mp4", "file_uri": "https://youtu.be/aqz-KE-bpKQ"}},
            id="youtube-link-with-port",
        ),
        pytest.param(
            {"type": "video_url", "video_url": {"url": "https://www.youtube.com/shorts/aqz-KE-bpKQ"}},
            {"file_data": {"mime_type": "video/mp4", "file_uri": "https://www.youtube.com/shorts/aqz-KE-bpKQ"}},
            id="youtube-short",
        ),
        pytest.param(
            {"type": "video_url", "video_url": {"url": "https://www.youtube.com/embed/aqz-KE-bpKQ"}},
            {"file_data": {"mime_type": "video/mp4", "file_uri": "https://www.youtube.com/embed/aqz-KE-bpKQ"}},
            id="youtube-embed",
        ),
    ],
)
def test_a_video_url_part_reaches_gemini_as_video(
    video_part: Mapping[str, object], expected_part: Mapping[str, object]
) -> None:
    contents: Final = transformation.gemini_convert_messages_with_history(
        [{"role": "user", "content": [{"type": "text", "text": "Describe the clip"}, video_part]}],
        model="gemini-3.8-flash",
        custom_llm_provider="vertex_ai",
    )

    assert contents[0]["parts"] == [{"text": "Describe the clip"}, expected_part]


_WEB_PAGE_LINKS: Final = (
    "https://www.youtube.com/live/aqz-KE-bpKQ",
    "https://www.youtube.com/@Google",
    "https://vimeo.com/76979871",
)


@pytest.mark.parametrize("custom_llm_provider", ["gemini", "vertex_ai"])
@pytest.mark.parametrize("page_url", _WEB_PAGE_LINKS)
def test_a_video_url_linking_a_web_page_is_a_bad_request_instead_of_sending_the_html(
    monkeypatch: pytest.MonkeyPatch, page_url: str, custom_llm_provider: str
) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", False)
    messages: Final = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Describe the clip"}, {"type": "video_url", "video_url": page_url}],
        }
    ]

    with respx.mock:
        respx.get(page_url).mock(
            return_value=httpx.Response(200, text="<html>clip</html>", headers={"content-type": "text/html"})
        )
        with pytest.raises(litellm.BadRequestError, match="but it holds text/html") as excinfo:
            transformation.gemini_convert_messages_with_history(
                messages, model="gemini-3.8-flash", custom_llm_provider=custom_llm_provider
            )
        dropped: Final = transformation.gemini_convert_messages_with_history(
            messages,
            model="gemini-3.8-flash",
            custom_llm_provider=custom_llm_provider,
            litellm_params={"drop_params": True},
        )

    assert (excinfo.value.llm_provider, excinfo.value.model) == (custom_llm_provider, "gemini-3.8-flash")
    assert dropped[0]["parts"] == [{"text": "Describe the clip"}]


_MIB: Final = 1024 * 1024
_FIND_CLIP_CALL: Final = {"id": "call_1", "type": "function", "function": {"name": "find_clip", "arguments": "{}"}}


@dataclass(slots=True)
class _OversizedVideoHost:
    sent_bytes: int = 0

    def body(self) -> Iterator[bytes]:
        for chunk in itertools.repeat(b"\0" * _MIB, int(2 * MAX_IMAGE_URL_DOWNLOAD_SIZE_MB)):
            self.sent_bytes += len(chunk)
            yield chunk


def _turns_with_a_linked_video(role: str, url: str) -> list[Mapping[str, object]]:
    video: Final = {"type": "video_url", "video_url": {"url": url}}
    if role == "user":
        return [{"role": "user", "content": [{"type": "text", "text": "Describe the clip"}, video]}]
    return [
        {"role": "user", "content": "Find the clip"},
        {"role": "assistant", "content": None, "tool_calls": [_FIND_CLIP_CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": [video]},
    ]


@pytest.mark.parametrize("custom_llm_provider", ["gemini", "vertex_ai"])
@pytest.mark.parametrize("role", ["user", "tool"])
def test_an_oversized_linked_video_stops_downloading_at_the_size_limit(role: str, custom_llm_provider: str) -> None:
    video_url: Final = f"https://8.8.8.8/{uuid.uuid4()}"
    host: Final = _OversizedVideoHost()

    with respx.mock as router:
        router.get(video_url).mock(
            side_effect=lambda _: httpx.Response(200, headers={"content-type": "video/mp4"}, content=host.body())
        )
        with pytest.raises(litellm.BadRequestError):
            transformation.gemini_convert_messages_with_history(
                _turns_with_a_linked_video(role, video_url),
                model="gemini-3.8-flash",
                custom_llm_provider=custom_llm_provider,
            )

    assert 0 < host.sent_bytes <= MAX_IMAGE_URL_DOWNLOAD_SIZE_MB * _MIB + _MIB


@pytest.mark.parametrize("custom_llm_provider", ["gemini", "vertex_ai"])
@pytest.mark.parametrize("role", ["user", "tool"])
def test_a_compressed_linked_video_is_a_bad_request_without_unpacking_it(role: str, custom_llm_provider: str) -> None:
    video_url: Final = f"https://8.8.8.8/{uuid.uuid4()}"

    with respx.mock as router:
        route: Final = router.get(video_url).respond(
            200, headers={"content-type": "video/mp4", "content-encoding": "gzip"}, content=gzip.compress(b"\0" * _MIB)
        )
        with pytest.raises(litellm.BadRequestError):
            transformation.gemini_convert_messages_with_history(
                _turns_with_a_linked_video(role, video_url),
                model="gemini-3.8-flash",
                custom_llm_provider=custom_llm_provider,
            )

    assert route.calls.last.request.headers["accept-encoding"] == "identity"


@pytest.mark.parametrize("custom_llm_provider", ["gemini", "vertex_ai"])
def test_a_message_video_that_fails_to_download_is_a_bad_request_without_its_link_or_dropped_under_drop_params(
    custom_llm_provider: str,
) -> None:
    signature: Final = uuid.uuid4().hex
    video_url: Final = f"https://8.8.8.8/{uuid.uuid4()}?X-Goog-Signature={signature}"
    turns: Final = _turns_with_a_linked_video("user", video_url)

    with respx.mock as router:
        router.get(video_url).respond(404)
        with pytest.raises(litellm.BadRequestError) as excinfo:
            transformation.gemini_convert_messages_with_history(
                turns, model="gemini-3.8-flash", custom_llm_provider=custom_llm_provider
            )
        dropped: Final = transformation.gemini_convert_messages_with_history(
            turns,
            model="gemini-3.8-flash",
            custom_llm_provider=custom_llm_provider,
            litellm_params={"drop_params": True},
        )

    assert (excinfo.value.llm_provider, excinfo.value.model) == (custom_llm_provider, "gemini-3.8-flash")
    assert signature not in "".join(traceback.format_exception(excinfo.value))
    assert dropped[0]["parts"] == [{"text": "Describe the clip"}]


_SYSTEM_MESSAGE_WITH_VIDEO: Final[openai.ChatCompletionSystemMessage] = {
    "role": "system",
    "content": [
        {"type": "text", "text": "Be brief"},
        {"type": "video_url", "video_url": {"url": "https://videos.example/clip.mp4"}},
    ],
}


@pytest.mark.parametrize(
    "messages",
    [
        [_SYSTEM_MESSAGE_WITH_VIDEO, {"role": "user", "content": "Describe the clip"}],
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            _SYSTEM_MESSAGE_WITH_VIDEO,
            {"role": "user", "content": "Describe the clip"},
        ],
    ],
    ids=["leading", "mid_conversation"],
)
def test_a_linked_video_in_a_system_message_is_a_bad_request_naming_the_model_and_provider(
    messages: list[openai.AllMessageValues],
) -> None:
    with pytest.raises(litellm.BadRequestError, match="system instructions take text only") as excinfo:
        transformation.transform_request_body(
            messages=messages,
            model="gemini-3.8-flash",
            optional_params={},
            custom_llm_provider="vertex_ai",
            litellm_params={},
            cached_content=None,
        )

    assert (excinfo.value.llm_provider, excinfo.value.model) == ("vertex_ai", "gemini-3.8-flash")


@pytest.mark.parametrize("custom_llm_provider", ["gemini", "vertex_ai"])
async def test_async_transform_lets_gemini_fetch_youtube_links_itself(
    async_only_image_fetch: AsyncOnlyImageFetch, custom_llm_provider: str
) -> None:
    watch_link: Final = "https://www.youtube.com/watch?v=aqz-KE-bpKQ"
    short_link: Final = "https://youtu.be/aqz-KE-bpKQ"

    body: Final = await transformation.async_transform_request_body(
        gemini_api_key=None,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe these"},
                    {"type": "video_url", "video_url": {"url": watch_link}},
                    {"type": "file", "file": {"file_id": short_link}},
                ],
            }
        ],
        api_base=None,
        model="gemini-3.8-flash",
        client=None,
        timeout=None,
        extra_headers=None,
        optional_params={},
        logging_obj=Mock(),
        custom_llm_provider=custom_llm_provider,
        litellm_params={},
        vertex_project="qa-project",
        vertex_location="us-central1",
        vertex_auth_header=None,
    )

    assert body["contents"][0]["parts"] == [
        {"text": "Describe these"},
        {"file_data": {"mime_type": "video/mp4", "file_uri": watch_link}},
        {"file_data": {"mime_type": "video/mp4", "file_uri": short_link}},
    ]
    assert async_only_image_fetch.fetched == []


def test_gemini_ai_studio_sync_completion_lets_gemini_fetch_youtube_links_itself(
    async_only_image_fetch: AsyncOnlyImageFetch,
) -> None:
    watch_link: Final = "https://www.youtube.com/watch?v=aqz-KE-bpKQ"
    short_link: Final = "https://youtu.be/aqz-KE-bpKQ"
    response: Final = litellm.completion(
        model="gemini/gemini-3.8-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe these"},
                    {"type": "video_url", "video_url": {"url": watch_link}},
                    {"type": "file", "file": {"file_id": short_link}},
                ],
            }
        ],
        api_key="fake-gemini-key",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_echo_request_parts))),
    )

    assert json.loads(response.choices[0].message.content) == [
        {"text": "Describe these"},
        {"file_data": {"mime_type": "video/mp4", "file_uri": watch_link}},
        {"file_data": {"mime_type": "video/mp4", "file_uri": short_link}},
    ]


def test_gemini_completion_rejects_a_video_url_part_without_a_url_instead_of_dropping_it() -> None:
    with pytest.raises(litellm.BadRequestError, match="'video_url' but it has no 'url'"):
        litellm.completion(
            model="gemini/gemini-flash-latest",
            api_key="fake-gemini-key",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Describe the clip"}, {"type": "video_url", "video_url": {}}],
                }
            ],
        )


def test_drop_params_drops_a_video_url_part_without_a_url() -> None:
    contents: Final = transformation.gemini_convert_messages_with_history(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Describe the clip"}, {"type": "video_url", "video_url": {}}],
            }
        ],
        model="gemini-3.8-flash",
        litellm_params={"drop_params": True},
    )

    assert contents[0]["parts"] == [{"text": "Describe the clip"}]


def _find_clip_turns(video_url: Mapping[str, object]) -> list[Mapping[str, object]]:
    return [
        {"role": "user", "content": "Find the clip"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "find_clip", "arguments": "{}"}}],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"type": "text", "text": '{"ok":true}'}, {"type": "video_url", "video_url": video_url}],
        },
    ]


def _turns_with_a_url_less_video(role: str) -> list[Mapping[str, object]]:
    url_less_video: Final = {"type": "video_url", "video_url": {}}
    if role == "system":
        return [
            {"role": "system", "content": [{"type": "text", "text": "Be brief"}, url_less_video]},
            {"role": "user", "content": "Find the clip"},
        ]
    if role == "assistant":
        return [
            {"role": "user", "content": "Find the clip"},
            {"role": "assistant", "content": [{"type": "text", "text": "Here it is"}, url_less_video]},
        ]
    return _find_clip_turns({})


@pytest.mark.parametrize("role", ["system", "assistant", "tool"])
def test_a_url_less_video_url_part_in_any_role_is_a_bad_request_instead_of_a_silent_drop(role: str) -> None:
    with pytest.raises(litellm.BadRequestError):
        litellm.completion(
            model="gemini/gemini-flash-latest", api_key="fake-gemini-key", messages=_turns_with_a_url_less_video(role)
        )


@pytest.mark.parametrize(
    "role, expected_last_turn, expected_system",
    [
        pytest.param(
            "system",
            {"role": "user", "parts": [{"text": "Find the clip"}]},
            {"parts": [{"text": "Be brief"}]},
            id="system",
        ),
        pytest.param("assistant", {"role": "model", "parts": [{"text": "Here it is"}]}, None, id="assistant"),
        pytest.param(
            "tool",
            {
                "role": "user",
                "parts": [{"function_response": {"name": "find_clip", "response": {"ok": True}, "id": "call_1"}}],
            },
            None,
            id="tool",
        ),
    ],
)
def test_drop_params_drops_a_url_less_video_url_part_in_any_role(
    role: str, expected_last_turn: Mapping[str, object], expected_system: Mapping[str, object] | None
) -> None:
    body: Final = transformation.transform_request_body(
        messages=_turns_with_a_url_less_video(role),
        model="gemini-3.8-flash",
        optional_params={},
        custom_llm_provider="gemini",
        litellm_params={"drop_params": True},
        cached_content=None,
    )

    assert body["contents"][-1] == expected_last_turn
    assert body.get("system_instruction") == expected_system


def test_an_assistant_video_url_part_reaches_gemini_as_video() -> None:
    contents: Final = transformation.gemini_convert_messages_with_history(
        [
            {"role": "user", "content": "Show me the clip"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Here it is"},
                    {"type": "refusal", "refusal": "I won't narrate it"},
                    {"type": "video_url", "video_url": {"url": "gs://qa-bucket/clip.mp4"}},
                ],
            },
        ],
        model="gemini-3.8-flash",
        custom_llm_provider="vertex_ai",
    )

    assert contents[-1]["parts"] == [
        {"text": "Here it is"},
        {"file_data": {"mime_type": "video/mp4", "file_uri": "gs://qa-bucket/clip.mp4"}},
    ]


def _user_video_turns(video_url: Mapping[str, object]) -> list[Mapping[str, object]]:
    return [{"role": "user", "content": [{"type": "video_url", "video_url": video_url}]}]


def _user_video_parts(body: RequestBody) -> object:
    return [part for part in body["contents"][-1]["parts"] if "inline_data" in part]


def _tool_result_video_parts(body: RequestBody) -> object:
    return body["contents"][-1]["parts"][0]["function_response"]["parts"]


@pytest.mark.parametrize(
    "video_url, turns, downloaded_parts",
    [
        (f"https://8.8.8.8/{uuid.uuid4()}.mp4", _find_clip_turns, _tool_result_video_parts),
        (f"http://8.8.8.8/{uuid.uuid4()}.mp4", _user_video_turns, _user_video_parts),
        (f"https://8.8.8.8/{uuid.uuid4()}", _user_video_turns, _user_video_parts),
    ],
    ids=["tool-result", "user-http", "user-no-extension"],
)
async def test_async_transform_downloads_a_video_off_the_event_loop(
    video_url: str,
    turns: Callable[[Mapping[str, object]], list[Mapping[str, object]]],
    downloaded_parts: Callable[[RequestBody], object],
) -> None:
    loop: Final = asyncio.get_running_loop()
    download_started: Final = asyncio.Event()
    released_by_the_loop: Final = threading.Event()

    def blocking_download(request: httpx.Request) -> httpx.Response:
        loop.call_soon_threadsafe(download_started.set)
        if not released_by_the_loop.wait(timeout=5):
            return httpx.Response(504)
        return httpx.Response(200, content=b"\x00\x00\x00\x18ftypmp42", headers={"content-type": "video/mp4"})

    async def release_the_download() -> None:
        await download_started.wait()
        released_by_the_loop.set()

    releaser: Final = asyncio.create_task(release_the_download())
    with respx.mock:
        respx.get(video_url).mock(side_effect=blocking_download)
        body: Final = await transformation.async_transform_request_body(
            gemini_api_key=None,
            messages=turns({"url": video_url}),
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
    await asyncio.wait_for(releaser, timeout=5)

    assert downloaded_parts(body) == [{"inline_data": {"data": "AAAAGGZ0eXBtcDQy", "mime_type": "video/mp4"}}]


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


def _gemini_request_body(messages: list[openai.AllMessageValues]) -> RequestBody:
    return transformation.transform_request_body(
        messages=messages,
        model="gemini-3.8-flash",
        optional_params={},
        custom_llm_provider="vertex_ai",
        litellm_params={},
        cached_content=None,
    )


def _exchange_with_a_reminder(turn: int) -> tuple[openai.AllMessageValues, ...]:
    return (
        {"role": "user", "content": f"question {turn}"},
        {"role": "system", "content": f"{turn} of 8 turns used"},
        {"role": "assistant", "content": f"answer {turn}"},
    )


def _conversation_with_a_reminder_after_each_user_turn(user_turns: int) -> list[openai.AllMessageValues]:
    exchanges: Final = tuple(chain.from_iterable(map(_exchange_with_a_reminder, range(1, user_turns + 1))))
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


def test_a_leading_system_message_with_no_content_sends_only_the_rest_of_the_conversation():
    body: Final = _gemini_request_body([{"role": "system"}, {"role": "user", "content": "question 1"}])

    assert "system_instruction" not in body
    assert body["contents"] == [{"role": "user", "parts": [{"text": "question 1"}]}]


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
    def call(call_id: str, name: str) -> openai.ChatCompletionAssistantToolCall:
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
