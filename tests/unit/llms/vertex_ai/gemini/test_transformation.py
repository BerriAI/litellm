
import json
import logging
import uuid
from collections.abc import Mapping
from typing import Final
from unittest.mock import MagicMock, Mock

import httpx
import pytest

import litellm
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.context_caching.known_cached_contents import known_cached_contents
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


def test_image_only_user_content_logs_the_blank_text_fixup_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    messages: Final = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]}
    ]

    with caplog.at_level(logging.DEBUG, logger=verbose_logger.name):
        contents: Final = transformation.gemini_convert_messages_with_history(messages)

    fixups: Final = tuple(r for r in caplog.records if "No text in user content" in r.getMessage())
    assert tuple(r.levelno for r in fixups) == (logging.DEBUG,)
    assert contents[0]["parts"][-1] == {"text": " "}


_BREAKPOINT_MODEL: Final = "gemini/gemini-2.5-flash"
_BREAKPOINT_CACHE_NAME: Final = "cachedContents/prompt-cache-breakpoint"
_REFERENCE_DOCUMENT: Final = "Section watch: the watchlist feature tracks companies. " * 600
_CREATED_CACHE: Final = {
    "name": _BREAKPOINT_CACHE_NAME,
    "model": "models/gemini-2.5-flash",
    "expireTime": "2099-01-01T00:00:00Z",
}
_GENERATED: Final = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "Section watch"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 4000, "candidatesTokenCount": 2, "totalTokenCount": 4002},
}
_MALFORMED_BREAKPOINTS: Final = (
    {"mode": "implicit"},
    {"mode": "explicit", "ttl": "1h"},
    {"mode": "explicit", "tll": "12h"},
    {"ttl": "30m"},
    {},
    "explicit",
)


@pytest.fixture
def forget_known_cached_contents():
    known_cached_contents.clear()
    yield
    known_cached_contents.clear()


def _url(args: tuple[object, ...], kwargs: Mapping[str, object]) -> str:
    return str(kwargs.get("url", args[0] if args else ""))


def _listed_caches(*args: object, **kwargs: object) -> httpx.Response:
    return httpx.Response(200, json={"cachedContents": []}, request=httpx.Request("GET", _url(args, kwargs)))


def _gemini_answer(*args: object, **kwargs: object) -> httpx.Response:
    url: Final = _url(args, kwargs)
    payload: Final = _CREATED_CACHE if "cachedContents" in url else _GENERATED
    return httpx.Response(200, json=payload, request=httpx.Request("POST", url))


def _fake_gemini(spec: type[HTTPHandler] | type[AsyncHTTPHandler]) -> MagicMock:
    client: Final = MagicMock(spec=spec)
    client.get.side_effect = _listed_caches
    client.post.side_effect = _gemini_answer
    return client


def _reference_messages(**marks: object) -> list[dict[str, object]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": _REFERENCE_DOCUMENT, **marks}]},
        {"role": "user", "content": "Which section mentions watchlists? One line."},
    ]


async def _complete_with(
    client: MagicMock, sync_mode: bool, messages: list[dict[str, object]], **params: object
) -> None:
    request: Final = {"model": _BREAKPOINT_MODEL, "messages": messages, "api_key": "fake-key", "client": client}
    if sync_mode:
        litellm.completion(**request, **params)
    else:
        await litellm.acompletion(**request, **params)


async def _sent_bodies(sync_mode: bool, messages: list[dict[str, object]], **params: object) -> dict[str, dict]:
    client: Final = _fake_gemini(HTTPHandler if sync_mode else AsyncHTTPHandler)
    await _complete_with(client, sync_mode, messages, **params)
    return {
        "cache" if "cachedContents" in _url(call.args, call.kwargs) else "generate": (
            call.kwargs["json"] if "json" in call.kwargs else json.loads(call.kwargs["data"])
        )
        for call in client.post.call_args_list
    }


@pytest.mark.usefixtures("forget_known_cached_contents")
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize(
    "breakpoint_ttl, gemini_ttl",
    [({}, "1800s"), ({"ttl": "30m"}, "1800s"), ({"ttl": "12h"}, "43200s")],
)
@pytest.mark.asyncio
async def test_a_prompt_cache_breakpoint_part_is_moved_into_a_gemini_cache_for_the_marker_ttl(
    sync_mode, breakpoint_ttl, gemini_ttl
):
    sent: Final = await _sent_bodies(
        sync_mode, _reference_messages(prompt_cache_breakpoint={"mode": "explicit", **breakpoint_ttl})
    )

    assert sent["cache"]["ttl"] == gemini_ttl
    assert _REFERENCE_DOCUMENT in json.dumps(sent["cache"])
    assert sent["generate"]["cachedContent"] == _BREAKPOINT_CACHE_NAME
    assert _REFERENCE_DOCUMENT not in json.dumps(sent["generate"])


@pytest.mark.usefixtures("forget_known_cached_contents")
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_a_prompt_cache_breakpoint_part_creates_the_same_cache_as_the_equivalent_cache_control(sync_mode):
    marked: Final = await _sent_bodies(
        sync_mode, _reference_messages(prompt_cache_breakpoint={"mode": "explicit", "ttl": "12h"})
    )
    known_cached_contents.clear()
    controlled: Final = await _sent_bodies(
        sync_mode, _reference_messages(cache_control={"type": "ephemeral", "ttl": "12h"})
    )

    assert marked == controlled


@pytest.mark.usefixtures("forget_known_cached_contents")
@pytest.mark.asyncio
async def test_cache_control_on_a_prompt_cache_breakpoint_part_keeps_its_own_ttl():
    sent: Final = await _sent_bodies(
        True,
        _reference_messages(
            cache_control={"type": "ephemeral", "ttl": "1h"},
            prompt_cache_breakpoint={"mode": "explicit", "ttl": "12h"},
        ),
    )

    assert sent["cache"]["ttl"] == "3600s"


@pytest.mark.usefixtures("forget_known_cached_contents")
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("marker", _MALFORMED_BREAKPOINTS)
@pytest.mark.asyncio
async def test_a_malformed_prompt_cache_breakpoint_is_a_400_before_anything_reaches_gemini(sync_mode, marker):
    client: Final = _fake_gemini(HTTPHandler if sync_mode else AsyncHTTPHandler)

    with pytest.raises(litellm.UnsupportedParamsError) as error:
        await _complete_with(client, sync_mode, _reference_messages(prompt_cache_breakpoint=marker))

    assert error.value.status_code == 400
    assert "prompt_cache_breakpoint" in str(error.value)
    client.get.assert_not_called()
    client.post.assert_not_called()


@pytest.mark.usefixtures("forget_known_cached_contents")
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("marker", _MALFORMED_BREAKPOINTS)
@pytest.mark.asyncio
async def test_drop_params_sends_a_malformed_prompt_cache_breakpoint_part_uncached(sync_mode, marker):
    sent: Final = await _sent_bodies(sync_mode, _reference_messages(prompt_cache_breakpoint=marker), drop_params=True)

    assert list(sent) == ["generate"]
    assert "cachedContent" not in sent["generate"]
    assert _REFERENCE_DOCUMENT in json.dumps(sent["generate"])


@pytest.mark.usefixtures("forget_known_cached_contents")
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_global_drop_params_sends_a_malformed_prompt_cache_breakpoint_part_uncached(sync_mode, monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", True)

    sent: Final = await _sent_bodies(sync_mode, _reference_messages(prompt_cache_breakpoint={"mode": "implicit"}))

    assert list(sent) == ["generate"]
    assert "cachedContent" not in sent["generate"]
    assert _REFERENCE_DOCUMENT in json.dumps(sent["generate"])
