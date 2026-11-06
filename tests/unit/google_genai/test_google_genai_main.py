#!/usr/bin/env python3
"""
Test to verify the Google GenAI generate_content adapter functionality
"""

import json
from collections.abc import Iterator, Mapping
from typing import Final

import pytest




import litellm


@pytest.mark.asyncio
async def test_agenerate_content_stream():
    """
    Test that the agenerate_content_stream function works
    """
    from unittest.mock import AsyncMock, patch

    from litellm.google_genai.main import (
        agenerate_content_stream,
        base_llm_http_handler,
    )

    with patch.object(
        base_llm_http_handler, "generate_content_handler", new=AsyncMock()
    ) as mock_post:
        result = await agenerate_content_stream(
            model="gemini/gemini-2.0-flash-001",
            contents="Hello, world!",
            stream=True,
        )
        mock_post.assert_called_once()
        assert mock_post.call_args.kwargs["stream"] is True


def _mock_gemini_post_response():
    """A minimal stand-in for a successful Gemini generateContent HTTP response."""
    from unittest.mock import MagicMock

    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {}
    resp.json.return_value = {
        "candidates": [
            {
                "content": {"parts": [{"text": "hi"}], "role": "model"},
                "finishReason": "STOP",
            }
        ]
    }
    return resp


NATIVE_TOP_LEVEL_FIELD_CASES = [
    (
        "safetySettings",
        [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}],
    ),
    ("toolConfig", {"functionCallingConfig": {"mode": "AUTO"}}),
    ("cachedContent", "cachedContents/abc123"),
    ("labels", {"team": "search"}),
    ("continuationToken", "dG9rLTE="),
]


@pytest.mark.parametrize("field_name, field_value", NATIVE_TOP_LEVEL_FIELD_CASES)
def test_native_top_level_field_forwarded_to_request_body(field_name, field_value):
    """
    Regression for https://github.com/BerriAI/litellm/issues/12671

    Google's native generateContent body carries fields like safetySettings at the
    top level (siblings of generationConfig). The proxy spreads them as loose kwargs
    into generate_content. They must reach Google's request body at the top level and
    must NOT be silently dropped nor nested under generationConfig.
    """
    from unittest.mock import patch

    from litellm.google_genai.main import generate_content
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    with patch.object(
        HTTPHandler, "post", return_value=_mock_gemini_post_response()
    ) as mock_post:
        generate_content(
            model="gemini/gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
            custom_llm_provider="gemini",
            api_key="test-key",
            **{field_name: field_value},
        )

    assert mock_post.called, "expected the request to reach the HTTP client"
    body = mock_post.call_args.kwargs["json"]
    assert (
        body[field_name] == field_value
    ), f"{field_name} should be forwarded to Google at the top level"
    assert field_name not in body.get("generationConfig", {}), (
        f"{field_name} must be a top-level sibling of generationConfig, "
        "not nested inside it"
    )


@pytest.mark.asyncio
async def test_native_safety_settings_forwarded_async():
    """The async path (used by the proxy's :generateContent route) must also forward
    native top-level fields."""
    from unittest.mock import AsyncMock, patch

    from litellm.google_genai.main import agenerate_content
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}
    ]

    with patch.object(
        AsyncHTTPHandler,
        "post",
        new_callable=AsyncMock,
        return_value=_mock_gemini_post_response(),
    ) as mock_post:
        await agenerate_content(
            model="gemini/gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
            custom_llm_provider="gemini",
            api_key="test-key",
            safetySettings=safety_settings,
        )

    assert mock_post.called
    body = mock_post.call_args.kwargs["json"]
    assert body["safetySettings"] == safety_settings
    assert "safetySettings" not in body.get("generationConfig", {})


def test_native_fields_coexist_with_generation_config():
    """Forwarding native top-level fields must not regress the already-working
    generationConfig path; both must land in their correct positions."""
    from unittest.mock import patch

    from litellm.google_genai.main import generate_content
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}
    ]

    with patch.object(
        HTTPHandler, "post", return_value=_mock_gemini_post_response()
    ) as mock_post:
        generate_content(
            model="gemini/gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
            custom_llm_provider="gemini",
            api_key="test-key",
            safetySettings=safety_settings,
            generationConfig={"temperature": 0, "responseMimeType": "application/json"},
        )

    body = mock_post.call_args.kwargs["json"]
    assert body["safetySettings"] == safety_settings
    generation_config = body["generationConfig"]
    assert generation_config["temperature"] == 0
    assert generation_config["responseMimeType"] == "application/json"


def test_explicit_extra_body_overrides_native_top_level_field():
    """An explicit extra_body value takes precedence over the same top-level field."""
    from unittest.mock import patch

    from litellm.google_genai.main import generate_content
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    native = [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}]
    override = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"}
    ]

    with patch.object(
        HTTPHandler, "post", return_value=_mock_gemini_post_response()
    ) as mock_post:
        generate_content(
            model="gemini/gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
            custom_llm_provider="gemini",
            api_key="test-key",
            safetySettings=native,
            extra_body={"safetySettings": override},
        )

    body = mock_post.call_args.kwargs["json"]
    assert body["safetySettings"] == override


def test_native_fields_and_system_instruction_forwarded_on_sync_stream():
    """The sync streaming path (generate_content_stream) must forward native top-level
    fields AND systemInstruction. The PR changed the merge here and newly added the
    systemInstruction kwarg; without coverage a regression on either ships green."""
    from unittest.mock import patch

    from litellm.google_genai.main import generate_content_stream
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}
    ]
    system_instruction = {"parts": [{"text": "Be terse"}]}

    with patch.object(
        HTTPHandler, "post", return_value=_mock_gemini_post_response()
    ) as mock_post:
        generate_content_stream(
            model="gemini/gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
            custom_llm_provider="gemini",
            api_key="test-key",
            safetySettings=safety_settings,
            systemInstruction=system_instruction,
        )

    assert mock_post.called
    body = mock_post.call_args.kwargs["json"]
    assert body["safetySettings"] == safety_settings
    assert body["systemInstruction"] == system_instruction
    assert "safetySettings" not in body.get("generationConfig", {})


@pytest.mark.asyncio
async def test_native_fields_forwarded_on_async_stream():
    """The async streaming path (agenerate_content_stream) backs the proxy's
    :streamGenerateContent route and must forward native top-level fields too."""
    from unittest.mock import AsyncMock, patch

    from litellm.google_genai.main import agenerate_content_stream
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}
    ]

    with patch.object(
        AsyncHTTPHandler,
        "post",
        new_callable=AsyncMock,
        return_value=_mock_gemini_post_response(),
    ) as mock_post:
        await agenerate_content_stream(
            model="gemini/gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
            custom_llm_provider="gemini",
            api_key="test-key",
            safetySettings=safety_settings,
        )

    assert mock_post.called
    body = mock_post.call_args.kwargs["json"]
    assert body["safetySettings"] == safety_settings
    assert "safetySettings" not in body.get("generationConfig", {})


@pytest.mark.parametrize("base_model, uses_native_api", [("gemini-2.5-flash", True), (None, False)])
def test_vertex_generate_content_routes_by_deployment_base_model(base_model, uses_native_api):
    import datetime

    from litellm.google_genai.main import GenerateContentHelper
    from litellm.litellm_core_utils.litellm_logging import Logging

    logging_obj = Logging(
        model="vertex_ai/openai/tuned-endpoint",
        messages=[],
        stream=False,
        call_type="agenerate_content",
        start_time=datetime.datetime.now(),
        litellm_call_id="route-test",
        function_id="route-test",
    )
    setup = GenerateContentHelper.setup_generate_content_call(
        model="vertex_ai/openai/tuned-endpoint",
        contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
        litellm_logging_obj=logging_obj,
        **({} if base_model is None else {"base_model": base_model}),
    )

    assert (setup.generate_content_provider_config is not None) is uses_native_api


_UPLIFT_PROBE: Final = "gemini-regional-uplift-probe"
_UPLIFT_INPUT_RATE: Final = 1e-06
_UPLIFT_OUTPUT_RATE: Final = 2e-06
_UPLIFT_MULTIPLIER: Final = 1.1


@pytest.fixture
def vertex_regional_uplift_rate_card(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    rates: Final = {
        "input_cost_per_token": _UPLIFT_INPUT_RATE,
        "output_cost_per_token": _UPLIFT_OUTPUT_RATE,
        "mode": "chat",
    }
    monkeypatch.setitem(litellm.model_cost, _UPLIFT_PROBE, {**rates, "litellm_provider": "vertex_ai-language-models"})
    monkeypatch.setattr(litellm, "vertex_language_models", {*litellm.vertex_language_models, _UPLIFT_PROBE})
    monkeypatch.setitem(
        litellm.model_cost,
        f"vertex_ai/{_UPLIFT_PROBE}",
        {
            **rates,
            "litellm_provider": "vertex_ai-language-models",
            "regional_endpoint_uplift_multiplier": _UPLIFT_MULTIPLIER,
        },
    )
    monkeypatch.delenv("VERTEXAI_LOCATION", raising=False)
    monkeypatch.delenv("VERTEX_LOCATION", raising=False)
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


async def _logged_vertex_generate_content_cost(location: str, *, stream: bool) -> float:
    import asyncio
    import datetime

    from litellm.google_genai.main import GenerateContentHelper
    from litellm.google_genai.streaming_iterator import BaseGoogleGenAIGenerateContentStreamingIterator
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.google_genai.main import GenerateContentResponse

    reply: Final = {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 1, "totalTokenCount": 8},
        "modelVersion": _UPLIFT_PROBE,
    }
    logging_obj: Final = Logging(
        model=f"vertex_ai/{_UPLIFT_PROBE}",
        messages=[],
        stream=stream,
        call_type="agenerate_content_stream" if stream else "agenerate_content",
        start_time=datetime.datetime(2026, 1, 1),
        litellm_call_id="uplift-test",
        function_id="uplift-test",
    )
    setup: Final = GenerateContentHelper.setup_generate_content_call(
        model=f"vertex_ai/{_UPLIFT_PROBE}",
        contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        litellm_logging_obj=logging_obj,
        vertex_location=location,
        stream=stream,
    )
    if stream:
        iterator: Final = BaseGoogleGenAIGenerateContentStreamingIterator(
            litellm_logging_obj=logging_obj,
            request_body={},
            model=setup.model,
            custom_llm_provider=setup.custom_llm_provider,
        )
        iterator.collected_chunks.append(f"data: {json.dumps(reply)}\n\n".encode())
        await iterator._handle_async_streaming_logging()
        await asyncio.gather(*(task for task in asyncio.all_tasks() if task is not asyncio.current_task()))
    else:
        logging_obj._process_hidden_params_and_response_cost(
            GenerateContentResponse(**reply),
            datetime.datetime(2026, 1, 1),
            datetime.datetime(2026, 1, 1),
            build_logging_payload=False,
        )
    return logging_obj.model_call_details["response_cost"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "non-streamed"])
@pytest.mark.parametrize("location, multiplier", [("global", 1.0), ("us", _UPLIFT_MULTIPLIER)])
async def test_vertex_generate_content_spend_follows_the_deployment_location(
    vertex_regional_uplift_rate_card: None, stream: bool, location: str, multiplier: float
) -> None:
    expected: Final = (7 * _UPLIFT_INPUT_RATE + 1 * _UPLIFT_OUTPUT_RATE) * multiplier

    recorded: Final = await _logged_vertex_generate_content_cost(location, stream=stream)

    assert recorded == pytest.approx(expected), (location, stream)


class _WireCapture:
    def __init__(self, url: str) -> None:
        self.url: Final = url
        self._items: Final[list[tuple[str, Mapping[str, str]]]] = []  # mutable-ok: test wire log

    def record(self, path: str, headers: Mapping[str, str]) -> None:
        self._items.append((path, headers))

    @property
    def requests(self) -> tuple[tuple[str, Mapping[str, str]], ...]:
        return tuple(self._items)


def _mock_wire_reply(path: str, stream_requested: bool) -> tuple[bytes, str]:
    if "streamGenerateContent" in path:
        return (
            b'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"pong"}]},'
            b'"finishReason":"STOP"}],'
            b'"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":1,"totalTokenCount":5}}\n\n',
            "text/event-stream",
        )
    if ":generateContent" in path:
        return (
            b'{"candidates":[{"content":{"role":"model","parts":[{"text":"pong"}]},'
            b'"finishReason":"STOP"}],'
            b'"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":1,"totalTokenCount":5}}',
            "application/json",
        )
    if stream_requested:
        return (
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
            b'"model":"gpt-4o-mini","choices":[{"index":0,'
            b'"delta":{"role":"assistant","content":"pong"},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n",
            "text/event-stream",
        )
    return (
        b'{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-4o-mini",'
        b'"choices":[{"index":0,"message":{"role":"assistant","content":"pong"},'
        b'"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}',
        "application/json",
    )


@pytest.fixture
def header_capture_wire() -> Iterator[_WireCapture]:
    """
    Loopback upstream that records outbound wire headers.

    A real socket (allowed by ``tests/unit/conftest.py`` for loopback hosts) is used instead of
    ``respx`` because async calls go through litellm's aiohttp transport, which ``respx`` cannot intercept.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    holder: Final[list[_WireCapture]] = []  # mutable-ok: single-slot fixture bootstrap reference

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length: Final = int(self.headers.get("Content-Length", "0"))
            raw_body: Final = self.rfile.read(length)
            holder[0].record(self.path, {k.lower(): v for k, v in self.headers.items()})
            payload: Final = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            body, content_type = _mock_wire_reply(self.path, bool(payload.get("stream")))
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    capture: Final = _WireCapture(f"http://127.0.0.1:{server.server_address[1]}")
    holder.append(capture)
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield capture
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _extract_chunk_texts(payload: object) -> tuple[str, ...]:
    from pydantic import TypeAdapter

    obj_map: Final = TypeAdapter[Mapping[str, object]](Mapping[str, object])
    map_seq: Final = TypeAdapter[tuple[Mapping[str, object], ...]](tuple[Mapping[str, object], ...])
    parsed: Final = obj_map.validate_python(payload.model_dump() if hasattr(payload, "model_dump") else payload)
    candidates: Final = map_seq.validate_python(parsed.get("candidates", ()))
    if not candidates:
        return ()
    content: Final = obj_map.validate_python(candidates[0].get("content", {}))
    parts: Final = map_seq.validate_python(content.get("parts", ()))
    return tuple(str(part["text"]) for part in parts if isinstance(part.get("text"), str) and part["text"])


def _sse_payload_texts(line: str) -> tuple[str, ...]:
    if not line.startswith("data: "):
        return ()
    payload: Final = line[6:].strip()
    if not payload or payload == "[DONE]":
        return ()
    return _extract_chunk_texts(json.loads(payload))


def _extract_sse_texts(chunks: tuple[bytes, ...]) -> tuple[str, ...]:
    from itertools import chain

    lines: Final = chain.from_iterable(raw.decode("utf-8").splitlines() for raw in chunks)
    return tuple(chain.from_iterable(_sse_payload_texts(line) for line in lines))


_UPSTREAM_KEY: Final = "upstream-secret"


def _target_route(target: str, wire: _WireCapture) -> tuple[str, str]:
    if target == "native":
        return "gemini/gemini-2.0-flash", wire.url
    return "openai/gpt-4o-mini", f"{wire.url}/v1"


async def _invoke_generate_content(
    entrypoint: str,
    *,
    model: str,
    api_base: str,
    headers: Mapping[str, str],
    extra_headers: Mapping[str, str],
) -> tuple[str, ...]:
    from collections.abc import AsyncIterator

    from pydantic import TypeAdapter

    from litellm.google_genai.main import (
        agenerate_content,
        agenerate_content_stream,
        generate_content,
        generate_content_stream,
    )

    contents: Final = [{"role": "user", "parts": [{"text": "ping"}]}]
    if entrypoint == "sync":
        return _extract_chunk_texts(
            generate_content(
                model=model,
                contents=contents,
                api_base=api_base,
                api_key=_UPSTREAM_KEY,
                headers=headers,
                extra_headers=dict(extra_headers),
            )
        )
    if entrypoint == "async":
        return _extract_chunk_texts(
            await agenerate_content(
                model=model,
                contents=contents,
                api_base=api_base,
                api_key=_UPSTREAM_KEY,
                headers=headers,
                extra_headers=dict(extra_headers),
            )
        )
    stream: Final = (
        generate_content_stream(
            model=model,
            contents=contents,
            api_base=api_base,
            api_key=_UPSTREAM_KEY,
            headers=headers,
            extra_headers=dict(extra_headers),
        )
        if entrypoint == "sync_stream"
        else await agenerate_content_stream(
            model=model,
            contents=contents,
            api_base=api_base,
            api_key=_UPSTREAM_KEY,
            headers=headers,
            extra_headers=dict(extra_headers),
        )
    )
    as_bytes: Final = TypeAdapter(bytes)
    # native sync streams implement both protocols; adapter sync streams are async-only
    if entrypoint == "sync_stream" and isinstance(stream, Iterator):
        return _extract_sse_texts(tuple(as_bytes.validate_python(chunk) for chunk in stream))
    assert isinstance(stream, AsyncIterator)
    return _extract_sse_texts(tuple([as_bytes.validate_python(chunk) async for chunk in stream]))


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["sync", "async", "sync_stream", "async_stream"])
@pytest.mark.parametrize("target", ["native", "adapter"])
async def test_generate_content_forwards_client_headers_behind_explicit_extra_headers(
    header_capture_wire: _WireCapture,
    target: str,
    entrypoint: str,
) -> None:
    model, api_base = _target_route(target, header_capture_wire)
    forwarded: Final = {"x-team-id": "platform", "x-custom": "from-proxy", "anthropic-beta": "context-1m-2025-08-07"}
    explicit: Final = {"X-Custom": "from-caller"}

    texts: Final = await _invoke_generate_content(
        entrypoint, model=model, api_base=api_base, headers=forwarded, extra_headers=explicit
    )

    assert texts == ("pong",)
    assert forwarded == {"x-team-id": "platform", "x-custom": "from-proxy", "anthropic-beta": "context-1m-2025-08-07"}
    assert explicit == {"X-Custom": "from-caller"}
    assert len(header_capture_wire.requests) == 1
    wire_headers: Final = header_capture_wire.requests[0][1]
    assert wire_headers["x-team-id"] == "platform"
    assert wire_headers["anthropic-beta"] == "context-1m-2025-08-07"
    assert wire_headers["x-custom"] == "from-caller"


@pytest.mark.parametrize(
    "extra_headers, client_headers, expected",
    [
        (None, None, None),
        (None, {}, None),
        ({}, None, {}),
        ({"x-b": "2"}, None, {"x-b": "2"}),
        (None, {"x-a": "1"}, {"x-a": "1"}),
        ({}, {"x-a": "1"}, {"x-a": "1"}),
        ({"X-Custom": "from-caller"}, {"x-custom": "from-proxy", "x-a": "1"}, {"x-a": "1", "X-Custom": "from-caller"}),
        ({"x-b": "2"}, "not-a-mapping", {"x-b": "2"}),
        (None, "not-a-mapping", None),
        ({"x-b": "2"}, {"x-a": 1}, {"x-b": "2"}),
    ],
    ids=[
        "nothing",
        "empty-forwarded",
        "empty-explicit",
        "explicit-only",
        "forwarded-only",
        "empty-explicit-with-forwarded",
        "case-insensitive-precedence",
        "non-mapping-forwarded-kept-explicit",
        "non-mapping-forwarded",
        "non-str-value-forwarded",
    ],
)
def test_merge_client_forwarded_headers(
    extra_headers: Mapping[str, str] | None,
    client_headers: object,
    expected: Mapping[str, str] | None,
) -> None:
    from litellm.google_genai.main import _merge_client_forwarded_headers

    merged: Final = _merge_client_forwarded_headers(extra_headers, client_headers)

    assert merged == expected
    assert merged is None or (merged is not extra_headers and merged is not client_headers)


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["native", "adapter"])
async def test_router_deployment_headers_reach_generate_content(
    header_capture_wire: _WireCapture,
    target: str,
) -> None:
    from litellm import Router

    model, api_base = _target_route(target, header_capture_wire)
    router: Final = Router(
        model_list=[
            {
                "model_name": "genai-deployment",
                "litellm_params": {
                    "model": model,
                    "api_base": api_base,
                    "api_key": _UPSTREAM_KEY,
                    "headers": {"x-deployment-tag": "blue"},
                },
            }
        ]
    )

    resp: Final = await router.agenerate_content(
        model="genai-deployment",
        contents=[{"role": "user", "parts": [{"text": "ping"}]}],
    )

    assert _extract_chunk_texts(resp) == ("pong",)
    assert len(header_capture_wire.requests) == 1
    assert header_capture_wire.requests[0][1]["x-deployment-tag"] == "blue"


@pytest.mark.asyncio
async def test_proxy_forwarded_headers_reach_generate_content_without_leaking_proxy_auth(
    header_capture_wire: _WireCapture,
) -> None:
    from starlette.datastructures import Headers

    from litellm.google_genai.main import agenerate_content
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.litellm_pre_call_utils import (
        LiteLLMProxyRequestSetup,
        clean_headers,
        litellm_auth_header_name,
    )

    incoming_headers: Final = Headers(
        {
            "x-goog-api-key": "sk-litellm-virtual-key",
            "authorization": "Bearer sk-litellm-virtual-key",
            "x-cost-center": "core-platform",
            "user-agent": "google-genai-sdk/1.0",
        }
    )
    cleaned: Final = clean_headers(
        incoming_headers,
        litellm_key_header_name=None,
        forward_llm_provider_auth_headers=False,
        authenticated_with_header=litellm_auth_header_name(incoming_headers),
    )
    backend_data: Final = LiteLLMProxyRequestSetup.add_litellm_data_for_backend_llm_call(
        headers=cleaned,
        request_data={"model": "gemini/gemini-2.0-flash"},
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-litellm-virtual-key"),
        general_settings={"forward_client_headers_to_llm_api": True},
    )

    resp: Final = await agenerate_content(
        model="gemini/gemini-2.0-flash",
        contents=[{"role": "user", "parts": [{"text": "ping"}]}],
        api_base=header_capture_wire.url,
        api_key=_UPSTREAM_KEY,
        headers=backend_data.get("headers"),
    )

    assert _extract_chunk_texts(resp) == ("pong",)
    assert len(header_capture_wire.requests) == 1
    wire_headers: Final = header_capture_wire.requests[0][1]
    assert wire_headers["x-goog-api-key"] == _UPSTREAM_KEY
    assert wire_headers["x-cost-center"] == "core-platform"
    assert "authorization" not in wire_headers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("incoming", "expected_upstream_key"),
    [
        ({"x-goog-api-key": "sk-litellm-virtual-key", "x-cost-center": "core-platform"}, _UPSTREAM_KEY),
        (
            {
                "authorization": "Bearer sk-litellm-virtual-key",
                "x-goog-api-key": "sk-litellm-virtual-key",
                "x-cost-center": "core-platform",
            },
            _UPSTREAM_KEY,
        ),
        (
            {
                "x-api-key": "",
                "x-goog-api-key": "sk-litellm-virtual-key",
                "x-cost-center": "core-platform",
            },
            _UPSTREAM_KEY,
        ),
        (
            {
                "authorization": "Bearer sk-litellm-virtual-key",
                "x-goog-api-key": "",
                "x-cost-center": "core-platform",
            },
            _UPSTREAM_KEY,
        ),
        (
            {
                "x-litellm-api-key": "sk-litellm-virtual-key",
                "x-goog-api-key": "AIza-caller-key",
                "x-cost-center": "core-platform",
            },
            "AIza-caller-key",
        ),
        (
            {
                "authorization": "Bearer sk-litellm-virtual-key",
                "x-goog-api-key": "AIza-caller-key",
                "x-cost-center": "core-platform",
            },
            "AIza-caller-key",
        ),
    ],
    ids=[
        "proxy-key-in-goog-header",
        "sdk-dual-bearer-and-goog-proxy-key",
        "empty-higher-precedence-header",
        "empty-goog-header-beside-bearer-key",
        "caller-key-beside-litellm-key",
        "caller-key-beside-bearer-key",
    ],
)
async def test_provider_auth_forwarding_never_sends_proxy_key_to_generate_content(
    header_capture_wire: _WireCapture,
    incoming: Mapping[str, str],
    expected_upstream_key: str,
) -> None:
    from starlette.datastructures import Headers

    from litellm.google_genai.main import agenerate_content
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.litellm_pre_call_utils import (
        LiteLLMProxyRequestSetup,
        clean_headers,
        litellm_auth_header_name,
    )

    incoming_headers: Final = Headers(incoming)
    cleaned: Final = clean_headers(
        incoming_headers,
        forward_llm_provider_auth_headers=True,
        authenticated_with_header=litellm_auth_header_name(incoming_headers),
    )
    backend_data: Final = LiteLLMProxyRequestSetup.add_litellm_data_for_backend_llm_call(
        headers=cleaned,
        request_data={"model": "gemini/gemini-2.0-flash"},
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-litellm-virtual-key"),
        general_settings={"forward_client_headers_to_llm_api": True},
    )

    resp: Final = await agenerate_content(
        model="gemini/gemini-2.0-flash",
        contents=[{"role": "user", "parts": [{"text": "ping"}]}],
        api_base=header_capture_wire.url,
        api_key=_UPSTREAM_KEY,
        headers=backend_data.get("headers"),
    )

    assert _extract_chunk_texts(resp) == ("pong",)
    assert len(header_capture_wire.requests) == 1
    wire_headers: Final = header_capture_wire.requests[0][1]
    assert wire_headers["x-goog-api-key"] == expected_upstream_key
    assert wire_headers["x-cost-center"] == "core-platform"
    assert "x-litellm-api-key" not in wire_headers
