#!/usr/bin/env python3
"""
Test to verify the Google GenAI proxy API endpoints
"""

import json
from collections.abc import Callable, Iterator, Mapping
from typing import Final
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from httpx_sse import EventSource
from jsonschema import validate
from pydantic import JsonValue, TypeAdapter

from litellm.types.utils import TokenCountResponse

_GOOGLE_PREFIXES: Final = ("/v1", "/v1beta", "")
_GOOGLE_V1_MODELS: Final = ("routed-google-model", "provider/routed-google-model")
_GOOGLE_V1_CONTENTS: Final = [{"role": "user", "parts": [{"text": "Hello"}]}]
_GOOGLE_V1_REPLY: Final = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "Hello from the provider"}]}, "index": 0}],
    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 5, "totalTokenCount": 8},
}
_JSON_OBJECT: Final = TypeAdapter(Mapping[str, JsonValue])


@pytest.fixture
def native_google_proxy(monkeypatch: pytest.MonkeyPatch, httpx_transport: None) -> Iterator[TestClient]:
    from litellm import Router
    from litellm.proxy import proxy_server

    router: Final = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gemini/gemini-3.8-flash",
                    "api_key": "test-key",
                    "api_base": "https://google-v1.test",
                },
            }
            for model in _GOOGLE_V1_MODELS
        ],
        num_retries=0,
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "master_key", None)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    client: Final = TestClient(proxy_server.app)
    try:
        yield client
    finally:
        client.close()
        router.retire()


@pytest.mark.parametrize("prefix", _GOOGLE_PREFIXES)
@pytest.mark.parametrize("model", _GOOGLE_V1_MODELS)
def test_google_generate_content_aliases_return_the_provider_content(
    native_google_proxy: TestClient, respx_mock: respx.MockRouter, prefix: str, model: str
) -> None:
    upstream: Final = respx_mock.post("https://google-v1.test/models/gemini-3.8-flash:generateContent").respond(
        200, json=_GOOGLE_V1_REPLY
    )

    response: Final = native_google_proxy.post(
        f"{prefix}/models/{model}:generateContent", json={"contents": _GOOGLE_V1_CONTENTS}
    )

    assert response.status_code == 200, response.text
    assert response.json() == _GOOGLE_V1_REPLY
    assert upstream.call_count == 1
    assert _JSON_OBJECT.validate_json(upstream.calls.last.request.content)["contents"] == _GOOGLE_V1_CONTENTS


@pytest.mark.parametrize("prefix", _GOOGLE_PREFIXES)
@pytest.mark.parametrize("model", _GOOGLE_V1_MODELS)
def test_google_stream_aliases_return_provider_events_without_an_openai_terminator(
    native_google_proxy: TestClient, respx_mock: respx.MockRouter, prefix: str, model: str
) -> None:
    frame: Final = f"data: {json.dumps(_GOOGLE_V1_REPLY)}\n\n"
    upstream: Final = respx_mock.post(
        url__regex=r"https://google-v1\.test/models/gemini-3\.8-flash:streamGenerateContent(?:\?.*)?$"
    ).respond(200, text=frame, headers={"content-type": "text/event-stream"})

    response: Final = native_google_proxy.post(
        f"{prefix}/models/{model}:streamGenerateContent?alt=sse", json={"contents": _GOOGLE_V1_CONTENTS}
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    events: Final = tuple(EventSource(response).iter_sse())
    assert all(event.data != "[DONE]" for event in events)
    assert tuple(_JSON_OBJECT.validate_json(event.data) for event in events) == (_GOOGLE_V1_REPLY,)
    assert upstream.call_count == 1
    assert _JSON_OBJECT.validate_json(upstream.calls.last.request.content)["contents"] == _GOOGLE_V1_CONTENTS


@pytest.mark.parametrize("prefix", _GOOGLE_PREFIXES)
@pytest.mark.parametrize("model", _GOOGLE_V1_MODELS)
def test_google_count_aliases_return_the_provider_token_count(
    native_google_proxy: TestClient, respx_mock: respx.MockRouter, prefix: str, model: str
) -> None:
    upstream: Final = respx_mock.post("https://google-v1.test/v1beta/models/gemini-3.8-flash:countTokens").respond(
        200, json={"totalTokens": 23}
    )

    response: Final = native_google_proxy.post(
        f"{prefix}/models/{model}:countTokens", json={"contents": _GOOGLE_V1_CONTENTS}
    )

    assert response.status_code == 200, response.text
    assert response.json()["totalTokens"] == 23
    assert response.json()["litellm_count_source"] == "provider"
    assert upstream.call_count == 1
    assert _JSON_OBJECT.validate_json(upstream.calls.last.request.content)["contents"] == _GOOGLE_V1_CONTENTS


def test_google_v1_aliases_preserve_openai_model_listing(native_google_proxy: TestClient) -> None:
    response: Final = native_google_proxy.get("/v1/models")

    assert response.status_code == 200, response.text
    assert response.json()["object"] == "list"
    assert "routed-google-model" in {model["id"] for model in response.json()["data"]}


def test_google_v1_aliases_preserve_openai_model_retrieval(native_google_proxy: TestClient) -> None:
    response: Final = native_google_proxy.get("/v1/models/routed-google-model")

    assert response.status_code == 200, response.text
    assert response.json()["id"] == "routed-google-model"
    assert response.json()["object"] == "model"


@pytest.mark.parametrize("prefix", _GOOGLE_PREFIXES)
@pytest.mark.parametrize("operation", ("generateContent", "streamGenerateContent", "countTokens"))
def test_google_openapi_accepts_requests_with_tools(prefix: str, operation: str) -> None:
    with _build_test_client() as client:
        response: Final = client.get("/openapi.json")

    assert response.status_code == 200, response.text
    schema: Final = response.json()["paths"][f"{prefix}/models/{{model_name}}:{operation}"]["post"]["requestBody"][
        "content"
    ]["application/json"]["schema"]
    validate(
        instance={
            "contents": _GOOGLE_V1_CONTENTS,
            "systemInstruction": {"parts": [{"text": "Be concise"}]},
            "tools": [{"functionDeclarations": [{"name": "lookup", "description": "Look up a value"}]}],
            "generationConfig": {"temperature": 0.2},
        },
        schema=schema,
    )


@pytest.mark.parametrize("prefix", _GOOGLE_PREFIXES)
def test_google_openapi_describes_streaming_responses_as_sse(prefix: str) -> None:
    with _build_test_client() as client:
        response: Final = client.get("/openapi.json")

    assert response.status_code == 200, response.text
    content: Final = response.json()["paths"][f"{prefix}/models/{{model_name}}:streamGenerateContent"]["post"]["responses"][
        "200"
    ]["content"]
    assert content == {"text/event-stream": {"schema": {"type": "string"}}}


def _build_test_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from litellm.proxy.google_endpoints.endpoints import router as google_router

    app = FastAPI()
    app.include_router(google_router)
    return TestClient(app)


def _patch_base_process(return_value=None):
    """Patch ProxyBaseLLMRequestProcessing.base_process_llm_request so endpoint
    tests don't run the full pipeline. Returns the AsyncMock so callers can
    inspect call args."""
    if return_value is None:
        return_value = {"test": "response"}
    return patch(
        "litellm.proxy.google_endpoints.endpoints.ProxyBaseLLMRequestProcessing.base_process_llm_request",
        new_callable=AsyncMock,
        return_value=return_value,
    )


def test_google_generate_content_endpoint():
    """generateContent routes through ProxyBaseLLMRequestProcessing with the
    agenerate_content route_type — that pipeline runs pre_call_hook +
    during_call_hook + post_call_success_hook for every guardrail callback."""
    try:
        client = _build_test_client()
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")

    with _patch_base_process() as mock_base:
        response = client.post(
            "/v1beta/models/test-model:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
        )

        assert response.status_code == 200
        mock_base.assert_called_once()
        kwargs = mock_base.call_args.kwargs
        assert kwargs["route_type"] == "agenerate_content"
        assert kwargs["model"] == "test-model"


def test_google_stream_generate_content_endpoint():
    """streamGenerateContent must route through the same processor with the
    streaming route_type so the guardrail pipeline runs."""
    try:
        client = _build_test_client()
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")

    with (
        _patch_base_process() as mock_base,
        patch(
            "litellm.proxy.google_endpoints.endpoints.ProxyBaseLLMRequestProcessing.__init__",
            return_value=None,
        ) as mock_init,
    ):
        response = client.post(
            "/v1beta/models/test-model:streamGenerateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
        )

        assert response.status_code == 200
        mock_base.assert_called_once()
        kwargs = mock_base.call_args.kwargs
        assert kwargs["route_type"] == "agenerate_content_stream"
        assert kwargs["model"] == "test-model"

        # stream=True must be forced into the data the processor receives.
        init_kwargs = mock_init.call_args.kwargs
        assert init_kwargs["data"]["stream"] is True
        assert init_kwargs["data"]["_litellm_raw_sse_stream"] is True
        assert init_kwargs["data"]["_litellm_skip_openai_stream_done"] is True
        assert init_kwargs["data"]["model"] == "test-model"
        assert init_kwargs["data"]["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]


def test_google_generate_content_data_flows_through_processor():
    """The body the client sends must reach ProxyBaseLLMRequestProcessing
    intact so the pipeline can apply guardrails to it."""
    try:
        client = _build_test_client()
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")

    with (
        _patch_base_process(),
        patch(
            "litellm.proxy.google_endpoints.endpoints.ProxyBaseLLMRequestProcessing.__init__",
            return_value=None,
        ) as mock_init,
    ):
        client.post(
            "/v1beta/models/test-model:generateContent",
            json={
                "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
                "systemInstruction": {"parts": [{"text": "Your name is Doodle."}]},
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"],
                    "imageConfig": {"aspectRatio": "9:16", "imageSize": "4K"},
                },
            },
        )

        data = mock_init.call_args.kwargs["data"]
        assert data["model"] == "test-model"
        assert data["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]
        assert data["systemInstruction"] == {"parts": [{"text": "Your name is Doodle."}]}
        # generationConfig arrives intact here; the rename to `config` is
        # done downstream in route_request (see test_route_llm_request).
        assert data["generationConfig"]["responseModalities"] == ["TEXT", "IMAGE"]
        assert data["generationConfig"]["imageConfig"]["aspectRatio"] == "9:16"


def test_google_generate_content_forwards_call_id_header():
    """The endpoint must forward the x-litellm-call-id header to the processor
    so the helper can stamp it on the logging object. Trace continuity from
    client → callbacks (S3, Langfuse, etc.) depends on this header surviving
    the hop through these endpoints."""
    try:
        client = _build_test_client()
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")

    with _patch_base_process() as mock_base:
        client.post(
            "/v1beta/models/test-model:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
            headers={"x-litellm-call-id": "trace-abc-123"},
        )

        forwarded_request = mock_base.call_args.kwargs["request"]
        assert forwarded_request.headers.get("x-litellm-call-id") == "trace-abc-123"


def test_google_count_tokens_unchanged():
    """countTokens has its own path and isn't affected by the pipeline change."""
    try:
        client = _build_test_client()
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")

    fake_response = TokenCountResponse(
        total_tokens=7,
        request_model="test-model",
        model_used="test-model",
        tokenizer_type="gemini_api",
        original_response={
            "totalTokens": 7,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 7}],
        },
        count_source="provider",
    )

    with patch(
        "litellm.proxy.proxy_server.count_request_tokens",
        new_callable=AsyncMock,
        return_value=fake_response,
    ):
        response = client.post(
            "/v1beta/models/test-model:countTokens",
            json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["totalTokens"] == 7
        assert body["litellm_count_source"] == "provider"


def test_google_count_tokens_unexpected_failure_uses_the_google_error_envelope():
    try:
        client = _build_test_client()
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")

    with patch(
        "litellm.proxy.proxy_server.count_request_tokens",
        new_callable=AsyncMock,
        side_effect=RuntimeError("tokenizer unavailable"),
    ):
        response = client.post(
            "/v1beta/models/test-model:countTokens",
            json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
        )

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == 500
    assert error["status"] == "INTERNAL"
    assert "tokenizer unavailable" not in error["message"]


def test_google_count_tokens_uses_normalized_total_when_provider_shape_differs():
    """A provider that counts in Anthropic shape (Bedrock, Anthropic) carries no totalTokens key, so the normalized total must win over the raw response's missing field."""
    try:
        client: Final = _build_test_client()
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")

    fake_response: Final = TokenCountResponse(
        total_tokens=2167,
        request_model="claude-opus-4-8",
        model_used="us.anthropic.claude-opus-5-5",
        tokenizer_type="bedrock_api",
        original_response={"input_tokens": 2167},
        count_source="token_count_model",
    )

    with patch(
        "litellm.proxy.proxy_server.count_request_tokens",
        new_callable=AsyncMock,
        return_value=fake_response,
    ):
        response: Final = client.post(
            "/v1beta/models/claude-opus-4-8:countTokens",
            json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
        )

        assert response.status_code == 200
        body: Final = response.json()
        assert body["totalTokens"] == 2167
        assert body["promptTokensDetails"] == []
        assert body["litellm_count_source"] == "token_count_model"
        assert body["litellm_count_model"] == "us.anthropic.claude-opus-5-5"


@pytest.mark.parametrize("path", ["/v1beta/models/count-log-claude:countTokens", "/models/count-log-claude:countTokens"])
def test_google_count_tokens_logs_one_line_with_the_callers_call_id_model_and_count(
    monkeypatch: pytest.MonkeyPatch, token_count_log_lines: Callable[[], tuple[dict[str, object], ...]], path: str
) -> None:
    import litellm.proxy.proxy_server as proxy_server
    from litellm import Router
    from litellm.llms.anthropic.count_tokens import handler as anthropic_count_tokens_handler
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    claude: Final = {"model": "anthropic/claude-zeta-9", "api_key": "test-key", "api_base": "https://count-tokens.test"}
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        Router(
            model_list=[
                {
                    "model_name": "count-log-claude",
                    "litellm_params": claude,
                    "model_info": {"id": "count-log-gemini-deployment"},
                }
            ]
        ),
    )
    monkeypatch.setattr(
        anthropic_count_tokens_handler,
        "get_async_httpx_client",
        lambda *args, **kwargs: AsyncHTTPHandler(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"input_tokens": 77}, request=request)
            )
        ),
    )

    response: Final = _build_test_client().post(
        path,
        json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
        headers={"x-litellm-call-id": "count-log-gemini"},
    )

    assert (response.status_code, response.json()["totalTokens"]) == (200, 77), response.text
    assert response.headers.get("x-litellm-call-id") == "count-log-gemini"
    assert token_count_log_lines() == (
        {
            "litellm_call_id": "count-log-gemini",
            "request_model": "count-log-claude",
            "model_id": "count-log-gemini-deployment",
            "model_used": "claude-zeta-9",
            "input_tokens": 77,
            "count_source": "provider",
            "status_code": 200,
        },
    )


def test_google_count_tokens_rejected_body_logs_the_400_it_answers(
    token_count_log_lines: Callable[[], tuple[dict[str, object], ...]],
) -> None:
    response: Final = _build_test_client().post(
        "/v1beta/models/count-log-claude:countTokens",
        json={"contents": "not a list"},
        headers={"x-litellm-call-id": "count-log-gemini-bad-body"},
    )

    assert response.status_code == 400, response.text
    assert token_count_log_lines() == (
        {
            "litellm_call_id": "count-log-gemini-bad-body",
            "request_model": "count-log-claude",
            "model_id": None,
            "model_used": None,
            "input_tokens": None,
            "count_source": None,
            "status_code": 400,
        },
    )


@pytest.mark.parametrize("client_call_id", [None, "count-echo-gemini"])
def test_google_count_tokens_rejected_body_answers_the_call_id_its_count_line_names(
    token_count_log_lines: Callable[[], tuple[dict[str, object], ...]], client_call_id: str | None
) -> None:
    response: Final = _build_test_client().post(
        "/v1beta/models/count-log-claude:countTokens",
        json={"contents": "not a list"},
        headers={} if client_call_id is None else {"x-litellm-call-id": client_call_id},
    )

    (line,) = token_count_log_lines()
    assert (response.status_code, response.headers.get("x-litellm-call-id")) == (400, line["litellm_call_id"])
    assert client_call_id in (None, line["litellm_call_id"])
