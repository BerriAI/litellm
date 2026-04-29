"""
Tests for cursor-based stream resume on the Responses API retrieve endpoint.

Covers the GET /v1/responses/{response_id}?stream=true&starting_after=N flow:

- Provider transforms (OpenAI, Azure) put ``stream`` and ``starting_after``
  into the returned ``data`` dict so the HTTP layer attaches them as query
  params.
- Provider transforms that do not support resume (Volcengine, Manus) accept
  the new arguments without breaking (signature parity with the base class).
- The async core dispatcher ``litellm.aget_responses(stream=True, ...)``
  returns a streaming iterator instead of a non-streaming
  :class:`ResponsesAPIResponse`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.manus.responses.transformation import ManusResponsesAPIConfig
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.llms.volcengine.responses.transformation import (
    VolcEngineResponsesAPIConfig,
)
from litellm.responses.streaming_iterator import (
    ResponsesAPIStreamingIterator,
    SyncResponsesAPIStreamingIterator,
)
from litellm.types.router import GenericLiteLLMParams


# ---------------------------------------------------------------------------
# Provider transform tests
# ---------------------------------------------------------------------------


def _params() -> GenericLiteLLMParams:
    return GenericLiteLLMParams(api_base="https://api.openai.com/v1/responses")


def test_openai_transform_includes_stream_and_starting_after() -> None:
    config = OpenAIResponsesAPIConfig()
    url, data = config.transform_get_response_api_request(
        response_id="resp_abc",
        api_base="https://api.openai.com/v1/responses",
        litellm_params=_params(),
        headers={},
        stream=True,
        starting_after=42,
    )
    assert url == "https://api.openai.com/v1/responses/resp_abc"
    assert data == {"stream": "true", "starting_after": 42}


def test_openai_transform_omits_resume_args_when_not_set() -> None:
    config = OpenAIResponsesAPIConfig()
    url, data = config.transform_get_response_api_request(
        response_id="resp_abc",
        api_base="https://api.openai.com/v1/responses",
        litellm_params=_params(),
        headers={},
    )
    assert url == "https://api.openai.com/v1/responses/resp_abc"
    assert data == {}


def test_azure_transform_includes_stream_and_starting_after() -> None:
    config = AzureOpenAIResponsesAPIConfig()
    url, data = config.transform_get_response_api_request(
        response_id="resp_abc",
        api_base="https://example.openai.azure.com/openai/responses?api-version=2025-04-01-preview",
        litellm_params=GenericLiteLLMParams(
            api_base="https://example.openai.azure.com/",
            api_version="2025-04-01-preview",
        ),
        headers={},
        stream=True,
        starting_after=6,
    )
    # URL must contain the response_id; query params live in ``data``.
    assert "resp_abc" in url
    assert data == {"stream": "true", "starting_after": 6}


def test_volcengine_transform_accepts_resume_args_without_using_them() -> None:
    config = VolcEngineResponsesAPIConfig()
    _, data = config.transform_get_response_api_request(
        response_id="resp_abc",
        api_base="https://example.volces.com/v1/responses",
        litellm_params=_params(),
        headers={},
        stream=True,
        starting_after=10,
    )
    # Volcengine does not support resume; args are silently ignored.
    assert data == {}


def test_manus_transform_accepts_resume_args_without_using_them() -> None:
    config = ManusResponsesAPIConfig()
    _, data = config.transform_get_response_api_request(
        response_id="resp_abc",
        api_base="https://api.manus.im/v1/responses",
        litellm_params=_params(),
        headers={},
        stream=True,
        starting_after=10,
    )
    assert data == {}


# ---------------------------------------------------------------------------
# Core dispatcher streaming test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aget_responses_stream_returns_streaming_iterator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``litellm.aget_responses(stream=True, starting_after=N)`` must return a
    :class:`ResponsesAPIStreamingIterator` rather than a single
    :class:`ResponsesAPIResponse`. We stub out the underlying HTTP client to
    keep the test offline and assert on the wiring (URL params + iterator
    type).
    """
    import litellm
    from litellm.llms.custom_httpx import llm_http_handler as handler_module

    captured: dict[str, Any] = {}

    async def fake_get(
        url: str,
        headers: dict | None = None,
        params: dict | None = None,
        stream: bool = False,
        timeout: Any = None,
        **_: Any,
    ) -> httpx.Response:
        captured["url"] = url
        captured["params"] = params
        captured["stream"] = stream
        # An empty SSE-style body — the iterator can be constructed from any
        # httpx.Response with ``aiter_lines``; we only care about wiring here.
        return httpx.Response(
            status_code=200,
            content=b"",
            request=httpx.Request("GET", url),
        )

    fake_client = MagicMock()
    fake_client.get = AsyncMock(side_effect=fake_get)
    monkeypatch.setattr(
        handler_module,
        "get_async_httpx_client",
        lambda *a, **kw: fake_client,
    )

    response_id = "resp_test_stream_resume_with_a_long_enough_id_to_satisfy_validators"
    result = await litellm.aget_responses(
        response_id=response_id,
        stream=True,
        starting_after=6,
        model="gpt-4.1-mini",
        custom_llm_provider="openai",
        api_key="sk-test",
        api_base="https://api.openai.com/v1/responses",
    )

    assert isinstance(result, ResponsesAPIStreamingIterator)
    assert result.model == "gpt-4.1-mini"
    assert captured["stream"] is True
    assert captured["params"] == {"stream": "true", "starting_after": 6}
    assert response_id in captured["url"]


def test_get_responses_stream_returns_sync_streaming_iterator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm
    from litellm.llms.custom_httpx import llm_http_handler as handler_module

    captured: dict[str, Any] = {}

    def fake_get(
        url: str,
        headers: dict | None = None,
        params: dict | None = None,
        stream: bool = False,
        timeout: Any = None,
        **_: Any,
    ) -> httpx.Response:
        captured["url"] = url
        captured["params"] = params
        captured["stream"] = stream
        captured["timeout"] = timeout
        return httpx.Response(
            status_code=200,
            content=b"",
            request=httpx.Request("GET", url),
        )

    fake_client = MagicMock()
    fake_client.get = MagicMock(side_effect=fake_get)
    monkeypatch.setattr(
        handler_module,
        "_get_httpx_client",
        lambda *a, **kw: fake_client,
    )

    response_id = "resp_test_stream_resume_with_a_long_enough_id_to_satisfy_validators"
    result = litellm.get_responses(
        response_id=response_id,
        stream=True,
        starting_after=6,
        model="gpt-4.1-mini",
        custom_llm_provider="openai",
        api_key="sk-test",
        api_base="https://api.openai.com/v1/responses",
    )

    assert isinstance(result, SyncResponsesAPIStreamingIterator)
    assert result.model == "gpt-4.1-mini"
    assert captured["stream"] is True
    assert captured["params"] == {"stream": "true", "starting_after": 6}
    assert response_id in captured["url"]


def test_http_handler_get_stream_uses_open_request() -> None:
    captured: dict[str, Any] = {}

    class DummyClient:
        def build_request(
            self,
            method: str,
            url: str,
            params: dict | None = None,
            headers: dict | None = None,
            timeout: Any = None,
        ) -> httpx.Request:
            captured["method"] = method
            captured["url"] = url
            captured["params"] = params
            captured["headers"] = headers
            captured["timeout"] = timeout
            return httpx.Request(method, url, params=params, headers=headers)

        def send(
            self,
            request: httpx.Request,
            stream: bool = False,
            follow_redirects: Any = None,
        ) -> httpx.Response:
            captured["stream"] = stream
            captured["follow_redirects"] = follow_redirects
            captured["request_url"] = str(request.url)
            return httpx.Response(status_code=200, content=b"", request=request)

    handler = HTTPHandler(client=DummyClient())
    response = handler.get(
        url="https://api.openai.com/v1/responses/resp_abc?existing=1",
        params={"stream": "true"},
        headers={"x-test": "1"},
        stream=True,
        timeout=12.0,
    )

    assert response.status_code == 200
    assert captured["method"] == "GET"
    assert captured["params"] == {"stream": "true", "existing": "1"}
    assert captured["stream"] is True
    assert captured["timeout"] == 12.0
