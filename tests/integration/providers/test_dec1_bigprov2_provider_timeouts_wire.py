"""Deployment timeouts and stream idle timeouts on each big provider's own HTTP client.

`litellm_params.timeout` bounds a non streamed call and `stream_timeout` bounds a streamed one, both as the
gap the client waits for the next bytes; the caller may lower either with `x-litellm-timeout` /
`x-litellm-stream-timeout` (`litellm/proxy/litellm_pre_call_utils.py`, `_get_timeout_from_request`). A call
that runs out gets HTTP 408 with a message starting `litellm.Timeout: ` before the stream opens, or, once a
streamed answer has started, an in-band error frame whose `code` is `"408"` followed by `[DONE]`.

The generic contract is pinned against an OpenAI-compatible upstream (`routing/test_dec1_routing_timeouts.py`,
`streaming/test_dec1_stream_errors_and_timeouts.py`) and Bedrock has its own suite
(`test_bedrock_stream_timeout_wire.py`). But each big provider is served by a different client: Anthropic and
Vertex Claude through `BaseLLMHTTPHandler` and the Anthropic stream iterator, Gemini and Vertex Gemini through
the Vertex `ModelResponseIterator`, Azure through the OpenAI SDK's `AzureOpenAI` client with its own timeout
object. A Rust port that applies the timeout in one place and forgets a client would hang those callers
(`index/test_dec1_*_.md`, "Not covered"). Every test passes `num_retries: 0` so one wait is measured.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario
from integration._support.timing import assert_enforced_within
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, wire_server
from integration.providers.dec1_bigprov2_support import (
    error_message,
    error_of,
    marker,
    silent_stream,
    sse_data,
    stalled_stream,
    withheld_json,
)
from pydantic import JsonValue

_TIMEOUT: Final = 1
_SLACK: Final = 14.0
"""Seconds past the configured timeout a timed out call may take; the peer itself never answers."""
_HEAD_TEXT: Final = "dec1 head"


def _anthropic_head() -> bytes:
    start: Final = {
        "type": "message_start",
        "message": {
            "id": "msg_dec1_timeout",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5-20250929",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 1},
        },
    }
    block: Final = {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
    delta: Final = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": _HEAD_TEXT}}
    return b"".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in (start, block, delta)
    )


def _gemini_head() -> bytes:
    frame: Final = {"candidates": [{"content": {"role": "model", "parts": [{"text": _HEAD_TEXT}]}, "index": 0}]}
    return f"data: {json.dumps(frame)}\r\n\r\n".encode()


def _openai_head() -> bytes:
    chunk: Final = {
        "id": "chatcmpl-dec1-timeout",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": _HEAD_TEXT}, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()


@dataclass(frozen=True, slots=True)
class _Provider:
    name: str
    head: Callable[[], bytes]
    register: Callable[..., str]


def _anthropic(scenario: Scenario, base: str, **params: JsonValue) -> str:
    return scenario.model(
        model="anthropic/claude-sonnet-4-5-20250929", api_base=base, api_key="dec1-bigprov2-anthropic-key", **params
    )


def _gemini(scenario: Scenario, base: str, **params: JsonValue) -> str:
    return scenario.model(
        model="gemini/gemini-2.5-flash", api_base=f"{base}/v1beta", api_key="dec1-bigprov2-gemini", **params
    )


def _vertex_gemini(scenario: Scenario, base: str, **params: JsonValue) -> str:
    return scenario.model(
        model="vertex_ai/gemini-2.5-flash",
        api_key=None,
        api_base=base,
        vertex_project="dec1-bigprov2-timeouts",
        vertex_location="us-central1",
        vertex_credentials=service_account_json("dec1-bigprov2-timeouts", base),
        **params,
    )


def _vertex_claude(scenario: Scenario, base: str, **params: JsonValue) -> str:
    return scenario.model(
        model="vertex_ai/claude-sonnet-4-5@20250929",
        api_key=None,
        api_base=base,
        vertex_project="dec1-bigprov2-timeouts",
        vertex_location="us-east5",
        vertex_credentials=service_account_json("dec1-bigprov2-timeouts", base),
        **params,
    )


def _azure(scenario: Scenario, base: str, **params: JsonValue) -> str:
    return scenario.model(
        model="azure/gpt-4o-dec1", api_base=base, api_key="dec1-bigprov2-azure-key", api_version="2024-10-21", **params
    )


_PROVIDERS: Final = (
    _Provider("anthropic", _anthropic_head, _anthropic),
    _Provider("gemini", _gemini_head, _gemini),
    _Provider("vertex_gemini", _gemini_head, _vertex_gemini),
    _Provider("vertex_claude", _anthropic_head, _vertex_claude),
    _Provider("azure", _openai_head, _azure),
)
_IDS: Final = tuple(provider.name for provider in _PROVIDERS)


def _peer(model_reply: Callable[[], Reply]) -> Callable[[Request], Reply]:
    """Answers the Google token exchange at once (Vertex), probes with an empty list, and model calls with
    `model_reply`."""

    def answer(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(body=b'{"object":"list","data":[]}')
        if request.target.startswith("/_oauth/token"):
            return Reply(body=b'{"access_token": "dec1-timeout-token", "token_type": "Bearer", "expires_in": 3600}')
        return model_reply()

    return answer


def _chat(
    gateway: Gateway, model: str, *, headers: dict[str, str] | None = None, **fields: JsonValue
) -> tuple[httpx.Response, float]:
    started: Final = time.monotonic()
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": marker()}], "num_retries": 0, **fields},
        headers=headers,
    )
    return response, time.monotonic() - started


@pytest.mark.parametrize("provider", _PROVIDERS, ids=_IDS)
def test_a_deployment_timeout_ends_a_withheld_answer_with_408(gateway: Gateway, provider: _Provider) -> None:
    """The deployment sets `timeout: 1`; the provider sends response headers and never the JSON body.

    The caller gets HTTP 408, `error.code` `"408"` and a message starting `litellm.Timeout: `, after at least one
    second and within the configured value plus slack. Each provider client must honour the deployment timeout itself.
    """
    with wire_server(_peer(withheld_json)) as wire, gateway.scenario() as scenario:
        model: Final = provider.register(scenario, wire.url.rstrip("/"), timeout=_TIMEOUT)
        response, elapsed = _chat(gateway, model)
        assert response.status_code == 408, response.text
        assert error_of(response).get("code") == "408", response.text
        assert error_message(response).startswith("litellm.Timeout: "), response.text
        assert_enforced_within(elapsed, configured=_TIMEOUT, slack=_SLACK, not_before=_TIMEOUT)


@pytest.mark.parametrize("provider", _PROVIDERS, ids=_IDS)
def test_the_x_litellm_timeout_header_bounds_a_withheld_answer(gateway: Gateway, provider: _Provider) -> None:
    """The deployment sets no timeout; the caller sends `x-litellm-timeout: 1` and the provider withholds its
    body. The caller gets 408 within the configured value plus slack instead of waiting for the default 600 second timeout."""
    with wire_server(_peer(withheld_json)) as wire, gateway.scenario() as scenario:
        model: Final = provider.register(scenario, wire.url.rstrip("/"))
        response, elapsed = _chat(gateway, model, headers={"x-litellm-timeout": str(_TIMEOUT)})
        assert response.status_code == 408, response.text
        assert_enforced_within(elapsed, configured=_TIMEOUT, slack=_SLACK)


@pytest.mark.parametrize("provider", _PROVIDERS, ids=_IDS)
def test_a_stream_silent_past_its_stream_timeout_is_an_http_408(gateway: Gateway, provider: _Provider) -> None:
    """The caller streams with `x-litellm-stream-timeout: 1`; the provider opens the event stream and never
    sends an event. Nothing has reached the caller yet, so the answer is a JSON error with HTTP 408 rather than
    a 200 stream, within the configured value plus slack."""
    with wire_server(_peer(lambda: silent_stream((provider.head(),)))) as wire, gateway.scenario() as scenario:
        model: Final = provider.register(scenario, wire.url.rstrip("/"))
        response, elapsed = _chat(gateway, model, stream=True, headers={"x-litellm-stream-timeout": str(_TIMEOUT)})
        assert response.status_code == 408, response.text
        assert response.headers.get("content-type", "").startswith("application/json"), response.headers
        assert_enforced_within(elapsed, configured=_TIMEOUT, slack=_SLACK)


@pytest.mark.parametrize("provider", _PROVIDERS, ids=_IDS)
def test_a_stream_that_stalls_after_content_ends_with_an_in_band_408(gateway: Gateway, provider: _Provider) -> None:
    """The caller streams with `x-litellm-stream-timeout: 1`; the provider sends one content event (`dec1 head`)
    and then stalls with the connection open. The caller's stream is HTTP 200, carries `dec1 head`, and ends with
    one error frame whose `error.code` is `"408"` followed by `[DONE]`, within the configured value plus slack. UNVERIFIED for Azure
    and Gemini: the in-band `code` is read from the hosted vLLM stall test, which shares the stream wrapper."""
    reply: Final = lambda: stalled_stream((provider.head(), provider.head()))
    with wire_server(_peer(reply)) as wire, gateway.scenario() as scenario:
        model: Final = provider.register(scenario, wire.url.rstrip("/"))
        response, elapsed = _chat(gateway, model, stream=True, headers={"x-litellm-stream-timeout": str(_TIMEOUT)})
        assert response.status_code == 200, response.text
        assert _HEAD_TEXT in response.text
        frames: Final = sse_data(response.text)
        assert "error" in frames[-1], response.text
        assert str(frames[-1]["error"]["code"]) == "408", frames[-1]  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        assert response.text.rstrip().endswith("data: [DONE]")
        assert_enforced_within(elapsed, configured=_TIMEOUT, slack=_SLACK)


@pytest.mark.parametrize("provider", _PROVIDERS, ids=_IDS)
def test_a_deployment_stream_timeout_alone_bounds_a_silent_stream(gateway: Gateway, provider: _Provider) -> None:
    """The deployment is registered with `stream_timeout: 1` and the caller sends no timeout header; the provider
    opens the stream and stays silent. The caller gets 408 within the configured value plus slack: the deployment value reaches each
    provider's streaming client without any request level override."""
    with wire_server(_peer(lambda: silent_stream((provider.head(),)))) as wire, gateway.scenario() as scenario:
        name: Final = provider.register(scenario, wire.url.rstrip("/"), stream_timeout=_TIMEOUT)
        response, elapsed = _chat(gateway, name, stream=True)
        assert response.status_code == 408, response.text
        assert_enforced_within(elapsed, configured=_TIMEOUT, slack=_SLACK)
