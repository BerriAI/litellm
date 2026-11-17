"""Python-era features on the providers whose `/v1/messages` now runs only on Rust.

`litellm/rust_bridge/catalog.py` marks `/v1/messages` RUST_REQUIRED for `anthropic`, for `azure_ai` Claude on
Foundry (#51721) and for Claude on Bedrock InvokeModel (#51330). There is no Python fallback for those
providers any more: `litellm/messages/dispatch.py` refuses a call that would land on Python with
`UnsupportedParamsError("provider <p> on /v1/messages runs only on the Rust path, which does not support ...")`.
`MESSAGES_PYTHON_FEATURES` is now the empty set, which is a promise that every feature the Python handler
used to serve is served natively: mock responses, the timeout variants (`request_timeout`), the response
cache, `drop_params` and `additional_drop_params`, deployment retries, `-latest` aliases. These tests pin
that each of those still behaves as it did on Python for every Rust-native provider, so a regression shows up
as a provider-specific failure instead of an opaque 400 from the dispatch refusal.

Vertex AI (draft #51294) is included where cheap, so the same contract is in place before its port lands.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request
from integration.messages_endpoint.dec1_messages_support import JSON_OBJECT, SPEND_COLUMNS, anthropic_error
from integration.messages_endpoint.dec1_msgprov2_support import (
    ANTHROPIC_BACKEND,
    NATIVE_PROVIDERS,
    RUST_NATIVE_PROVIDERS,
    anthropic_answer,
    native_deployment,
    native_error,
)
from integration.messages_endpoint.dec1_msgprov_support import (
    anthropic_body_echo,
    body_of,
    invoke_echo,
    unique_text,
    vertex_echo,
)
from pydantic import JsonValue

_MOCK_TEXT: Final = "synthetic mocked answer"


def _echo(provider: str) -> Callable[[Request], Reply]:
    return {
        "anthropic": anthropic_answer(ANTHROPIC_BACKEND),
        "azure_ai": anthropic_body_echo,
        "bedrock_invoke": invoke_echo,
        "vertex_ai": vertex_echo,
    }[provider]


def _gated(provider: str, gate: threading.Event) -> Callable[[Request], Reply]:
    """The provider's echo, held before its first byte until `gate` is set."""
    echo: Final = _echo(provider)

    def respond(request: Request) -> Reply:
        reply: Final = echo(request)
        return Reply(
            status=reply.status,
            body=reply.body,
            content_type=reply.content_type,
            chunks=reply.chunks,
            headers=reply.headers,
            gate_before_first=gate,
        )

    return respond


@pytest.mark.parametrize("provider", RUST_NATIVE_PROVIDERS)
def test_a_deployment_mock_response_answers_without_calling_the_provider(gateway: Gateway, provider: str) -> None:
    """A deployment with `mock_response: "synthetic mocked answer"` answers `/v1/messages` with HTTP 200 and a
    message whose first text block is that string, and the provider peer receives no POST. Mock responses are a
    native feature now (`_NATIVE_COMMON_FEATURES` in dispatch.py), so a Rust-only provider must not refuse
    them. UNVERIFIED: the rest of the mocked message (id, usage) is not pinned."""
    with native_deployment(gateway, provider, mock_response=_MOCK_TEXT) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        content: Final = JSON_OBJECT.validate_json(response.content)["content"]
        assert isinstance(content, list) and object_value(content[0])["text"] == _MOCK_TEXT
        assert rig.calls() == ()


@pytest.mark.parametrize("provider", RUST_NATIVE_PROVIDERS)
def test_a_deployment_request_timeout_cuts_off_a_slow_provider_with_408(gateway: Gateway, provider: str) -> None:
    """A deployment with `request_timeout: 0.3` whose provider holds its answer is answered with HTTP 408 and an
    error message containing `Timeout`, and the provider was called exactly once. `request_timeout` is one of the
    timeout variants that used to force Python (`detectors.timeout_variants`); the Rust path resolves it through
    `_native_kwargs`."""
    gate: Final = threading.Event()
    try:
        with native_deployment(gateway, provider, _gated(provider, gate), request_timeout=0.3) as rig:
            response: Final = rig.post(rig.body())
            assert response.status_code == 408, response.text
            assert "Timeout" in str(anthropic_error(response)["message"]), response.text
            assert len(eventually(rig.calls, lambda calls: len(calls) >= 1, seconds=5)) == 1
    finally:
        gate.set()


@pytest.mark.parametrize("provider", RUST_NATIVE_PROVIDERS)
def test_an_identical_request_is_served_from_the_response_cache(gateway: Gateway, provider: str) -> None:
    """The shared proxy has `cache: true`. The same non-streamed body sent twice reaches the provider once; the
    second answer has the same `id` and `content` as the first, and a second SpendLogs row whose `request_id`
    starts with `<first id>_cache_hit` is written. The native path owns the cache since #50426, so a port that
    skips the lookup would double provider calls and spend."""
    with native_deployment(gateway, provider) as rig:
        body: Final = rig.body(unique_text("cache"))
        first: Final = rig.post(body)
        assert first.status_code == 200, first.text
        second: Final = rig.post(body)
        assert second.status_code == 200, second.text
        one: Final = JSON_OBJECT.validate_json(first.content)
        two: Final = JSON_OBJECT.validate_json(second.content)
        assert (two["id"], two["content"]) == (one["id"], one["content"])
        assert len(rig.calls()) == 1
        rows: Final = eventually(
            lambda: read_rows(SPEND_COLUMNS + "WHERE model_group=%s", (rig.model,)),
            lambda found: len(found) >= 2,
            seconds=90,
        )
        assert any(str(row["request_id"]).startswith(f"{one['id']}_cache_hit") for row in rows), rows


@pytest.mark.parametrize("provider", RUST_NATIVE_PROVIDERS)
def test_no_cache_in_the_body_sends_every_request_to_the_provider(gateway: Gateway, provider: str) -> None:
    """With `cache: {"no-cache": true}` in the body the same request sent twice reaches the provider twice, and
    neither upstream body contains a `cache` member: the proxy consumes the control field."""
    with native_deployment(gateway, provider) as rig:
        body: Final = rig.body(unique_text("nocache"), cache={"no-cache": True})
        assert rig.post(body).status_code == 200
        assert rig.post(body).status_code == 200
        calls: Final = rig.calls()
        assert len(calls) == 2
        assert all("cache" not in body_of(call) for call in calls)


@pytest.mark.parametrize("provider", RUST_NATIVE_PROVIDERS)
def test_additional_drop_params_removes_the_field_on_the_rust_path(gateway: Gateway, provider: str) -> None:
    """A deployment with `additional_drop_params: ["metadata"]` sends no `metadata` member upstream even when the
    caller sends `metadata.user_id`. `drop_params` and `additional_drop_params` were Python features
    (`Feature.DROP_PARAMS`) and are no longer in `MESSAGES_PYTHON_FEATURES`, so the Rust path must apply them
    (the Python fix was #50753)."""
    with native_deployment(gateway, provider, additional_drop_params=["metadata"]) as rig:
        assert rig.post(rig.body(metadata={"user_id": "synthetic-user-03"})).status_code == 200
        assert "metadata" not in body_of(rig.one_call())


@pytest.mark.parametrize("provider", RUST_NATIVE_PROVIDERS)
def test_a_deployment_retry_budget_recovers_from_one_server_error(gateway: Gateway, provider: str) -> None:
    """A deployment with `num_retries: 1` whose provider fails the first call with a 500 (an
    InternalServerException for Bedrock) and answers the second is answered HTTP 200, and the provider saw two
    POSTs. Retries were a Python feature (`Feature.RETRIES_FALLBACKS`) and are run by the router for native
    providers. UNVERIFIED: whether Bedrock's 500, which the proxy maps to 503, is retried like Anthropic's."""
    seen: Final[list[int]] = []
    echo: Final = _echo(provider)

    def respond(request: Request) -> Reply:
        seen.append(1)
        if len(seen) == 1:
            return native_error(provider, 500, "api_error", "synthetic first failure")
        return echo(request)

    with native_deployment(gateway, provider, respond, num_retries=1) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        assert len(rig.calls()) == 2


@pytest.mark.parametrize("provider", RUST_NATIVE_PROVIDERS)
def test_a_client_error_is_not_retried_even_with_a_retry_budget(gateway: Gateway, provider: str) -> None:
    """With `num_retries: 2` a provider 400 is answered at once as HTTP 400 `invalid_request_error` after a single
    provider call: client errors are never retried, on Python or Rust."""
    reply: Final = native_error(provider, 400, "invalid_request_error", "synthetic bad request")
    with native_deployment(gateway, provider, lambda _r: reply, num_retries=2) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 400, response.text
        assert anthropic_error(response)["type"] == "invalid_request_error"
        assert len(rig.calls()) == 1


@pytest.mark.parametrize("provider", NATIVE_PROVIDERS)
def test_a_missing_max_tokens_is_refused_before_the_provider(gateway: Gateway, provider: str) -> None:
    """A body without `max_tokens` is answered with HTTP 400 `invalid_request_error` on every native provider and
    the provider is not called. The message names `max_tokens`. A Rust port must validate before sending, as the
    Python config does (`max_tokens is required for Anthropic /v1/messages API`)."""
    with native_deployment(gateway, provider) as rig:
        body: Final[dict[str, JsonValue]] = {
            "model": rig.model,
            "messages": [{"role": "user", "content": unique_text()}],
        }
        response: Final = rig.post(body)
        assert response.status_code == 400, response.text
        error: Final = anthropic_error(response)
        assert error["type"] == "invalid_request_error"
        assert "max_tokens" in str(error["message"])
        assert rig.calls() == ()


@pytest.mark.parametrize("provider", RUST_NATIVE_PROVIDERS)
def test_a_rust_native_provider_never_answers_with_the_rust_only_refusal_for_a_plain_call(
    gateway: Gateway, provider: str
) -> None:
    """A plain call with every optional field a client commonly sends (system, tools, tool_choice, stop_sequences,
    temperature, metadata, thinking) is served with HTTP 200 and the body never contains the dispatch refusal text
    `runs only on the Rust path`. If a detector is ever added back to `MESSAGES_PYTHON_FEATURES` for one of these
    inputs, the Rust-only providers would start refusing ordinary traffic; this catches it."""
    fields: Final[dict[str, JsonValue]] = {
        "system": "You are terse.",
        "tools": [
            {
                "name": "get_weather",
                "description": "Look up the weather for a city",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
        "tool_choice": {"type": "auto"},
        "stop_sequences": ["###"],
        "metadata": {"user_id": "synthetic-user-04"},
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }
    with native_deployment(gateway, provider) as rig:
        response: Final = rig.post(rig.body(max_tokens=2048, **fields))
        assert response.status_code == 200, response.text
        assert "runs only on the Rust path" not in response.text
