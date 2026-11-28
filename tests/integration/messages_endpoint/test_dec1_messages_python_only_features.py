"""Requests that still go to Python must look like plain requests to the caller.

`litellm/rust_bridge/catalog.py` lists `MESSAGES_PYTHON_FEATURES`: request and config shapes for
which `/v1/messages` on an Anthropic deployment stays on the Python implementation while plain
requests can be served by Rust. The split is invisible to callers only if both sides answer the
same way, so each test here sends one request that carries such a feature and asserts the ordinary
contract: the same success body with the proxy model name, the same error envelope, the same
provider request apart from what the feature itself changes.

Covered here, with the catalog feature in brackets: `mock_response` in the body (honoured only for a
key whose metadata allows it) and on the deployment [MOCK_RESPONSE], replayed `thinking` blocks in
history [THINKING_SIGNATURE_RETRY], a tool carrying a non-spec `custom` object [TOOL_CUSTOM_FIELD],
strict tools under `drop_params`
[STRICT_TOOLS_RETRY], the `x-litellm-include-cost-in-streaming-usage` header
[STREAM_COST_INJECTION], the response cache and its `no-cache` control [RESPONSE_CACHE], a
`-latest` model alias [LATEST_ALIAS], and `litellm_settings.modify_params` [MODIFY_PARAMS] on a
proxy this module boots itself.

One thing to know when reading results: the shared integration proxy runs with
`litellm_settings.cache: true`, and the `response_cache` detector marks every call Python-only
unless the request says `cache: {"no-cache": true}`. So on this proxy a "plain" request already
takes the Python path, and only the `no_cache` cells of the paired tests in the other
`test_dec1_messages_*` files can reach Rust.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request
from integration.messages_endpoint.dec1_messages_support import (
    BACKEND,
    JSON_OBJECT,
    NO_CACHE,
    PRICES,
    always,
    anthropic_error,
    deployment,
    echo_message,
    marker,
    message_json,
    message_reply,
    sse_events,
    stream_reply,
    upstream_error,
)
from pydantic import JsonValue

pytestmark: Final = pytest.mark.shares_worker_state

_MOCK_ID: Final = "msg_013Zva2CMHLNnXjNJJKqJ2EF"
_COST_HEADER: Final = "x-litellm-include-cost-in-streaming-usage"
_MOCK_ALLOWED: Final[dict[str, JsonValue]] = {"allow_client_mock_response": True}
_TOOL: Final[dict[str, JsonValue]] = {
    "name": "get_weather",
    "description": "Look up synthetic weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
}


def test_a_body_mock_response_is_ignored_for_a_key_that_may_not_mock(gateway: Gateway) -> None:
    """`mock_response` in the body is an untrusted control field: the proxy strips it unless the key
    or its team has `metadata.allow_client_mock_response: true` (`litellm_pre_call_utils.py`
    `_UNTRUSTED_ROOT_CONTROL_FIELDS`). A key without that flag must get the real provider answer,
    the provider must be called once, and `mock_response` must not be forwarded in the provider body."""
    message_id: Final = f"msg_{marker()}"
    with deployment(gateway, always(message_reply(message_id))) as rig:
        key: Final = rig.scenario.key(models=[rig.model])
        response: Final = rig.post(rig.body(mock_response="should be ignored"), key=key)
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["id"] == message_id
        assert "mock_response" not in rig.one_body()


def test_a_body_mock_response_answers_an_anthropic_message_without_calling_the_provider(gateway: Gateway) -> None:
    """A key whose metadata sets `allow_client_mock_response: true` sends `mock_response: "<text>"`.
    The answer must be HTTP 200 with an Anthropic message whose single text block is that text,
    `role: assistant`, `stop_reason: end_turn`, the fixed id `msg_013Zva2CMHLNnXjNJJKqJ2EF` and usage
    2095 input / 503 output tokens (`native/utils.py` `mock_response`). The provider is not called."""
    text: Final = f"mocked {marker()}"
    with deployment(gateway, echo_message) as rig:
        key: Final = rig.scenario.key(models=[rig.model], metadata=_MOCK_ALLOWED)
        response: Final = rig.post(rig.body(mock_response=text), key=key)
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert body["id"] == _MOCK_ID
        assert (body["type"], body["role"], body["stop_reason"]) == ("message", "assistant", "end_turn")
        assert body["content"] == [{"type": "text", "text": text}]
        assert body["usage"] == {"input_tokens": 2095, "output_tokens": 503}
        assert rig.calls() == ()


def test_a_deployment_mock_response_answers_every_call_without_the_provider(gateway: Gateway) -> None:
    """The deployment itself is configured with `mock_response: "deployment mock"`. A plain request,
    with no mock field in the body, must get HTTP 200 with that text as the only content block, and
    the provider must not be called."""
    with deployment(gateway, echo_message, mock_response="deployment mock") as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["content"] == [{"type": "text", "text": "deployment mock"}]
        assert rig.calls() == ()


@pytest.mark.parametrize(
    ("mock", "status", "error_type"),
    (
        ("litellm.RateLimitError", 429, "rate_limit_error"),
        ("litellm.InternalServerError", 500, "api_error"),
        ("litellm.ContextWindowExceededError", 400, "invalid_request_error"),
    ),
)
def test_a_mocked_error_is_answered_in_the_same_envelope_as_a_provider_error(
    gateway: Gateway, mock: str, status: int, error_type: str
) -> None:
    """For a key allowed to mock, `mock_response` set to `litellm.RateLimitError`, `litellm.InternalServerError` or
    `litellm.ContextWindowExceededError` raises that error instead of calling the provider. The caller
    must see what a real provider failure of that kind gives: HTTP 429, 500 or 400 in the Anthropic
    envelope with `rate_limit_error`, `api_error` or `invalid_request_error`, and no provider call."""
    with deployment(gateway, echo_message) as rig:
        key: Final = rig.scenario.key(models=[rig.model], metadata=_MOCK_ALLOWED)
        response: Final = rig.post(rig.body(mock_response=mock), key=key)
        assert response.status_code == status, response.text
        assert anthropic_error(response)["type"] == error_type
        assert rig.calls() == ()


def test_history_with_a_replayed_thinking_block_is_answered_like_a_plain_request(gateway: Gateway) -> None:
    """The caller replays an assistant turn that starts with a signed `thinking` block, which keeps
    the call on Python for the signature-retry path. With a provider that accepts the history, the
    caller must get the ordinary result: HTTP 200, the provider message with the proxy model name,
    exactly one provider call, and the user turns and the assistant text block delivered unchanged."""
    message_id: Final = f"msg_{marker()}"
    first_prompt: Final = f"think about {marker()}"
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": first_prompt},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "step one", "signature": "c3ludGhldGljLXNpZ25hdHVyZQ=="},
                {"type": "text", "text": "First answer."},
            ],
        },
        {"role": "user", "content": "and then?"},
    ]
    with deployment(gateway, always(message_reply(message_id))) as rig:
        response: Final = rig.post(
            rig.body(messages=messages, max_tokens=4096, thinking={"type": "enabled", "budget_tokens": 2048})
        )
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content) == {**message_json(message_id), "model": rig.model}
        sent: Final = rig.one_body()["messages"]
        assert isinstance(sent, list) and len(sent) == 3
        assert sent[0] == messages[0] and sent[2] == messages[2]
        assistant_blocks: Final = object_value(sent[1])["content"]
        assert isinstance(assistant_blocks, list), sent
        assert {"type": "text", "text": "First answer."} in assistant_blocks


def test_a_tool_with_a_custom_object_is_forwarded_as_sent_without_drop_params(gateway: Gateway) -> None:
    """A tool carrying a non-spec `custom: {"note": "caller side"}` object keeps the call on Python.
    Without `drop_params`, and with nothing in `custom` to hoist, the provider must receive the tool
    exactly as sent, `custom` included (`common_utils.py` `_tool_without_custom_field`), and the
    caller gets an ordinary HTTP 200."""
    tool: Final[dict[str, JsonValue]] = {**_TOOL, "custom": {"note": "caller side"}}
    with deployment(gateway, echo_message) as rig:
        assert rig.post(rig.body(tools=[tool])).status_code == 200
        assert rig.one_body()["tools"] == [tool]


def test_a_tool_custom_object_is_removed_when_the_deployment_drops_params(gateway: Gateway) -> None:
    """The same tool sent to a deployment configured with `drop_params: true` must reach the provider
    without the `custom` key and otherwise unchanged, and the caller gets an ordinary HTTP 200."""
    tool: Final[dict[str, JsonValue]] = {**_TOOL, "custom": {"note": "caller side"}}
    with deployment(gateway, echo_message, drop_params=True) as rig:
        assert rig.post(rig.body(tools=[tool])).status_code == 200
        assert rig.one_body()["tools"] == [_TOOL]


def test_a_strict_tool_under_drop_params_is_sent_strict_once_when_the_provider_accepts_it(gateway: Gateway) -> None:
    """A tool with `strict: true` on a `drop_params: true` deployment keeps the call on Python, which
    would retry without `strict` only if Claude rejected the schema. When the provider accepts it, the
    caller must get HTTP 200 after exactly one provider call that still carries `strict: true`."""
    tool: Final[dict[str, JsonValue]] = {**_TOOL, "strict": True}
    with deployment(gateway, echo_message, drop_params=True) as rig:
        assert rig.post(rig.body(tools=[tool])).status_code == 200
        assert rig.one_body()["tools"] == [tool]


def test_a_strict_tool_error_unrelated_to_strict_limits_is_relayed_without_a_second_call(gateway: Gateway) -> None:
    """On the same strict-tools-under-drop_params shape, a provider 400 that has nothing to do with
    strict schema limits must be answered HTTP 400 `invalid_request_error` with the provider's text,
    like any plain request. UNVERIFIED: exactly one provider call is expected, on the reading that
    the strict retry only fires for Claude's strict-limit rejections."""
    text: Final = f"messages: synthetic unrelated refusal {marker()}"
    tool: Final[dict[str, JsonValue]] = {**_TOOL, "strict": True}
    with deployment(gateway, always(upstream_error(400, "invalid_request_error", text)), drop_params=True) as rig:
        response: Final = rig.post(rig.body(tools=[tool]))
        assert response.status_code == 400, response.text
        error: Final = anthropic_error(response)
        assert error["type"] == "invalid_request_error"
        assert text in str(error["message"])
        assert len(rig.calls()) == 1


def test_the_stream_cost_header_adds_only_a_cost_to_the_final_usage(gateway: Gateway) -> None:
    """With `x-litellm-include-cost-in-streaming-usage: true` on a priced deployment, the stream must
    carry the same six events in the same order with the same text and token counts as a plain
    stream. The only addition is a numeric `cost` inside `message_delta.usage`.

    Pinned current behaviour: the cost prices the `message_delta` frame's own `usage` object and
    nothing else (`common_request_processing.py` `_inject_cost_into_usage_dict` and
    `_stream_usage_for_event`). The scripted stream reports the 11 input tokens only in
    `message_start` and sends `{"output_tokens": 7}` in `message_delta`, so the injected cost is
    7 * 0.002 = 0.014, not the 0.025 the whole call costs. Real Anthropic streams repeat the input
    counts in `message_delta`, where the two agree."""
    with deployment(gateway, always(stream_reply(f"msg_{marker()}")), **PRICES) as rig:
        _plain_response, plain_text = rig.stream(rig.body(stream=True))
        _cost_response, cost_text = rig.stream(rig.body(stream=True), headers={_COST_HEADER: "true"})
        plain: Final = sse_events(plain_text)
        costed: Final = sse_events(cost_text)
        assert [event for event, _ in costed] == [event for event, _ in plain]
        assert [data for event, data in costed if event != "message_delta"] == [
            data for event, data in plain if event != "message_delta"
        ]
        usage: Final = object_value(dict(costed)["message_delta"]["usage"])
        assert usage["output_tokens"] == 7
        assert float(str(usage["cost"])) == pytest.approx(7 * 0.002)
        assert "cost" not in object_value(dict(plain)["message_delta"]["usage"])


def test_the_stream_cost_header_changes_nothing_on_a_non_streaming_call(gateway: Gateway) -> None:
    """The cost header only applies to streams. A non-streaming call that sends it must return the
    provider message with exactly the provider's `usage` keys, with no `cost` key added to the body."""
    message_id: Final = f"msg_{marker()}"
    with deployment(gateway, always(message_reply(message_id)), **PRICES) as rig:
        response: Final = rig.post(rig.body(), headers={_COST_HEADER: "true"})
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content) == {**message_json(message_id), "model": rig.model}


def test_a_repeated_request_is_served_from_the_cache_with_the_same_message(gateway: Gateway) -> None:
    """The shared proxy has the response cache on. The same request is sent until one answer arrives
    without a new provider call (the store is asynchronous, so the first repeats may miss). That
    cached answer must be HTTP 200 with the same `id`, `content`, `stop_reason`, `usage` and proxy
    model name as the first answer: a cache hit is indistinguishable from a provider answer in the body."""
    with deployment(gateway, echo_message) as rig:
        body: Final = rig.body(f"cache me {marker()}")
        first: Final = rig.post(body)
        assert first.status_code == 200, first.text
        rig.calls()

        def repeat() -> tuple[int, dict[str, JsonValue], int]:
            response: Final = rig.post(body)
            return response.status_code, JSON_OBJECT.validate_json(response.content), len(rig.calls())

        status, cached, _calls = eventually(repeat, lambda seen: seen[2] == 0, seconds=60)
        assert status == 200
        original: Final = JSON_OBJECT.validate_json(first.content)
        assert {name: cached[name] for name in ("id", "content", "stop_reason", "usage", "model")} == {
            name: original[name] for name in ("id", "content", "stop_reason", "usage", "model")
        }


def test_no_cache_forces_a_provider_call_for_a_request_that_is_already_cached(gateway: Gateway) -> None:
    """Once a request is being served from the cache, sending the same request with
    `cache: {"no-cache": true}` must reach the provider again (one new provider call) and return the
    new provider message, whose id differs from the cached one."""
    with deployment(gateway, echo_message) as rig:
        body: Final = rig.body(f"cache me {marker()}")
        first: Final = rig.post(body)
        assert first.status_code == 200, first.text
        rig.calls()
        eventually(lambda: (rig.post(body), len(rig.calls()))[1], lambda calls: calls == 0, seconds=60)
        fresh: Final = rig.post({**body, **NO_CACHE})
        assert fresh.status_code == 200, fresh.text
        assert len(rig.calls()) == 1
        assert fresh.json()["id"] != first.json()["id"]


def test_a_latest_alias_deployment_calls_the_provider_with_a_dated_model(gateway: Gateway) -> None:
    """The deployment is `anthropic/claude-sonnet-latest`, an alias row in the model cost map, which
    keeps the call on Python. The caller must get an ordinary HTTP 200 with the proxy model name.
    UNVERIFIED: the provider is expected to receive the alias's resolved target (a `claude-sonnet-`
    name that is not `claude-sonnet-latest`, see `model_latest_aliases.py`); the exact target moves
    with the cost map, so only that shape is asserted."""
    with deployment(gateway, echo_message, backend="claude-sonnet-latest") as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["model"] == rig.model
        sent_model: Final = str(rig.one_body()["model"])
        assert sent_model.startswith("claude-sonnet-") and sent_model != "claude-sonnet-latest", sent_model


@pytest.fixture(scope="module")
def modify_params_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    """A proxy booted from the shared config with `litellm_settings.modify_params: true` and the
    response cache off, so the only Python-only feature in play is MODIFY_PARAMS."""
    directory: Final = tmp_path_factory.mktemp("dec1-messages-modify-params")
    config: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    settings: Final = JSON_OBJECT.validate_python(config.get("litellm_settings", {}))
    path: Final = directory / "modify-params.yaml"
    path.write_text(yaml.safe_dump({**config, "litellm_settings": {**settings, "cache": False, "modify_params": True}}))
    with gateway_from_environment() as shared, owned_proxy(shared, directory, {}, config=path) as owned:
        yield owned


def test_modify_params_leaves_a_valid_request_and_its_answer_unchanged(modify_params_gateway: Gateway) -> None:
    """On a proxy with `modify_params: true`, a request that needs no repair must behave like a plain
    one: the provider receives the backend model, the caller's `max_tokens`, `system` and messages
    unchanged, and the caller receives the provider message with the proxy model name."""
    message_id: Final = f"msg_{marker()}"
    prompt: Final = f"ping {marker()}"
    with deployment(modify_params_gateway, always(message_reply(message_id))) as rig:
        response: Final = rig.post(rig.body(prompt, max_tokens=128, system="synthetic policy"))
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content) == {**message_json(message_id), "model": rig.model}
        sent: Final = rig.one_body()
        assert (sent["model"], sent["max_tokens"], sent["system"]) == (BACKEND, 128, "synthetic policy")
        assert sent["messages"] == [{"role": "user", "content": prompt}]


def test_modify_params_relays_a_provider_error_in_the_same_envelope(modify_params_gateway: Gateway) -> None:
    """On the `modify_params: true` proxy a provider 429 must still be answered HTTP 429 in the
    Anthropic envelope with `error.type == "rate_limit_error"` and the provider's text in the message."""
    text: Final = f"synthetic rate limit {marker()}"

    def respond(_request: Request) -> Reply:
        return upstream_error(429, "rate_limit_error", text)

    with deployment(modify_params_gateway, respond) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 429, response.text
        error: Final = anthropic_error(response)
        assert error["type"] == "rate_limit_error"
        assert text in str(error["message"])
