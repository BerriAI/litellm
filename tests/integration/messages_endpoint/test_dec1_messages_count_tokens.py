"""`POST /v1/messages/count_tokens` for an Anthropic deployment.

The caller sends the same shape as a Messages request (without `max_tokens`) and gets back
`{"input_tokens": <n>}`. For an Anthropic deployment the proxy asks the provider: one
`POST <api_base>/v1/messages/count_tokens` with the deployment key in `x-api-key`,
`anthropic-version: 2023-06-01`, the `token-counting-2024-11-01` beta, the backend model name and
only the countable fields `model`, `messages`, `system`, `tools`
(`anthropic_endpoints/endpoints.py` `_TOKEN_COUNT_FIELDS`). The proxy adds
`"litellm_count_source": "provider"` to the answer to say the number came from the provider and not
from a local estimate.

Forwarding of thinking, tool_choice and output_config, beta merging, upstream error relay and the
model-access refusal are already pinned in `providers/anthropic/test_anthropic_count_tokens_wire.py`.
This file adds the request-validation errors, the auth forms, the response shape and the fields
that must stay out of the count request. This route stays on Python for now
(`rust_bridge/catalog.py`: `Route.TOKEN_COUNTER` is `PYTHON_ONLY`), but it sits behind the same auth
and HTTP front door that are being replaced.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.messages_endpoint.dec1_messages_support import (
    BACKEND,
    DEFAULT_ANTHROPIC_VERSION,
    JSON_OBJECT,
    UPSTREAM_KEY,
    Rig,
    anthropic_error,
    deployment,
    marker,
)
from pydantic import JsonValue

_PATH: Final = "/v1/messages/count_tokens"
_COUNT_BETA: Final = "token-counting-2024-11-01"
_TOOL: Final[dict[str, JsonValue]] = {
    "name": "lookup",
    "description": "Look up a synthetic record.",
    "input_schema": {"type": "object", "properties": {"record_id": {"type": "string"}}, "required": ["record_id"]},
}


def _counts(tokens: int) -> Callable[[Request], Reply]:
    def respond(_request: Request) -> Reply:
        return Reply(body=json.dumps({"input_tokens": tokens}).encode())

    return respond


def _count_body(rig: Rig, **fields: JsonValue) -> dict[str, JsonValue]:
    return {"model": rig.model, "messages": [{"role": "user", "content": f"count {marker()}"}], **fields}


def test_the_provider_count_is_returned_as_input_tokens_with_its_source(gateway: Gateway) -> None:
    """The provider answers `{"input_tokens": 73}`. The caller must get HTTP 200 and exactly
    `{"input_tokens": 73, "litellm_count_source": "provider"}`: the provider's number, not a local
    tokenizer estimate, and no `total_tokens` or `tokenizer_type` key from the proxy's own
    `/utils/token_counter` shape."""
    with deployment(gateway, _counts(73)) as rig:
        response: Final = rig.post(_count_body(rig), path=_PATH)
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content) == {"input_tokens": 73, "litellm_count_source": "provider"}


def test_the_count_request_goes_to_the_provider_count_tokens_path_with_the_deployment_key(gateway: Gateway) -> None:
    """One caller count must be exactly one provider `POST /v1/messages/count_tokens` (not
    `/v1/messages`), carrying the deployment key in `x-api-key`, `anthropic-version: 2023-06-01` and
    the `token-counting-2024-11-01` beta, and no `authorization` header with the caller's proxy key."""
    with deployment(gateway, _counts(5)) as rig:
        assert rig.post(_count_body(rig), path=_PATH).status_code == 200
        call: Final = rig.one_call()
        assert (call.method, call.target) == ("POST", _PATH)
        assert call.headers["x-api-key"] == UPSTREAM_KEY
        assert call.headers["anthropic-version"] == DEFAULT_ANTHROPIC_VERSION
        assert _COUNT_BETA in call.headers["anthropic-beta"]
        assert "authorization" not in call.headers


def test_the_count_request_names_the_backend_model_and_forwards_system_and_tools(gateway: Gateway) -> None:
    """The caller counts a request with a string `system` and one tool. The provider body must name
    the backend model `claude-sonnet-4-5-20250929` and carry `messages`, `system` and `tools` equal to
    what the caller sent."""
    with deployment(gateway, _counts(5)) as rig:
        body: Final = _count_body(rig, system="synthetic policy", tools=[_TOOL])
        assert rig.post(body, path=_PATH).status_code == 200
        sent: Final = rig.one_body()
        assert sent["model"] == BACKEND
        assert sent["messages"] == body["messages"]
        assert sent["system"] == "synthetic policy"
        assert sent["tools"] == [_TOOL]


def test_generation_only_fields_are_left_out_of_the_count_request(gateway: Gateway) -> None:
    """Claude Code sends the full Messages body to the count route, including `max_tokens`,
    `stream`, `temperature` and `metadata`. Anthropic's count endpoint rejects those, so the provider
    body must contain none of them, and no proxy field such as `litellm_metadata`."""
    with deployment(gateway, _counts(5)) as rig:
        body: Final = _count_body(
            rig,
            max_tokens=64,
            stream=True,
            temperature=0.2,
            metadata={"user_id": "synthetic-user"},
            litellm_metadata={"tags": ["synthetic-tag"]},
        )
        assert rig.post(body, path=_PATH).status_code == 200
        sent: Final = rig.one_body()
        leaked: Final = {"max_tokens", "stream", "temperature", "metadata", "litellm_metadata"}.intersection(sent)
        assert leaked == set(), sent


def test_a_count_without_a_model_is_refused_with_400_model_parameter_is_required(gateway: Gateway) -> None:
    """A count body with `messages` but no `model` must be refused with HTTP 400 in the Anthropic
    envelope, `error.type == "invalid_request_error"`, and a message containing
    `model parameter is required` (`endpoints.py` `count_tokens`)."""
    response: Final = gateway.request("POST", _PATH, {"messages": [{"role": "user", "content": "count me"}]})
    assert response.status_code == 400, response.text
    error: Final = anthropic_error(response)
    assert error["type"] == "invalid_request_error"
    assert "model parameter is required" in str(error["message"])


@pytest.mark.parametrize("messages", ("absent", "empty"))
def test_a_count_without_messages_is_refused_with_400_messages_parameter_is_required(
    gateway: Gateway, messages: str
) -> None:
    """A count body whose `messages` is missing or an empty list must be refused with HTTP 400 in
    the Anthropic envelope and a message containing `messages parameter is required`, and the
    provider must not be called."""
    with deployment(gateway, _counts(5)) as rig:
        body: Final[dict[str, JsonValue]] = {"model": rig.model, **({"messages": []} if messages == "empty" else {})}
        response: Final = rig.post(body, path=_PATH)
        assert response.status_code == 400, response.text
        error: Final = anthropic_error(response)
        assert error["type"] == "invalid_request_error"
        assert "messages parameter is required" in str(error["message"])
        assert rig.calls() == ()


def test_a_count_accepts_the_proxy_key_in_x_api_key(gateway: Gateway) -> None:
    """The Anthropic SDK sends its key in `x-api-key` on the count route too. A virtual key allowed
    on the model, sent only in `x-api-key`, must get HTTP 200 with the provider's count."""
    with deployment(gateway, _counts(41)) as rig:
        key: Final = rig.scenario.key(models=[rig.model])
        response: Final = rig.gateway.client.post(_PATH, json=_count_body(rig), headers={"x-api-key": key})
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["input_tokens"] == 41


def test_a_count_with_the_beta_query_parameter_is_served_the_same(gateway: Gateway) -> None:
    """Claude Code posts to `/v1/messages/count_tokens?beta=true`. It must be served like the bare
    path: HTTP 200, the provider's count, and a provider call to `/v1/messages/count_tokens` without
    a query string."""
    with deployment(gateway, _counts(12)) as rig:
        response: Final = rig.post(_count_body(rig), path=f"{_PATH}?beta=true")
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["input_tokens"] == 12
        assert rig.one_call().target == _PATH


def test_a_count_does_not_generate(gateway: Gateway) -> None:
    """Counting must never cost a generation. After a successful count the provider must have
    received no `POST /v1/messages` at all, only the single count call."""
    with deployment(gateway, _counts(5)) as rig:
        assert rig.post(_count_body(rig), path=_PATH).status_code == 200
        assert [call.target for call in rig.calls()] == [_PATH]


def test_malformed_json_on_the_count_route_is_a_400_in_the_anthropic_envelope(gateway: Gateway) -> None:
    """Truncated JSON posted to the count route must be answered HTTP 400 in the Anthropic envelope
    with `error.type == "invalid_request_error"` and a message starting with `Invalid JSON payload`,
    the same as on `/v1/messages`."""
    response: Final = gateway.client.post(
        _PATH,
        content=b'{"model": ',
        headers={"Authorization": f"Bearer {gateway.key}", "content-type": "application/json"},
    )
    assert response.status_code == 400, response.text
    error: Final = anthropic_error(response)
    assert error["type"] == "invalid_request_error"
    assert str(error["message"]).startswith("Invalid JSON payload")
