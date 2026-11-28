"""Headers the proxy sends to Anthropic for a `/v1/messages` call.

The caller authenticates to the proxy with a proxy key. The provider must be called with the
DEPLOYMENT's key in `x-api-key`, never with the caller's proxy key, and with an `anthropic-version`
header: the caller's own value when one was sent, otherwise `2023-06-01`
(`transformation.py` `DEFAULT_ANTHROPIC_API_VERSION`). `anthropic-beta` values the caller sent are
forwarded, and the proxy adds the beta a request field needs (structured outputs, context
management). The merged list is de-duplicated and sorted alphabetically
(`_update_headers_with_anthropic_beta`). On the Rust route the same two caller headers are the only
ones forwarded (`gateway-inference/src/messages.rs` `ANTHROPIC_API_HEADERS`).

Each test reads the headers the owned TCP peer received. Header names arrive lower-cased.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration.messages_endpoint.dec1_messages_support import (
    DEFAULT_ANTHROPIC_VERSION,
    JSON_OBJECT,
    NO_CACHE,
    UPSTREAM_KEY,
    deployment,
    echo_message,
    marker,
)
from pydantic import JsonValue

_CLIENT_BETA: Final = "client-future-beta-2099-01-01"
_SECOND_BETA: Final = "another-future-beta-2099-02-02"
_STRUCTURED_OUTPUTS_BETA: Final = "structured-outputs-2025-11-13"
_CONTEXT_MANAGEMENT_BETA: Final = "context-management-2025-06-27"
_OUTPUT_FORMAT: Final[dict[str, JsonValue]] = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    },
}


def _betas(value: str) -> list[str]:
    return sorted(piece.strip() for piece in value.split(",") if piece.strip())


def test_the_deployment_key_is_sent_as_x_api_key(gateway: Gateway) -> None:
    """The deployment was created with `api_key: synthetic-anthropic-key`. The provider request must
    carry exactly that value in `x-api-key`, whatever key the caller used against the proxy."""
    with deployment(gateway, echo_message) as rig:
        assert rig.post(rig.body()).status_code == 200
        assert rig.one_call().headers["x-api-key"] == UPSTREAM_KEY


def test_the_caller_bearer_proxy_key_is_not_forwarded_to_the_provider(gateway: Gateway) -> None:
    """The caller authenticates with `Authorization: Bearer <proxy key>`. The provider request must
    have no `authorization` header at all and the proxy key must not appear in any header value.
    Forwarding it would leak a proxy credential to a third party."""
    with deployment(gateway, echo_message) as rig:
        assert rig.post(rig.body()).status_code == 200
        headers: Final = rig.one_call().headers
        assert "authorization" not in headers, headers
        assert all(gateway.key not in value for value in headers.values()), headers


def test_the_caller_x_api_key_proxy_key_is_replaced_by_the_deployment_key(gateway: Gateway) -> None:
    """The caller authenticates the way the Anthropic SDK does, with the proxy key in `x-api-key`.
    Upstream, `x-api-key` must hold the deployment key and the proxy key must appear in no header."""
    with deployment(gateway, echo_message) as rig:
        response: Final = rig.gateway.client.post(
            "/v1/messages", json=rig.body(), headers={"x-api-key": gateway.key, "anthropic-version": "2023-06-01"}
        )
        assert response.status_code == 200, response.text
        headers: Final = rig.one_call().headers
        assert headers["x-api-key"] == UPSTREAM_KEY
        assert all(gateway.key not in value for value in headers.values()), headers


def test_anthropic_version_defaults_to_2023_06_01_when_the_caller_sends_none(gateway: Gateway) -> None:
    """A caller that sends no `anthropic-version` header (a plain HTTP client) must still produce a
    provider request with `anthropic-version: 2023-06-01`, because Anthropic rejects a request without it."""
    with deployment(gateway, echo_message) as rig:
        assert rig.post(rig.body()).status_code == 200
        assert rig.one_call().headers["anthropic-version"] == DEFAULT_ANTHROPIC_VERSION


def test_the_caller_anthropic_version_is_forwarded_instead_of_the_default(gateway: Gateway) -> None:
    """A caller that sends `anthropic-version: 2099-12-31` must have exactly that value forwarded.
    The default `2023-06-01` must not replace it or be appended to it."""
    with deployment(gateway, echo_message) as rig:
        assert rig.post(rig.body(), headers={"anthropic-version": "2099-12-31"}).status_code == 200
        assert rig.one_call().headers["anthropic-version"] == "2099-12-31"


def test_no_anthropic_beta_header_is_sent_for_a_plain_request(gateway: Gateway) -> None:
    """A request with no beta header and no beta-gated field must reach the provider without any
    `anthropic-beta` header. An empty or invented beta value makes Anthropic answer 400."""
    with deployment(gateway, echo_message) as rig:
        assert rig.post(rig.body()).status_code == 200
        assert "anthropic-beta" not in rig.one_call().headers


def test_a_caller_anthropic_beta_value_is_forwarded(gateway: Gateway) -> None:
    """A caller sending `anthropic-beta: client-future-beta-2099-01-01`, a value the proxy has never
    heard of, must have it forwarded to Anthropic unchanged. The first-party Anthropic route does not
    filter unknown betas (`should_filter_anthropic_beta_headers` is false for `anthropic`)."""
    with deployment(gateway, echo_message) as rig:
        assert rig.post(rig.body(), headers={"anthropic-beta": _CLIENT_BETA}).status_code == 200
        assert _betas(rig.one_call().headers["anthropic-beta"]) == [_CLIENT_BETA]


def test_two_caller_betas_in_one_header_are_both_forwarded(gateway: Gateway) -> None:
    """A caller sending two comma-separated betas in one `anthropic-beta` header must have both
    forwarded in a single header, each exactly once."""
    with deployment(gateway, echo_message) as rig:
        sent: Final = f"{_CLIENT_BETA}, {_SECOND_BETA}"
        assert rig.post(rig.body(), headers={"anthropic-beta": sent}).status_code == 200
        assert _betas(rig.one_call().headers["anthropic-beta"]) == sorted([_CLIENT_BETA, _SECOND_BETA])


def test_output_format_adds_the_structured_outputs_beta(gateway: Gateway) -> None:
    """A request with a top-level `output_format` JSON schema and no beta header must reach the
    provider with `anthropic-beta: structured-outputs-2025-11-13`, added by the proxy, and with
    `output_format` unchanged in the body."""
    with deployment(gateway, echo_message) as rig:
        assert rig.post(rig.body(output_format=_OUTPUT_FORMAT)).status_code == 200
        call: Final = rig.one_call()
        assert _betas(call.headers["anthropic-beta"]) == [_STRUCTURED_OUTPUTS_BETA]
        assert JSON_OBJECT.validate_json(call.body)["output_format"] == _OUTPUT_FORMAT


def test_output_config_format_adds_the_structured_outputs_beta(gateway: Gateway) -> None:
    """The newer spelling, `output_config: {"format": <schema>}`, must add the same
    `structured-outputs-2025-11-13` beta as the older top-level `output_format`."""
    with deployment(gateway, echo_message) as rig:
        assert rig.post(rig.body(output_config={"format": _OUTPUT_FORMAT})).status_code == 200
        assert _betas(rig.one_call().headers["anthropic-beta"]) == [_STRUCTURED_OUTPUTS_BETA]


def test_a_derived_beta_is_merged_with_the_caller_betas_sorted_and_deduplicated(gateway: Gateway) -> None:
    """The caller sends its own beta plus the structured-outputs beta, and the body has `output_format`.
    The provider must receive one `anthropic-beta` header holding the two distinct values in
    alphabetical order joined by commas, with the structured-outputs value present once."""
    with deployment(gateway, echo_message) as rig:
        sent: Final = f"{_STRUCTURED_OUTPUTS_BETA},{_CLIENT_BETA}"
        assert rig.post(rig.body(output_format=_OUTPUT_FORMAT), headers={"anthropic-beta": sent}).status_code == 200
        assert rig.one_call().headers["anthropic-beta"] == f"{_CLIENT_BETA},{_STRUCTURED_OUTPUTS_BETA}"


def test_context_management_edits_add_the_context_management_beta(gateway: Gateway) -> None:
    """A request with `context_management.edits` holding a `clear_tool_uses_20250919` edit must reach
    the provider with `anthropic-beta: context-management-2025-06-27` and the `context_management`
    object unchanged in the body."""
    context_management: Final[dict[str, JsonValue]] = {"edits": [{"type": "clear_tool_uses_20250919"}]}
    with deployment(gateway, echo_message) as rig:
        assert rig.post(rig.body(context_management=context_management)).status_code == 200
        call: Final = rig.one_call()
        assert _betas(call.headers["anthropic-beta"]) == [_CONTEXT_MANAGEMENT_BETA]
        assert JSON_OBJECT.validate_json(call.body)["context_management"] == context_management


def test_the_provider_request_is_json(gateway: Gateway) -> None:
    """The provider request must declare `content-type: application/json` and its body must be a
    JSON object, whatever content-type spelling the caller used."""
    with deployment(gateway, echo_message) as rig:
        response: Final = rig.post(rig.body(), headers={"content-type": "application/json; charset=utf-8"})
        assert response.status_code == 200, response.text
        assert rig.one_call().headers["content-type"].startswith("application/json")


def test_proxy_control_headers_are_not_forwarded_to_the_provider(gateway: Gateway) -> None:
    """The caller sends proxy-only headers: `x-litellm-call-id`, `x-litellm-tags` and
    `x-litellm-api-key`-style bookkeeping. None of the `x-litellm-*` headers may reach Anthropic."""
    with deployment(gateway, echo_message) as rig:
        response: Final = rig.post(
            rig.body(), headers={"x-litellm-call-id": marker("call"), "x-litellm-tags": "synthetic-tag"}
        )
        assert response.status_code == 200, response.text
        forwarded: Final = sorted(name for name in rig.one_call().headers if name.startswith("x-litellm"))
        assert forwarded == []


def test_arbitrary_caller_headers_are_not_forwarded_by_default(gateway: Gateway) -> None:
    """Without `forward_client_headers_to_llm_api`, a caller header the proxy does not know
    (`x-synthetic-caller-header`) and the caller's `cookie` must not reach the provider. Only the two
    Anthropic API headers are forwarded by default."""
    with deployment(gateway, echo_message) as rig:
        response: Final = rig.post(
            rig.body(), headers={"x-synthetic-caller-header": "caller-value", "cookie": "session=synthetic"}
        )
        assert response.status_code == 200, response.text
        headers: Final = rig.one_call().headers
        assert "x-synthetic-caller-header" not in headers, headers
        assert "cookie" not in headers, headers


def test_the_beta_query_parameter_is_accepted_and_not_forwarded(gateway: Gateway) -> None:
    """The Anthropic SDK's beta namespace posts to `/v1/messages?beta=true`. The proxy must serve it
    like `/v1/messages` and call the provider at `/v1/messages` with no query string."""
    with deployment(gateway, echo_message) as rig:
        response: Final = rig.post(rig.body(), path="/v1/messages?beta=true")
        assert response.status_code == 200, response.text
        assert rig.one_call().target == "/v1/messages"


@pytest.mark.parametrize("uncached", (False, True), ids=("cache_default", "no_cache"))
def test_version_and_beta_forwarding_is_the_same_with_and_without_the_response_cache(
    gateway: Gateway, uncached: bool
) -> None:
    """The response-cache path is Python-only while a `no-cache` call can take the Rust route. Either
    way a caller's `anthropic-version: 2099-12-31` and `anthropic-beta` must arrive unchanged and the
    deployment key must be in `x-api-key`."""
    with deployment(gateway, echo_message) as rig:
        response: Final = rig.post(
            rig.body(**(NO_CACHE if uncached else {})),
            headers={"anthropic-version": "2099-12-31", "anthropic-beta": _CLIENT_BETA},
        )
        assert response.status_code == 200, response.text
        headers: Final = rig.one_call().headers
        assert headers["anthropic-version"] == "2099-12-31"
        assert _betas(headers["anthropic-beta"]) == [_CLIENT_BETA]
        assert headers["x-api-key"] == UPSTREAM_KEY
