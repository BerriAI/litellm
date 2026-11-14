"""How a caller authenticates to `POST /v1/messages` and what a refused caller sees.

The proxy accepts its own key in either of the two forms Anthropic clients use:
`Authorization: Bearer <key>` (Claude Code with `ANTHROPIC_AUTH_TOKEN`, most HTTP clients) and
`x-api-key: <key>` (the Anthropic SDK with `api_key`). When both are present the `Authorization`
header is the one that is checked (`user_api_key_auth.py` `get_api_key`). A refused caller always
gets Anthropic's error envelope `{"type": "error", "error": {"type": ..., "message": ...}}` because
the route is `/v1/messages` (`anthropic_error_payload.py` `uses_anthropic_error_envelope`), with the
Anthropic error type that matches the status: 401 `authentication_error`, 403 `permission_error`,
422 `invalid_request_error`. A refused request must never reach the provider.

Auth is one of the layers moving to Rust, and the Rust gateway's own auth errors are built by a
different crate, so these envelopes are the easiest thing to change by accident.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration.messages_endpoint.dec1_messages_support import (
    JSON_OBJECT,
    PRICES,
    anthropic_error,
    deployment,
    echo_message,
    marker,
)


def test_a_bearer_authorization_header_authenticates_the_caller(gateway: Gateway) -> None:
    """A virtual key allowed to call the model is sent as `Authorization: Bearer <key>` with no
    `x-api-key`. The proxy must answer 200 with the provider's message and call the provider once."""
    with deployment(gateway, echo_message) as rig:
        key: Final = rig.scenario.key(models=[rig.model])
        response: Final = rig.gateway.client.post(
            "/v1/messages", json=rig.body(), headers={"Authorization": f"Bearer {key}"}
        )
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["type"] == "message"
        assert len(rig.calls()) == 1


def test_an_x_api_key_header_authenticates_the_caller(gateway: Gateway) -> None:
    """The same virtual key sent only as `x-api-key: <key>`, the Anthropic SDK's form, with no
    `Authorization` header, must be accepted the same way: 200 and one provider call."""
    with deployment(gateway, echo_message) as rig:
        key: Final = rig.scenario.key(models=[rig.model])
        response: Final = rig.gateway.client.post("/v1/messages", json=rig.body(), headers={"x-api-key": key})
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["type"] == "message"
        assert len(rig.calls()) == 1


def test_a_bearer_prefix_inside_x_api_key_is_not_required(gateway: Gateway) -> None:
    """`x-api-key` carries the bare key. The master key sent bare in `x-api-key` must authenticate;
    the caller does not have to write `Bearer ` in that header."""
    with deployment(gateway, echo_message) as rig:
        response: Final = rig.gateway.client.post("/v1/messages", json=rig.body(), headers={"x-api-key": gateway.key})
        assert response.status_code == 200, response.text


def test_authorization_wins_when_both_headers_carry_different_keys(gateway: Gateway) -> None:
    """A valid key in `Authorization: Bearer` next to a garbage `x-api-key` must succeed, and the
    reverse (garbage Bearer, valid `x-api-key`) must be refused with 401: when both are present only
    the `Authorization` header is checked."""
    with deployment(gateway, echo_message) as rig:
        accepted: Final = rig.gateway.client.post(
            "/v1/messages",
            json=rig.body(),
            headers={"Authorization": f"Bearer {gateway.key}", "x-api-key": "sk-synthetic-not-a-key"},
        )
        assert accepted.status_code == 200, accepted.text
        refused: Final = rig.gateway.client.post(
            "/v1/messages",
            json=rig.body(),
            headers={"Authorization": "Bearer sk-synthetic-not-a-key", "x-api-key": gateway.key},
        )
        assert refused.status_code == 401, refused.text
        assert anthropic_error(refused)["type"] == "authentication_error"
        assert len(rig.calls()) == 1


def test_a_request_without_any_key_is_refused_with_401_authentication_error(gateway: Gateway) -> None:
    """A request with neither `Authorization` nor `x-api-key` must get HTTP 401 and an Anthropic
    envelope with `error.type == "authentication_error"`. The provider must not be called.
    UNVERIFIED: the exact message is expected to contain `No api key passed in`
    (`user_api_key_auth.py`), asserted here as a substring."""
    with deployment(gateway, echo_message) as rig:
        response: Final = rig.gateway.client.post("/v1/messages", json=rig.body())
        assert response.status_code == 401, response.text
        error: Final = anthropic_error(response)
        assert error["type"] == "authentication_error"
        assert "No api key passed in" in str(error["message"])
        assert rig.calls() == ()


@pytest.mark.parametrize("form", ("bearer", "x-api-key"))
def test_an_unknown_key_is_refused_with_401_authentication_error(gateway: Gateway, form: str) -> None:
    """A well-formed `sk-` key that does not exist, sent in either header form, must get HTTP 401
    with an Anthropic envelope of type `authentication_error`, and the provider must not be called.
    The response body must not echo the key that was sent."""
    with deployment(gateway, echo_message) as rig:
        unknown: Final = f"sk-{marker('unknown')}"
        headers: Final = {"Authorization": f"Bearer {unknown}"} if form == "bearer" else {"x-api-key": unknown}
        response: Final = rig.gateway.client.post("/v1/messages", json=rig.body(), headers=headers)
        assert response.status_code == 401, response.text
        assert anthropic_error(response)["type"] == "authentication_error"
        assert unknown not in response.text
        assert rig.calls() == ()


@pytest.mark.parametrize("form", ("bearer", "x-api-key"))
@pytest.mark.parametrize("stream", (False, True), ids=("non_stream", "stream"))
def test_a_key_without_access_to_the_model_is_refused_with_403_permission_error(
    gateway: Gateway, form: str, stream: bool
) -> None:
    """A virtual key restricted to another model asks for this deployment, in either header form and
    with or without `stream: true`. The answer must be HTTP 403 as JSON (not an SSE stream) with exactly
    `{"type": "error", "error": {"type": "permission_error", "message": "The requested model '<model>' is not
    available for this API key, or the model name is invalid. Check the models available to you and try again.",
    "litellm_error_type": "key_model_access_denied"}}`, and the provider must not be called."""
    with deployment(gateway, echo_message) as rig:
        key: Final = rig.scenario.key(models=["some-other-synthetic-model"])
        headers: Final = {"Authorization": f"Bearer {key}"} if form == "bearer" else {"x-api-key": key}
        response: Final = rig.gateway.client.post("/v1/messages", json=rig.body(stream=stream), headers=headers)
        assert response.status_code == 403, response.text
        assert JSON_OBJECT.validate_json(response.content) == {
            "type": "error",
            "error": {
                "type": "permission_error",
                "message": (
                    f"The requested model '{rig.model}' is not available for this API key, or the model name is "
                    "invalid. Check the models available to you and try again."
                ),
                "litellm_error_type": "key_model_access_denied",
            },
        }
        assert rig.calls() == ()


def test_a_blocked_key_is_refused_with_401_and_the_block_message(gateway: Gateway) -> None:
    """A virtual key is created, then blocked through `/key/block`. Its next `/v1/messages` call must
    get HTTP 401, type `authentication_error`, and a message containing
    `Key is blocked. Update via \\`/key/unblock\\` if you're an admin.`. The provider is not called."""
    with deployment(gateway, echo_message) as rig:
        key: Final = rig.scenario.key(models=[rig.model])
        gateway.post("/key/block", {"key": key})
        response: Final = eventually(
            lambda: rig.post(rig.body(), key=key),
            lambda answer: answer.status_code == 401,
            seconds=30,
        )
        error: Final = anthropic_error(response)
        assert error["type"] == "authentication_error"
        assert "Key is blocked. Update via `/key/unblock` if you're an admin." in str(error["message"])


def test_a_key_over_its_budget_is_refused_with_422_in_the_anthropic_envelope(gateway: Gateway) -> None:
    """A virtual key with a `max_budget` far below the price of one call makes a first priced call,
    which succeeds. Once that spend is recorded, the next call must be refused before the provider
    with HTTP 422 (`litellm.budget_exceeded_status_code`), an Anthropic envelope of type
    `invalid_request_error` and `litellm_error_type == "budget_exceeded"`. The test polls with fresh
    prompts until the budget takes effect, since spend is written asynchronously."""
    with deployment(gateway, echo_message, **PRICES) as rig:
        key: Final = rig.scenario.key(models=[rig.model], max_budget=0.000001)
        assert rig.post(rig.body(), key=key).status_code == 200
        refused: Final = eventually(
            lambda: rig.post(rig.body(), key=key),
            lambda answer: answer.status_code != 200,
            seconds=90,
        )
        assert refused.status_code == 422, refused.text
        error: Final = anthropic_error(refused)
        assert error["type"] == "invalid_request_error"
        assert error["litellm_error_type"] == "budget_exceeded"


def test_an_auth_failure_echoes_the_caller_x_request_id_as_request_id(gateway: Gateway) -> None:
    """Anthropic error bodies carry a top-level `request_id`. When the caller sends `x-request-id`, an
    auth refusal on this route must include `"request_id": "<that value>"` next to `type` and `error`
    (`anthropic_error_payload.py` reads the header), and must omit the key when no header was sent."""
    with deployment(gateway, echo_message) as rig:
        request_id: Final = f"req_{marker()}"
        with_id: Final = rig.gateway.client.post(
            "/v1/messages",
            json=rig.body(),
            headers={"Authorization": "Bearer sk-synthetic-not-a-key", "x-request-id": request_id},
        )
        assert with_id.status_code == 401, with_id.text
        assert JSON_OBJECT.validate_json(with_id.content)["request_id"] == request_id
        without_id: Final = rig.gateway.client.post(
            "/v1/messages", json=rig.body(), headers={"Authorization": "Bearer sk-synthetic-not-a-key"}
        )
        assert "request_id" not in JSON_OBJECT.validate_json(without_id.content)


def test_the_same_key_is_refused_the_same_way_on_count_tokens(gateway: Gateway) -> None:
    """`/v1/messages/count_tokens` shares the Anthropic envelope. An unknown key there must get HTTP 401
    with `error.type == "authentication_error"`, exactly like `/v1/messages`, and no provider call."""
    with deployment(gateway, echo_message) as rig:
        response: Final = rig.gateway.client.post(
            "/v1/messages/count_tokens",
            json={"model": rig.model, "messages": [{"role": "user", "content": "count me"}]},
            headers={"x-api-key": f"sk-{marker('unknown')}"},
        )
        assert response.status_code == 401, response.text
        assert anthropic_error(response)["type"] == "authentication_error"
        assert rig.calls() == ()
