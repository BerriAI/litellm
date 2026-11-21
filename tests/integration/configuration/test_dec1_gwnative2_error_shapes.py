"""Error bodies the native Rust gateway preview writes, per route and per provider answer.

The contract, read from `litellm/rust/crates/gateway-inference/src/error.rs`:

On `POST /v1/messages` every failure is the Anthropic envelope `{"type": "error", "error": {"type", "message"}}`
(`Error::body`). A provider body that already has that shape passes through untouched, extra keys and its own
`request_id` included. Any other provider body is wrapped: its message comes from `detail.message`, then the
first non-empty of `Message` and `message`, and otherwise the raw body text. The error `type` follows the status
(`error_type`: 400 invalid_request_error, 401 authentication_error, 403 permission_error, 404 not_found_error,
413 request_too_large, 429 rate_limit_error, 529 overloaded_error, anything else api_error), and the status is
the provider's own. The caller's `x-request-id` is added as `request_id` only when non-empty and only when the
envelope has none. On `/chat/completions` the same failures use the OpenAI envelope
`{"error": {"message", "type", "param": null, "code": <int status>}}` and never carry a request id.

Python builds the same Anthropic envelope in `AnthropicExceptionMapping`, so these shapes are what SDK error
parsers already depend on. One gateway and one scripted provider serve the module.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest

from tests.integration._support.wire import Reply
from tests.integration.configuration.dec1_gwnative_support import (
    NativeGateway,
    ScriptedProvider,
    anthropic_deployment,
    bedrock_deployment,
    chat_body,
    marker,
    master_key_config,
    messages_body,
    native_gateway,
    openai_like_deployment,
    response_json,
    scripted_provider,
)


@pytest.fixture(scope="module")
def provider() -> Iterator[ScriptedProvider]:
    with scripted_provider() as scripted:
        yield scripted


@pytest.fixture(scope="module")
def gateway(provider: ScriptedProvider, tmp_path_factory: pytest.TempPathFactory) -> Iterator[NativeGateway]:
    url: Final = provider.url
    config: Final = master_key_config(
        [
            anthropic_deployment("claude", url),
            bedrock_deployment("bedrock-claude", url),
            openai_like_deployment("chat", url),
        ]
    )
    with native_gateway(config, tmp_path_factory.mktemp("gwnative2-errors")) as booted:
        yield booted


@pytest.mark.parametrize(
    ("status", "body", "error_type", "message"),
    [
        (403, b'{"detail": {"message": "dec1 denied"}}', "permission_error", "dec1 denied"),
        (429, b'{"Message": "dec1 slow down"}', "rate_limit_error", "dec1 slow down"),
        (404, b'{"message": "dec1 no such model"}', "not_found_error", "dec1 no such model"),
        (401, b'{"message": "dec1 bad provider key"}', "authentication_error", "dec1 bad provider key"),
        (413, b'{"message": "dec1 prompt too big"}', "request_too_large", "dec1 prompt too big"),
        (529, b'{"message": "dec1 overloaded"}', "overloaded_error", "dec1 overloaded"),
        (503, b'{"message": "dec1 unavailable"}', "api_error", "dec1 unavailable"),
    ],
)
def test_a_wrapped_provider_error_keeps_the_provider_status_and_maps_the_error_type(
    gateway: NativeGateway, provider: ScriptedProvider, status: int, body: bytes, error_type: str, message: str
) -> None:
    """The Anthropic deployment answers `status` with a non-Anthropic JSON body. The caller must get the same
    status and exactly `{"type": "error", "error": {"type": error_type, "message": message}}`, the message taken
    from `detail.message`, `Message` or `message` (`provider_message` and `error_type` in `error.rs:180-219`)."""
    case: Final = marker()
    provider.script(case, lambda _: Reply(status=status, body=body))
    response: Final = gateway.post("/v1/messages", messages_body(case))
    assert response.status_code == status, response.text
    assert response_json(response) == {"type": "error", "error": {"type": error_type, "message": message}}


def test_an_empty_capitalised_message_falls_through_to_the_lowercase_one(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The provider answers 400 `{"Message": "", "message": "dec1 second choice"}`. `provider_message` takes
    the first NON-EMPTY of `Message` then `message` (`error.rs:215-218`), so the caller's message must be
    `dec1 second choice` with type `invalid_request_error`."""
    case: Final = marker()
    provider.script(case, lambda _: Reply(status=400, body=b'{"Message": "", "message": "dec1 second choice"}'))
    response: Final = gateway.post("/v1/messages", messages_body(case))
    assert response.status_code == 400, response.text
    assert response_json(response)["error"] == {"type": "invalid_request_error", "message": "dec1 second choice"}


def test_a_json_provider_body_without_a_message_becomes_the_message_verbatim(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The provider answers 500 `{"error":"dec1 opaque"}` (no `detail`, `Message` or `message`). The whole raw
    body text must become the message, unchanged byte for byte, with type `api_error`
    (`provider_message(...).unwrap_or(&raw)` in `error.rs:124`)."""
    case: Final = marker()
    provider.script(case, lambda _: Reply(status=500, body=b'{"error":"dec1 opaque"}'))
    response: Final = gateway.post("/v1/messages", messages_body(case))
    assert response.status_code == 500, response.text
    assert response_json(response) == {
        "type": "error",
        "error": {"type": "api_error", "message": '{"error":"dec1 opaque"}'},
    }


def test_a_plain_text_provider_error_becomes_the_message(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """The provider answers 502 with `text/plain` body `dec1 bad gateway`. The caller must get 502 and
    `{"type": "error", "error": {"type": "api_error", "message": "dec1 bad gateway"}}` (`envelope(status, &raw)`
    when the body is not JSON, `error.rs:126`)."""
    case: Final = marker()
    provider.script(case, lambda _: Reply(status=502, body=b"dec1 bad gateway", content_type="text/plain"))
    response: Final = gateway.post("/v1/messages", messages_body(case))
    assert response.status_code == 502, response.text
    assert response_json(response) == {"type": "error", "error": {"type": "api_error", "message": "dec1 bad gateway"}}


def test_an_anthropic_provider_error_keeps_its_extra_keys(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """The provider answers 400 with an Anthropic error whose `error` object carries an extra
    `dec1_hint` key. The body must pass through unchanged, extra key included (`is_anthropic_error` only
    checks `type`, `error.type` and `error.message`, `error.rs:203-209`)."""
    case: Final = marker()
    provider.script(
        case,
        lambda _: Reply(
            status=400,
            body=b'{"type": "error", "error": {"type": "invalid_request_error", "message": "dec1 bad", "dec1_hint": 3}}',
        ),
    )
    response: Final = gateway.post("/v1/messages", messages_body(case))
    assert response.status_code == 400, response.text
    assert response_json(response) == {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "dec1 bad", "dec1_hint": 3},
    }


def test_the_providers_request_id_wins_over_the_callers(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """The provider's Anthropic error body has `request_id: req_dec1_upstream` and the caller sent
    `x-request-id: dec1-caller`. The caller must see `request_id` `req_dec1_upstream`, because `with_request_id`
    adds the caller's id only when the envelope has none (`error.rs:221-229`, unit case
    `upstream_request_id_wins`)."""
    case: Final = marker()
    provider.script(
        case,
        lambda _: Reply(
            status=429,
            body=b'{"type": "error", "error": {"type": "rate_limit_error", "message": "dec1"}, "request_id": "req_dec1_upstream"}',
        ),
    )
    response: Final = gateway.post("/v1/messages", messages_body(case), headers={"x-request-id": "dec1-caller"})
    assert response.status_code == 429, response.text
    assert response_json(response)["request_id"] == "req_dec1_upstream"


def test_an_empty_x_request_id_adds_no_request_id(gateway: NativeGateway) -> None:
    """An unknown-model call on `/v1/messages` with `x-request-id` present but empty. The envelope must have
    no `request_id` key (`Some(id) if !id.is_empty()` in `with_request_id`, `error.rs:223`)."""
    response: Final = gateway.post("/v1/messages", messages_body(marker(), "dec1-nope"), headers={"x-request-id": ""})
    assert response.status_code == 400, response.text
    assert "request_id" not in response_json(response)


def test_a_wrapped_bedrock_error_also_gets_the_callers_request_id(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The Bedrock deployment answers 400 `{"message": "dec1 validation"}` and the caller sent
    `x-request-id: dec1-bedrock-req`. The caller must get exactly the wrapped envelope plus
    `request_id: dec1-bedrock-req` (`Error::body` with `with_request_id`)."""
    case: Final = marker()
    provider.script(case, lambda _: Reply(status=400, body=b'{"message": "dec1 validation"}'))
    response: Final = gateway.post(
        "/v1/messages", messages_body(case, "bedrock-claude"), headers={"x-request-id": "dec1-bedrock-req"}
    )
    assert response.status_code == 400, response.text
    assert response_json(response) == {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "dec1 validation"},
        "request_id": "dec1-bedrock-req",
    }


def test_messages_without_a_model_say_the_model_is_required_in_the_anthropic_envelope(gateway: NativeGateway) -> None:
    """`POST /v1/messages` with messages but no `model`. The answer must be 400 with exactly
    `{"type": "error", "error": {"type": "invalid_request_error", "message": "invalid request body: model is
    required"}}` (`BodyError::MissingModel` rendered through `Report`, `error.rs:46` and `:146`)."""
    body: Final = messages_body(marker())
    del body["model"]
    response: Final = gateway.post("/v1/messages", body)
    assert response.status_code == 400, response.text
    assert response_json(response) == {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "invalid request body: model is required"},
    }


def test_messages_with_a_json_array_body_say_an_object_is_expected(gateway: NativeGateway) -> None:
    """`POST /v1/messages` with body `[1, 2]`. The answer must be 400 with message
    `invalid request body: expected a JSON object` in the Anthropic envelope (`BodyError::NotAnObject`)."""
    response: Final = gateway.post("/v1/messages", content=b"[1, 2]")
    assert response.status_code == 400, response.text
    assert response_json(response) == {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "invalid request body: expected a JSON object"},
    }


def test_messages_errors_are_served_as_application_json(gateway: NativeGateway) -> None:
    """An unknown-model call on `/v1/messages`. The answer's `content-type` must be `application/json`, since
    the handler returns `(status, Json(body))` (`messages.rs` `create`), so SDKs decode it as an error object."""
    response: Final = gateway.post("/v1/messages", messages_body(marker(), "dec1-nope"))
    assert response.headers["content-type"] == "application/json"


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (500, "api_error"),
        (503, "api_error"),
    ],
)
def test_chat_completions_map_provider_errors_to_the_openai_envelope(
    gateway: NativeGateway, provider: ScriptedProvider, status: int, error_type: str
) -> None:
    """The `openai_like` deployment answers `status` with an OpenAI error body. The caller must get the same
    status, an `error` object with `type` error_type, `param` null and integer `code` equal to the status
    (`openai_body` and `route_status` in `error.rs:101-165`). The message is the report chain, so only its
    presence is asserted."""
    case: Final = marker()
    provider.script(
        case,
        lambda _: Reply(status=status, body=b'{"error": {"message": "dec1 chat failure", "type": "x", "code": null}}'),
    )
    response: Final = gateway.post("/chat/completions", chat_body(case))
    assert response.status_code == status, response.text
    error: Final = response_json(response)["error"]
    assert isinstance(error, dict)
    assert (error["type"], error["param"], error["code"]) == (error_type, None, status)
    assert isinstance(error["message"], str)
    assert error["message"]


def test_chat_completions_errors_never_carry_a_request_id(gateway: NativeGateway) -> None:
    """An unknown-model call on `/chat/completions` with `x-request-id: dec1-chat-req`. The OpenAI envelope has
    no request id slot (`openai_body`), so the body must be exactly the one `error` key and the id must not
    appear anywhere in it."""
    response: Final = gateway.post(
        "/chat/completions", chat_body(marker(), "dec1-nope"), headers={"x-request-id": "dec1-chat-req"}
    )
    assert response.status_code == 400, response.text
    assert set(response_json(response)) == {"error"}
    assert "dec1-chat-req" not in response.text


def test_chat_completions_invalid_json_names_the_body_and_the_parser_error(gateway: NativeGateway) -> None:
    """`POST /chat/completions` with body `{"model": "chat",`. The answer must be 400, type
    `invalid_request_error`, code 400, and a message starting `invalid request body: invalid JSON: `
    (unit test `a_body_fault_names_the_body_and_keeps_its_cause_once`, `error.rs:380`)."""
    response: Final = gateway.post("/chat/completions", content=b'{"model": "chat",')
    assert response.status_code == 400, response.text
    error: Final = response_json(response)["error"]
    assert isinstance(error, dict)
    assert (error["type"], error["code"]) == ("invalid_request_error", 400)
    assert isinstance(error["message"], str)
    assert error["message"].startswith("invalid request body: invalid JSON: ")


def test_an_empty_body_on_messages_is_invalid_json(gateway: NativeGateway) -> None:
    """`POST /v1/messages` with an empty body and `content-type: application/json`. The body is read as
    bytes and parsed as JSON (`object` in `request.rs`), so the answer must be 400 `invalid_request_error` in
    the Anthropic envelope with a message starting `invalid request body: invalid JSON: `."""
    response: Final = gateway.post("/v1/messages", content=b"")
    assert response.status_code == 400, response.text
    error: Final = response_json(response)["error"]
    assert isinstance(error, dict)
    assert error["type"] == "invalid_request_error"
    assert isinstance(error["message"], str)
    assert error["message"].startswith("invalid request body: invalid JSON: ")


def test_trailing_bytes_after_the_json_object_are_refused(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """A messages body that is a valid object followed by `garbage`. `serde_json::from_slice` refuses trailing
    characters, so the answer must be 400 with an `invalid JSON` message and the provider must see nothing.
    Python's `json.loads` refuses the same body ("Extra data")."""
    case: Final = marker()
    raw: Final = (
        b'{"model": "claude", "max_tokens": 4, "messages": [{"role": "user", "content": "'
        + case.encode()
        + b'"}]} garbage'
    )
    response: Final = gateway.post("/v1/messages", content=raw)
    assert response.status_code == 400, response.text
    assert "invalid JSON" in response.text
    assert provider.requests_for(case) == ()
