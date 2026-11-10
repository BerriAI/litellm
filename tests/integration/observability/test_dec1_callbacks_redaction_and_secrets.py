"""What callbacks must not receive: message content when logging is turned off, and credentials always.

Message redaction is read from the generic_api payload. With `litellm_settings.turn_off_message_logging: true`
every payload's `messages` is replaced by one placeholder user message and the assistant text in `response` by
the placeholder "redacted-by-litellm", while the caller still gets the real reply and the payload keeps its
tokens, cost and identity fields. A caller can ask for redaction on one request (body field or header), but can
only opt out of the admin's global redaction when the key or team carries
`allow_client_message_redaction_opt_out: true` (litellm/proxy/litellm_pre_call_utils.py and
litellm/litellm_core_utils/redact_messages.py).

The second half pins that no payload and no custom-logger record ever contains the provider api key, the master
key, the caller's virtual key or the callback sink's own bearer token, and that callback credentials a caller
puts in the request body are neither forwarded to the provider nor logged.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import JsonValue, list_value, object_value
from integration.observability.dec1_callbacks_support import (
    CHAT_COST,
    CHAT_REJECTED,
    CHAT_USAGE,
    EMBEDDING,
    MESSAGES,
    PROVIDER_KEY,
    PROVIDER_REJECTION,
    RECORDER,
    REPLY_TEXT,
    RESPONSES,
    SINK_SECRET,
    CallbackRig,
    callback_rig,
    chat,
    key_hash,
    marker,
    new_call_id,
    post,
    provider_bodies,
    stream_chat,
    streamed_text,
)

from litellm.constants import REDACTED_BY_LITELLM

pytestmark: Final = [pytest.mark.timeout(240), pytest.mark.shares_worker_state]

REDACTED_MESSAGES: Final = [{"role": "user", "content": REDACTED_BY_LITELLM}]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CallbackRig]:
    with callback_rig(
        tmp_path_factory.mktemp("dec1-callbacks-secrets"), {"callbacks": ["generic_api", RECORDER]}
    ) as created:
        yield created


@pytest.fixture(scope="module")
def redacting(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CallbackRig]:
    with callback_rig(
        tmp_path_factory.mktemp("dec1-callbacks-redacting"),
        {"callbacks": ["generic_api", RECORDER], "turn_off_message_logging": True},
    ) as created:
        yield created


def _logged_reply(event: dict[str, JsonValue]) -> JsonValue:
    choice: Final = object_value(list_value(object_value(event["response"])["choices"])[0])
    return object_value(choice["message"])["content"]


def test_global_redaction_replaces_messages_and_reply_in_the_payload(redacting: CallbackRig) -> None:
    """With `turn_off_message_logging: true` a caller sends a chat with a unique text. The caller still receives
    the provider's real reply, but the payload's `messages` must be exactly one user message whose content is
    "redacted-by-litellm", the logged assistant content must be "redacted-by-litellm", and the unique text must
    appear nowhere in the payload."""
    call_id: Final = new_call_id()
    text: Final = marker()
    response: Final = chat(redacting, call_id, text)
    assert response.json()["choices"][0]["message"]["content"] == REPLY_TEXT
    event: Final = redacting.event(call_id)
    assert event["messages"] == REDACTED_MESSAGES
    assert _logged_reply(event) == REDACTED_BY_LITELLM
    assert text not in json.dumps(event) and REPLY_TEXT not in json.dumps(event)


def test_global_redaction_keeps_usage_cost_and_identity(redacting: CallbackRig) -> None:
    """Redaction removes content only. The redacted payload must still report tokens 11/4/15, response_cost
    0.019, the model group, the key's hash and alias, and the caller's request tag."""
    call_id: Final = new_call_id()
    alias, tag = "alias-" + marker(), marker()
    with redacting.proxy.scenario() as scenario:
        key: Final = scenario.key(key_alias=alias)
        assert chat(redacting, call_id, key=key, metadata={"tags": [tag]}).status_code == 200
        event: Final = redacting.event(call_id)
        metadata: Final = object_value(event["metadata"])
        assert (event["prompt_tokens"], event["completion_tokens"], event["total_tokens"]) == (*CHAT_USAGE, 15)
        assert event["response_cost"] == pytest.approx(CHAT_COST)
        assert (metadata["user_api_key_hash"], metadata["user_api_key_alias"]) == (key_hash(key), alias)
        assert tag in list_value(event["request_tags"])


def test_global_redaction_applies_to_streamed_responses(redacting: CallbackRig) -> None:
    """A streamed chat under global redaction: the caller receives the full reply text over SSE, and the single
    payload logged after the stream has redacted `messages` and a redacted assistant content."""
    call_id: Final = new_call_id()
    status, _, sse = stream_chat(redacting, call_id, marker())
    assert status == 200 and streamed_text(sse) == REPLY_TEXT
    event: Final = redacting.event(call_id)
    assert event["messages"] == REDACTED_MESSAGES
    assert REPLY_TEXT not in json.dumps(event["response"])


def test_global_redaction_applies_to_failure_payloads_but_keeps_the_error(redacting: CallbackRig) -> None:
    """A provider-rejected chat under global redaction: the failure payload's `messages` are redacted and the
    caller's text appears nowhere in it, while `error_information` still carries error_code "400" and the
    provider's message."""
    call_id: Final = new_call_id()
    text: Final = marker()
    assert chat(redacting, call_id, text, model=CHAT_REJECTED).status_code == 400
    event: Final = redacting.event(call_id, "failure")
    information: Final = object_value(event["error_information"])
    assert text not in json.dumps(event)
    assert information["error_code"] == "400" and PROVIDER_REJECTION in str(information["error_message"])


@pytest.mark.parametrize(
    ("route", "field"),
    [
        pytest.param("/v1/embeddings", "input", id="embeddings"),
        pytest.param("/v1/responses", "input", id="responses"),
        pytest.param("/v1/messages", "messages", id="messages"),
    ],
)
def test_global_redaction_hides_the_input_on_every_surface(redacting: CallbackRig, route: str, field: str) -> None:
    """Under global redaction an embeddings, /v1/responses or /v1/messages request carrying a unique text must
    produce a success payload in which that text appears nowhere."""
    call_id: Final = new_call_id()
    text: Final = marker()
    bodies: Final[dict[str, dict[str, JsonValue]]] = {
        "/v1/embeddings": {"model": EMBEDDING, "input": text},
        "/v1/responses": {"model": RESPONSES, "input": text},
        "/v1/messages": {"model": MESSAGES, "max_tokens": 16, "messages": [{"role": "user", "content": text}]},
    }
    assert field in bodies[route]
    assert post(redacting, route, bodies[route], call_id).status_code == 200
    assert text not in json.dumps(redacting.event(call_id, "success"))


def test_caller_cannot_opt_out_of_global_redaction_with_a_body_field(redacting: CallbackRig) -> None:
    """Under global redaction a caller with an ordinary key sends `turn_off_message_logging: false` in the body.
    The proxy strips that opt-out, so the payload must still be redacted and must not contain the caller's text."""
    call_id: Final = new_call_id()
    text: Final = marker()
    assert chat(redacting, call_id, text, turn_off_message_logging=False).status_code == 200
    event: Final = redacting.event(call_id)
    assert event["messages"] == REDACTED_MESSAGES
    assert text not in json.dumps(event)


def test_caller_cannot_opt_out_of_global_redaction_with_a_header(redacting: CallbackRig) -> None:
    """Under global redaction a caller with an ordinary key sends the header
    `litellm-disable-message-redaction: true`. The header is dropped for keys without the opt-out permission, so
    the payload must still be redacted."""
    call_id: Final = new_call_id()
    text: Final = marker()
    assert chat(redacting, call_id, text, headers={"litellm-disable-message-redaction": "true"}).status_code == 200
    assert text not in json.dumps(redacting.event(call_id))


def test_key_with_opt_out_permission_can_turn_global_redaction_off(redacting: CallbackRig) -> None:
    """A key whose metadata has `allow_client_message_redaction_opt_out: true` sends
    `litellm-disable-message-redaction: true`. For that request the payload must carry the caller's real message
    text and the provider's real reply, even though the proxy redacts globally."""
    call_id: Final = new_call_id()
    text: Final = marker()
    with redacting.proxy.scenario() as scenario:
        key: Final = scenario.key(metadata={"allow_client_message_redaction_opt_out": True})
        response: Final = chat(redacting, call_id, text, key=key, headers={"litellm-disable-message-redaction": "true"})
        assert response.status_code == 200, response.text
        event: Final = redacting.event(call_id)
        assert event["messages"] == [{"role": "user", "content": text}]
        assert _logged_reply(event) == REPLY_TEXT


def test_caller_can_request_redaction_for_one_request_with_a_body_field(rig: CallbackRig) -> None:
    """On a proxy that does not redact globally, a caller sends `turn_off_message_logging: true` in the body. That
    request's payload must be redacted, the field must not be forwarded to the provider, and the next request
    without the field must be logged in full."""
    redacted_call, plain_call = new_call_id(), new_call_id()
    hidden, shown = marker(), marker()
    assert chat(rig, redacted_call, hidden, turn_off_message_logging=True).status_code == 200
    assert chat(rig, plain_call, shown).status_code == 200
    assert hidden not in json.dumps(rig.event(redacted_call))
    assert rig.event(plain_call)["messages"] == [{"role": "user", "content": shown}]
    sent: Final = tuple(body for body in provider_bodies(rig) if hidden in json.dumps(body))
    assert len(sent) == 1 and "turn_off_message_logging" not in sent[0], sent


def test_caller_can_request_redaction_for_one_request_with_a_header(rig: CallbackRig) -> None:
    """On a proxy that does not redact globally, a caller sends the header
    `x-litellm-enable-message-redaction: true`. That request's payload must have redacted `messages` and a
    redacted assistant content."""
    call_id: Final = new_call_id()
    text: Final = marker()
    assert chat(rig, call_id, text, headers={"x-litellm-enable-message-redaction": "true"}).status_code == 200
    event: Final = rig.event(call_id)
    assert event["messages"] == REDACTED_MESSAGES
    assert _logged_reply(event) == REDACTED_BY_LITELLM


def test_redact_user_api_key_info_does_not_strip_the_standard_payload(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """`litellm_settings.redact_user_api_key_info: true` removes user_api_key_* metadata for the integrations that
    call the redaction helper (Langfuse, Langsmith, Logfire, OTel). The StandardLoggingPayload delivered to
    generic_api and to custom loggers is not passed through it, so its `metadata.user_api_key_hash` and
    `metadata.user_api_key_alias` must still identify the key. This pins today's behaviour; if the rewrite starts
    stripping them, consumers that bill by key from the generic payload break."""
    with (
        callback_rig(
            tmp_path_factory.mktemp("dec1-callbacks-redact-key-info"),
            {"callbacks": ["generic_api", RECORDER], "redact_user_api_key_info": True},
        ) as rig,
        rig.proxy.scenario() as scenario,
    ):
        call_id: Final = new_call_id()
        alias: Final = "alias-" + marker()
        key: Final = scenario.key(key_alias=alias)
        assert chat(rig, call_id, key=key).status_code == 200
        metadata: Final = object_value(rig.event(call_id)["metadata"])
        assert (metadata["user_api_key_hash"], metadata["user_api_key_alias"]) == (key_hash(key), alias)


def test_no_credential_appears_in_any_payload_or_custom_logger_record(rig: CallbackRig) -> None:
    """One virtual key makes a successful chat, a provider-rejected chat, a streamed chat, an embedding, a
    /v1/messages call and a /v1/responses call. Across every byte the generic_api endpoint received and every
    record the custom logger reported (including its copy of the StandardLoggingPayload), none of these may
    appear: the provider api key configured on the deployments, the proxy master key, the caller's virtual key,
    and the generic_api sink's own bearer secret."""
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        calls: Final = tuple(new_call_id() for _ in range(6))
        assert chat(rig, calls[0], key=key).status_code == 200
        assert chat(rig, calls[1], key=key, model=CHAT_REJECTED).status_code == 400
        assert stream_chat(rig, calls[2], key=key)[0] == 200
        assert (
            post(rig, "/v1/embeddings", {"model": EMBEDDING, "input": "secrets"}, calls[3], key=key).status_code == 200
        )
        messages_body: Final[dict[str, JsonValue]] = {
            "model": MESSAGES,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "secrets"}],
        }
        assert post(rig, "/v1/messages", messages_body, calls[4], key=key).status_code == 200
        assert (
            post(rig, "/v1/responses", {"model": RESPONSES, "input": "secrets"}, calls[5], key=key).status_code == 200
        )
        for call_id in calls:
            rig.event(call_id)
        delivered: Final = rig.sink.bodies() + rig.probe.bodies()
        for name, secret in (
            ("provider api key", PROVIDER_KEY),
            ("master key", rig.proxy.key),
            ("virtual key", key),
            ("sink secret", SINK_SECRET),
        ):
            assert secret.encode() not in delivered, name


def test_requester_custom_headers_never_include_authorization(rig: CallbackRig) -> None:
    """The caller sends an Authorization header (its key) and a custom `x-dec1-trace` header. Whatever the payload
    records under `metadata.requester_custom_headers`, it must not contain an authorization entry or the key."""
    call_id: Final = new_call_id()
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        assert chat(rig, call_id, key=key, headers={"x-dec1-trace": "dec1-trace-value"}).status_code == 200
        headers: Final = object_value(rig.event(call_id)["metadata"]).get("requester_custom_headers") or {}
        assert "authorization" not in {str(name).lower() for name in object_value(headers)}
        assert key not in json.dumps(headers)


def test_request_supplied_langfuse_credentials_are_not_forwarded_or_logged(rig: CallbackRig) -> None:
    """A caller puts `langfuse_public_key`, `langfuse_secret_key` and `langfuse_host` in the chat body on a proxy
    with no Langfuse callback. The request answers 200, the provider's request body contains none of the three
    fields, and the secret value appears nowhere in the generic_api payload for the call."""
    call_id: Final = new_call_id()
    text, secret = marker(), "sk-lf-dec1-" + marker()
    response: Final = chat(
        rig,
        call_id,
        text,
        langfuse_public_key="pk-lf-dec1",
        langfuse_secret_key=secret,
        langfuse_host="http://127.0.0.1:9",
    )
    assert response.status_code == 200, response.text
    sent: Final = tuple(body for body in provider_bodies(rig) if text in json.dumps(body))
    assert len(sent) == 1
    assert {"langfuse_public_key", "langfuse_secret_key", "langfuse_host"} & sent[0].keys() == set()
    assert secret not in json.dumps(rig.event(call_id))


def test_request_supplied_blocked_callback_destinations_are_dropped(rig: CallbackRig) -> None:
    """A caller puts `dd_api_key`, `dd_site` and `gcs_bucket_name` in the chat body and in `metadata`. These are on
    the request-blocked list, so the request answers 200, none of them reach the provider, and none of their
    values appear in the payload for the call."""
    call_id: Final = new_call_id()
    text, secret = marker(), "dd-dec1-" + marker()
    fields: Final[dict[str, JsonValue]] = {"dd_api_key": secret, "dd_site": "dec1.invalid", "gcs_bucket_name": secret}
    response: Final = chat(rig, call_id, text, metadata=dict(fields), **fields)
    assert response.status_code == 200, response.text
    sent: Final = tuple(body for body in provider_bodies(rig) if text in json.dumps(body))
    assert len(sent) == 1 and fields.keys() & sent[0].keys() == set(), sent
    assert secret not in json.dumps(rig.event(call_id))


def test_env_reference_in_a_request_callback_param_is_refused(rig: CallbackRig) -> None:
    """A caller sends `langfuse_secret_key: "os.environ/LITELLM_MASTER_KEY"` hoping the proxy resolves it. The
    proxy must not answer 200 and must not call the provider, and neither the response nor any payload for the
    call may contain the master key. UNVERIFIED: the exact status (the ValueError names the param and says
    environment references are no longer resolved) was not traced to an HTTP code."""
    call_id: Final = new_call_id()
    text: Final = marker()
    response: Final = chat(rig, call_id, text, langfuse_secret_key="os.environ/LITELLM_MASTER_KEY")
    assert response.status_code >= 400, response.text
    assert rig.proxy.key not in response.text
    assert all(text not in json.dumps(body) for body in provider_bodies(rig))
    assert all(rig.proxy.key not in json.dumps(event) for event in rig.events(call_id))


def test_error_payload_for_a_bad_key_does_not_echo_the_authorization_header(rig: CallbackRig) -> None:
    """A caller sends an unknown key. The 401 failure payload, including its error message and traceback fields,
    must not contain the presented key string, and neither may the custom logger's record of the failure."""
    call_id: Final = new_call_id()
    bad_key: Final = "sk-dec1-secret-" + marker()
    assert chat(rig, call_id, key=bad_key).status_code == 401
    rig.event(call_id, "failure")
    rig.hook(call_id, "async_log_failure_event")
    assert bad_key.encode() not in rig.sink.bodies() + rig.probe.bodies()
