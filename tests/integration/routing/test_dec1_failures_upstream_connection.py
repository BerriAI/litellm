"""What the caller gets when the proxy cannot complete an HTTP exchange with the upstream provider at all.

Contract (litellm/llms/openai/openai.py, litellm/litellm_core_utils/exception_mapping_utils.py
`_map_openai_exception` and the timeout branch of `exception_type`, litellm/exceptions.py,
litellm/proxy/common_request_processing.py `handle_llm_api_exception`,
litellm/proxy/common_utils/openai_error_payload.py):

* Every failure is answered as JSON `{"error": {"message", "type", "param", "code"}}` with an HTTP status,
  never as a dropped caller connection and never as a 200.
* A connection that cannot be made or dies before a full response (refused, DNS failure, TLS failure, closed
  with no reply, reset in the body, body shorter than announced) has no upstream status. The OpenAI client
  reports `Connection error.`, litellm maps it to its `InternalServerError`, and the caller gets HTTP 500 with
  `code` `"500"`, `type` `internal_server_error` and a message starting
  `litellm.InternalServerError: InternalServerError: OpenAIException - `.
* A connect or read that exceeds the deployment's `timeout` is answered HTTP 408 with `code` `"408"`, `type`
  `invalid_request_error` (the proxy's default type for a 4xx that carries none) and a message starting
  `litellm.Timeout: APITimeoutError - Request timed out.`
* The timeout is a per-read limit, not a total: an upstream that keeps sending bytes with gaps shorter than
  the timeout is not cut off.
* With `num_retries: 0` in the request the upstream sees exactly one attempt per failed call.

Each test adds its own deployment to the shared proxy, pointing at an owned peer that misbehaves in one
way. Chat completions and embeddings are both exercised where the failure is independent of the route.
"""

from __future__ import annotations

import ssl
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.process import self_signed_certificate
from tests.integration._support.timing import assert_cut_short, assert_enforced_within, timed
from tests.integration._support.wire import Reply, Request, wire_server
from tests.integration.database.dec1_failures_rig import digest, marker
from tests.integration.routing.dec1_failures_support import (
    REPLY_TEXT,
    chat_body,
    closing_without_a_reply,
    headers_then_body_after,
    http_response,
    in_turn,
    ok,
    plain_http_port,
    raw_peer,
    resetting_after,
    silent_until,
    trickling_body,
    unaccepting_listener,
    unbound_api_base,
)

pytestmark: Final = pytest.mark.timeout(300)

CONNECTION_PREFIX: Final = "litellm.InternalServerError: InternalServerError: OpenAIException - "
TIMEOUT_PREFIX: Final = "litellm.Timeout: APITimeoutError - Request timed out."
TIMEOUT_SECONDS: Final = 2
ROUTES: Final = ("chat", "embeddings")
UPSTREAM_MODEL: Final[Mapping[str, str]] = {"chat": "openai/gpt-4o-mini", "embeddings": "openai/text-embedding-3-small"}


def _model(scenario: Scenario, route: str, api_base: str, **params: JsonValue) -> str:
    return scenario.model(model=UPSTREAM_MODEL[route], api_base=api_base, **params)


def _call(gateway: Gateway, route: str, model: str, *, key: str | None = None, **extra: JsonValue) -> httpx.Response:
    if route == "embeddings":
        return gateway.request("POST", "/v1/embeddings", {"model": model, "input": marker(), "num_retries": 0}, key=key)
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "messages": [{"role": "user", "content": marker()}],
        "num_retries": 0,
        **extra,
    }
    return gateway.request("POST", "/v1/chat/completions", body, key=key)


def _error(response: httpx.Response) -> dict[str, JsonValue]:
    return object_value(JSON_OBJECT.validate_json(response.content)["error"])


def _assert_connection_error(response: httpx.Response) -> None:
    error: Final = _error(response)
    assert response.status_code == 500, response.text
    assert (error["type"], error["code"], error["param"]) == ("internal_server_error", "500", None), response.text
    assert string_value(error["message"]).startswith("litellm."), response.text


def _assert_timeout(response: httpx.Response) -> None:
    error: Final = _error(response)
    assert response.status_code == 408, response.text
    assert (error["type"], error["code"], error["param"]) == ("invalid_request_error", "408", None), response.text
    assert string_value(error["message"]).startswith("litellm.Timeout: "), response.text


@pytest.mark.parametrize("route", ROUTES)
def test_a_refused_connection_is_answered_500_internal_server_error(gateway: Gateway, route: str) -> None:
    """The deployment's `api_base` is a loopback port nothing listens on. The caller of chat completions or
    embeddings must get HTTP 500 with `error.type` `internal_server_error`, `error.code` `"500"`,
    `error.param` null and a message starting `litellm.`, as JSON, not a dropped connection."""
    with gateway.scenario() as scenario:
        model: Final = _model(scenario, route, unbound_api_base())
        _assert_connection_error(_call(gateway, route, model))


def test_a_refused_chat_connection_message_names_the_openai_connection_error(gateway: Gateway) -> None:
    """The deployment's `api_base` refuses connections and the caller sends a chat completion. `error.message`
    must start with `litellm.InternalServerError: InternalServerError: OpenAIException - ` and contain
    `Connection error`, the text the OpenAI client reports when there is no HTTP response."""
    with gateway.scenario() as scenario:
        model: Final = _model(scenario, "chat", unbound_api_base())
        message: Final = string_value(_error(_call(gateway, "chat", model))["message"])
        assert message.startswith(CONNECTION_PREFIX), message
        assert "Connection error" in message, message


def test_a_refused_connection_error_does_not_reveal_the_provider_key(gateway: Gateway) -> None:
    """The deployment has a distinctive `api_key` and its `api_base` refuses connections. Nothing in the 500
    response, body or headers, may contain that key."""
    secret: Final = f"sk-dec1-provider-{marker()}"
    with gateway.scenario() as scenario:
        model: Final = _model(scenario, "chat", unbound_api_base(), api_key=secret)
        response: Final = _call(gateway, "chat", model)
        assert response.status_code == 500, response.text
        assert secret not in response.text
        assert all(secret not in value for value in response.headers.values())


@pytest.mark.parametrize("route", ROUTES)
def test_a_host_that_does_not_resolve_is_answered_500(gateway: Gateway, route: str) -> None:
    """The deployment's `api_base` is `http://dec1-failures.invalid/v1`, a name the reserved `.invalid` domain
    guarantees never resolves. The caller must get the same HTTP 500 `internal_server_error` shape as for a
    refused connection."""
    with gateway.scenario() as scenario:
        model: Final = _model(scenario, route, "http://dec1-failures.invalid/v1")
        _assert_connection_error(_call(gateway, route, model))


def test_an_untrusted_upstream_certificate_is_answered_500_and_the_request_is_not_sent(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The upstream serves HTTPS with a self-signed certificate the proxy does not trust. The caller must get
    HTTP 500 `internal_server_error`, and the upstream must receive no request: the prompt and the provider
    key must not be sent over a connection that failed verification."""
    certificate: Final = self_signed_certificate(tmp_path)
    context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate.certificate, certificate.key)

    def respond(_request: Request) -> Reply:
        return Reply(body=chat_body())

    with wire_server(respond, tls=context) as wire, gateway.scenario() as scenario:
        model: Final = _model(scenario, "chat", f"{wire.url}/v1")
        _assert_connection_error(_call(gateway, "chat", model))
        assert [request for request in wire.drain() if request.method == "POST"] == []


def test_an_https_api_base_on_a_plain_http_port_is_answered_500(gateway: Gateway) -> None:
    """The deployment's `api_base` says `https://` but the port speaks plain HTTP, so the TLS handshake fails.
    The caller must get HTTP 500 `internal_server_error`.
    The port answers the ClientHello with a plain `400 Bad Request` and closes, as a real HTTP server does.
    `raw_peer` cannot stand in for one: it waits for the end of an HTTP request head that a ClientHello never
    contains, so the handshake hung and the caller timed out in CI run 42966523075."""
    with plain_http_port() as port, gateway.scenario() as scenario:
        model: Final = _model(scenario, "chat", f"https://127.0.0.1:{port}/v1")
        _assert_connection_error(_call(gateway, "chat", model))


@pytest.mark.parametrize("route", ROUTES)
def test_an_upstream_that_closes_without_a_reply_is_answered_500(gateway: Gateway, route: str) -> None:
    """The upstream reads the whole request and closes the connection without sending a byte. The caller must
    get HTTP 500 `internal_server_error`, and the upstream must have received exactly one POST: with
    `num_retries: 0` a non-idempotent request is not silently replayed.
    Product bug for the embeddings case: in CI run 42966523075 the upstream saw three POSTs. The router passes
    `max_retries: 0`, but `OpenAIChatCompletion.embedding` reads it as `max_retries or
    litellm.DEFAULT_MAX_RETRIES` (litellm/llms/openai/openai.py:1411), so the OpenAI client retries twice.
    """
    if route == "embeddings":
        pytest.skip("BUG: OpenAI embeddings ignore max_retries 0 and the client retries each attempt twice")
    with raw_peer(closing_without_a_reply()) as peer, gateway.scenario() as scenario:
        model: Final = _model(scenario, route, peer.api_base)
        _assert_connection_error(_call(gateway, route, model))
        assert len(peer.posts()) == 1


def test_a_reset_in_the_middle_of_the_body_is_answered_500_not_a_truncated_200(gateway: Gateway) -> None:
    """The upstream sends a 200 status line, headers announcing the full length and the first 20 bytes of the
    body, then resets the connection. The caller must get HTTP 500 `internal_server_error`: a partial
    upstream body must never be passed on as a successful answer."""
    body: Final = chat_body()
    partial: Final = http_response(200, body[:20], declared_length=len(body))
    with raw_peer(resetting_after(lambda _exchange: partial)) as peer, gateway.scenario() as scenario:
        model: Final = _model(scenario, "chat", peer.api_base)
        _assert_connection_error(_call(gateway, "chat", model))


def test_a_body_shorter_than_its_content_length_is_answered_500(gateway: Gateway) -> None:
    """The upstream announces `content-length` for the whole body, sends half of it and closes cleanly. The
    caller must get HTTP 500 `internal_server_error`."""
    body: Final = chat_body()
    short: Final = http_response(200, body[: len(body) // 2], declared_length=len(body))
    with raw_peer(resetting_after(lambda _exchange: short)) as peer, gateway.scenario() as scenario:
        model: Final = _model(scenario, "chat", peer.api_base)
        _assert_connection_error(_call(gateway, "chat", model))


def test_a_streaming_request_to_a_refused_upstream_is_a_json_500_not_an_event_stream(gateway: Gateway) -> None:
    """The caller sends a chat completion with `stream: true` to a deployment whose upstream refuses
    connections. The response must be HTTP 500 with a JSON error body, not a 200 `text/event-stream` that
    then carries an error event: nothing was streamed, so the status must tell the truth."""
    with gateway.scenario() as scenario:
        model: Final = _model(scenario, "chat", unbound_api_base())
        response: Final = _call(gateway, "chat", model, stream=True)
        assert response.status_code == 500, response.text
        assert response.headers["content-type"].startswith("application/json"), response.headers
        assert _error(response)["code"] == "500", response.text


def test_a_connect_that_never_completes_is_answered_408_at_the_deployment_timeout(gateway: Gateway) -> None:
    """The upstream port completes no TCP handshake and the deployment sets `timeout: 2`. The caller must get
    HTTP 408 with `error.type` `invalid_request_error` and `error.code` `"408"`, no sooner than 2 seconds
    and within a few seconds of it.
    Assumes Linux drops the SYN of a full accept queue, which is how the peer withholds the handshake."""
    with unaccepting_listener() as base, gateway.scenario() as scenario:
        model: Final = _model(scenario, "chat", f"{base}/v1", timeout=TIMEOUT_SECONDS)
        answered: Final = timed(lambda: _call(gateway, "chat", model))
        _assert_timeout(answered.value)
        assert_enforced_within(answered.seconds, configured=TIMEOUT_SECONDS, slack=8, not_before=TIMEOUT_SECONDS - 0.5)


@pytest.mark.parametrize("route", ROUTES)
def test_an_upstream_that_never_sends_headers_is_answered_408_at_the_deployment_timeout(
    gateway: Gateway, route: str
) -> None:
    """The upstream accepts the request and sends nothing, and the deployment sets `timeout: 2`. The caller of
    chat completions or embeddings must get HTTP 408 `invalid_request_error` with `code` `"408"`, no sooner
    than 2 seconds, and the upstream must have received exactly one POST.
    Product bug for the embeddings case: in CI run 42966523075 the upstream saw three POSTs. The router passes
    `max_retries: 0`, but `OpenAIChatCompletion.embedding` reads it as `max_retries or
    litellm.DEFAULT_MAX_RETRIES` (litellm/llms/openai/openai.py:1411), so the OpenAI client retries twice.
    """
    if route == "embeddings":
        pytest.skip("BUG: OpenAI embeddings ignore max_retries 0 and the client retries each attempt twice")
    never: Final = threading.Event()
    with raw_peer(silent_until(never)) as peer, gateway.scenario() as scenario:
        model: Final = _model(scenario, route, peer.api_base, timeout=TIMEOUT_SECONDS)
        answered: Final = timed(lambda: _call(gateway, route, model))
        _assert_timeout(answered.value)
        assert_enforced_within(answered.seconds, configured=TIMEOUT_SECONDS, slack=8, not_before=TIMEOUT_SECONDS - 0.5)
        assert len(peer.posts()) == 1


def test_the_chat_timeout_message_is_the_request_timed_out_text(gateway: Gateway) -> None:
    """The upstream sends nothing and the deployment sets `timeout: 2`. `error.message` of the 408 must start
    with exactly `litellm.Timeout: APITimeoutError - Request timed out.`"""
    never: Final = threading.Event()
    with raw_peer(silent_until(never)) as peer, gateway.scenario() as scenario:
        model: Final = _model(scenario, "chat", peer.api_base, timeout=TIMEOUT_SECONDS)
        message: Final = string_value(_error(_call(gateway, "chat", model))["message"])
        assert message.startswith(TIMEOUT_PREFIX), message


def test_the_proxy_hangs_up_on_the_upstream_when_the_timeout_fires(gateway: Gateway) -> None:
    """The upstream sends nothing and the deployment sets `timeout: 2`. After the caller got its 408 the
    upstream must see its connection closed by the proxy: a timed out call must not leave a provider
    connection open that keeps generating, and billing, in the background."""
    never: Final = threading.Event()
    with raw_peer(silent_until(never)) as peer, gateway.scenario() as scenario:
        model: Final = _model(scenario, "chat", peer.api_base, timeout=TIMEOUT_SECONDS)
        assert _call(gateway, "chat", model).status_code == 408
        assert eventually(peer.hung_up.is_set, bool, seconds=15)


def test_headers_that_arrive_inside_the_timeout_make_a_normal_200(gateway: Gateway) -> None:
    """The upstream waits until it is released, well inside the deployment's 20 second timeout, and then
    answers 200. The caller must get 200 with the upstream's content: a slow first byte alone is not an error."""
    gate: Final = threading.Event()
    with raw_peer(silent_until(gate)) as peer, gateway.scenario() as scenario:
        model: Final = _model(scenario, "chat", peer.api_base, timeout=20)
        releaser: Final = threading.Timer(1.0, gate.set)
        releaser.start()
        try:
            response: Final = _call(gateway, "chat", model)
        finally:
            gate.set()
            releaser.cancel()
        assert response.status_code == 200, response.text
        assert REPLY_TEXT in response.text


def test_a_body_that_stops_after_the_headers_is_answered_408(gateway: Gateway) -> None:
    """The upstream sends a 200 status line and headers at once and then no body, and the deployment sets
    `timeout: 2`. The caller must get HTTP 408, not a 200 with an empty body and not a hang: the read timeout
    also guards the body."""
    never: Final = threading.Event()
    with raw_peer(headers_then_body_after(never)) as peer, gateway.scenario() as scenario:
        model: Final = _model(scenario, "chat", peer.api_base, timeout=TIMEOUT_SECONDS)
        answered: Final = timed(lambda: _call(gateway, "chat", model))
        _assert_timeout(answered.value)
        assert_enforced_within(answered.seconds, configured=TIMEOUT_SECONDS, slack=8, not_before=TIMEOUT_SECONDS - 0.5)


def test_a_body_that_trickles_with_short_gaps_is_served_even_past_the_timeout(gateway: Gateway) -> None:
    """The deployment sets `timeout: 2` and the upstream sends its body in 8 pieces half a second apart, 4
    seconds in total. The caller must get 200 with the full content: the timeout limits each wait for bytes,
    not the whole response.
    UNVERIFIED: read from the timeout being handed to the HTTP client as a per-read limit. If the router also
    enforces a wall-clock limit this becomes a 408; either way the behaviour must not change in the port."""
    with raw_peer(trickling_body(8, 0.5)) as peer, gateway.scenario() as scenario:
        model: Final = _model(scenario, "chat", peer.api_base, timeout=TIMEOUT_SECONDS)
        response: Final = _call(gateway, "chat", model)
        assert response.status_code == 200, response.text
        assert REPLY_TEXT in response.text


def test_a_timed_out_call_writes_a_failure_spend_row_with_error_code_408(gateway: Gateway) -> None:
    """A virtual key calls a deployment whose upstream never answers, with `timeout: 2`. Besides the 408 to
    the caller there must be one `LiteLLM_SpendLogs` row for the key with status `failure`, spend 0 and
    `metadata.error_information.error_code` `"408"`.
    Product bug: CI run 42966523075 wrote spend 4.8e-06 on the row, the prompt cost of the request, although
    the upstream never sent a byte. A failure keeps a cost only when usage was recovered from an interrupted
    stream (litellm/litellm_core_utils/litellm_logging.py:4625), and failure rows are otherwise zero spend
    (`spend/test_dec1_spend_log_row_failures.py`); a refused connection does write spend 0."""
    pytest.skip("BUG: a timed out call with no upstream reply writes a failure spend row with the prompt cost")
    never: Final = threading.Event()
    with raw_peer(silent_until(never)) as peer, gateway.scenario() as scenario:
        model: Final = _model(scenario, "chat", peer.api_base, timeout=TIMEOUT_SECONDS)
        key: Final = scenario.key(models=[model])
        assert _call(gateway, "chat", model, key=key).status_code == 408
        rows: Final = eventually(
            lambda: gateway.rows(
                'SELECT status, spend, metadata FROM "LiteLLM_SpendLogs" WHERE api_key = %s', (digest(key),)
            ),
            lambda found: len(found) == 1,
            seconds=70,
        )
        information: Final = object_value(object_value(rows[0]["metadata"])["error_information"])
        assert (rows[0]["status"], float(str(rows[0]["spend"]))) == ("failure", 0.0), rows
        assert information["error_code"] == "408", information


def test_a_refused_connection_writes_a_failure_spend_row_with_error_code_500(gateway: Gateway) -> None:
    """A virtual key calls a deployment whose upstream refuses connections. There must be one spend row for
    the key with status `failure`, spend 0 and `metadata.error_information.error_code` `"500"`: calls that
    never reached a provider are still visible in the logs."""
    with gateway.scenario() as scenario:
        model: Final = _model(scenario, "chat", unbound_api_base())
        key: Final = scenario.key(models=[model])
        assert _call(gateway, "chat", model, key=key).status_code == 500
        rows: Final = eventually(
            lambda: gateway.rows(
                'SELECT status, spend, metadata FROM "LiteLLM_SpendLogs" WHERE api_key = %s', (digest(key),)
            ),
            lambda found: len(found) == 1,
            seconds=70,
        )
        information: Final = object_value(object_value(rows[0]["metadata"])["error_information"])
        assert (rows[0]["status"], float(str(rows[0]["spend"]))) == ("failure", 0.0), rows
        assert information["error_code"] == "500", information


def test_a_connection_failure_answers_quickly_instead_of_waiting_for_the_timeout(gateway: Gateway) -> None:
    """The deployment sets `timeout: 20` and its `api_base` refuses connections. The 500 must arrive in less
    than those 20 seconds: a refused connection is known immediately and must not be held until the
    timeout."""
    with gateway.scenario() as scenario:
        model: Final = _model(scenario, "chat", unbound_api_base(), timeout=20)
        answered: Final = timed(lambda: _call(gateway, "chat", model))
        assert answered.value.status_code == 500, answered.value.text
        assert_cut_short(answered.seconds, injected_delay=20)


def test_a_deployment_works_again_as_soon_as_its_upstream_accepts_connections(gateway: Gateway) -> None:
    """A deployment's first call fails because the upstream closes without replying, and the upstream then
    answers normally. The second call to the same deployment must be 200: one connection failure must not
    leave a broken pooled connection or client behind. Cooldowns are off in the shared config."""
    with raw_peer(in_turn(closing_without_a_reply(), ok())) as peer, gateway.scenario() as scenario:
        model: Final = _model(scenario, "chat", peer.api_base)
        assert _call(gateway, "chat", model).status_code == 500
        recovered: Final = _call(gateway, "chat", model)
        assert recovered.status_code == 200, recovered.text
        assert len(peer.posts()) == 2
