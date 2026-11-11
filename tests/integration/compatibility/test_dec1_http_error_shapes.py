"""The JSON shape of every error the front door can produce.

The proxy has more than one error envelope, and clients (OpenAI SDKs, Anthropic SDKs, dashboards) parse each one.
Errors raised by proxy code on OpenAI-style routes use `{"error": {"message", "type", "param", "code"}}` with
`code` as a string (`ProxyException.to_dict`, `litellm/proxy/_types.py:3916`). Errors on `/v1/messages` use the
Anthropic envelope `{"type": "error", "error": {"type", "message"}}`. Errors raised by the web framework before any
proxy code runs (unknown path, wrong method, schema validation) use FastAPI's `{"detail": ...}`. A gateway in
front of the proxy must pass all of these through unchanged, and when it has to fail a request itself it should
pick the envelope the client of that route expects. These tests pin which envelope each failure uses.
"""

from __future__ import annotations

import json
import uuid
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply
from integration.compatibility.dec1_http_support import CHAT, chat_body, front_door, provider_error

ENVELOPE_KEYS: Final = {"message", "type", "param", "code"}


def _error(response: httpx.Response) -> dict[str, object]:
    return dict(object_value(object_value(response.json())["error"]))


def test_a_missing_key_is_a_401_openai_envelope_with_type_auth_error(gateway: Gateway) -> None:
    """A chat call is sent with no Authorization header at all. The proxy must answer 401 with the OpenAI envelope
    whose `type` is `auth_error`, whose `code` is the string `"401"` and whose message starts with
    `Authentication Error` (`auth_exception_handler.py:134`)."""
    response: Final = gateway.client.post(CHAT, json=chat_body("dec1-any-model"))
    error: Final = _error(response)
    assert response.status_code == 401
    assert set(error) >= ENVELOPE_KEYS
    assert error["type"] == "auth_error"
    assert error["code"] == "401"
    assert str(error["message"]).startswith("Authentication Error")


def test_an_unknown_key_is_a_401_that_never_echoes_the_key(gateway: Gateway) -> None:
    """A chat call is sent with a well-formed but unknown `sk-` key. The proxy must answer 401 in the OpenAI
    envelope with `type` `token_not_found_in_db` and `code` `"401"` (`ProxyErrorTypes.token_not_found_in_db`,
    raised by `get_key_object` in `auth_checks.py` when the hashed key has no row), unlike the missing-key case,
    which is `auth_error`. The full key must not appear anywhere in the response body: error bodies end up in
    client logs."""
    key: Final = f"sk-dec1-{uuid.uuid4().hex}"
    response: Final = gateway.request("POST", CHAT, chat_body("dec1-any-model"), key=key)
    assert response.status_code == 401
    error: Final = _error(response)
    assert set(error) >= ENVELOPE_KEYS
    assert (error["type"], error["code"]) == ("token_not_found_in_db", "401")
    assert key not in response.text


def test_an_unknown_model_is_a_400_naming_the_model(gateway: Gateway) -> None:
    """The master key asks for a model name that no deployment serves. The proxy must answer 400 with the OpenAI
    envelope and a message containing `Invalid model name passed in model=<name>`
    (`ProxyModelNotFoundError`, `route_llm_request.py:238`). UNVERIFIED: the envelope `code` is assumed to be the
    string `"400"`."""
    name: Final = f"dec1-missing-{uuid.uuid4().hex}"
    response: Final = gateway.request("POST", CHAT, chat_body(name))
    error: Final = _error(response)
    assert response.status_code == 400
    assert f"Invalid model name passed in model={name}" in str(error["message"])
    assert error["code"] == "400"


@pytest.mark.parametrize("status", (400, 401, 403, 404, 422, 429, 500, 503))
def test_a_provider_error_status_is_relayed_in_the_openai_envelope(gateway: Gateway, status: int) -> None:
    """The provider answers a chat call with the given status and an OpenAI-style error body. The proxy must
    answer with the same status, content type `application/json`, and an `error` object carrying all four of
    `message`, `type`, `param` and `code`, with `code` equal to the status as a string."""
    with front_door(gateway, lambda request: provider_error(status)) as door:
        response: Final = gateway.request("POST", CHAT, chat_body(door.chat_model(), num_retries=0))
    error: Final = _error(response)
    assert response.status_code == status, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert set(error) >= ENVELOPE_KEYS
    assert error["code"] == str(status)


def test_a_provider_error_message_reaches_the_client(gateway: Gateway) -> None:
    """The provider answers 400 with the message `dec1 scripted refusal`. The proxy's error message must contain
    that text, so a caller can see what the provider objected to."""
    with front_door(gateway, lambda request: provider_error(400, "dec1 scripted refusal")) as door:
        response: Final = gateway.request("POST", CHAT, chat_body(door.chat_model(), num_retries=0))
    assert response.status_code == 400
    assert "dec1 scripted refusal" in str(_error(response)["message"])


def test_a_provider_that_drops_the_connection_is_a_json_5xx_not_a_broken_response(gateway: Gateway) -> None:
    """The provider reads the request and closes the socket without answering. The proxy must still answer the
    client with a complete JSON error envelope and a 5xx status. UNVERIFIED: the exact status is 500 for a
    connection error (`APIConnectionError`), so the test accepts any of 500, 502 and 503."""
    with front_door(gateway, lambda request: Reply(drop_connection=True)) as door:
        response: Final = gateway.request("POST", CHAT, chat_body(door.chat_model(), num_retries=0))
    assert response.status_code in (500, 502, 503), response.text
    assert set(_error(response)) >= ENVELOPE_KEYS


def test_a_provider_html_error_page_is_wrapped_in_json(gateway: Gateway) -> None:
    """The provider answers 502 with an HTML page, as a load balancer in front of it would. The proxy must answer
    with content type `application/json` and an `error` object: the client must never receive the provider's HTML
    as the body of an API response."""
    page: Final = Reply(status=502, body=b"<html><body>Bad Gateway</body></html>", content_type="text/html")
    with front_door(gateway, lambda request: page) as door:
        response: Final = gateway.request("POST", CHAT, chat_body(door.chat_model(), num_retries=0))
    assert response.status_code >= 500, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert set(_error(response)) >= ENVELOPE_KEYS


def test_an_unknown_path_and_a_wrong_method_use_the_detail_shape(gateway: Gateway) -> None:
    """The master key GETs a path that does not exist and GETs the POST-only chat route. The answers must be
    exactly `{"detail": "Not Found"}` with 404 and `{"detail": "Method Not Allowed"}` with 405. These come from
    the web framework, not from proxy code, so they have no `error` key."""
    missing: Final = gateway.request("GET", f"/dec1-missing-{uuid.uuid4().hex}")
    wrong_method: Final = gateway.request("GET", CHAT)
    assert (missing.status_code, missing.json()) == (404, {"detail": "Not Found"})
    assert (wrong_method.status_code, wrong_method.json()) == (405, {"detail": "Method Not Allowed"})


def test_a_management_body_of_the_wrong_type_is_a_422_detail_list(gateway: Gateway) -> None:
    """The master key posts `{"max_budget": "not-a-number"}` to `/key/generate`. The proxy must answer 422 with
    `{"detail": [...]}` where each entry has `loc`, `msg` and `type`, the framework's validation shape.
    UNVERIFIED: assumes no custom validation handler rewrites this (none is registered in `proxy_server.py`)."""
    response: Final = gateway.request("POST", "/key/generate", {"max_budget": "not-a-number"})
    detail: Final = response.json()["detail"]
    assert response.status_code == 422, response.text
    assert isinstance(detail, list)
    assert {"loc", "msg", "type"} <= set(detail[0])


def test_messages_route_auth_failure_uses_the_anthropic_envelope(gateway: Gateway) -> None:
    """A `/v1/messages` call is sent with an unknown key. The proxy must answer 401 with the Anthropic envelope:
    top-level `type` equal to `error` and an `error` object with `type` and `message`. Anthropic SDKs cannot parse
    the OpenAI envelope."""
    body: Final = {"model": "dec1-any", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    response: Final = gateway.request("POST", "/v1/messages", body, key=f"sk-dec1-{uuid.uuid4().hex}")
    payload: Final = response.json()
    assert response.status_code == 401
    assert payload["type"] == "error"
    assert {"type", "message"} <= set(payload["error"])


def test_messages_route_error_echoes_the_clients_request_id(gateway: Gateway) -> None:
    """A `/v1/messages` call with an unknown key carries `x-request-id: dec1-req-<id>`. The Anthropic error body
    must carry the same value as `request_id`, which is how Anthropic clients correlate failures. A forwarder that
    rewrites or drops `x-request-id` breaks that correlation."""
    request_id: Final = f"dec1-req-{uuid.uuid4().hex}"
    body: Final = {"model": "dec1-any", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    response: Final = gateway.request(
        "POST", "/v1/messages", body, key=f"sk-dec1-{uuid.uuid4().hex}", headers={"x-request-id": request_id}
    )
    assert response.status_code == 401
    assert response.json()["request_id"] == request_id


def test_every_error_body_is_valid_json_with_an_exact_content_length(gateway: Gateway) -> None:
    """Four failing requests are sent: no key, unknown path, wrong method and malformed JSON. Each response must
    declare `application/json`, parse as JSON, and have a `Content-Length` equal to the number of body bytes, so a
    keep-alive client can reuse the connection after any error."""
    responses: Final = (
        gateway.client.post(CHAT, json=chat_body("dec1-any-model")),
        gateway.request("GET", f"/dec1-missing-{uuid.uuid4().hex}"),
        gateway.request("GET", CHAT),
        gateway.client.post(
            CHAT,
            content=b'{"model": ',
            headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
        ),
    )
    assert [response.status_code for response in responses] == [401, 404, 405, 400]
    for response in responses:
        assert response.headers["content-type"].startswith("application/json")
        assert response.headers["content-length"] == str(len(response.content))
        assert isinstance(json.loads(response.content), dict)
