"""What the proxy does with the headers a client sends, under the default settings.

By default nothing the client sends is passed on to the LLM provider: not the virtual key, not custom `x-`
headers, not `OpenAI-Organization`, not cookies. Forwarding is opt-in (`forward_client_headers_to_llm_api`,
`forward_openai_org_id`), and the opt-in behaviour is pinned in `test_dec1_http_forwarding_settings.py`. A second
group of headers is consumed by the proxy as request controls and lands in the spend log: `x-litellm-tags`,
`x-litellm-spend-logs-metadata`, `x-litellm-customer-id`, `x-litellm-end-user-id` and the client's `User-Agent`
(`litellm_pre_call_utils.py`). A forwarder in front of the proxy must deliver all of them untouched, in any
letter case, and a reimplementation must read the same names. Each test reads the provider's received headers
from a recording peer and the spend row from Postgres.
"""

from __future__ import annotations

import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration.compatibility.dec1_http_support import CHAT, chat_body, front_door, provider_error, spend_rows
from pydantic import JsonValue


def _row_after_one_call(gateway: Gateway, headers: dict[str, str]) -> dict[str, JsonValue]:
    with front_door(gateway) as door:
        key: Final = door.scenario.key()
        response: Final = gateway.request("POST", CHAT, chat_body(door.chat_model()), key=key, headers=headers)
        assert response.status_code == 200, response.text
        (row,) = spend_rows(gateway, key)
        return row


def test_the_provider_receives_the_deployment_key_and_never_the_callers_key(gateway: Gateway) -> None:
    """A virtual key calls a chat model whose deployment is configured with `integration-provider-key`. The
    provider must receive `Authorization: Bearer integration-provider-key`, and the caller's virtual key must not
    appear in any header value the provider received."""
    with front_door(gateway) as door:
        key: Final = door.scenario.key()
        response: Final = gateway.request("POST", CHAT, chat_body(door.chat_model()), key=key)
        assert response.status_code == 200, response.text
        (call,) = door.calls()
        assert call.headers["authorization"] == "Bearer integration-provider-key"
        assert [name for name, value in call.headers.items() if key in value] == []


def test_a_key_sent_in_x_litellm_api_key_authenticates_and_is_not_passed_on(gateway: Gateway) -> None:
    """A virtual key is sent as `x-litellm-api-key: Bearer <key>` with no Authorization header. The proxy must
    serve the chat call with 200, and the provider must receive neither an `x-litellm-api-key` header nor the key
    in any other header value. That header is for proxy auth only (`clean_headers`, `litellm_pre_call_utils.py:1585`)."""
    with front_door(gateway) as door:
        key: Final = door.scenario.key()
        response: Final = gateway.client.post(
            CHAT, json=chat_body(door.chat_model()), headers={"x-litellm-api-key": f"Bearer {key}"}
        )
        assert response.status_code == 200, response.text
        (call,) = door.calls()
        assert "x-litellm-api-key" not in call.headers
        assert [name for name, value in call.headers.items() if key in value] == []


def test_custom_client_headers_are_not_passed_to_the_provider_by_default(gateway: Gateway) -> None:
    """The caller adds `x-dec1-trace`, `anthropic-beta`, `OpenAI-Organization` and `Cookie` headers to a chat
    call on the shared proxy, where `forward_client_headers_to_llm_api` and `forward_openai_org_id` are unset.
    The call must succeed and the provider must receive none of those four headers."""
    sent: Final = {
        "x-dec1-trace": "trace-123",
        "anthropic-beta": "dec1-beta-2026",
        "OpenAI-Organization": "org-dec1",
        "Cookie": "session=dec1",
    }
    with front_door(gateway) as door:
        response: Final = gateway.request("POST", CHAT, chat_body(door.chat_model()), headers=sent)
        assert response.status_code == 200, response.text
        (call,) = door.calls()
        assert [name for name in sent if name.lower() in call.headers] == []


def test_forwarding_and_peer_headers_are_not_passed_to_the_provider_by_default(gateway: Gateway) -> None:
    """The caller adds `X-Forwarded-For`, `Forwarded` and `x-litellm-rust-peer` headers to a chat call on the
    shared proxy. The provider must receive none of them. `x-litellm-rust-peer` is the header a Rust forwarder
    adds on every request to carry the client's TCP address, so it must never leave the proxy."""
    sent: Final = {
        "X-Forwarded-For": "203.0.113.9",
        "Forwarded": "for=203.0.113.9",
        "x-litellm-rust-peer": "203.0.113.9:40000",
    }
    with front_door(gateway) as door:
        response: Final = gateway.request("POST", CHAT, chat_body(door.chat_model()), headers=sent)
        assert response.status_code == 200, response.text
        (call,) = door.calls()
        assert [name for name in sent if name.lower() in call.headers] == []


def test_the_clients_user_agent_is_not_the_user_agent_the_provider_sees(gateway: Gateway) -> None:
    """The caller sends `User-Agent: dec1-client/9.9` on a chat call. The provider must see a different
    `user-agent` (the proxy's own HTTP client), never the caller's. UNVERIFIED: assumes default settings do not
    forward User-Agent; only `x-` and `anthropic-beta` headers are ever forwardable (`litellm_pre_call_utils.py:1743`)."""
    with front_door(gateway) as door:
        response: Final = gateway.request(
            "POST", CHAT, chat_body(door.chat_model()), headers={"User-Agent": "dec1-client/9.9"}
        )
        assert response.status_code == 200, response.text
        (call,) = door.calls()
        assert call.headers.get("user-agent") != "dec1-client/9.9"


def test_the_clients_user_agent_is_recorded_on_the_spend_row(gateway: Gateway) -> None:
    """A fresh key calls a chat model with `User-Agent: dec1-client/9.9`. The spend row's `metadata.user_agent`
    must be exactly that string. A forwarder that rewrites or drops User-Agent would corrupt per-client usage
    reports."""
    row: Final = _row_after_one_call(gateway, {"User-Agent": "dec1-client/9.9"})
    assert object_value(row["metadata"])["user_agent"] == "dec1-client/9.9"


@pytest.mark.parametrize("name", ("x-litellm-tags", "X-LiteLLM-Tags", "X-LITELLM-TAGS"))
def test_tags_header_lands_in_request_tags_in_any_letter_case(gateway: Gateway, name: str) -> None:
    """A fresh key calls a chat model with the tags header spelled in lower, mixed or upper case and the value
    `dec1-alpha, dec1-beta`. The spend row's `request_tags` must contain both `dec1-alpha` and `dec1-beta`, split
    on the comma with surrounding spaces removed (`litellm_pre_call_utils.py:2375`)."""
    row: Final = _row_after_one_call(gateway, {name: "dec1-alpha, dec1-beta"})
    assert {"dec1-alpha", "dec1-beta"} <= set(list_value(row["request_tags"]))


def test_spend_logs_metadata_header_lands_in_the_row_metadata(gateway: Gateway) -> None:
    """A fresh key calls a chat model with `x-litellm-spend-logs-metadata: {"dec1_job": "<id>", "attempt": 2}`.
    The spend row's `metadata.spend_logs_metadata` must equal that JSON object exactly
    (`_get_spend_logs_metadata_from_request_headers`, `litellm_pre_call_utils.py:1731`)."""
    job: Final = uuid.uuid4().hex
    row: Final = _row_after_one_call(
        gateway, {"x-litellm-spend-logs-metadata": json.dumps({"dec1_job": job, "attempt": 2})}
    )
    assert object_value(row["metadata"])["spend_logs_metadata"] == {"dec1_job": job, "attempt": 2}


@pytest.mark.parametrize("name", ("x-litellm-customer-id", "x-litellm-end-user-id"))
def test_customer_id_headers_set_the_end_user_on_the_spend_row(gateway: Gateway, name: str) -> None:
    """A fresh key calls a chat model with a customer id in one of the two standard headers and no `user` in the
    body. The spend row's `end_user` column must equal the header value. Both headers work with no configuration
    (`STANDARD_CUSTOMER_ID_HEADERS`, `constants.py:1735`), which is how Claude Code style tools attribute usage."""
    customer: Final = f"dec1-customer-{uuid.uuid4().hex}"
    row: Final = _row_after_one_call(gateway, {name: customer})
    assert row["end_user"] == customer


def test_customer_id_header_wins_over_the_user_field_in_the_body(gateway: Gateway) -> None:
    """A fresh key calls a chat model with `x-litellm-customer-id` set to one id and the body's `user` field set
    to another. The spend row's `end_user` must be the header's value: the standard headers are checked before
    the body (`get_end_user_id_from_request_body`, `auth_utils.py:1939`)."""
    header_customer: Final = f"dec1-header-{uuid.uuid4().hex}"
    with front_door(gateway) as door:
        key: Final = door.scenario.key()
        response: Final = gateway.request(
            "POST",
            CHAT,
            chat_body(door.chat_model(), user=f"dec1-body-{uuid.uuid4().hex}"),
            key=key,
            headers={"x-litellm-customer-id": header_customer},
        )
        assert response.status_code == 200, response.text
        (row,) = spend_rows(gateway, key)
        assert row["end_user"] == header_customer


def test_control_headers_do_not_reach_the_provider(gateway: Gateway) -> None:
    """The caller sends `x-litellm-tags`, `x-litellm-spend-logs-metadata`, `x-litellm-customer-id` and
    `x-litellm-num-retries` on a chat call under default settings. The proxy consumes them, so the provider must
    receive none of the four and the forwarded JSON body must not contain a `tags` key."""
    sent: Final = {
        "x-litellm-tags": "dec1-alpha",
        "x-litellm-spend-logs-metadata": '{"dec1": true}',
        "x-litellm-customer-id": "dec1-customer",
        "x-litellm-num-retries": "0",
    }
    with front_door(gateway) as door:
        response: Final = gateway.request("POST", CHAT, chat_body(door.chat_model()), headers=sent)
        assert response.status_code == 200, response.text
        (call,) = door.calls()
        assert [name for name in sent if name in call.headers] == []
        assert "tags" not in json.loads(call.body)


def test_a_lower_case_authorization_header_name_authenticates(gateway: Gateway) -> None:
    """The caller authenticates with `authorization: Bearer <master key>` using a lower-case header name, as
    HTTP/2 front ends and some forwarders normalise it. The chat call must be served with 200: header names are
    case-insensitive end to end."""
    with front_door(gateway) as door:
        response: Final = gateway.client.post(
            CHAT, json=chat_body(door.chat_model()), headers={"authorization": f"Bearer {gateway.key}"}
        )
        assert response.status_code == 200, response.text
        assert len(door.calls()) == 1


def test_num_retries_zero_header_limits_a_failing_call_to_one_provider_attempt(gateway: Gateway) -> None:
    """The provider answers every chat call with 500. The caller sends `x-litellm-num-retries: 0`. The proxy must
    answer 500 after exactly one provider attempt: the header overrides the router's retry count for this
    request (`_get_num_retries_from_request`, `litellm_pre_call_utils.py:1721`). Retry behaviour itself is pinned
    in `routing/test_dec1_routing_retries.py`."""
    with front_door(gateway, lambda request: provider_error(500)) as door:
        response: Final = gateway.request(
            "POST", CHAT, chat_body(door.chat_model()), headers={"x-litellm-num-retries": "0"}
        )
        assert response.status_code == 500, response.text
        assert len(door.calls()) == 1
