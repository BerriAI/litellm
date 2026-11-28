"""Where a credential may be carried, and what odd shapes of it do, on the Python proxy's auth path.

Contract (litellm/proxy/auth/user_api_key_auth.py `get_api_key` and `get_bearer_token`, the master key comparison
near `secrets.compare_digest(api_key, master_key)`, and the non `sk-` refusal that masks the received value):

* The proxy reads a credential from `Authorization` and five provider style headers (`API-Key`, `x-api-key`,
  `x-goog-api-key`, `Ocp-Apim-Subscription-Key`, `x-litellm-api-key`), and from the `key` query parameter on
  Google `generateContent` routes only. The master key works in every one of them, not only as `Bearer`.
  The Rust gateway's own extractor (`litellm/rust/crates/gateway-auth/src/http.rs` `Bearer`) reads only
  `Authorization: Bearer`, so a Rust front door that authenticates by itself would refuse these callers.
* Header names are case insensitive. Two `Authorization` headers: Python reads the first one, the Rust
  extractor refuses both. Non ASCII and very long values are refused with a 4xx, never a 5xx, and are never
  echoed back.
* A credential sent in the `key` query parameter never reaches the provider, neither as a header nor in the
  request target.

Tests use the suite proxy (`gateway` fixture) and register their models through `gateway.scenario()`; the ones
that inspect provider traffic point a model at a local recording peer (`wire_server`).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.client import Gateway, Scenario
from integration._support.wire import Wire, wire_server
from integration.security.dec1_authsec_support import (
    CHAT,
    DEPLOYMENT_KEY,
    INVALID_VIRTUAL_KEY_PREFIX,
    NO_KEY_MESSAGE,
    UNKNOWN_KEY_MESSAGE,
    UNKNOWN_KEY_TYPE,
    assert_auth_error,
    assert_not_echoed,
    assert_served,
    chat_body,
    header_values_carrying,
    marker,
    message_of,
    posts,
    provider_reply,
    raw_get,
    raw_post,
)

PROVIDER_STYLE_HEADERS: Final = (
    "API-Key",
    "x-api-key",
    "x-goog-api-key",
    "Ocp-Apim-Subscription-Key",
    "x-litellm-api-key",
)


@dataclass(frozen=True, slots=True)
class Recorded:
    gateway: Gateway
    scenario: Scenario
    upstream: Wire
    model: str


@contextmanager
def recorded(gateway: Gateway) -> Iterator[Recorded]:
    """A chat model on the shared proxy whose provider is a local recording peer."""
    with wire_server(provider_reply) as upstream, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{upstream.url}/v1", api_key=DEPLOYMENT_KEY)
        yield Recorded(gateway, scenario, upstream, model)


@pytest.mark.parametrize("header", PROVIDER_STYLE_HEADERS)
def test_the_master_key_is_an_admin_in_every_provider_style_header(gateway: Gateway, header: str) -> None:
    """The caller sends the master key in one provider style header and no `Authorization`, and calls the
    admin only route `GET /user/list`. The answer must be 200: `get_api_key` treats the raw header value as the
    credential and it equals the master key. The Rust gateway extractor reads only `Authorization: Bearer`, so a
    port that reuses it on the Python front door would turn these admin callers into 401s."""
    response: Final = raw_get(gateway, "/user/list", {header: gateway.key})
    assert response.status_code == 200, response.text


def test_the_master_key_in_x_litellm_api_key_with_a_bearer_prefix_is_an_admin(gateway: Gateway) -> None:
    """The caller sends `x-litellm-api-key: Bearer <master key>` and calls `GET /user/list`. The answer must be
    200: `x-litellm-api-key` is read with `_get_bearer_token_or_received_api_key`, which strips `Bearer `."""
    response: Final = raw_get(gateway, "/user/list", {"x-litellm-api-key": f"Bearer {gateway.key}"})
    assert response.status_code == 200, response.text


def test_the_master_key_behind_a_basic_scheme_is_an_admin(gateway: Gateway) -> None:
    """The caller sends `Authorization: Basic <master key>` (the raw key, not base64) and calls
    `GET /user/list`. The answer must be 200: `get_bearer_token` strips `Basic ` for Langfuse style clients and
    compares the remainder. Pinned so a stricter Rust extractor is a deliberate change."""
    response: Final = raw_get(gateway, "/user/list", {"Authorization": f"Basic {gateway.key}"})
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    "header",
    ["AUTHORIZATION", "authorization", "X-API-KEY", "X-Goog-Api-Key", "api-key", "OCP-APIM-SUBSCRIPTION-KEY"],
)
def test_credential_header_names_are_case_insensitive(gateway: Gateway, header: str) -> None:
    """A virtual key calls a chat model, sending the key under a header name spelled in an unusual case
    (`AUTHORIZATION` carries `Bearer <key>`, the others the raw key). The call must be served with 200: HTTP
    header names are case insensitive and the proxy must not depend on one spelling."""
    with recorded(gateway) as rig:
        key: Final = rig.scenario.key(models=[rig.model])
        value: Final = f"Bearer {key}" if header.lower() == "authorization" else key
        assert_served(raw_post(gateway, CHAT, chat_body(rig.model), {header: value}))


def test_two_authorization_headers_use_the_first_value(gateway: Gateway) -> None:
    """The caller sends two `Authorization` headers, a valid virtual key first and an unknown `sk-` key second.
    Today the Python proxy reads the first value and serves the call with 200. The Rust gateway extractor refuses
    any request with two `Authorization` headers with 401 (`http.rs` `bearer`), so this pins the Python side of a
    known divergence the port has to settle on purpose."""
    with recorded(gateway) as rig:
        key: Final = rig.scenario.key(models=[rig.model])
        headers: Final = [("Authorization", f"Bearer {key}"), ("Authorization", f"Bearer sk-{marker()}")]
        assert_served(raw_post(gateway, CHAT, chat_body(rig.model), headers))


def test_a_valid_second_authorization_header_never_rescues_an_unknown_first_one(gateway: Gateway) -> None:
    """The caller sends two `Authorization` headers, an unknown `sk-` key first and a valid virtual key second.
    The call must be refused with 401 `Authentication Error, Invalid API key provided.` (error type
    "token_not_found_in_db", litellm/proxy/auth/resolvers/exceptions.py:33-39) and the provider must not be called. Whatever rule the Rust port picks for duplicates, it must never authenticate on a later copy of
    the header when an earlier one is wrong."""
    with recorded(gateway) as rig:
        key: Final = rig.scenario.key(models=[rig.model])
        headers: Final = [("Authorization", f"Bearer sk-{marker()}"), ("Authorization", f"Bearer {key}")]
        refused: Final = raw_post(gateway, CHAT, chat_body(rig.model), headers)
        assert_auth_error(refused, 401, UNKNOWN_KEY_MESSAGE, error_type=UNKNOWN_KEY_TYPE)
        assert posts(rig.upstream) == ()


def test_a_request_with_no_credential_anywhere_is_refused_with_no_api_key(gateway: Gateway) -> None:
    """The caller sends a chat call with no credential header at all. The proxy must answer 401 with exactly
    `Authentication Error, No api key passed in.` and must not call the provider."""
    with recorded(gateway) as rig:
        refused: Final = raw_post(gateway, CHAT, chat_body(rig.model), {})
        assert_auth_error(refused, 401, NO_KEY_MESSAGE)
        assert posts(rig.upstream) == ()


def test_a_non_ascii_bearer_value_is_a_401_and_is_not_echoed(gateway: Gateway) -> None:
    """The caller sends `Authorization: Bearer sk-` followed by UTF-8 bytes of non ASCII letters. The proxy must
    refuse with 401 (not 400 or 500) and the body must not contain the received bytes. Python decodes header
    bytes as latin-1, so the value still starts with `sk-`, is hashed and is not found. UNVERIFIED: the exact
    message; read from `user_api_key_auth.py` it is the unknown key refusal."""
    secret: Final = "sk-dec1-éèü-" + marker("tail")
    with recorded(gateway) as rig:
        refused: Final = raw_post(
            gateway, CHAT, chat_body(rig.model), [("Authorization", b"Bearer " + secret.encode("utf-8"))]
        )
        assert refused.status_code == 401, refused.text
        assert secret.encode("utf-8").decode("latin-1") not in refused.text
        assert posts(rig.upstream) == ()


def test_a_four_kilobyte_unknown_key_is_refused_without_echoing_it(gateway: Gateway) -> None:
    """The caller presents an unknown `sk-` key of 4096 characters. The proxy must answer 401 with exactly
    `Authentication Error, Invalid API key provided.` and error type "token_not_found_in_db", and neither the key nor its SHA-256 may appear in the
    body or the response headers."""
    secret: Final = "sk-" + ("a" * 4000) + marker("x")[:93]
    with recorded(gateway) as rig:
        refused: Final = raw_post(gateway, CHAT, chat_body(rig.model), {"Authorization": f"Bearer {secret}"})
        assert_auth_error(refused, 401, UNKNOWN_KEY_MESSAGE, error_type=UNKNOWN_KEY_TYPE)
        assert_not_echoed(refused, secret)


def test_a_very_long_bearer_value_is_a_client_error_not_a_server_error(gateway: Gateway) -> None:
    """The caller presents a 60 KB bearer value. The answer must be a 4xx (the server may refuse the header block
    with 400 or 431, or auth may answer 401), never a 5xx, and the body must not contain the value. UNVERIFIED:
    which of those statuses; it depends on the HTTP parser's header limit (uvicorn with httptools or h11)."""
    secret: Final = "sk-" + ("b" * 60_000)
    with recorded(gateway) as rig:
        refused: Final = raw_post(gateway, CHAT, chat_body(rig.model), {"Authorization": f"Bearer {secret}"})
        assert 400 <= refused.status_code < 500, refused.status_code
        assert secret not in refused.text
        assert posts(rig.upstream) == ()


def test_a_long_non_sk_credential_shows_only_four_characters_at_each_end(gateway: Gateway) -> None:
    """The caller sends a 40 character bearer value that does not start with `sk-`. The 401 message must be
    exactly `LiteLLM Virtual Key expected. Received=<first 4>****<last 4>, expected to start with 'sk-'.`, so
    the middle 32 characters are never echoed (`_masked_key` in `user_api_key_auth.py`)."""
    secret: Final = "tok" + marker("m")[:37]
    with recorded(gateway) as rig:
        refused: Final = raw_post(gateway, CHAT, chat_body(rig.model), {"Authorization": f"Bearer {secret}"})
        assert_auth_error(
            refused, 401, f"{INVALID_VIRTUAL_KEY_PREFIX}{secret[:4]}****{secret[-4:]}, expected to start with 'sk-'."
        )
        assert secret not in refused.text


def test_a_nine_character_non_sk_credential_is_not_mostly_echoed(gateway: Gateway) -> None:
    """The caller sends the nine character bearer value `hunter2go`. The refusal must not reveal more than a
    small part of it: no run of five or more of its characters may appear in the body. Today `_masked_key`
    shows the first four and last four characters of anything longer than eight, which is eight of nine
    characters here. The Rust core (#51374 `Token::masked` with `min_custom_key_length`) echoes nothing for
    values shorter than `MINIMUM_CUSTOM_KEY_LENGTH` (16)."""
    pytest.skip("BUG: a 9 to 15 character non sk- credential is echoed with only its middle characters masked")
    secret: Final = "hunter2go"
    with recorded(gateway) as rig:
        refused: Final = raw_post(gateway, CHAT, chat_body(rig.model), {"Authorization": f"Bearer {secret}"})
        assert refused.status_code == 401, refused.text
        assert all(secret[start : start + 5] not in message_of(refused) for start in range(len(secret) - 4))


def test_a_prefix_of_the_master_key_is_not_the_master_key(gateway: Gateway) -> None:
    """The caller sends the master key without its last character and calls `GET /user/list`. The proxy must
    refuse with 401 and the body must not contain the master key. The comparison is exact
    (`secrets.compare_digest`), not a prefix match."""
    refused: Final = raw_get(gateway, "/user/list", {"Authorization": f"Bearer {gateway.key[:-1]}"})
    assert refused.status_code == 401, refused.text
    assert gateway.key not in refused.text


def test_the_master_key_in_a_different_letter_case_is_not_the_master_key(gateway: Gateway) -> None:
    """The caller sends the master key with its letters upper cased and calls `GET /user/list`. The proxy must
    refuse with 401: the master key comparison is case sensitive, and so must be the Rust `MasterKey::matches`."""
    swapped: Final = gateway.key.upper() if gateway.key != gateway.key.upper() else gateway.key.lower()
    if swapped == gateway.key:
        pytest.skip("the configured master key has no letters to change the case of")
    refused: Final = raw_get(gateway, "/user/list", {"Authorization": f"Bearer {swapped}"})
    assert refused.status_code == 401, refused.text


def test_a_query_key_authenticates_generate_content_and_never_reaches_the_provider(gateway: Gateway) -> None:
    """A Gemini model points at a recording peer; a virtual key allowed that model calls
    `POST /v1beta/models/<model>:generateContent?key=<virtual key>` with no credential header. The call must be
    served with 200, and the provider request must carry the deployment key in `x-goog-api-key` while the virtual
    key appears neither in any provider header nor anywhere in the provider request target (path or query)."""
    with wire_server(provider_reply) as upstream, gateway.scenario() as scenario:
        model: Final = scenario.model(model="gemini/gemini-2.5-flash", api_base=upstream.url, api_key=DEPLOYMENT_KEY)
        key: Final = scenario.key(models=[model])
        response: Final = gateway.client.post(
            f"/v1beta/models/{model}:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": marker()}]}]},
            params={"key": key},
        )
        assert response.status_code == 200, response.text
        (call,) = posts(upstream)
        assert call.headers.get("x-goog-api-key") == DEPLOYMENT_KEY
        assert header_values_carrying(call, key) == []
        assert key not in call.target


def test_a_query_key_is_not_a_credential_on_embeddings(gateway: Gateway) -> None:
    """A valid virtual key is sent only as `?key=<virtual key>` on `POST /v1/embeddings`. The proxy must refuse
    with 401 `Authentication Error, No api key passed in.`: the query key is honoured only on routes whose path
    contains `generateContent` (`RouteChecks.is_generate_content_route`)."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        refused: Final = gateway.client.post(
            "/v1/embeddings", json={"model": "text-embedding-3-small", "input": "x"}, params={"key": key}
        )
        assert_auth_error(refused, 401, NO_KEY_MESSAGE)


def test_an_unknown_query_key_on_generate_content_is_refused_without_echo(gateway: Gateway) -> None:
    """An unknown `sk-` key is sent as `?key=` on `generateContent`. The proxy must refuse with 401 and neither
    the key nor its hash may appear in the response."""
    secret: Final = f"sk-{marker()}"
    with wire_server(provider_reply) as upstream, gateway.scenario() as scenario:
        model: Final = scenario.model(model="gemini/gemini-2.5-flash", api_base=upstream.url, api_key=DEPLOYMENT_KEY)
        refused: Final = gateway.client.post(
            f"/v1beta/models/{model}:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "x"}]}]},
            params={"key": secret},
        )
        assert refused.status_code == 401, refused.text
        assert_not_echoed(refused, secret)
        assert posts(upstream) == ()
