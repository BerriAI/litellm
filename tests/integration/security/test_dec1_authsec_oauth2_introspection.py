"""OAuth2 token introspection: no answer is cached, revocation is immediate, and the token stays out of logs.

Contract (litellm/proxy/auth/oauth2_check.py `Oauth2Handler.check_oauth2_token`; the plan in
litellm/rust/crates/gateway-auth-core/src/plan/introspection.rs picks `OAuth2Introspection` for LLM routes, info
routes and `/lens` when `enable_oauth2_auth` is on):

* Every request with an opaque bearer token on an introspected route calls `OAUTH_TOKEN_INFO_ENDPOINT` (a GET
  with `Authorization: Bearer <token>`); there is no cache in front of it. So a token the identity provider stops
  vouching for is refused on the very next request. A Rust port that adds a cache would keep revoked tokens
  alive for its TTL and must make that a visible decision.
* A token the endpoint answers with 4xx is refused with 401 and the message starts
  "Authentication Error, Oauth 2.0 Token validation failed: ". The refusal must not echo the token.
* The opaque token is not `sk-` shaped, so the log redaction for keys does not catch it; the DEBUG line
  "Oauth2 token validation for token=%s, response from endpoint=%s" prints it.

Every test here needs an enterprise license (`LITELLM_LICENSE` in the run environment), like
`authorization/test_dec1_authmodes_oauth2.py`, which covers the request shapes and field mapping.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.process import graceful_stop_seconds
from integration.security.dec1_authsec_support import (
    CHAT,
    OWNED_MODEL,
    OAuthDoor,
    assert_served,
    chat_body,
    oauth_door,
)

STATUS_FAILURE: Final = "Authentication Error, Oauth 2.0 Token validation failed: "

pytestmark: Final = [
    pytest.mark.timeout(2 * graceful_stop_seconds() + 180),
    pytest.mark.xdist_group("dec1-authsec-oauth2"),
]


def _token() -> str:
    return f"dec1opaque{uuid.uuid4().hex}"


@pytest.fixture(scope="module")
def oauth(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OAuthDoor]:
    with oauth_door(tmp_path_factory.mktemp("dec1-authsec-oauth2")) as rig:
        yield rig


def _vouch(oauth: OAuthDoor, token: str, sub: str = "dec1-authsec-oauth-user") -> None:
    oauth.info.grant(token, {"sub": sub, "role": "internal_user"})


def test_every_request_introspects_the_token_again(oauth: OAuthDoor) -> None:
    """An opaque token the double vouches for makes two chat calls. Both must be served, and the token-info
    double must have been asked about this token exactly twice: nothing caches the introspection answer."""
    token: Final = _token()
    _vouch(oauth, token)
    for _ in range(2):
        assert_served(oauth.door.gateway.request("POST", CHAT, chat_body(OWNED_MODEL), key=token))
    assert oauth.info.calls_for(token) == 2


def test_a_token_revoked_at_the_identity_provider_is_refused_on_the_next_request(oauth: OAuthDoor) -> None:
    """An opaque token is served once, then the double stops vouching for it (answers 401). The next chat call
    with that token must be refused with 401 whose message starts with the token validation failure prefix, and
    the provider must not be called for it."""
    token: Final = _token()
    _vouch(oauth, token)
    assert_served(oauth.door.gateway.request("POST", CHAT, chat_body(OWNED_MODEL), key=token))
    oauth.info.revoke(token)
    oauth.door.calls()
    refused: Final = oauth.door.gateway.request("POST", CHAT, chat_body(OWNED_MODEL), key=token)
    assert refused.status_code == 401, refused.text
    assert refused.json()["error"]["message"].startswith(STATUS_FAILURE), refused.text
    assert oauth.door.calls() == ()


def test_a_refused_token_is_not_echoed_in_the_refusal(oauth: OAuthDoor) -> None:
    """An opaque token the double does not know calls a chat model. The 401 body must not contain the token. The
    message embeds the httpx status error text, which names the token-info URL but not the bearer value."""
    token: Final = _token()
    refused: Final = oauth.door.gateway.request("POST", CHAT, chat_body(OWNED_MODEL), key=token)
    assert refused.status_code == 401, refused.text
    assert token not in refused.text


def test_the_token_is_sent_only_to_the_token_info_endpoint_never_to_the_provider(oauth: OAuthDoor) -> None:
    """A vouched opaque token makes a chat call. The provider double must receive the deployment key and must not
    see the opaque token in any header value."""
    token: Final = _token()
    _vouch(oauth, token)
    oauth.door.calls()
    assert_served(oauth.door.gateway.request("POST", CHAT, chat_body(OWNED_MODEL), key=token))
    (call,) = oauth.door.calls()
    assert [name for name, value in call.headers.items() if token in value] == []


def test_a_management_route_does_not_introspect_and_refuses_the_opaque_token(oauth: OAuthDoor) -> None:
    """A vouched opaque token calls `POST /key/generate`, a management route outside the introspection plan. It
    must be refused with 401 (it falls to the virtual key path and is not `sk-` shaped), and the token-info
    double must not be asked about it on that call."""
    token: Final = _token()
    _vouch(oauth, token)
    before: Final = oauth.info.calls_for(token)
    refused: Final = oauth.door.gateway.request("POST", "/key/generate", {}, key=token)
    assert refused.status_code == 401, refused.text
    assert oauth.info.calls_for(token) == before


def test_the_debug_log_never_holds_the_opaque_token(tmp_path_factory: pytest.TempPathFactory) -> None:
    """A proxy with `enable_oauth2_auth` and `LITELLM_LOG=DEBUG` serves one call for a vouched opaque token. After
    the proxy stops, its captured log must not contain the token. Suspected to fail: `check_oauth2_token` logs
    "Oauth2 token validation for token=%s" with the raw token at DEBUG, and the opaque value matches none of the
    `sk-` or `Bearer` redaction patterns in `litellm_core_utils/secret_redaction.py`."""
    pytest.skip("BUG: check_oauth2_token logs the raw OAuth2 token at DEBUG level")
    token: Final = _token()
    with oauth_door(tmp_path_factory.mktemp("dec1-authsec-oauth2-debug"), environment={"LITELLM_LOG": "DEBUG"}) as rig:
        _vouch(rig, token)
        assert_served(rig.door.gateway.request("POST", CHAT, chat_body(OWNED_MODEL), key=token))
        log: Final = rig.door.owned.log
    assert token.encode() not in log.read_bytes()
