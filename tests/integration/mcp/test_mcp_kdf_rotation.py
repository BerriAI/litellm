"""Bridge envelopes and gateway session tokens are keyed by HKDF-SHA256 over the master key.

Every cell mints a bearer in the test under keys the test derives itself (HKDF-SHA256 for the current
construction, scrypt for the legacy one) and presents it to a real proxy. The proxy is the only party
that derives keys from the master key, so admission proves which KDF it runs and whether the legacy
grace window is honoured.
"""

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import httpx
import pytest
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from integration._support.client import Gateway
from integration._support.mcp import McpPeer, mcp_peer, register_mcp, tool_calls
from integration._support.oauth_server import oauth_server
from integration._support.process import owned_proxy
from pydantic import SecretStr

from litellm.proxy._experimental.mcp_server.outbound_credentials.envelope import (
    EnvelopeKeys,
    SealedEnvelope,
    UpstreamTokenGrant,
    key_hash_identity,
    mint_envelope,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.session_token import (
    MintedSessionToken,
    SessionKeys,
    SessionPrincipal,
    mint_session_token,
)
from litellm.proxy.utils import hash_token

ACCEPT: Final = {"Accept": "application/json, text/event-stream"}
TOOLS_LIST: Final = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
ENVELOPE_SIGNING: Final = b"litellm-mcp-bridge:envelope-signing:"
ENVELOPE_ENCRYPTION: Final = b"litellm-mcp-bridge:envelope-encryption:"
SESSION_SIGNING: Final = b"litellm-mcp-gateway:session-signing:"
GRACE: Final = {"LITELLM_MCP_LEGACY_KDF_GRACE": "true"}


def _hkdf(master_key: str, info: bytes) -> str:
    return HKDF(algorithm=SHA256(), length=32, salt=None, info=info).derive(master_key.encode()).hex()


def _scrypt(master_key: str, salt: bytes) -> str:
    return hashlib.scrypt(master_key.encode(), salt=salt, n=2**15, r=8, p=1, maxmem=2**27, dklen=32).hex()


def _hkdf_envelope_keys(master_key: str) -> EnvelopeKeys:
    return EnvelopeKeys(
        signing_key=SecretStr(_hkdf(master_key, ENVELOPE_SIGNING)),
        encryption_key=SecretStr(_hkdf(master_key, ENVELOPE_ENCRYPTION)),
    )


def _scrypt_envelope_keys(master_key: str) -> EnvelopeKeys:
    return EnvelopeKeys(
        signing_key=SecretStr(_scrypt(master_key, ENVELOPE_SIGNING)),
        encryption_key=SecretStr(_scrypt(master_key, ENVELOPE_ENCRYPTION)),
    )


def _hkdf_session_keys(master_key: str) -> SessionKeys:
    return SessionKeys(signing_key=SecretStr(_hkdf(master_key, SESSION_SIGNING)))


def _scrypt_session_keys(master_key: str) -> SessionKeys:
    return SessionKeys(signing_key=SecretStr(_scrypt(master_key, SESSION_SIGNING)))


def _envelope(identity_server: str, key: str, keys: EnvelopeKeys, upstream_token: str) -> str:
    sealed: Final = mint_envelope(
        key_hash_identity(server_id=identity_server, key_hash=hash_token(key)),
        UpstreamTokenGrant(access_token=SecretStr(upstream_token), token_type="Bearer", expires_in=600),
        keys,
        datetime.now(timezone.utc),
    )
    assert isinstance(sealed, SealedEnvelope), sealed
    return sealed.token.get_secret_value()


def _session(user_id: str, keys: SessionKeys) -> str:
    minted: Final = mint_session_token(
        SessionPrincipal(user_id=user_id, client_id="dcr-integration"), keys, datetime.now(timezone.utc)
    )
    assert isinstance(minted, MintedSessionToken), minted
    return minted.token.get_secret_value()


def _register_bridge(scenario, peer: McpPeer, alias: str, issuer: str) -> str:
    return register_mcp(
        scenario,
        peer,
        alias,
        auth_type="oauth_delegate",
        dcr_bridge=True,
        issuer=issuer,
        authorization_url=issuer + "/authorize",
        token_url=issuer + "/token",
        registration_url=issuer + "/register",
    )


def _tools_list(client: httpx.Client, path: str, bearer: str) -> httpx.Response:
    return client.post(path, headers={**ACCEPT, "Authorization": f"Bearer {bearer}"}, json=TOOLS_LIST)


def _peer_authorizations(peer: McpPeer) -> tuple[bytes | None, ...]:
    return tuple(
        value if isinstance(value := call["headers"].get(b"authorization"), bytes) else None
        for call in peer.drain()
        if isinstance(call["headers"], dict)
    )


def _assert_admitted(response: httpx.Response, peer: McpPeer, upstream_token: str) -> None:
    assert response.status_code == 200, response.text
    assert "tools" in response.text, response.text
    forwarded: Final = _peer_authorizations(peer)
    assert forwarded and set(forwarded) == {f"Bearer {upstream_token}".encode()}, forwarded


def _assert_rejected(response: httpx.Response, peer: McpPeer) -> None:
    assert response.status_code == 401, response.text
    assert "invalid_token" in response.headers.get("www-authenticate", ""), response.headers
    assert peer.drain() == ()


def test_envelope_minted_under_hkdf_sha256_keys_is_admitted(gateway: Gateway) -> None:
    with mcp_peer() as peer, oauth_server() as auth, gateway.scenario() as scenario:
        alias: Final = "kdf" + uuid.uuid4().hex[:8]
        identity: Final = _register_bridge(scenario, peer, alias, auth.issuer)
        key: Final = scenario.key(user_id=scenario.user(), object_permission={"mcp_servers": [identity]})
        upstream_token: Final = "up-" + uuid.uuid4().hex
        peer.drain()
        envelope: Final = _envelope(identity, key, _hkdf_envelope_keys(gateway.key), upstream_token)
        _assert_admitted(_tools_list(gateway.client, f"/{alias}/mcp", envelope), peer, upstream_token)


def test_envelope_minted_under_legacy_scrypt_keys_is_rejected_without_grace(gateway: Gateway) -> None:
    with mcp_peer() as peer, oauth_server() as auth, gateway.scenario() as scenario:
        alias: Final = "kdf" + uuid.uuid4().hex[:8]
        identity: Final = _register_bridge(scenario, peer, alias, auth.issuer)
        key: Final = scenario.key(user_id=scenario.user(), object_permission={"mcp_servers": [identity]})
        peer.drain()
        envelope: Final = _envelope(identity, key, _scrypt_envelope_keys(gateway.key), "up-" + uuid.uuid4().hex)
        _assert_rejected(_tools_list(gateway.client, f"/{alias}/mcp", envelope), peer)


@pytest.mark.parametrize("kdf", ("hkdf", "scrypt"))
def test_envelope_of_either_kdf_is_admitted_during_the_legacy_grace_window(
    gateway: Gateway, tmp_path: Path, kdf: str
) -> None:
    with (
        owned_proxy(gateway, tmp_path, GRACE) as graced,
        mcp_peer() as peer,
        oauth_server() as auth,
        graced.scenario() as scenario,
    ):
        alias: Final = "kdf" + uuid.uuid4().hex[:8]
        identity: Final = _register_bridge(scenario, peer, alias, auth.issuer)
        key: Final = scenario.key(user_id=scenario.user(), object_permission={"mcp_servers": [identity]})
        upstream_token: Final = "up-" + uuid.uuid4().hex
        peer.drain()
        keys: Final = _hkdf_envelope_keys(graced.key) if kdf == "hkdf" else _scrypt_envelope_keys(graced.key)
        envelope: Final = _envelope(identity, key, keys, upstream_token)
        _assert_admitted(_tools_list(graced.client, f"/{alias}/mcp", envelope), peer, upstream_token)


def test_envelope_under_a_foreign_master_key_is_rejected_during_grace(gateway: Gateway, tmp_path: Path) -> None:
    with (
        owned_proxy(gateway, tmp_path, GRACE) as graced,
        mcp_peer() as peer,
        oauth_server() as auth,
        graced.scenario() as scenario,
    ):
        alias: Final = "kdf" + uuid.uuid4().hex[:8]
        identity: Final = _register_bridge(scenario, peer, alias, auth.issuer)
        key: Final = scenario.key(user_id=scenario.user(), object_permission={"mcp_servers": [identity]})
        peer.drain()
        foreign: Final = "sk-foreign-" + uuid.uuid4().hex
        for keys in (_hkdf_envelope_keys(foreign), _scrypt_envelope_keys(foreign)):
            _assert_rejected(_tools_list(graced.client, f"/{alias}/mcp", _envelope(identity, key, keys, "up")), peer)


def test_grace_variable_set_to_anything_but_true_keeps_legacy_envelopes_rejected(
    gateway: Gateway, tmp_path: Path
) -> None:
    with (
        owned_proxy(gateway, tmp_path, {"LITELLM_MCP_LEGACY_KDF_GRACE": "yes"}) as proxy,
        mcp_peer() as peer,
        oauth_server() as auth,
        proxy.scenario() as scenario,
    ):
        alias: Final = "kdf" + uuid.uuid4().hex[:8]
        identity: Final = _register_bridge(scenario, peer, alias, auth.issuer)
        key: Final = scenario.key(user_id=scenario.user(), object_permission={"mcp_servers": [identity]})
        peer.drain()
        envelope: Final = _envelope(identity, key, _scrypt_envelope_keys(proxy.key), "up")
        _assert_rejected(_tools_list(proxy.client, f"/{alias}/mcp", envelope), peer)


def test_session_token_minted_under_hkdf_sha256_key_is_admitted(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        token: Final = _session(scenario.user(), _hkdf_session_keys(gateway.key))
        response: Final = _tools_list(gateway.client, "/mcp", token)
        assert response.status_code == 200, response.text
        assert "tools" in response.text, response.text


def test_session_token_minted_under_legacy_scrypt_key_is_rejected_without_grace(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        token: Final = _session(scenario.user(), _scrypt_session_keys(gateway.key))
        response: Final = _tools_list(gateway.client, "/mcp", token)
        assert response.status_code == 401, response.text
        assert "invalid_token" in response.headers.get("www-authenticate", ""), response.headers


@pytest.mark.parametrize("kdf", ("hkdf", "scrypt"))
def test_session_token_of_either_kdf_is_admitted_during_the_legacy_grace_window(
    gateway: Gateway, tmp_path: Path, kdf: str
) -> None:
    with owned_proxy(gateway, tmp_path, GRACE) as graced, graced.scenario() as scenario:
        keys: Final = _hkdf_session_keys(graced.key) if kdf == "hkdf" else _scrypt_session_keys(graced.key)
        response: Final = _tools_list(graced.client, "/mcp", _session(scenario.user(), keys))
        assert response.status_code == 200, response.text
        assert "tools" in response.text, response.text


def test_tampered_session_token_is_rejected_during_grace(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy(gateway, tmp_path, GRACE) as graced, graced.scenario() as scenario:
        token: Final = _session(scenario.user(), _scrypt_session_keys(graced.key))
        tampered: Final = token[:-2] + ("AA" if token[-2:] != "AA" else "BB")
        response: Final = _tools_list(graced.client, "/mcp", tampered)
        assert response.status_code == 401, response.text
        assert "invalid_token" in response.headers.get("www-authenticate", ""), response.headers
