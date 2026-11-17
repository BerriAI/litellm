"""Entra ID client secret flow for Azure deployments, with the token endpoint diverted to an owned TLS peer.

A deployment with `tenant_id`, `client_id` and `client_secret` and no key gets a bearer token from Entra ID.
On the Python path (`azure/` chat and `azure_ai/` chat) `initialize_azure_sdk_client` builds a
`ClientSecretCredential` token provider (`litellm/llms/azure/common_utils.py:925`,
`_cached_entra_id_token_provider` at `common_utils.py:268`) for the scope `azure_scope` or
`https://cognitiveservices.azure.com/.default`. On the Rust path (`azure_ai` Claude on `/v1/messages`, since
#51721 on 2026-11-14) `litellm-auth-azure` runs the same client secret grant natively
(`litellm/rust/crates/auth-azure/src/native.rs`). Both read `AZURE_AUTHORITY_HOST`, so this module boots its
own proxy with that variable pointing at a TLS peer on loopback that plays both Entra ID and Azure, and
trusts the peer's self signed certificate through `SSL_CERT_FILE`.

The token request is a form POST to `/<tenant>/oauth2/v2.0/token` with `grant_type=client_credentials`,
`client_id`, `client_secret` and `scope`. The minted token must reach Azure as `Authorization: Bearer`, and a
cached token must be reused across calls instead of one Entra round trip per request.

UNVERIFIED for the Python path: MSAL may try authority validation (instance discovery against the public
cloud, or an OpenID configuration GET) before the token POST. The peer answers the OpenID configuration GET,
but if instance discovery reaches the network these tests fail in CI for a harness reason, not a product one.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final
from urllib.parse import parse_qs, urlsplit

import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration._support.ports import allocate_port
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers.dec1_azure_support import (
    anthropic_message_json,
    chat_json,
    marker,
    register_model,
)

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 120)]

_TENANT: Final = "dec1-tenant"
_CLIENT_ID: Final = "dec1-client-id"
_CLIENT_SECRET: Final = "synthetic-dec1-client-secret"
_MINTED: Final = "minted-entra-token-dec1"
_WRONG_SECRET: Final = "wrong-secret"
_TOKEN_PATH: Final = f"/{_TENANT}/oauth2/v2.0/token"
_REMOVED_ENVIRONMENT: Final = (
    "AZURE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_AI_API_KEY",
    "AZURE_AD_TOKEN",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_SCOPE",
    "AZURE_USERNAME",
    "AZURE_PASSWORD",
    "AZURE_FEDERATED_TOKEN_FILE",
    "IDENTITY_ENDPOINT",
    "MSI_ENDPOINT",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
)


def _peer(request: Request) -> Reply:
    path: Final = urlsplit(request.target).path
    if request.method == "GET" and path.endswith("/.well-known/openid-configuration"):
        base: Final = f"https://127.0.0.1:{request.headers.get('host', '').split(':')[-1]}/{_TENANT}"
        return Reply(
            body=json.dumps(
                {
                    "token_endpoint": f"{base}/oauth2/v2.0/token",
                    "authorization_endpoint": f"{base}/oauth2/v2.0/authorize",
                    "issuer": f"{base}/v2.0",
                }
            ).encode()
        )
    if path == _TOKEN_PATH:
        form: Final = parse_qs(request.body.decode())
        if form.get("client_secret", [""])[-1] == _WRONG_SECRET:
            return Reply(
                status=401,
                body=json.dumps(
                    {"error": "invalid_client", "error_description": "AADSTS7000215: Invalid client secret provided."}
                ).encode(),
            )
        return Reply(
            body=json.dumps(
                {"access_token": _MINTED, "token_type": "Bearer", "expires_in": 3600, "ext_expires_in": 3600}
            ).encode()
        )
    if path.endswith("/anthropic/v1/messages"):
        return Reply(body=json.dumps(anthropic_message_json("pong")).encode())
    if request.method != "POST":
        return Reply(body=b'{"object":"list","data":[]}')
    return Reply(body=json.dumps(chat_json("pong")).encode())


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    wire: Wire

    def token_requests(self, requests: tuple[Request, ...]) -> tuple[Request, ...]:
        return tuple(request for request in requests if urlsplit(request.target).path == _TOKEN_PATH)

    def provider_calls(self, requests: tuple[Request, ...]) -> tuple[Request, ...]:
        return tuple(
            request for request in requests if request.method == "POST" and urlsplit(request.target).path != _TOKEN_PATH
        )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("dec1-azure-entra-client-secret")
    port: Final = allocate_port()
    cert, key = write_self_signed_cert(directory)
    environment: Final = {"AZURE_AUTHORITY_HOST": f"https://127.0.0.1:{port}", "SSL_CERT_FILE": str(cert)}
    with (
        wire_server(_peer, tls=server_context(cert, key), port=port) as wire,
        gateway_from_environment() as shared,
        owned_proxy(shared, directory, environment, remove_environment=_REMOVED_ENVIRONMENT) as proxy,
    ):
        yield _Rig(proxy, wire)


def _client_secret_params(rig: _Rig, client_id: str) -> dict[str, str]:
    """Each test passes its own `client_id`, so the per credential token cache never carries a token across tests."""
    return {
        "api_base": rig.wire.url,
        "tenant_id": _TENANT,
        "client_id": client_id,
        "client_secret": _CLIENT_SECRET,
    }


def _chat(gateway: Gateway, model: str) -> int:
    return gateway.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": marker()}]}
    ).status_code


def test_azure_chat_with_a_client_secret_sends_the_minted_entra_token(rig: _Rig) -> None:
    """An `azure/gpt-4o-dec1` deployment with tenant, client id and client secret and no key serves one chat call.

    The peer must receive a token POST to `/dec1-tenant/oauth2/v2.0/token` and then the chat POST with
    `Authorization: Bearer minted-entra-token-dec1` and no `api-key`; the caller gets HTTP 200.
    """
    with rig.gateway.scenario() as scenario:
        model: Final = register_model(
            scenario,
            model="azure/gpt-4o-dec1",
            api_version="2024-10-21",
            **_client_secret_params(rig, f"{_CLIENT_ID}-1"),
        )
        assert _chat(rig.gateway, model) == 200
        requests: Final = rig.wire.drain()
        assert len(rig.token_requests(requests)) >= 1
        (call,) = rig.provider_calls(requests)
        assert call.headers.get("authorization") == f"Bearer {_MINTED}"
        assert "api-key" not in call.headers, call.headers


def test_the_client_secret_grant_names_the_cognitive_services_scope(rig: _Rig) -> None:
    """Same deployment with no `azure_scope`.

    The form body of the token POST must have `grant_type=client_credentials`, `client_id=dec1-client-id-2`,
    `client_secret=synthetic-dec1-client-secret` and `scope=https://cognitiveservices.azure.com/.default`
    (`common_utils.py:906`).
    """
    with rig.gateway.scenario() as scenario:
        model: Final = register_model(
            scenario,
            model="azure/gpt-4o-dec2",
            api_version="2024-10-21",
            **_client_secret_params(rig, f"{_CLIENT_ID}-2"),
        )
        assert _chat(rig.gateway, model) == 200
        tokens: Final = rig.token_requests(rig.wire.drain())
        assert tokens, "no token request reached the authority host"
        form: Final = {name: values[-1] for name, values in parse_qs(tokens[-1].body.decode()).items()}
        assert form["grant_type"] == "client_credentials"
        assert form["client_id"] == f"{_CLIENT_ID}-2"
        assert form["client_secret"] == _CLIENT_SECRET
        assert form["scope"] == "https://cognitiveservices.azure.com/.default"


def test_a_custom_azure_scope_is_used_in_the_grant(rig: _Rig) -> None:
    """The deployment also sets `azure_scope: api://dec1-app/.default`.

    The token POST must ask for `scope=api://dec1-app/.default` instead of the Cognitive Services default.
    """
    with rig.gateway.scenario() as scenario:
        model: Final = register_model(
            scenario,
            model="azure/gpt-4o-dec3",
            api_version="2024-10-21",
            azure_scope="api://dec1-app/.default",
            **_client_secret_params(rig, f"{_CLIENT_ID}-3"),
        )
        assert _chat(rig.gateway, model) == 200
        tokens: Final = rig.token_requests(rig.wire.drain())
        form: Final = {name: values[-1] for name, values in parse_qs(tokens[-1].body.decode()).items()}
        assert form["scope"] == "api://dec1-app/.default"


def test_the_minted_token_is_reused_across_calls(rig: _Rig) -> None:
    """Three chat calls in a row on one client secret deployment.

    The token lives 3600 seconds, so the peer must see exactly one token POST and three chat POSTs: one Entra
    round trip per token lifetime, not per request (`_cached_entra_id_token_provider`).
    """
    with rig.gateway.scenario() as scenario:
        model: Final = register_model(
            scenario,
            model="azure/gpt-4o-dec4",
            api_version="2024-10-21",
            **_client_secret_params(rig, f"{_CLIENT_ID}-4"),
        )
        for _ in range(3):
            assert _chat(rig.gateway, model) == 200
        requests: Final = rig.wire.drain()
        assert len(rig.token_requests(requests)) == 1
        assert len(rig.provider_calls(requests)) == 3


def test_foundry_claude_messages_with_a_client_secret_send_the_minted_token(rig: _Rig) -> None:
    """An `azure_ai/claude-sonnet-4-5` deployment with tenant, client id and secret and no key serves `/v1/messages`.

    This is the Rust path (#51721): the peer must receive the token POST and then `POST /anthropic/v1/messages`
    with `Authorization: Bearer minted-entra-token-dec1` and no `x-api-key`.
    """
    with rig.gateway.scenario() as scenario:
        model: Final = register_model(
            scenario, model="azure_ai/claude-sonnet-4-5", **_client_secret_params(rig, f"{_CLIENT_ID}-5")
        )
        response: Final = rig.gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, "max_tokens": 64, "messages": [{"role": "user", "content": marker()}]},
            headers={"anthropic-version": "2023-06-01"},
        )
        assert response.status_code == 200, response.text
        requests: Final = rig.wire.drain()
        (call,) = rig.provider_calls(requests)
        assert urlsplit(call.target).path == "/anthropic/v1/messages"
        assert call.headers.get("authorization") == f"Bearer {_MINTED}"
        assert "x-api-key" not in call.headers, call.headers


def test_a_rejected_client_secret_fails_the_call_with_401(rig: _Rig) -> None:
    """A deployment whose `client_secret` is `wrong-secret`; the peer answers its token POST with 401 `invalid_client`.

    The caller must get HTTP 401, since Entra rejections are re-raised as authentication errors
    (`_entra_rejections_as_auth_errors`, `common_utils.py:961`), and the peer must receive no chat POST.
    UNVERIFIED: the exact status the wrapped `ClientAuthenticationError` maps to through the router.
    """
    with rig.gateway.scenario() as scenario:
        model: Final = register_model(
            scenario,
            model="azure/gpt-4o-dec5",
            api_version="2024-10-21",
            **{**_client_secret_params(rig, f"{_CLIENT_ID}-6"), "client_secret": _WRONG_SECRET},
        )
        assert _chat(rig.gateway, model) == 401
        assert rig.provider_calls(rig.wire.drain()) == ()
