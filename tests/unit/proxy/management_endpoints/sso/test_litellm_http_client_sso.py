import datetime
import ipaddress
import json
import ssl
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi import Request
from fastapi_sso.sso.base import DiscoveryDocument, OpenID, SSOBase

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # params is an untyped dict
)
from litellm.proxy._types import ProxyException
from litellm.proxy.management_endpoints.sso import (
    CustomMicrosoftSSO,
    LiteLLMGoogleSSO,
    create_generic_sso_provider,
)
from litellm.types.llms.custom_http import httpxSpecialProvider

_ACCESS_TOKEN: Final = "idp-access-token"
_USERINFO: Final = {
    "sub": "u-9445",
    "id": "u-9445",
    "email": "user9445@example.com",
    "email_verified": True,
    "mail": "user9445@example.com",
}


@dataclass(slots=True)
class _SelfSignedIdP:
    base_url: str
    requests: list[tuple[str, str, dict[str, str], str]] = field(default_factory=list)


def _write_self_signed_cert(directory: Path) -> tuple[Path, Path]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))]), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    cert_path = directory / "cert.pem"
    key_path = directory / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


@pytest.fixture
def self_signed_idp(tmp_path: Path) -> Iterator[_SelfSignedIdP]:
    """An HTTPS IdP on 127.0.0.1 whose certificate no trust store knows, like an internal-CA IdP."""
    cert_path, key_path = _write_self_signed_cert(tmp_path)
    idp = _SelfSignedIdP(base_url="")

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, body: Mapping[str, object]) -> None:
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode()
            idp.requests.append(("POST", self.path, {k.lower(): v for k, v in self.headers.items()}, body))
            self._reply({"access_token": _ACCESS_TOKEN, "token_type": "Bearer", "expires_in": 3600})

        def do_GET(self) -> None:
            idp.requests.append(("GET", self.path, {k.lower(): v for k, v in self.headers.items()}, ""))
            if self.path == "/discovery":
                self._reply(
                    {
                        "authorization_endpoint": f"{idp.base_url}/authorize",
                        "token_endpoint": f"{idp.base_url}/token",
                        "userinfo_endpoint": f"{idp.base_url}/userinfo",
                    }
                )
                return
            self._reply(_USERINFO)

        def log_message(self, format: str, *args: object) -> None:  # matches BaseHTTPRequestHandler
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    idp.base_url = f"https://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield idp
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def fresh_sso_http_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A new litellm client cache so the SSO client is built from this test's TLS settings."""
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    monkeypatch.delenv("SSL_VERIFY", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    yield


def _callback_request(code: str = "abc") -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "scheme": "https",
            "server": ("proxy.example.com", 443),
            "path": "/sso/callback",
            "query_string": f"code={code}".encode(),
            "headers": [],
        }
    )


def _generic_provider(idp: _SelfSignedIdP, _monkeypatch: pytest.MonkeyPatch) -> SSOBase:
    provider = create_generic_sso_provider(
        name="oidc",
        discovery_document=DiscoveryDocument(
            authorization_endpoint=f"{idp.base_url}/authorize",
            token_endpoint=f"{idp.base_url}/token",
            userinfo_endpoint=f"{idp.base_url}/userinfo",
        ),
        response_convertor=lambda userinfo, _session: OpenID(
            id=str(userinfo["sub"]), email=str(userinfo["email"]), provider="oidc"
        ),
    )
    return provider(
        client_id="litellm-ui",
        client_secret="s3cret",
        redirect_uri="https://proxy.example.com/sso/callback",
        allow_insecure_http=True,
    )


def _google_provider(idp: _SelfSignedIdP, _monkeypatch: pytest.MonkeyPatch) -> SSOBase:
    sso = LiteLLMGoogleSSO(
        client_id="litellm-ui", client_secret="s3cret", redirect_uri="https://proxy.example.com/sso/callback"
    )
    sso.discovery_url = f"{idp.base_url}/discovery"
    return sso


def _microsoft_provider(idp: _SelfSignedIdP, monkeypatch: pytest.MonkeyPatch) -> SSOBase:
    monkeypatch.setenv("MICROSOFT_AUTHORIZATION_ENDPOINT", f"{idp.base_url}/authorize")
    monkeypatch.setenv("MICROSOFT_TOKEN_ENDPOINT", f"{idp.base_url}/token")
    monkeypatch.setenv("MICROSOFT_USERINFO_ENDPOINT", f"{idp.base_url}/userinfo")
    return CustomMicrosoftSSO(
        client_id="litellm-ui",
        client_secret="s3cret",
        redirect_uri="https://proxy.example.com/sso/callback",
        tenant="common",
    )


_PROVIDERS: Final = {
    "generic": _generic_provider,
    "google": _google_provider,
    "microsoft": _microsoft_provider,
}


@pytest.mark.parametrize("build_provider", _PROVIDERS.values(), ids=_PROVIDERS.keys())
@pytest.mark.asyncio
async def test_login_honours_litellm_ssl_verify_against_self_signed_idp(
    build_provider: Callable[[_SelfSignedIdP, pytest.MonkeyPatch], SSOBase],
    self_signed_idp: _SelfSignedIdP,
    fresh_sso_http_client: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for LIT-9445: the token exchange and userinfo call went out on a bare httpx client,
    so an IdP on an internal CA failed with CERTIFICATE_VERIFY_FAILED even with litellm.ssl_verify=False."""
    monkeypatch.setattr(litellm, "ssl_verify", False)
    sso = build_provider(self_signed_idp, monkeypatch)

    result = await sso.verify_and_process(_callback_request(code="code-9445"))

    assert result is not None
    assert result.email == "user9445@example.com"
    assert result.id == "u-9445"
    assert sso.access_token == _ACCESS_TOKEN
    token_requests = [body for _method, path, _headers, body in self_signed_idp.requests if path == "/token"]
    assert len(token_requests) == 1
    assert "code=code-9445" in token_requests[0]
    assert "grant_type=authorization_code" in token_requests[0]
    userinfo_headers = [headers for _method, path, headers, _body in self_signed_idp.requests if path == "/userinfo"]
    assert len(userinfo_headers) == 1
    assert userinfo_headers[0]["authorization"] == f"Bearer {_ACCESS_TOKEN}"
    await get_async_httpx_client(llm_provider=httpxSpecialProvider.SSO_HANDLER).client.aclose()


@pytest.mark.asyncio
async def test_login_does_not_leave_the_bearer_token_on_the_shared_client(
    self_signed_idp: _SelfSignedIdP,
    fresh_sso_http_client: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SSO client is shared across logins, so one user's bearer token must not stick to it."""
    monkeypatch.setattr(litellm, "ssl_verify", False)
    sso = _generic_provider(self_signed_idp, monkeypatch)

    await sso.verify_and_process(_callback_request())

    client = get_async_httpx_client(llm_provider=httpxSpecialProvider.SSO_HANDLER).client
    assert "authorization" not in client.headers
    await client.aclose()


@pytest.mark.asyncio
async def test_untrusted_idp_certificate_fails_closed_naming_the_token_endpoint(
    self_signed_idp: _SelfSignedIdP,
    fresh_sso_http_client: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With verification on, an untrusted IdP certificate is a ProxyException that names the endpoint
    and the TLS error, not a bare httpx error that the proxy reports as an internal or database error."""
    monkeypatch.setattr(litellm, "ssl_verify", True)
    sso = _generic_provider(self_signed_idp, monkeypatch)

    with pytest.raises(ProxyException) as exc_info:
        await sso.verify_and_process(_callback_request())

    assert str(exc_info.value.code) == "401"
    assert f"{self_signed_idp.base_url}/token" in exc_info.value.message
    assert "certificate" in exc_info.value.message.lower()
    assert self_signed_idp.requests == []
    await get_async_httpx_client(llm_provider=httpxSpecialProvider.SSO_HANDLER).client.aclose()
