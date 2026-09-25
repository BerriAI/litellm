import base64
import hashlib
import json
import secrets
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final
from urllib.parse import parse_qs, urlencode, urlsplit

from integration._support.wire import Reply, Request, Wire, wire_server

TOKEN_EXCHANGE: Final = "urn:ietf:params:oauth:grant-type:token-exchange"


@dataclass(slots=True)
class AuthorizationServer:
    wire: Wire
    clients: dict[str, str] = field(default_factory=dict)
    codes: dict[str, dict[str, str]] = field(default_factory=dict)
    access_tokens: dict[str, dict[str, str]] = field(default_factory=dict)
    refresh_tokens: dict[str, dict[str, str]] = field(default_factory=dict)
    revoked: set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def issuer(self) -> str:
        return self.wire.url

    def drain(self) -> tuple[Request, ...]:
        return self.wire.drain()

    def token_requests(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {name: values[0] for name, values in parse_qs(item.body.decode()).items()}
            for item in self.drain()
            if item.target.startswith("/token")
        )

    def is_live(self, token: str) -> bool:
        with self.lock:
            return token in self.access_tokens and token not in self.revoked

    def issue(self, grant: str, client_id: str, subject: str, scope: str) -> dict[str, object]:
        access: Final = f"at-{grant}-{secrets.token_urlsafe(8)}"
        refresh: Final = f"rt-{secrets.token_urlsafe(8)}"
        with self.lock:
            self.access_tokens[access] = {"client_id": client_id, "subject": subject, "scope": scope, "grant": grant}
            self.refresh_tokens[refresh] = {"client_id": client_id, "subject": subject, "scope": scope}
        return {
            "access_token": access,
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": refresh,
            "scope": scope,
        }


def _pkce_matches(challenge: str, verifier: str) -> bool:
    digest: Final = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode() == challenge


def _json(status: int, body: dict[str, object]) -> Reply:
    return Reply(status=status, body=json.dumps(body).encode())


def _client_credentials(request: Request, form: dict[str, str]) -> tuple[str, str | None]:
    header: Final = request.headers.get("authorization", "")
    if header.lower().startswith("basic "):
        decoded: Final = base64.b64decode(header.split(" ", 1)[1]).decode()
        client_id, _, secret = decoded.partition(":")
        return client_id, secret
    return form.get("client_id", ""), form.get("client_secret")


@contextmanager
def oauth_server(*, scopes: tuple[str, ...] = ("tools.read", "tools.call")) -> Iterator[AuthorizationServer]:
    holder: list[AuthorizationServer] = []

    def respond(request: Request) -> Reply:
        server: Final = holder[0]
        path: Final = urlsplit(request.target).path
        query: Final = {name: values[0] for name, values in parse_qs(urlsplit(request.target).query).items()}
        form: Final = {name: values[0] for name, values in parse_qs(request.body.decode()).items()}
        if path.startswith("/.well-known/oauth-authorization-server") or path == "/.well-known/openid-configuration":
            return _json(
                200,
                {
                    "issuer": server.issuer,
                    "authorization_endpoint": server.issuer + "/authorize",
                    "token_endpoint": server.issuer + "/token",
                    "registration_endpoint": server.issuer + "/register",
                    "revocation_endpoint": server.issuer + "/revoke",
                    "introspection_endpoint": server.issuer + "/introspect",
                    "scopes_supported": list(scopes),
                    "response_types_supported": ["code"],
                    "grant_types_supported": [
                        "authorization_code",
                        "refresh_token",
                        "client_credentials",
                        TOKEN_EXCHANGE,
                    ],
                    "code_challenge_methods_supported": ["S256"],
                    "token_endpoint_auth_methods_supported": ["client_secret_post", "client_secret_basic", "none"],
                },
            )
        if path == "/register" and request.method == "POST":
            metadata: Final = json.loads(request.body or b"{}")
            client_id: Final = f"dcr-{uuid.uuid4().hex[:12]}"
            secret: Final = f"secret-{secrets.token_urlsafe(8)}"
            with server.lock:
                server.clients[client_id] = secret
            return _json(
                201,
                {
                    "client_id": client_id,
                    "client_secret": secret,
                    "client_id_issued_at": 0,
                    "redirect_uris": metadata.get("redirect_uris", []),
                    "grant_types": metadata.get("grant_types", ["authorization_code"]),
                    "token_endpoint_auth_method": metadata.get("token_endpoint_auth_method", "client_secret_post"),
                },
            )
        if path == "/authorize" and request.method == "GET":
            missing: Final = tuple(
                name for name in ("client_id", "redirect_uri", "code_challenge", "state") if name not in query
            )
            if missing or query.get("code_challenge_method", "S256") != "S256" or query.get("response_type") != "code":
                return _json(400, {"error": "invalid_request", "missing": list(missing), "received": query})
            code: Final = f"code-{secrets.token_urlsafe(8)}"
            with server.lock:
                server.codes[code] = {
                    "client_id": query["client_id"],
                    "redirect_uri": query["redirect_uri"],
                    "code_challenge": query["code_challenge"],
                    "scope": query.get("scope", " ".join(scopes)),
                }
            location: Final = (
                query["redirect_uri"]
                + ("&" if "?" in query["redirect_uri"] else "?")
                + urlencode({"code": code, "state": query["state"]})
            )
            return Reply(status=302, body=b"", headers={"location": location})
        if path == "/token" and request.method == "POST":
            grant: Final = form.get("grant_type", "")
            client_id, client_secret = _client_credentials(request, form)
            if grant == "authorization_code":
                with server.lock:
                    issued: Final = server.codes.pop(form.get("code", ""), None)
                if issued is None:
                    return _json(400, {"error": "invalid_grant", "error_description": "unknown or reused code"})
                if issued["client_id"] != client_id:
                    return _json(400, {"error": "invalid_client", "error_description": "code issued to another client"})
                if not _pkce_matches(issued["code_challenge"], form.get("code_verifier", "")):
                    return _json(400, {"error": "invalid_grant", "error_description": "pkce verifier mismatch"})
                return _json(200, server.issue("authorization_code", client_id, "integration-user", issued["scope"]))
            if grant == "refresh_token":
                with server.lock:
                    known: Final = server.refresh_tokens.pop(form.get("refresh_token", ""), None)
                if known is None:
                    return _json(400, {"error": "invalid_grant", "error_description": "unknown refresh token"})
                return _json(200, server.issue("refresh_token", known["client_id"], known["subject"], known["scope"]))
            if grant == "client_credentials":
                with server.lock:
                    expected: Final = server.clients.get(client_id)
                if not client_id or (expected is not None and expected != client_secret) or not client_secret:
                    return _json(401, {"error": "invalid_client"})
                return _json(200, server.issue("client_credentials", client_id, client_id, form.get("scope", "")))
            if grant == TOKEN_EXCHANGE:
                subject: Final = form.get("subject_token", "")
                if not subject:
                    return _json(400, {"error": "invalid_request", "error_description": "subject_token required"})
                if not client_id:
                    return _json(401, {"error": "invalid_client"})
                token: Final = server.issue("token_exchange", client_id, f"exchanged:{subject}", form.get("scope", ""))
                return _json(200, {**token, "issued_token_type": "urn:ietf:params:oauth:token-type:access_token"})
            return _json(400, {"error": "unsupported_grant_type", "grant_type": grant})
        if path == "/revoke" and request.method == "POST":
            with server.lock:
                server.revoked.add(form.get("token", ""))
            return Reply(status=200, body=b"{}")
        if path == "/introspect" and request.method == "POST":
            token: Final = form.get("token", "")
            with server.lock:
                info: Final = server.access_tokens.get(token)
                active: Final = info is not None and token not in server.revoked
            return _json(200, {"active": active, **(info or {})})
        return _json(404, {"error": "not_found", "path": path, "method": request.method})

    with wire_server(respond) as wire:
        holder.append(AuthorizationServer(wire))
        yield holder[0]
