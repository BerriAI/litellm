"""Browser sign-in for ``lite login --pkce``: OAuth 2.1 authorization code + PKCE S256
against the proxy's own authorization server, as a public client on a loopback redirect.
The proxy publishes everything this needs at ``/.well-known/litellm-cli-auth``, so a CLI
in any other language can run the same steps from that document alone."""

from __future__ import annotations

import hashlib
import secrets
import socket
import threading
import time
import webbrowser
from base64 import urlsafe_b64encode
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm.litellm_core_utils.cli_token_utils import CLI_TOKEN_FRESHNESS_BUFFER_SECONDS

if TYPE_CHECKING:
    from .auth import CliTokenData

CLI_AUTH_DISCOVERY_PATH: Final = "/.well-known/litellm-cli-auth"
CALLBACK_PATH: Final = "/callback"
LOGIN_TIMEOUT_SECONDS: Final = 300
_HTTP_TIMEOUT_SECONDS: Final = 15
_CLIENT_NAME: Final = "litellm-cli"


class CliAuthContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract_version: Literal[1]
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str
    revocation_endpoint: str
    resource: str
    code_challenge_methods_supported: tuple[str, ...]


class _RegisteredClient(BaseModel):
    model_config = ConfigDict(frozen=True)

    client_id: str = Field(min_length=1)


class _TokenResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    access_token: str = Field(min_length=1)
    expires_in: int = Field(gt=0)
    refresh_token: str = Field(min_length=1)
    user_id: str | None = None
    team_id: str | None = None


@dataclass(frozen=True, slots=True)
class PkceFailure:
    reason: str


@dataclass(frozen=True, slots=True)
class RevocationUnavailable:
    reason: str


@dataclass(frozen=True, slots=True)
class PkceCredential:
    access_token: str
    refresh_token: str
    expires_at: float
    client_id: str
    token_endpoint: str
    revocation_endpoint: str
    resource: str
    user_id: str | None
    team_id: str | None


@dataclass(frozen=True, slots=True)
class CallbackCode:
    code: str


@dataclass(frozen=True, slots=True)
class CallbackDenied:
    error: str
    description: str | None


CallbackOutcome = CallbackCode | CallbackDenied


class Http(Protocol):
    def get(self, url: str, *, timeout: float) -> requests.Response: ...

    def post(
        self,
        url: str,
        *,
        data: Mapping[str, str] | None = None,
        json: Mapping[str, object] | None = None,
        timeout: float,
        allow_redirects: bool,
    ) -> requests.Response: ...


class LoopbackServer(HTTPServer):
    """The OS-assigned loopback listener the browser is sent back to. Only the response
    carrying the pending sign-in's ``state`` settles it; anything else (a stray request, a
    stale tab, an attacker poking the port) gets a 400 and the wait continues. A connection
    that opens and then sends nothing is dropped after ``connection_timeout_seconds`` so it
    cannot hold the single-threaded wait past its deadline."""

    def __init__(self, expected_state: str, connection_timeout_seconds: float = 5) -> None:
        super().__init__(("127.0.0.1", 0), _CallbackHandler)
        self.expected_state: Final = expected_state
        self.connection_timeout_seconds: Final = connection_timeout_seconds
        self.outcome: CallbackOutcome | None = None
        self.timeout = 1

    @property
    def redirect_uri(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}{CALLBACK_PATH}"

    def get_request(self) -> tuple[socket.socket, object]:
        accepted: Final[tuple[socket.socket, object]] = super().get_request()
        accepted[0].settimeout(self.connection_timeout_seconds)
        return accepted

    def wait(
        self, timeout_seconds: float, clock: Callable[[], float] = time.monotonic
    ) -> CallbackOutcome | PkceFailure:
        deadline: Final = clock() + timeout_seconds
        while self.outcome is None:
            if clock() >= deadline:
                return PkceFailure("timed out waiting for the browser sign-in to finish")
            self.handle_request()
        return self.outcome


class _CallbackHandler(BaseHTTPRequestHandler):
    server: LoopbackServer  # pyright: ignore[reportIncompatibleVariableOverride]  # only ever constructed by LoopbackServer

    def do_GET(self) -> None:
        parsed: Final = urlparse(self.path)
        if parsed.path != CALLBACK_PATH:
            self._respond(404, "Not found.")
            return
        params: Final = parse_qs(parsed.query)
        if _first(params, "state") != self.server.expected_state:
            self._respond(400, "This response does not belong to the pending sign-in; still waiting.")
            return
        error: Final = _first(params, "error")
        if error is not None:
            self.server.outcome = CallbackDenied(error=error, description=_first(params, "error_description"))
            self._respond(200, "Sign-in was not approved. You can close this window.")
            return
        code: Final = _first(params, "code")
        if code is None:
            self._respond(400, "The sign-in response carried no authorization code; still waiting.")
            return
        self.server.outcome = CallbackCode(code=code)
        self._respond(200, "Signed in to LiteLLM. You can close this window and return to the terminal.")

    def log_message(self, format: str, *args: object) -> None:
        return

    def _respond(self, status: int, text: str) -> None:
        body: Final = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _first(params: Mapping[str, Sequence[str]], key: str) -> str | None:
    values: Final = params.get(key)
    return values[0] if values else None


def discover_cli_auth(base_url: str, http: Http) -> CliAuthContract | PkceFailure:
    url: Final = f"{base_url.rstrip('/')}{CLI_AUTH_DISCOVERY_PATH}"
    try:
        response: Final = http.get(url, timeout=_HTTP_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        return PkceFailure(f"could not reach {url}: {exc}")
    if response.status_code != 200:
        return PkceFailure(
            f"{url} answered {response.status_code}; this proxy version does not support `lite login --pkce`"
        )
    try:
        contract: Final = CliAuthContract.model_validate(response.json())
    except (ValueError, ValidationError) as exc:
        return PkceFailure(f"{url} returned an unsupported discovery document: {exc}")
    if "S256" not in contract.code_challenge_methods_supported:
        return PkceFailure("the proxy does not support PKCE S256")
    if _canonical_url(contract.issuer) != _canonical_url(base_url):
        return PkceFailure(f"{url} is issued for {contract.issuer}, not {base_url}; pass that address as --base-url")
    foreign: Final = _endpoints_outside(contract, _origin(base_url))
    if foreign:
        return PkceFailure(
            f"{url} names endpoints outside {base_url} ({', '.join(foreign)}); refusing to send credentials there"
        )
    return contract


def _endpoints_outside(contract: CliAuthContract, origin: str | None) -> tuple[str, ...]:
    endpoints: Final = (
        contract.authorization_endpoint,
        contract.token_endpoint,
        contract.registration_endpoint,
        contract.revocation_endpoint,
        contract.resource,
    )
    return tuple(endpoint for endpoint in endpoints if origin is None or _origin(endpoint) != origin)


def _origin(url: str) -> str | None:
    """``scheme://host:port`` with the default port made explicit, so the same server spelled
    two ways (``https://llm.example.com`` and ``https://LLM.example.com:443/``) compares equal
    and two different servers never do."""
    parsed: Final = urlparse(url)
    try:
        port: Final = parsed.port
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None
    host: Final = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    return f"{parsed.scheme}://{host}:{port or (443 if parsed.scheme == 'https' else 80)}"


def _canonical_url(url: str) -> str | None:
    """The origin plus the path with its trailing slash dropped: the RFC 8414 section 3.3
    identity check, so a document can only ever be accepted for the proxy it was fetched from."""
    origin: Final = _origin(url)
    return None if origin is None else f"{origin}{urlparse(url).path.rstrip('/')}"


class _ClientRegistration(TypedDict):
    client_name: ReadOnly[str]
    redirect_uris: ReadOnly[tuple[str, ...]]
    grant_types: ReadOnly[tuple[str, ...]]
    response_types: ReadOnly[tuple[str, ...]]
    token_endpoint_auth_method: ReadOnly[Literal["none"]]


def _form(**fields: str) -> Mapping[str, str]:
    return MappingProxyType(fields)


def _refused_redirect(request_name: str, response: requests.Response) -> PkceFailure | None:
    """Every POST to the proxy is sent with ``allow_redirects=False``: a 307 or 308 would make
    ``requests`` replay the form, code and verifier or refresh token included, wherever ``Location``
    points, past the origin check discovery passed."""
    if not 300 <= response.status_code < 400:
        return None
    return PkceFailure(
        f"{request_name} redirected to {response.headers.get('Location', 'another address')}; refusing to follow it"
    )


def register_client(contract: CliAuthContract, redirect_uri: str, http: Http) -> str | PkceFailure:
    registration: Final[_ClientRegistration] = {
        "client_name": _CLIENT_NAME,
        "redirect_uris": (redirect_uri,),
        "grant_types": ("authorization_code", "refresh_token"),
        "response_types": ("code",),
        "token_endpoint_auth_method": "none",
    }
    try:
        response: Final = http.post(
            contract.registration_endpoint, json=registration, timeout=_HTTP_TIMEOUT_SECONDS, allow_redirects=False
        )
    except requests.RequestException as exc:
        return PkceFailure(f"client registration failed: {exc}")
    redirected: Final = _refused_redirect("client registration", response)
    if redirected is not None:
        return redirected
    if response.status_code not in (200, 201):
        return PkceFailure(f"client registration failed with {response.status_code}: {_error_detail(response)}")
    try:
        return _RegisteredClient.model_validate(response.json()).client_id
    except (ValueError, ValidationError) as exc:
        return PkceFailure(f"client registration returned an unexpected body: {exc}")


def pkce_pair() -> tuple[str, str]:
    verifier: Final = secrets.token_urlsafe(64)
    digest: Final = hashlib.sha256(verifier.encode("ascii")).digest()
    return verifier, urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def authorize_url(contract: CliAuthContract, client_id: str, redirect_uri: str, state: str, code_challenge: str) -> str:
    query: Final = urlencode(
        _form(
            response_type="code",
            client_id=client_id,
            redirect_uri=redirect_uri,
            state=state,
            code_challenge=code_challenge,
            code_challenge_method="S256",
            resource=contract.resource,
        )
    )
    return f"{contract.authorization_endpoint}?{query}"


def redeem_code(
    contract: CliAuthContract,
    client_id: str,
    redirect_uri: str,
    code: str,
    code_verifier: str,
    http: Http,
    now: Callable[[], float] = time.time,
) -> PkceCredential | PkceFailure:
    return _token_request(
        token_endpoint=contract.token_endpoint,
        revocation_endpoint=contract.revocation_endpoint,
        resource=contract.resource,
        client_id=client_id,
        form=_form(
            grant_type="authorization_code",
            code=code,
            redirect_uri=redirect_uri,
            client_id=client_id,
            code_verifier=code_verifier,
            resource=contract.resource,
        ),
        http=http,
        now=now,
    )


def refresh_credential(
    token_endpoint: str,
    revocation_endpoint: str,
    resource: str,
    client_id: str,
    refresh_token: str,
    http: Http,
    now: Callable[[], float] = time.time,
) -> PkceCredential | PkceFailure:
    return _token_request(
        token_endpoint=token_endpoint,
        revocation_endpoint=revocation_endpoint,
        resource=resource,
        client_id=client_id,
        form=_form(grant_type="refresh_token", refresh_token=refresh_token, client_id=client_id, resource=resource),
        http=http,
        now=now,
    )


def _token_request(
    token_endpoint: str,
    revocation_endpoint: str,
    resource: str,
    client_id: str,
    form: Mapping[str, str],
    http: Http,
    now: Callable[[], float],
) -> PkceCredential | PkceFailure:
    try:
        response: Final = http.post(token_endpoint, data=form, timeout=_HTTP_TIMEOUT_SECONDS, allow_redirects=False)
    except requests.RequestException as exc:
        return PkceFailure(f"token request failed: {exc}")
    redirected: Final = _refused_redirect("token request", response)
    if redirected is not None:
        return redirected
    if response.status_code != 200:
        return PkceFailure(f"token request failed with {response.status_code}: {_error_detail(response)}")
    try:
        token: Final = _TokenResponse.model_validate(response.json())
    except (ValueError, ValidationError) as exc:
        return PkceFailure(f"token endpoint returned an unexpected body: {exc}")
    return PkceCredential(
        access_token=token.access_token,
        refresh_token=token.refresh_token,
        expires_at=now() + token.expires_in,
        client_id=client_id,
        token_endpoint=token_endpoint,
        revocation_endpoint=revocation_endpoint,
        resource=resource,
        user_id=token.user_id,
        team_id=token.team_id,
    )


def revoke_credential(
    revocation_endpoint: str, client_id: str, refresh_token: str, http: Http
) -> PkceFailure | RevocationUnavailable | None:
    try:
        response: Final = http.post(
            revocation_endpoint,
            data=_form(token=refresh_token, token_type_hint="refresh_token", client_id=client_id),
            timeout=_HTTP_TIMEOUT_SECONDS,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        return PkceFailure(f"revocation request failed: {exc}")
    redirected: Final = _refused_redirect("revocation request", response)
    if redirected is not None:
        return redirected
    if response.status_code == 503:
        return RevocationUnavailable(f"revocation failed with 503: {_error_detail(response)}")
    if response.status_code != 200:
        return PkceFailure(f"revocation failed with {response.status_code}: {_error_detail(response)}")
    return None


_ERROR_BODY: Final = TypeAdapter(Mapping[str, object])


def _error_detail(response: requests.Response) -> str:
    try:
        body: Final = _ERROR_BODY.validate_json(response.content)
    except ValidationError:
        return response.text[:200]
    return str(body.get("error_description") or body.get("error") or body.get("detail") or body)[:200]


def run_pkce_login(
    base_url: str,
    http: Http,
    open_browser: Callable[[str], object] = webbrowser.open,
    echo: Callable[[str], None] = print,
    timeout_seconds: float = LOGIN_TIMEOUT_SECONDS,
) -> PkceCredential | PkceFailure:
    contract: Final = discover_cli_auth(base_url, http)
    if isinstance(contract, PkceFailure):
        return contract
    state: Final = secrets.token_urlsafe(32)
    verifier, challenge = pkce_pair()
    with LoopbackServer(state) as server:
        client_id: Final = register_client(contract, server.redirect_uri, http)
        if isinstance(client_id, PkceFailure):
            return client_id
        url: Final = authorize_url(contract, client_id, server.redirect_uri, state, challenge)
        echo(f"Opening browser to: {url}")
        echo("Approve the sign-in in your browser. Waiting...")
        threading.Thread(target=open_browser, args=(url,), name="lite-login-browser", daemon=True).start()
        outcome: Final = server.wait(timeout_seconds)
    match outcome:
        case PkceFailure():
            return outcome
        case CallbackDenied():
            return PkceFailure(f"sign-in was not approved ({outcome.error}): {outcome.description or 'no details'}")
        case CallbackCode():
            return redeem_code(contract, client_id, server.redirect_uri, outcome.code, verifier, http)


def pkce_token_record(base_url: str, credential: PkceCredential) -> CliTokenData:
    record: Final[CliTokenData] = {
        "base_url": base_url.rstrip("/"),
        "key": credential.access_token,
        "user_id": credential.user_id or "cli-user",
        "user_email": "unknown",
        "user_role": "cli",
        "auth_header_name": "Authorization",
        "jwt_token": "",
        "timestamp": time.time(),
        "expires_at": credential.expires_at,
        "refresh_token": credential.refresh_token,
        "client_id": credential.client_id,
        "token_endpoint": credential.token_endpoint,
        "revocation_endpoint": credential.revocation_endpoint,
        "resource": credential.resource,
        "team_id": credential.team_id,
    }
    return record


def _ignore_warning(_message: str) -> None:
    return None


def fresh_api_key(
    token_data: Mapping[str, object],
    save: Callable[[CliTokenData], None],
    http: Http,
    *,
    reload: Callable[[], Mapping[str, object] | None],
    now: Callable[[], float] = time.time,
    warn: Callable[[str], None] = _ignore_warning,
) -> str | None:
    """The stored key, refreshed first when it is about to expire and a refresh token is
    on file. The refresh fires at the same moment ``is_cli_token_fresh`` stops calling the
    key fresh, so a command that checks freshness and then asks for the key never disagrees
    with itself. The rotated pair is saved before the new key is returned, so a crash after
    this point never strands the CLI with a burned refresh token. A refresh that fails
    reads the record again, because a sibling ``lite`` process may have rotated the pair
    first, in which case the key it saved for this same proxy is the live one; when no sibling
    did, the reason the proxy gave goes to ``warn`` so a revoked or refused refresh token is
    never a silent failure. A record without ``expires_at`` (the classic ``lite login``
    credential) is returned as stored."""
    key: Final = token_data.get("key")
    if not isinstance(key, str) or not key:
        return None
    expires_at: Final = token_data.get("expires_at")
    if not isinstance(expires_at, (int, float)):
        return key
    if now() < expires_at - CLI_TOKEN_FRESHNESS_BUFFER_SECONDS:
        return key
    still_valid: Final = key if now() < expires_at else None
    refresh_inputs: Final = _refresh_inputs(token_data)
    if refresh_inputs is None:
        return still_valid
    refreshed: Final = refresh_credential(*refresh_inputs, http=http, now=now)
    if isinstance(refreshed, PkceFailure):
        sibling_key: Final = _key_rotated_by_a_sibling(reload(), token_data, now())
        if sibling_key is None:
            warn(f"Could not renew the key: {refreshed.reason}")
        return sibling_key or still_valid
    base_url: Final = token_data.get("base_url")
    save(pkce_token_record(base_url if isinstance(base_url, str) else "", refreshed))
    return refreshed.access_token


_CREDENTIAL_IDENTITY_FIELDS: Final = ("base_url", "token_endpoint", "resource", "user_id", "team_id")


def _key_rotated_by_a_sibling(
    record: Mapping[str, object] | None, token_data: Mapping[str, object], now: float
) -> str | None:
    """The key a sibling process saved, but only when it continues this very credential:
    same proxy, same token endpoint, same resource, same user and team, and not yet expired.
    A concurrent ``lite login`` against a different proxy, or as someone else on this one,
    replaces the same file, and its key must never be sent as this credential."""
    if record is None or record.get("refresh_token") == token_data.get("refresh_token"):
        return None
    if any(record.get(field) != token_data.get(field) for field in _CREDENTIAL_IDENTITY_FIELDS):
        return None
    expires_at: Final = record.get("expires_at")
    if not isinstance(expires_at, (int, float)) or now >= expires_at:
        return None
    key: Final = record.get("key")
    return key if isinstance(key, str) and key else None


def _refresh_inputs(token_data: Mapping[str, object]) -> tuple[str, str, str, str, str] | None:
    values: Final = tuple(
        token_data.get(field)
        for field in ("token_endpoint", "revocation_endpoint", "resource", "client_id", "refresh_token")
    )
    if not all(isinstance(value, str) and value for value in values):
        return None
    token_endpoint, revocation_endpoint, resource, client_id, refresh_token = values
    return str(token_endpoint), str(revocation_endpoint), str(resource), str(client_id), str(refresh_token)


def revoke_stored_credential(
    token_data: Mapping[str, object], http: Http
) -> PkceFailure | RevocationUnavailable | None:
    refresh_inputs: Final = _refresh_inputs(token_data)
    if refresh_inputs is None:
        return None
    _, revocation_endpoint, _, client_id, refresh_token = refresh_inputs
    return revoke_credential(revocation_endpoint, client_id, refresh_token, http)
