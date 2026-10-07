from __future__ import annotations

import base64
import json
import os
import re
import secrets
import signal
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import jwt
import psutil
import pytest
import yaml
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually, gateway_from_environment, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.process import OwnedProxy, group_members, owned_proxy_process
from tests.integration._support.provider import SharedProvider
from tests.integration._support.wire import Reply, Request, Wire, wire_server

pytestmark: Final = pytest.mark.timeout(300)

CLI_SOURCE: Final = "litellm-cli"
CLI_STATE_PREFIX: Final = "litellm-session-token"
DEVICE_CODE_GRANT: Final = "urn:ietf:params:oauth:grant-type:device_code"
POLL_SECRET_HEADER: Final = "x-litellm-cli-poll-secret"
DISABLED_PAGE_TITLE: Final = "<title>Admin UI Disabled</title>"
LOGIN_FORM_TITLE: Final = "<title>LiteLLM Login</title>"
CLI_LOGIN_PAGE_TITLE: Final = "<title>LiteLLM CLI Login</title>"
CLI_SUCCESS_PAGE_TITLE: Final = "<title>CLI Authentication Successful - LiteLLM</title>"
SESSION_GONE: Final = "CLI login session not found or expired"
SUBJECT: Final = "cli-sso-audit-subject"
SUBJECT_EMAIL: Final = "cli-sso-audit-subject@example.com"
CLIENT_ID: Final = "integration-oidc-client"
CLIENT_SECRET: Final = "integration-oidc-secret"
MESSAGE_MODEL: Final = "anthropic/claude-haiku-4-5"
BURST: Final = 8
COMPLETE_TOKEN_FIELD: Final = re.compile(r'name="browser_complete_token" value="([^"]+)"')
COMPLETE_FORM_ACTION: Final = re.compile(r'action="([^"]+/sso/cli/complete/[^"]+)"')


@dataclass(frozen=True, slots=True)
class Idp:
    wire: Wire
    token_outage: threading.Event
    subject: str


@dataclass(frozen=True, slots=True)
class CliSession:
    login_id: str
    poll_secret: str
    user_code: str


@dataclass(frozen=True, slots=True)
class DeviceGrant:
    device_code: str
    user_code: str
    verification_uri: str


@dataclass(frozen=True, slots=True)
class OneWorkerProxy:
    idp: Idp
    owned: OwnedProxy


def _subject_inside(value: str, prefix: str) -> str:
    return value.removeprefix(prefix).rsplit("-", 1)[0]


def _idp_reply(request: Request, token_outage: threading.Event, subject: str) -> Reply:
    target: Final = urlparse(request.target)
    if target.path == "/authorize":
        query: Final = parse_qs(target.query)
        issued: Final = f"code-{query.get('login_hint', [subject])[0]}-{uuid.uuid4().hex}"
        location: Final = f"{query['redirect_uri'][0]}?{urlencode({'code': issued, 'state': query['state'][0]})}"
        return Reply(status=302, body=b"", headers={"location": location})
    if target.path == "/token":
        if token_outage.is_set():
            return Reply(status=503, body=b'{"error": "temporarily_unavailable"}')
        expected: Final = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
        if request.headers.get("authorization") != f"Basic {expected}":
            return Reply(status=401, body=b'{"error": "invalid_client"}')
        form: Final = parse_qs(request.body.decode())
        code: Final = form.get("code", [""])[0]
        if form.get("grant_type") != ["authorization_code"] or not code.startswith("code-"):
            return Reply(status=400, body=b'{"error": "invalid_grant"}')
        access_token: Final = f"idp-access-{_subject_inside(code, 'code-')}-{uuid.uuid4().hex}"
        return Reply(
            body=json.dumps({"access_token": access_token, "token_type": "Bearer", "expires_in": 3600}).encode()
        )
    if target.path == "/userinfo":
        bearer: Final = request.headers.get("authorization", "")
        if not bearer.startswith("Bearer idp-access-"):
            return Reply(status=401, body=b'{"error": "invalid_token"}')
        signed_in: Final = _subject_inside(bearer, "Bearer idp-access-")
        return Reply(
            body=json.dumps(
                {"sub": signed_in, "preferred_username": signed_in, "email": f"{signed_in}@example.com"}
            ).encode()
        )
    return Reply(status=404, body=b'{"error": "not_found"}')


def _sso_environment(idp_url: str) -> Mapping[str, str]:
    return {
        "GENERIC_CLIENT_ID": CLIENT_ID,
        "GENERIC_CLIENT_SECRET": CLIENT_SECRET,
        "GENERIC_AUTHORIZATION_ENDPOINT": f"{idp_url}/authorize",
        "GENERIC_TOKEN_ENDPOINT": f"{idp_url}/token",
        "GENERIC_USERINFO_ENDPOINT": f"{idp_url}/userinfo",
        "OAUTHLIB_INSECURE_TRANSPORT": "1",
    }


def _gateway_enabled_config(directory: Path) -> Path:
    stock: Final = yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
    config: Final = directory / f"cli_sso_{uuid.uuid4().hex}.yaml"
    config.write_text(
        json.dumps({**stock, "general_settings": {**stock["general_settings"], "enable_claude_code_gateway": True}})
    )
    return config


@contextmanager
def _fake_idp(subject: str) -> Iterator[Idp]:
    outage: Final = threading.Event()
    with wire_server(lambda request: _idp_reply(request, outage, subject)) as wire:
        yield Idp(wire, outage, subject)


@pytest.fixture(scope="module")
def idp() -> Iterator[Idp]:
    with _fake_idp(SUBJECT) as fake:
        yield fake


@pytest.fixture(scope="module")
def ui_disabled(idp: Idp, tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("cli-sso-ui-disabled")
    with (
        gateway_from_environment() as rig,
        owned_proxy_process(
            rig,
            directory,
            {"DISABLE_ADMIN_UI": "true", **_sso_environment(idp.wire.url)},
            config=_gateway_enabled_config(directory),
            remove_environment=("PROXY_BASE_URL",),
            workers=2,
        ) as owned,
    ):
        yield owned


@pytest.fixture(scope="module")
def one_worker(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OneWorkerProxy]:
    directory: Final = tmp_path_factory.mktemp("cli-sso-one-worker")
    with (
        _fake_idp(f"cli-sso-first-sign-in-{uuid.uuid4().hex[:12]}") as idp,
        gateway_from_environment() as rig,
        owned_proxy_process(
            rig,
            directory,
            {"DISABLE_ADMIN_UI": "true", **_sso_environment(idp.wire.url)},
            remove_environment=("PROXY_BASE_URL",),
            workers=1,
        ) as owned,
    ):
        yield OneWorkerProxy(idp, owned)


def _proxy_url(proxy: Gateway) -> str:
    return str(proxy.client.base_url).rstrip("/")


def _browser() -> httpx.Client:
    return httpx.Client(follow_redirects=False, trust_env=False, timeout=30)


def _start_lite_login(proxy: Gateway) -> CliSession:
    response: Final = proxy.client.post("/sso/cli/start")
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    return CliSession(
        string_value(body["login_id"]), string_value(body["poll_secret"]), string_value(body["user_code"])
    )


def _cli_link(proxy: Gateway, login_id: str) -> str:
    return f"{_proxy_url(proxy)}/sso/key/generate?{urlencode({'source': CLI_SOURCE, 'key': login_id})}"


def _assert_idp_redirect(proxy: Gateway, idp: Idp, link: httpx.Response, login_id: str) -> str:
    assert DISABLED_PAGE_TITLE not in link.text, link.text
    assert link.is_redirect, f"{link.status_code} {link.text}"
    location: Final = link.headers["location"]
    parsed: Final = urlparse(location)
    query: Final = parse_qs(parsed.query)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == f"{idp.wire.url}/authorize", location
    assert query["state"] == [f"{CLI_STATE_PREFIX}:{login_id}"], location
    assert query["redirect_uri"] == [f"{_proxy_url(proxy)}/sso/callback"], location
    assert query["client_id"] == [CLIENT_ID], location
    return location


def _walk_idp(browser: httpx.Client, authorize_url: str) -> httpx.Response:
    at_idp: Final = browser.get(authorize_url)
    assert at_idp.status_code == 302, f"{at_idp.status_code} {at_idp.text}"
    return browser.get(at_idp.headers["location"])


def _complete_in_browser(browser: httpx.Client, callback: httpx.Response, user_code: str) -> httpx.Response:
    assert callback.status_code == 200, f"{callback.status_code} {callback.text}"
    assert CLI_LOGIN_PAGE_TITLE in callback.text, callback.text
    token: Final = COMPLETE_TOKEN_FIELD.search(callback.text)
    action: Final = COMPLETE_FORM_ACTION.search(callback.text)
    assert token is not None and action is not None, callback.text
    return browser.post(action.group(1), data={"user_code": user_code, "browser_complete_token": token.group(1)})


def _poll(proxy: Gateway, session: CliSession, *, poll_secret: str | None = None) -> httpx.Response:
    return proxy.client.get(
        f"/sso/cli/poll/{session.login_id}",
        headers={POLL_SECRET_HEADER: session.poll_secret if poll_secret is None else poll_secret},
    )


def _sign_in(
    proxy: Gateway, idp: Idp, browser: httpx.Client, session: CliSession, *, subject: str | None = None
) -> None:
    link: Final = browser.get(_cli_link(proxy, session.login_id))
    authorize: Final = _assert_idp_redirect(proxy, idp, link, session.login_id)
    hinted: Final = authorize if subject is None else f"{authorize}&{urlencode({'login_hint': subject})}"
    callback: Final = _walk_idp(browser, hinted)
    done: Final = _complete_in_browser(browser, callback, session.user_code)
    assert done.status_code == 200 and CLI_SUCCESS_PAGE_TITLE in done.text, f"{done.status_code} {done.text}"


def _ready_key(proxy: Gateway, session: CliSession, *, subject: str = SUBJECT) -> str:
    ready: Final = _poll(proxy, session)
    assert ready.status_code == 200, f"{ready.status_code} {ready.text}"
    body: Final = JSON_OBJECT.validate_json(ready.content)
    assert body["status"] == "ready" and body["user_id"] == subject, ready.text
    return string_value(body["key"])


def _send_message(proxy: Gateway, provider: SharedProvider, key: str) -> str:
    message_id: Final = f"msg_{uuid.uuid4().hex}"
    marker: Final = f"audit-{uuid.uuid4().hex}"
    provider.expect(
        Reply(
            body=json.dumps(
                {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-4-5",
                    "content": [{"type": "text", "text": "scripted reply"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 9, "output_tokens": 5},
                }
            ).encode()
        )
    )
    response: Final = proxy.request(
        "POST",
        "/v1/messages",
        {"model": MESSAGE_MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": marker}]},
        key=key,
    )
    assert response.status_code == 200, response.text
    assert JSON_OBJECT.validate_json(response.content)["id"] == message_id, response.text
    upstream: Final = provider.received()
    assert len(upstream) == 1 and upstream[0].target == "/v1/messages", [item.target for item in upstream]
    assert marker in upstream[0].body.decode(), upstream[0].body
    return message_id


def _user_rows() -> Sequence[Mapping[str, JsonValue]]:
    return read_rows('SELECT user_id, user_email FROM "LiteLLM_UserTable" WHERE user_id = %s', (SUBJECT,))


def _worker_pids(root_pid: int) -> frozenset[int]:
    def is_worker(process: psutil.Process) -> bool:
        try:
            return "spawn_main" in " ".join(process.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return False

    return frozenset(process.pid for process in group_members(root_pid) if is_worker(process))


def test_cli_login_link_redirects_to_the_idp_when_the_ui_is_disabled(ui_disabled: OwnedProxy, idp: Idp) -> None:
    proxy: Final = ui_disabled.gateway
    session: Final = _start_lite_login(proxy)
    with _browser() as browser:
        link: Final = browser.get(_cli_link(proxy, session.login_id))
    _assert_idp_redirect(proxy, idp, link, session.login_id)


def test_lite_login_completes_and_the_key_serves_messages(
    ui_disabled: OwnedProxy, idp: Idp, provider: SharedProvider
) -> None:
    proxy: Final = ui_disabled.gateway
    session: Final = _start_lite_login(proxy)
    with _browser() as browser:
        early: Final = browser.post(
            f"{_proxy_url(proxy)}/sso/cli/complete/{session.login_id}",
            data={"user_code": session.user_code, "browser_complete_token": "x"},
        )
        assert early.status_code == 400 and "CLI login is not ready" in early.text, f"{early.status_code} {early.text}"
        assert JSON_OBJECT.validate_json(_poll(proxy, session).content) == {"status": "pending"}
        _sign_in(proxy, idp, browser, session)
    forged: Final = _poll(proxy, session, poll_secret="not-the-poll-secret")
    assert forged.status_code == 403, f"{forged.status_code} {forged.text}"
    key: Final = _ready_key(proxy, session)
    users: Final = _user_rows()
    assert [(row["user_id"], row["user_email"]) for row in users] == [(SUBJECT, SUBJECT_EMAIL)], users
    message_id: Final = _send_message(proxy, provider, key)
    spend: Final = eventually(
        lambda: read_rows('SELECT "user" FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (message_id,)),
        lambda rows: len(rows) == 1,
        seconds=70,
    )
    assert spend[0]["user"] == SUBJECT, spend


def _device_authorization(proxy: Gateway) -> DeviceGrant:
    response: Final = proxy.client.post("/claude_code_gateway/oauth/device_authorization")
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    return DeviceGrant(
        string_value(body["device_code"]), string_value(body["user_code"]), string_value(body["verification_uri"])
    )


def _device_token(proxy: Gateway, grant: DeviceGrant) -> httpx.Response:
    return proxy.client.post(
        "/claude_code_gateway/oauth/token", data={"grant_type": DEVICE_CODE_GRANT, "device_code": grant.device_code}
    )


def test_claude_code_device_flow_completes_when_the_ui_is_disabled(
    ui_disabled: OwnedProxy, idp: Idp, provider: SharedProvider
) -> None:
    proxy: Final = ui_disabled.gateway
    grant: Final = _device_authorization(proxy)
    login_id: Final = parse_qs(urlparse(grant.verification_uri).query)["key"][0]
    pending: Final = _device_token(proxy, grant)
    assert pending.status_code == 400, f"{pending.status_code} {pending.text}"
    assert JSON_OBJECT.validate_json(pending.content)["error"] == "authorization_pending", pending.text
    with _browser() as browser:
        link: Final = browser.get(grant.verification_uri)
        callback: Final = _walk_idp(browser, _assert_idp_redirect(proxy, idp, link, login_id))
        done: Final = _complete_in_browser(browser, callback, grant.user_code)
    assert done.status_code == 200 and CLI_SUCCESS_PAGE_TITLE in done.text, f"{done.status_code} {done.text}"
    issued: Final = _device_token(proxy, grant)
    assert issued.status_code == 200, f"{issued.status_code} {issued.text}"
    access_token: Final = string_value(JSON_OBJECT.validate_json(issued.content)["access_token"])
    replay: Final = _device_token(proxy, grant)
    assert replay.status_code == 400, f"{replay.status_code} {replay.text}"
    assert JSON_OBJECT.validate_json(replay.content)["error"] == "expired_token", replay.text
    _send_message(proxy, provider, access_token)


@pytest.mark.parametrize(
    ("method", "path", "params"),
    (
        ("GET", "/sso/key/generate", ()),
        ("GET", "/sso/key/generate", (("source", "1"),)),
        ("GET", "/sso/key/generate", (("source", ""),)),
        ("GET", "/sso/key/generate", (("source", "LITELLM-CLI"), ("key", f"cli-{'a' * 32}"))),
        ("GET", "/sso/key/generate", (("return_to", "/mcp/"),)),
        ("GET", "/sso/saml/login", ()),
        ("POST", "/sso/saml/callback", ()),
    ),
    ids=("no-source", "source-int", "source-empty", "source-case", "return-to", "saml-login", "saml-callback"),
)
def test_non_cli_entries_stay_refused_when_the_ui_is_disabled(
    ui_disabled: OwnedProxy, method: str, path: str, params: tuple[tuple[str, str], ...]
) -> None:
    response: Final = ui_disabled.gateway.client.request(method, path, params=params)
    assert response.status_code == 200 and DISABLED_PAGE_TITLE in response.text, (
        f"{response.status_code} {response.text}"
    )


@pytest.mark.parametrize(
    ("params", "detail"),
    (
        ((("source", CLI_SOURCE),), "Invalid CLI login session id"),
        ((("source", CLI_SOURCE), ("key", "1")), "Invalid CLI login session id"),
        ((("source", CLI_SOURCE), ("key", "cli-" + "k" * 5000)), "Invalid CLI login session id"),
        ((("source", CLI_SOURCE), ("key", f"cli-{'a' * 32}"), ("key", f"cli-{'b' * 32}")), SESSION_GONE),
        ((("source", CLI_SOURCE), ("key", "sk-legacy-cli-key")), "Your litellm CLI is out of date"),
        ((("source", CLI_SOURCE), ("key", f"cli-{secrets.token_urlsafe(24)}")), SESSION_GONE),
        ((("source", CLI_SOURCE), ("source", CLI_SOURCE), ("key", f"cli-{secrets.token_urlsafe(24)}")), SESSION_GONE),
    ),
    ids=("missing", "int", "five-kb", "twice", "legacy-sk", "unknown", "source-twice"),
)
def test_malformed_cli_keys_answer_400_when_the_ui_is_disabled(
    ui_disabled: OwnedProxy, params: tuple[tuple[str, str], ...], detail: str
) -> None:
    response: Final = ui_disabled.gateway.client.get("/sso/key/generate", params=params)
    assert response.status_code == 400, f"{response.status_code} {response.text}"
    assert DISABLED_PAGE_TITLE not in response.text, response.text
    assert detail in string_value(JSON_OBJECT.validate_json(response.content)["detail"]), response.text


def test_login_form_and_cli_validation_when_the_flag_is_unset(gateway: Gateway) -> None:
    form: Final = gateway.client.get("/sso/key/generate")
    assert form.status_code == 200 and LOGIN_FORM_TITLE in form.text, f"{form.status_code} {form.text}"
    unknown: Final = gateway.client.get(
        "/sso/key/generate", params={"source": CLI_SOURCE, "key": f"cli-{secrets.token_urlsafe(24)}"}
    )
    assert unknown.status_code == 400 and SESSION_GONE in unknown.text, f"{unknown.status_code} {unknown.text}"
    for method, path in (("GET", "/sso/saml/login"), ("POST", "/sso/saml/callback")):
        saml: Final = gateway.client.request(method, path)
        assert DISABLED_PAGE_TITLE not in saml.text and saml.status_code != 200, (
            f"{path}: {saml.status_code} {saml.text}"
        )


def test_cli_link_is_reentrant_until_the_poll_consumes_the_session(ui_disabled: OwnedProxy, idp: Idp) -> None:
    proxy: Final = ui_disabled.gateway
    session: Final = _start_lite_login(proxy)
    with _browser() as browser:
        first: Final = _assert_idp_redirect(
            proxy, idp, browser.get(_cli_link(proxy, session.login_id)), session.login_id
        )
        second: Final = _assert_idp_redirect(
            proxy, idp, browser.get(_cli_link(proxy, session.login_id)), session.login_id
        )
        assert parse_qs(urlparse(first).query)["state"] == parse_qs(urlparse(second).query)["state"]
        done: Final = _complete_in_browser(browser, _walk_idp(browser, second), session.user_code)
        assert done.status_code == 200, f"{done.status_code} {done.text}"
        _ready_key(proxy, session)
        gone: Final = browser.get(_cli_link(proxy, session.login_id))
    assert gone.status_code == 400 and SESSION_GONE in gone.text, f"{gone.status_code} {gone.text}"
    consumed: Final = _poll(proxy, session)
    assert consumed.status_code == 400 and SESSION_GONE in consumed.text, f"{consumed.status_code} {consumed.text}"


def test_burst_of_logins_recovers_from_an_idp_token_outage(ui_disabled: OwnedProxy, idp: Idp) -> None:
    proxy: Final = ui_disabled.gateway
    sessions: Final = tuple(_start_lite_login(proxy) for _ in range(BURST))

    def failed_exchange(session: CliSession) -> httpx.Response:
        with _browser() as browser:
            link: Final = browser.get(_cli_link(proxy, session.login_id))
            return _walk_idp(browser, _assert_idp_redirect(proxy, idp, link, session.login_id))

    def liveliness() -> int:
        return proxy.client.get("/health/liveliness").status_code

    idp.wire.drain()
    idp.token_outage.set()
    try:
        with ThreadPoolExecutor(max_workers=BURST + 1) as pool:
            probe: Final = pool.submit(liveliness)
            failures: Final = tuple(pool.map(failed_exchange, sessions))
        assert probe.result() == 200
    finally:
        idp.token_outage.clear()
    token_attempts: Final = tuple(request for request in idp.wire.drain() if request.target == "/token")
    assert len(token_attempts) == BURST, len(token_attempts)
    for failure in failures:
        assert failure.status_code >= 400, f"{failure.status_code} {failure.text!r}"
        assert CLI_LOGIN_PAGE_TITLE not in failure.text, failure.text
    for session in sessions:
        assert JSON_OBJECT.validate_json(_poll(proxy, session).content) == {"status": "pending"}, session.login_id

    def recovered_login(session: CliSession) -> str:
        with _browser() as browser:
            _sign_in(proxy, idp, browser, session)
        return _ready_key(proxy, session)

    with ThreadPoolExecutor(max_workers=BURST) as pool:
        keys: Final = tuple(pool.map(recovered_login, sessions))
    assert len(set(keys)) == BURST, keys
    for session in sessions:
        again: Final = _poll(proxy, session)
        assert again.status_code == 400 and SESSION_GONE in again.text, f"{again.status_code} {again.text}"
    assert [row["user_id"] for row in _user_rows()] == [SUBJECT]


def test_login_survives_a_worker_kill(idp: Idp, tmp_path: Path) -> None:
    with (
        gateway_from_environment() as rig,
        owned_proxy_process(
            rig,
            tmp_path,
            {"DISABLE_ADMIN_UI": "true", **_sso_environment(idp.wire.url)},
            remove_environment=("PROXY_BASE_URL",),
            workers=2,
        ) as owned,
    ):
        proxy: Final = owned.gateway
        workers: Final = eventually(lambda: _worker_pids(owned.process.pid), lambda pids: len(pids) == 2, seconds=30)
        victim: Final = min(workers)
        session: Final = _start_lite_login(proxy)
        with _browser() as browser:
            link: Final = browser.get(_cli_link(proxy, session.login_id))
            at_idp: Final = browser.get(_assert_idp_redirect(proxy, idp, link, session.login_id))
            assert at_idp.status_code == 302, f"{at_idp.status_code} {at_idp.text}"
            os.kill(victim, signal.SIGKILL)

            def callback_attempt() -> httpx.Response | None:
                try:
                    return browser.get(at_idp.headers["location"])
                except httpx.TransportError:
                    return None

            callback: Final = eventually(
                callback_attempt, lambda response: response is not None and response.status_code == 200, seconds=60
            )
            assert callback is not None
            done: Final = _complete_in_browser(browser, callback, session.user_code)
            assert done.status_code == 200 and CLI_SUCCESS_PAGE_TITLE in done.text, f"{done.status_code} {done.text}"
        _ready_key(proxy, session)
        respawned: Final = eventually(
            lambda: _worker_pids(owned.process.pid), lambda pids: len(pids) == 2 and victim not in pids, seconds=60
        )
        assert victim not in respawned, respawned


def test_first_sign_in_user_serves_messages_and_signs_in_again_on_one_worker(
    one_worker: OneWorkerProxy, provider: SharedProvider
) -> None:
    proxy: Final = one_worker.owned.gateway
    idp: Final = one_worker.idp
    first: Final = _start_lite_login(proxy)
    with _browser() as browser:
        _sign_in(proxy, idp, browser, first)
    _send_message(proxy, provider, _ready_key(proxy, first, subject=idp.subject))
    again: Final = _start_lite_login(proxy)
    with _browser() as browser:
        _sign_in(proxy, idp, browser, again)
    _ready_key(proxy, again, subject=idp.subject)


def test_a_user_created_after_a_missed_lookup_is_budgeted_at_once_on_that_worker(
    one_worker: OneWorkerProxy, provider: SharedProvider
) -> None:
    proxy: Final = one_worker.owned.gateway
    user_id: Final = f"cli-sso-late-user-{uuid.uuid4().hex[:12]}"
    with proxy.scenario() as scenario:
        key: Final = scenario.key(user_id=user_id, models=[MESSAGE_MODEL])
        _send_message(proxy, provider, key)
        created: Final = proxy.request("POST", "/user/new", {"user_id": user_id, "max_budget": 0})
        assert created.status_code == 200, created.text
        refused: Final = proxy.request(
            "POST",
            "/v1/messages",
            {"model": MESSAGE_MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": "over budget"}]},
            key=key,
        )
        assert refused.status_code == 422 and f"ExceededBudget: User={user_id}" in refused.text, (
            f"{refused.status_code} {refused.text}"
        )
        assert provider.received() == ()


def test_a_user_an_admin_created_after_a_missed_lookup_signs_in_at_once_on_that_worker(
    one_worker: OneWorkerProxy, provider: SharedProvider
) -> None:
    proxy: Final = one_worker.owned.gateway
    subject: Final = f"cli-sso-admin-created-{uuid.uuid4().hex[:12]}"
    with proxy.scenario() as scenario:
        _send_message(proxy, provider, scenario.key(user_id=subject, models=[MESSAGE_MODEL]))
        created: Final = proxy.request(
            "POST",
            "/user/new",
            {"user_id": subject, "user_email": f"{subject}@example.com", "user_role": "internal_user"},
        )
        assert created.status_code == 200, created.text
        session: Final = _start_lite_login(proxy)
        with _browser() as browser:
            _sign_in(proxy, one_worker.idp, browser, session, subject=subject)
        _send_message(proxy, provider, _ready_key(proxy, session, subject=subject))
        rows: Final = read_rows('SELECT user_email FROM "LiteLLM_UserTable" WHERE user_id = %s', (subject,))
        assert [row["user_email"] for row in rows] == [f"{subject}@example.com"], rows


def test_dashboard_sign_in_creates_the_user_and_hands_the_browser_a_session(tmp_path: Path) -> None:
    subject: Final = f"ui-sso-first-sign-in-{uuid.uuid4().hex[:12]}"
    with (
        _fake_idp(subject) as idp,
        gateway_from_environment() as rig,
        owned_proxy_process(
            rig,
            tmp_path,
            _sso_environment(idp.wire.url),
            remove_environment=("PROXY_BASE_URL", "DISABLE_ADMIN_UI"),
            workers=1,
        ) as owned,
    ):
        proxy: Final = owned.gateway
        with _browser() as browser:
            entry: Final = browser.get(f"{_proxy_url(proxy)}/sso/key/generate")
            assert entry.is_redirect, f"{entry.status_code} {entry.text}"
            signed_in: Final = _walk_idp(browser, entry.headers["location"])
        assert signed_in.status_code == 303, f"{signed_in.status_code} {signed_in.text}"
        assert urlparse(signed_in.headers["location"]).query == "login=success", signed_in.headers["location"]
        session: Final = JSON_OBJECT.validate_python(
            jwt.decode(signed_in.cookies["token"], rig.key, algorithms=["HS256"])
        )
        assert session["user_id"] == subject and session["login_method"] == "sso", session
        me: Final = proxy.request("GET", "/user/info", key=string_value(session["key"]))
        assert me.status_code == 200, f"{me.status_code} {me.text}"
        assert JSON_OBJECT.validate_json(me.content)["user_id"] == subject, me.text
        rows: Final = read_rows('SELECT user_email FROM "LiteLLM_UserTable" WHERE user_id = %s', (subject,))
        assert [row["user_email"] for row in rows] == [f"{subject}@example.com"], rows


@pytest.mark.parametrize("flag", ("false", ""), ids=("false", "empty"))
def test_flag_values_that_do_not_disable_keep_the_gates_open(idp: Idp, tmp_path: Path, flag: str) -> None:
    with (
        gateway_from_environment() as rig,
        owned_proxy_process(
            rig,
            tmp_path,
            {"DISABLE_ADMIN_UI": flag, **_sso_environment(idp.wire.url)},
            remove_environment=("PROXY_BASE_URL",),
            workers=2,
        ) as owned,
    ):
        proxy: Final = owned.gateway
        form_entry: Final = proxy.client.get("/sso/key/generate")
        assert form_entry.is_redirect, f"{form_entry.status_code} {form_entry.text}"
        assert urlparse(form_entry.headers["location"]).path == "/authorize", form_entry.headers["location"]
        session: Final = _start_lite_login(proxy)
        with _browser() as browser:
            _assert_idp_redirect(proxy, idp, browser.get(_cli_link(proxy, session.login_id)), session.login_id)
        for method, path in (("GET", "/sso/saml/login"), ("POST", "/sso/saml/callback")):
            saml: Final = proxy.client.request(method, path)
            assert DISABLED_PAGE_TITLE not in saml.text and saml.status_code != 200, f"{path}: {saml.status_code}"
