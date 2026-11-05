from __future__ import annotations

import json
import re
from typing import Final

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from litellm_enterprise.proxy.liteadmin import AdminSession, NativeAdminContext, native_admin_context, router
from pydantic import SecretStr, TypeAdapter

from litellm.proxy._types import LiteLLM_UserTable

TOKEN: Final = "a" * 43
PATH: Final = "/liteadmin/slack/connect/" + TOKEN
ORIGIN: Final = "https://gateway.example.com"


class Worker:
    def __init__(self, *, email: str = "alice@example.com", status: int = 200) -> None:
        self.email = email
        self.status = status
        self.session: object = None
        self.role = "proxy_admin"

    def request(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["X-LiteLLM-Admin-Agent-Token"] == "s" * 32
        if request.method == "POST":
            self.session = json.loads(request.content)
            return httpx.Response(self.status, json={"status": "connected"})
        return httpx.Response(
            self.status,
            json={
                "workspace_id": "Tworkspace",
                "slack_user_id": "Ualice",
                "email": self.email,
            },
        )


def client_for(
    worker: Worker, *, role: str | None = None, logged_in: bool = True, email: str = "alice@example.com"
) -> TestClient:
    async def session_user(request: Request) -> str | None:
        return "alice" if logged_in else None

    async def load_user(user_id: str) -> LiteLLM_UserTable:
        return LiteLLM_UserTable(user_id=user_id, user_email=email, user_role=role or worker.role)

    def mint(user: LiteLLM_UserTable) -> AdminSession:
        return AdminSession(user_id=user.user_id, credential=SecretStr("personal-session"), expires_at=86400)

    context: Final = NativeAdminContext(
        "http://private-worker:10000",
        SecretStr("s" * 32),
        httpx.AsyncClient(transport=httpx.MockTransport(worker.request)),
        session_user,
        load_user,
        mint,
    )
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[native_admin_context] = lambda: context
    return TestClient(app, base_url=ORIGIN)


def csrf_from(client: TestClient) -> str:
    page: Final = client.get(PATH)
    assert page.status_code == 200
    match: Final = re.search('name="csrf" value="([^"]+)"', page.text)
    assert match is not None
    return match[1]


def test_connect_uses_existing_login_without_a_hosted_oauth_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXY_BASE_URL", ORIGIN)
    with client_for(Worker(), logged_in=False) as client:
        response: Final = client.get(PATH, follow_redirects=False)
    assert response.status_code == 303
    assert (
        response.headers["location"] == ORIGIN + "/sso/key/generate?return_to=%2Fliteadmin%2Fslack%2Fconnect%2F" + TOKEN
    )
    assert response.headers["cache-control"] == "no-store"


def test_connect_hands_off_personal_session_only_over_private_worker_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXY_BASE_URL", ORIGIN)
    worker: Final = Worker()
    with client_for(worker) as client:
        csrf: Final = csrf_from(client)
        response: Final = client.post(PATH, data={"csrf": csrf}, headers={"Origin": ORIGIN})
    assert response.status_code == 200
    assert "Account connected" in response.text
    assert worker.session == {"user_id": "alice", "credential": "personal-session", "expires_at": 86400.0}
    assert "personal-session" not in response.text
    assert "Max-Age=0" in response.headers["set-cookie"]


@pytest.mark.parametrize("role,email", [("internal_user", "alice@example.com"), ("proxy_admin", "bob@example.com")])
def test_connect_rejects_nonadmin_and_another_slack_users_link(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    email: str,
) -> None:
    monkeypatch.setenv("PROXY_BASE_URL", ORIGIN)
    worker: Final = Worker(email=email)
    with client_for(worker, role=role) as client:
        response: Final = client.get(PATH)
    assert response.status_code == 403
    assert worker.session is None


@pytest.mark.parametrize("origin,csrf", [("https://attacker.example", None), ("null", None), (ORIGIN, "b" * 43)])
def test_connect_requires_same_origin_and_browser_csrf(
    monkeypatch: pytest.MonkeyPatch,
    origin: str,
    csrf: str | None,
) -> None:
    monkeypatch.setenv("PROXY_BASE_URL", ORIGIN)
    worker: Final = Worker()
    with client_for(worker) as client:
        valid: Final = csrf_from(client)
        response: Final = client.post(PATH, data={"csrf": csrf or valid}, headers={"Origin": origin})
    assert response.status_code == 403
    assert worker.session is None


@pytest.mark.parametrize("status,expected", [(410, 410), (403, 403), (500, 503), (302, 503)])
def test_worker_denial_expiry_and_failure_never_mint_a_session(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    expected: int,
) -> None:
    monkeypatch.setenv("PROXY_BASE_URL", ORIGIN)
    worker: Final = Worker(status=status)
    with client_for(worker) as client:
        response: Final = client.get(PATH)
    assert response.status_code == expected
    assert worker.session is None


def test_csrf_cookie_cannot_cross_links(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXY_BASE_URL", ORIGIN)
    worker: Final = Worker()
    with client_for(worker) as client:
        csrf: Final = csrf_from(client)
        response: Final = client.post(PATH.replace(TOKEN, "b" * 43), data={"csrf": csrf}, headers={"Origin": ORIGIN})
    assert response.status_code == 403
    assert worker.session is None


@pytest.mark.parametrize(
    "url,secret,enterprise,database,status",
    [
        ("", "s" * 32, True, True, 404),
        ("http://worker:10000", "s" * 32, False, True, 403),
        ("http://worker:10000", "s" * 32, True, False, 503),
        ("file:///etc/passwd", "s" * 32, True, True, 503),
        ("https://user:password@worker", "s" * 32, True, True, 503),
        ("https://worker/path", "s" * 32, True, True, 503),
        ("https://worker", "short", True, True, 503),
        ("http://[broken", "s" * 32, True, True, 503),
        ("http://worker:broken", "s" * 32, True, True, 503),
    ],
)
def test_native_configuration_requires_enterprise_database_and_private_worker_credentials(
    url: str,
    secret: str,
    enterprise: bool,
    database: bool,
    status: int,
) -> None:
    from fastapi import HTTPException
    from litellm_enterprise.proxy.liteadmin import validate_native_configuration

    with pytest.raises(HTTPException) as error:
        validate_native_configuration(url, secret, enterprise, database)
    assert error.value.status_code == status


def test_connect_rechecks_admin_permission_after_consent_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXY_BASE_URL", ORIGIN)
    worker: Final = Worker()
    with client_for(worker) as client:
        csrf: Final = csrf_from(client)
        worker.role = "internal_user"
        response: Final = client.post(PATH, data={"csrf": csrf}, headers={"Origin": ORIGIN})
    assert response.status_code == 403
    assert worker.session is None


def test_consent_escapes_slack_email(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXY_BASE_URL", ORIGIN)
    email: Final = '<script>alert("x")</script>@example.com'
    with client_for(Worker(email=email), email=email) as client:
        page: Final = client.get(PATH)
    assert page.status_code == 200
    assert "<script>" not in page.text
    assert "&lt;script&gt;" in page.text


def test_native_session_is_encrypted_personal_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import datetime, timezone

    from litellm_enterprise.proxy.liteadmin import mint_admin_session

    from litellm.proxy.auth.auth_checks import ExperimentalUIJWTToken

    monkeypatch.setenv("LITELLM_SALT_KEY", "local-test-only-encryption-key")
    before: Final = datetime.now(timezone.utc).timestamp()
    session: Final = mint_admin_session(LiteLLM_UserTable(user_id="alice", user_role="proxy_admin"))
    decoded: Final = ExperimentalUIJWTToken.get_key_object_from_ui_hash_key(session.credential.get_secret_value())
    assert decoded is not None
    assert decoded.user_id == "alice"
    assert decoded.is_session_token is True
    assert decoded.user_role == "proxy_admin"
    assert decoded.expires is not None
    assert TypeAdapter(datetime).validate_python(decoded.expires).timestamp() == session.expires_at
    assert before + 86400 <= session.expires_at <= datetime.now(timezone.utc).timestamp() + 86400
    assert "alice" not in session.credential.get_secret_value()


def test_worker_configuration_accepts_private_service_address() -> None:
    from litellm_enterprise.proxy.liteadmin import validate_native_configuration

    validate_native_configuration("http://liteadmin.default.svc:10000", "s" * 32, True, True)


def test_connect_preserves_gateway_prefix_through_login_and_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXY_BASE_URL", ORIGIN + "/gateway")
    with client_for(Worker(), logged_in=False) as client:
        login: Final = client.get(PATH, follow_redirects=False)
    assert login.status_code == 303
    assert login.headers["location"] == (
        ORIGIN + "/gateway/sso/key/generate?return_to=%2Fgateway%2Fliteadmin%2Fslack%2Fconnect%2F" + TOKEN
    )
    worker: Final = Worker()
    with client_for(worker) as client:
        csrf: Final = csrf_from(client)
        connected: Final = client.post(PATH, data={"csrf": csrf}, headers={"Origin": ORIGIN})
    assert connected.status_code == 200
    assert worker.session is not None
