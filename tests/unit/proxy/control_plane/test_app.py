from datetime import timedelta
from itertools import repeat
from typing import Final

import jwt
import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from pydantic import SecretStr

from litellm.proxy.control_plane.app import LicenseRequired, create_app
from litellm.proxy.control_plane.audit.pipeline import Batch, Delivered, DeliveryResult
from litellm.proxy.control_plane.auth.dependencies import get_token_verifier
from litellm.proxy.control_plane.auth.jwt import PyJwkClientLookup, SessionTokenCodec, build_token_verifier
from litellm.proxy.control_plane.auth.login import LOCKOUT_FAILURES, LoginService, LoginThrottle
from litellm.proxy.control_plane.auth.rbac import VIEWER_ROLE, build_principal
from litellm.proxy.control_plane.auth.router import get_login_service
from litellm.proxy.control_plane.container import InvalidRolePolicy, build_container
from litellm.proxy.control_plane.settings import ControlPlaneSettings

MASTER_KEY: Final = "01234567890123456789012345678901"


class Licensed:
    def is_premium(self) -> bool:
        return True


class Unlicensed:
    def is_premium(self) -> bool:
        return False


class BrokenCloseSink:
    name: Final = "broken"

    async def deliver(self, batch: Batch) -> DeliveryResult:
        return Delivered(len(batch))

    async def close(self) -> None:
        raise RuntimeError("sink refused to close")


class ShutdownCounter(SpanProcessor):  # mutable-ok: counts provider shutdowns for assertions
    def __init__(self) -> None:
        self.shutdowns = 0

    def shutdown(self) -> None:
        self.shutdowns += 1


def _settings(**overrides: object) -> ControlPlaneSettings:
    return ControlPlaneSettings.model_validate(
        {"LITELLM_MASTER_KEY": MASTER_KEY, "UI_USERNAME": "admin", "UI_PASSWORD": "pw", **overrides}
    )


def _client(settings: ControlPlaneSettings | None = None) -> TestClient:
    return TestClient(create_app(build_container(settings or _settings()), Licensed()))


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_app_refuses_to_build_without_a_valid_license() -> None:
    with pytest.raises(LicenseRequired):
        create_app(build_container(_settings()), Unlicensed())


def test_health_routes_need_no_credentials() -> None:
    client: Final = _client()
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ok"}


def test_unknown_path_returns_an_rfc9457_not_found_body() -> None:
    response: Final = _client().get("/api/v1/not-a-route")
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json() == {
        "type": "urn:litellm:error:not-found",
        "title": "Not Found",
        "status": 404,
        "detail": "Not Found",
    }


def test_wrong_method_returns_an_rfc9457_error_and_preserves_allow() -> None:
    response: Final = _client().post("/api/v1/auth/me")
    assert response.status_code == 405
    assert response.headers["content-type"] == "application/problem+json"
    assert "GET" in response.headers["allow"]
    assert response.json() == {
        "type": "urn:litellm:error:method-not-allowed",
        "title": "Method Not Allowed",
        "status": 405,
        "detail": "Method Not Allowed",
    }


def test_jwks_lookup_errors_make_auth_me_return_401_instead_of_500() -> None:
    settings: Final = _settings(oidc_issuer="https://idp.example.test", oidc_audience="control-plane")
    container: Final = build_container(settings)

    class FailingJwksClient:
        def get_signing_key_from_jwt(self, _token: str) -> str:
            raise OSError("JWKS server is unavailable")

    lookup: Final = PyJwkClientLookup(
        None,
        "https://idp.example.test",
        fetch=lambda _url: {"jwks_uri": "https://idp.example.test/keys"},
        client_factory=lambda _url: FailingJwksClient(),
    )
    verifier: Final = build_token_verifier(settings, container.policy, keys=lookup)
    app: Final = create_app(container, Licensed())
    app.dependency_overrides[get_token_verifier] = lambda: verifier
    token: Final = jwt.encode({"iss": "https://idp.example.test", "sub": "alice"}, "secret", algorithm="HS256")
    response: Final = TestClient(app, raise_server_exceptions=False).get("/api/v1/auth/me", headers=_bearer(token))
    assert response.status_code == 401
    assert response.json()["type"] == "urn:litellm:error:unauthorized"


def test_missing_bearer_is_401_error_json_with_a_challenge() -> None:
    response: Final = _client().get("/api/v1/auth/me")
    assert response.status_code == 401
    assert response.headers["content-type"] == "application/problem+json"
    assert response.headers["www-authenticate"].startswith("Bearer ")
    assert response.json()["type"] == "urn:litellm:error:unauthorized"


def test_garbage_bearer_is_401() -> None:
    response: Final = _client().get("/api/v1/auth/me", headers=_bearer("nope"))
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid bearer credentials."


def test_missing_and_non_bearer_credentials_are_audited_as_authentication_failures() -> None:
    client: Final = _client()
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.get("/api/v1/auth/me", headers={"Authorization": "Basic dTpw"}).status_code == 401
    records: Final = client.get("/api/v1/audit/ocsf", headers=_bearer(MASTER_KEY)).json()["items"]
    assert [(record["class_uid"], record["actor"]["user"]["uid"], record["status_detail"]) for record in records] == [
        (3002, "anonymous", "missing bearer"),
        (3002, "anonymous", "missing bearer"),
    ]


def test_master_key_is_a_fleet_wide_cp_admin() -> None:
    body: Final = _client().get("/api/v1/auth/me", headers=_bearer(MASTER_KEY)).json()
    assert body["roles"] == ["admin"]
    assert body["dataplanes"] is None
    assert body["token_kind"] == "master_key"
    assert "dataplanes:write" in body["scopes"]


def test_login_mints_a_session_that_me_accepts() -> None:
    client: Final = _client()
    login: Final = client.post("/api/v1/auth/login", json={"username": "admin", "password": "pw"})
    assert login.status_code == 200, login.text
    assert (login.headers["cache-control"], login.headers["pragma"]) == ("no-store", "no-cache")
    token: Final = login.json()
    assert token["token_type"] == "bearer" and token["expires_in"] == _settings().session_ttl_seconds
    me: Final = client.get("/api/v1/auth/me", headers=_bearer(token["access_token"])).json()
    assert (me["subject"], me["roles"], me["token_kind"]) == ("admin", ["admin"], "session")


def test_login_throttle_is_client_scoped_and_includes_retry_after() -> None:
    settings: Final = _settings()
    container: Final = build_container(settings)
    clock: Final = iter(repeat(1000.0))
    password: Final = settings.ui_password
    assert password is not None
    login_service: Final = LoginService(
        sessions=SessionTokenCodec(
            MASTER_KEY,
            settings.session_issuer,
            timedelta(seconds=settings.session_ttl_seconds),
            container.policy,
        ),
        policy=container.policy,
        ttl_seconds=settings.session_ttl_seconds,
        admin_username=settings.ui_username,
        admin_password=password.get_secret_value(),
        throttle=LoginThrottle(lambda: next(clock)),
    )
    app: Final = create_app(container, Licensed())
    app.dependency_overrides[get_login_service] = lambda: login_service
    with (
        TestClient(app, client=("198.51.100.1", 50000)) as client_a,
        TestClient(app, client=("198.51.100.2", 50000)) as client_b,
    ):
        failures: Final = tuple(
            client_a.post("/api/v1/auth/login", json={"username": "admin", "password": "wrong"})
            for _ in range(LOCKOUT_FAILURES)
        )
        assert tuple(response.status_code for response in failures) == (401,) * LOCKOUT_FAILURES
        locked: Final = client_a.post("/api/v1/auth/login", json={"username": "admin", "password": "pw"})
        success: Final = client_b.post("/api/v1/auth/login", json={"username": "admin", "password": "pw"})

    assert locked.status_code == 429
    assert locked.headers["retry-after"] == "300"
    assert success.status_code == 200, success.text


@pytest.mark.parametrize("body", [{"username": "admin", "password": "wrong"}, {"username": "root", "password": "pw"}])
def test_wrong_credentials_are_401(body: dict[str, str]) -> None:
    response: Final = _client().post("/api/v1/auth/login", json=body)
    assert response.status_code == 401
    assert response.json()["type"] == "urn:litellm:error:unauthorized"


@pytest.mark.parametrize(
    "body",
    [
        {"username": "ádmin", "password": "pw"},
        {"username": "admin", "password": "päss"},
    ],
)
def test_non_ascii_login_credentials_are_401(body: dict[str, str]) -> None:
    response: Final = _client().post("/api/v1/auth/login", json=body)
    assert response.status_code == 401
    assert response.json()["type"] == "urn:litellm:error:unauthorized"


def test_non_ascii_bearer_is_401() -> None:
    response: Final = _client().get("/api/v1/auth/me", headers={"Authorization": b"Bearer caf\xe9"})
    assert response.status_code == 401


def test_login_is_disabled_when_no_admin_user_is_configured() -> None:
    settings: Final = ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": MASTER_KEY})
    response: Final = _client(settings).post("/api/v1/auth/login", json={"username": "", "password": ""})
    assert response.status_code == 422
    assert _client(settings).post("/api/v1/auth/login", json={"username": "a", "password": "b"}).status_code == 401


def test_invalid_login_body_is_422_error_json() -> None:
    response: Final = _client().post("/api/v1/auth/login", json={"username": "admin"})
    assert response.status_code == 422
    assert response.json()["type"] == "urn:litellm:error:invalid-request-body"


def test_oversized_control_plane_request_is_rejected_before_json_parsing() -> None:
    response: Final = _client().post(
        "/api/v1/auth/login",
        content=b'{"username":"admin","password":"' + b"x" * (1100 * 1024) + b'"}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["type"] == "urn:litellm:error:request-too-large"


def test_session_scopes_are_reported_and_clamped_to_the_role() -> None:
    settings: Final = _settings()
    container: Final = build_container(settings)
    codec: Final = SessionTokenCodec(MASTER_KEY, settings.session_issuer, timedelta(hours=1), container.policy)
    viewer: Final = build_principal(
        container.policy, "v", None, frozenset({VIEWER_ROLE}), None, frozenset({"eu"}), "session"
    )
    body: Final = _client(settings).get("/api/v1/auth/me", headers=_bearer(codec.mint(viewer))).json()
    assert body["dataplanes"] == ["eu"]
    assert "proxy:write" not in body["scopes"] and "usage:read" in body["scopes"]


def test_master_key_setting_is_a_secret() -> None:
    assert isinstance(_settings().master_key, SecretStr)
    assert MASTER_KEY not in repr(_settings())


def test_custom_roles_from_settings_are_honoured_on_me() -> None:
    settings: Final = _settings(roles={"finops": {"scopes": ["usage:read"], "inherits": ["usage_viewer"]}})
    container: Final = build_container(settings)
    codec: Final = SessionTokenCodec(MASTER_KEY, settings.session_issuer, timedelta(hours=1), container.policy)
    finops: Final = build_principal(container.policy, "f", None, frozenset({"finops"}), None, frozenset(), "session")
    client: Final = TestClient(create_app(container, Licensed()))
    body: Final = client.get("/api/v1/auth/me", headers=_bearer(codec.mint(finops))).json()
    assert (body["roles"], body["scopes"]) == (["finops"], ["usage:read"])


def test_a_role_granting_an_unknown_scope_fails_startup() -> None:
    with pytest.raises(InvalidRolePolicy, match="unknown scope 'nope'"):
        build_container(_settings(roles={"x": {"scopes": ["nope"]}}))


def test_a_role_inheriting_an_unknown_role_fails_startup() -> None:
    with pytest.raises(InvalidRolePolicy, match="inherits unknown role 'ghost'"):
        build_container(_settings(roles={"x": {"inherits": ["ghost"]}}))


def test_tracing_shuts_down_even_when_an_audit_sink_fails_to_close() -> None:
    counter: Final = ShutdownCounter()
    provider: Final = TracerProvider()
    provider.add_span_processor(counter)
    container: Final = build_container(_settings(), tracer_provider=provider, audit_sinks=(BrokenCloseSink(),))
    with TestClient(create_app(container, Licensed())):
        pass
    assert counter.shutdowns == 1
