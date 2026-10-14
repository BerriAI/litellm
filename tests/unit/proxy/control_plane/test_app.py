from datetime import timedelta
from typing import Final

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from pydantic import SecretStr

from litellm.proxy.control_plane.app import LicenseRequired, create_app
from litellm.proxy.control_plane.audit.pipeline import Batch, Delivered, DeliveryResult
from litellm.proxy.control_plane.auth.jwt import SessionTokenCodec
from litellm.proxy.control_plane.auth.rbac import VIEWER_ROLE, build_principal
from litellm.proxy.control_plane.container import InvalidRolePolicy, build_container
from litellm.proxy.control_plane.settings import ControlPlaneSettings

MASTER_KEY: Final = "sk-cp-master"


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
    client = _client()
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ok"}


def test_missing_bearer_is_401_problem_json_with_a_challenge() -> None:
    response = _client().get("/api/v1/auth/me")
    assert response.status_code == 401
    assert response.headers["content-type"] == "application/problem+json"
    assert response.headers["www-authenticate"].startswith("Bearer ")
    assert response.json()["type"] == "urn:litellm:error:unauthorized"


def test_garbage_bearer_is_401() -> None:
    assert _client().get("/api/v1/auth/me", headers=_bearer("nope")).status_code == 401


def test_master_key_is_a_fleet_wide_cp_admin() -> None:
    body = _client().get("/api/v1/auth/me", headers=_bearer(MASTER_KEY)).json()
    assert body["roles"] == ["admin"]
    assert body["dataplanes"] is None
    assert body["token_kind"] == "master_key"
    assert "dataplanes:write" in body["scopes"]


def test_login_mints_a_session_that_me_accepts() -> None:
    client = _client()
    login = client.post("/api/v1/auth/login", json={"username": "admin", "password": "pw"})
    assert login.status_code == 200, login.text
    token = login.json()
    assert token["token_type"] == "bearer" and token["expires_in"] == _settings().session_ttl_seconds
    me = client.get("/api/v1/auth/me", headers=_bearer(token["access_token"])).json()
    assert (me["subject"], me["roles"], me["token_kind"]) == ("admin", ["admin"], "session")


@pytest.mark.parametrize("body", [{"username": "admin", "password": "wrong"}, {"username": "root", "password": "pw"}])
def test_wrong_credentials_are_401(body: dict[str, str]) -> None:
    response = _client().post("/api/v1/auth/login", json=body)
    assert response.status_code == 401
    assert response.json()["type"] == "urn:litellm:error:unauthorized"


def test_login_is_disabled_when_no_admin_user_is_configured() -> None:
    settings = ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": MASTER_KEY})
    response = _client(settings).post("/api/v1/auth/login", json={"username": "", "password": ""})
    assert response.status_code == 422
    assert _client(settings).post("/api/v1/auth/login", json={"username": "a", "password": "b"}).status_code == 401


def test_invalid_login_body_is_422_problem_json() -> None:
    response = _client().post("/api/v1/auth/login", json={"username": "admin"})
    assert response.status_code == 422
    assert response.json()["type"] == "urn:litellm:error:invalid-request-body"


def test_session_scopes_are_reported_and_clamped_to_the_role() -> None:
    settings = _settings()
    container = build_container(settings)
    codec = SessionTokenCodec(MASTER_KEY, settings.session_issuer, timedelta(hours=1), container.policy)
    viewer = build_principal(container.policy, "v", None, frozenset({VIEWER_ROLE}), None, frozenset({"eu"}), "session")
    body = _client(settings).get("/api/v1/auth/me", headers=_bearer(codec.mint(viewer))).json()
    assert body["dataplanes"] == ["eu"]
    assert "proxy:write" not in body["scopes"] and "usage:read" in body["scopes"]


def test_master_key_setting_is_a_secret() -> None:
    assert isinstance(_settings().master_key, SecretStr)
    assert MASTER_KEY not in repr(_settings())


def test_custom_roles_from_settings_are_honoured_on_me() -> None:
    settings = _settings(roles={"finops": {"scopes": ["usage:read"], "inherits": ["usage_viewer"]}})
    container = build_container(settings)
    codec = SessionTokenCodec(MASTER_KEY, settings.session_issuer, timedelta(hours=1), container.policy)
    finops = build_principal(container.policy, "f", None, frozenset({"finops"}), None, frozenset(), "session")
    client = TestClient(create_app(container, Licensed()))
    body = client.get("/api/v1/auth/me", headers=_bearer(codec.mint(finops))).json()
    assert (body["roles"], body["scopes"]) == (["finops"], ["usage:read"])


def test_a_role_granting_an_unknown_scope_fails_startup() -> None:
    with pytest.raises(InvalidRolePolicy, match="unknown scope 'nope'"):
        build_container(_settings(roles={"x": {"scopes": ["nope"]}}))


def test_a_role_inheriting_an_unknown_role_fails_startup() -> None:
    with pytest.raises(InvalidRolePolicy, match="inherits unknown role 'ghost'"):
        build_container(_settings(roles={"x": {"inherits": ["ghost"]}}))


def test_tracing_is_shut_down_even_when_an_audit_sink_fails_to_close() -> None:
    counter = ShutdownCounter()
    provider = TracerProvider()
    provider.add_span_processor(counter)
    container = build_container(_settings(), tracer_provider=provider, audit_sinks=(BrokenCloseSink(),))
    with pytest.raises(RuntimeError, match="sink refused to close"):
        with TestClient(create_app(container, Licensed())):
            pass
    assert counter.shutdowns == 1
