from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Final

import httpx
from fastapi.testclient import TestClient

from litellm.litellm_core_utils.oauth_token_exchange import OAuthTokenExchangeConfig
from litellm.proxy.control_plane.app import create_app
from litellm.proxy.control_plane.auth.jwt import SessionTokenCodec
from litellm.proxy.control_plane.auth.rbac import USAGE_VIEWER_ROLE, VIEWER_ROLE, RolePolicy, build_principal
from litellm.proxy.control_plane.container import build_container
from litellm.proxy.control_plane.settings import ControlPlaneSettings, read_dataplanes_file

MASTER_KEY: Final = "sk-cp-master"
DATAPLANES: Final = {
    "eu": {"url": "https://eu.test", "service_token": {"token": "sk-eu"}},
    "us": {"url": "https://us.test", "name": "US East", "service_token": {"token": "sk-us"}},
}


class Licensed:
    def is_premium(self) -> bool:
        return True


@dataclass
class FakeTransport:  # mutable-ok: records calls for assertions
    outcomes: Iterator[httpx.Response | Exception]
    calls: list[tuple[str, dict[str, str] | None]] = field(default_factory=list)

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, str] | None = None,
        json: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float,
    ) -> httpx.Response:
        assert method == "GET", "router never writes"
        self.calls.append((url, headers))
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


async def _no_exchange(config: OAuthTokenExchangeConfig, subject_token: str) -> str:
    raise AssertionError("no oidc dataplane configured")


def _response(status: int, body: object = None) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("GET", "https://x"))


@dataclass(frozen=True)
class _Minter:
    codec: SessionTokenCodec
    policy: RolePolicy


def _rig(*outcomes: httpx.Response | Exception) -> tuple[TestClient, FakeTransport, _Minter]:
    settings = ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": MASTER_KEY, "dataplanes": DATAPLANES})
    transport = FakeTransport(iter(outcomes))
    container = build_container(settings, transport=transport, exchanger=_no_exchange)
    codec = SessionTokenCodec(MASTER_KEY, settings.session_issuer, timedelta(hours=1), container.policy)
    return TestClient(create_app(container, Licensed())), transport, _Minter(codec, container.policy)


def _viewer(minter: _Minter, *dataplanes: str, role: str = VIEWER_ROLE) -> dict[str, str]:
    principal = build_principal(
        minter.policy, "alice", "alice@x.test", frozenset({role}), None, frozenset(dataplanes), "session"
    )
    return {"Authorization": f"Bearer {minter.codec.mint(principal)}"}


def test_list_shows_only_the_dataplanes_the_caller_is_bound_to() -> None:
    client, _, codec = _rig()
    fleet = client.get("/api/v1/dataplanes", headers={"Authorization": f"Bearer {MASTER_KEY}"}).json()
    assert [item["id"] for item in fleet["items"]] == ["eu", "us"]
    assert fleet["items"][1] == {
        "id": "us",
        "name": "US East",
        "url": "https://us.test",
        "credentials": ["service_token"],
    }
    bound = client.get("/api/v1/dataplanes", headers=_viewer(codec, "us")).json()
    assert [item["id"] for item in bound["items"]] == ["us"]


def test_unbound_dataplane_is_403_and_unknown_is_404() -> None:
    client, _, codec = _rig()
    assert client.get("/api/v1/dataplanes/eu", headers=_viewer(codec, "us")).status_code == 403
    missing = client.get("/api/v1/dataplanes/apac", headers=_viewer(codec))
    assert missing.status_code == 404 and missing.json()["type"] == "urn:litellm:error:not-found"


def test_usage_viewer_lacks_the_dataplanes_read_scope() -> None:
    client, _, codec = _rig()
    denied = client.get("/api/v1/dataplanes", headers=_viewer(codec, role=USAGE_VIEWER_ROLE))
    assert denied.status_code == 403 and "dataplanes:read" in denied.json()["detail"]


def test_whoami_forwards_the_callers_identity_and_reports_what_the_dataplane_saw() -> None:
    client, transport, codec = _rig(
        _response(200, {"user_id": "alice", "user_info": {"user_email": "alice@x.test", "user_role": "proxy_admin"}})
    )
    body = client.get("/api/v1/dataplanes/us/whoami", headers=_viewer(codec, "us")).json()
    assert body == {
        "id": "us",
        "credential": "service_token",
        "user_id": "alice",
        "user_email": "alice@x.test",
        "user_role": "proxy_admin",
    }
    url, headers = transport.calls[0]
    assert url == "https://us.test/user/info"
    assert headers is not None
    assert (
        headers["Authorization"],
        headers["X-LiteLLM-Impersonate-User"],
        headers["X-LiteLLM-Impersonate-Email"],
    ) == ("Bearer sk-us", "alice", "alice@x.test")


def test_whoami_with_a_subjectless_caller_reports_the_credential_and_no_user() -> None:
    client, transport, _ = _rig(_response(404, {"error": {"message": "User None not found"}}))
    body = client.get("/api/v1/dataplanes/eu/whoami", headers={"Authorization": f"Bearer {MASTER_KEY}"}).json()
    assert body == {"id": "eu", "credential": "service_token", "user_id": None, "user_email": None, "user_role": None}
    _, headers = transport.calls[0]
    assert headers is not None and "X-LiteLLM-Impersonate-User" not in headers


def test_dataplane_rejection_and_outage_surface_as_502_problems() -> None:
    client, _, codec = _rig(_response(403, {"error": "lacks impersonate"}), *([httpx.ConnectError("down")] * 3))
    rejected = client.get("/api/v1/dataplanes/eu/whoami", headers=_viewer(codec))
    assert rejected.status_code == 502 and "HTTP 403" in rejected.json()["detail"]
    down = client.get("/api/v1/dataplanes/eu/whoami", headers=_viewer(codec))
    assert down.status_code == 502 and "3 attempt" in down.json()["detail"]


def test_health_reports_ok_degraded_and_unreachable_without_failing_the_request() -> None:
    client, _, codec = _rig(
        _response(200, {"status": "healthy"}), _response(500, "warming"), *([httpx.ReadTimeout("slow")] * 3)
    )
    headers = _viewer(codec)
    assert client.get("/api/v1/dataplanes/eu/health", headers=headers).json()["status"] == "ok"
    degraded = client.get("/api/v1/dataplanes/eu/health", headers=headers).json()
    assert (degraded["status"], degraded["http_status"]) == ("degraded", 500)
    unreachable = client.get("/api/v1/dataplanes/eu/health", headers=headers).json()
    assert (unreachable["status"], unreachable["http_status"]) == (
        "unreachable",
        None,
    ) and "ReadTimeout" in unreachable["detail"]


def test_dataplanes_file_has_the_same_shape_as_the_env_json(tmp_path: Path) -> None:
    path = tmp_path / "dataplanes.yaml"
    path.write_text("dataplanes:\n  eu:\n    url: https://eu.test\n    service_token:\n      token: sk-eu\n")
    specs = read_dataplanes_file(str(path))
    assert list(specs) == ["eu"] and specs["eu"].service_token is not None
    assert specs["eu"].service_token.token.get_secret_value() == "sk-eu"


def test_a_redirect_or_not_modified_liveness_reply_is_degraded_not_ok() -> None:
    client, _, codec = _rig(_response(304), _response(204))
    not_modified = client.get("/api/v1/dataplanes/eu/health", headers=_viewer(codec)).json()
    assert (not_modified["status"], not_modified["http_status"]) == ("degraded", 304)
    assert client.get("/api/v1/dataplanes/eu/health", headers=_viewer(codec)).json()["status"] == "ok"


def test_malformed_user_info_body_is_a_502_problem_not_a_crash() -> None:
    client, _, codec = _rig(_response(200, {"user_info": []}), _response(200, ["alice"]))
    for _ in range(2):
        malformed = client.get("/api/v1/dataplanes/eu/whoami", headers=_viewer(codec))
        assert malformed.status_code == 502 and "malformed" in malformed.json()["detail"]
