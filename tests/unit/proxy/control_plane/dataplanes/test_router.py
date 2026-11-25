from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Final

import httpx
import pytest
from fastapi.testclient import TestClient

from litellm.litellm_core_utils.oauth_token_exchange import OAuthTokenExchangeConfig
from litellm.proxy.control_plane.app import create_app
from litellm.proxy.control_plane.auth.jwt import SessionTokenCodec
from litellm.proxy.control_plane.auth.rbac import USAGE_VIEWER_ROLE, VIEWER_ROLE, RolePolicy, build_principal
from litellm.proxy.control_plane.container import build_container
from litellm.proxy.control_plane.settings import ControlPlaneSettings, read_dataplanes_file
from tests.unit.proxy.control_plane.store.support import MemoryStore, MemoryStream
from tests.unit.proxy.control_plane.store.support import container as runtime_container
from tests.unit.proxy.control_plane.support import RoutedTransport, response

MASTER_KEY: Final = "01234567890123456789012345678901"
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

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, str] | None = None,
        json: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float,
    ) -> AsyncIterator[httpx.Response]:
        assert method == "GET", "router never writes"
        self.calls.append((url, headers))
        outcome: Final = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        yield outcome


async def _no_exchange(config: OAuthTokenExchangeConfig, subject_token: str) -> str:
    raise AssertionError("no oidc dataplane configured")


def _response(status: int, body: object = None) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("GET", "https://x"))


@dataclass(frozen=True)
class _Minter:
    codec: SessionTokenCodec
    policy: RolePolicy


def _rig(*outcomes: httpx.Response | Exception) -> tuple[TestClient, FakeTransport, _Minter]:
    settings: Final = ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": MASTER_KEY, "dataplanes": DATAPLANES})
    transport: Final = FakeTransport(iter(outcomes))
    container: Final = build_container(settings, transport=transport, exchanger=_no_exchange)
    codec: Final = SessionTokenCodec(MASTER_KEY, settings.session_issuer, timedelta(hours=1), container.policy)
    return TestClient(create_app(container, Licensed())), transport, _Minter(codec, container.policy)


def _viewer(minter: _Minter, *dataplanes: str, role: str = VIEWER_ROLE) -> dict[str, str]:
    principal: Final = build_principal(
        minter.policy, "alice", "alice@x.test", frozenset({role}), None, frozenset(dataplanes), "session"
    )
    return {"Authorization": f"Bearer {minter.codec.mint(principal)}"}


def test_list_shows_only_the_dataplanes_the_caller_is_bound_to() -> None:
    client, _, codec = _rig()
    fleet: Final = client.get("/api/v1/dataplanes", headers={"Authorization": f"Bearer {MASTER_KEY}"}).json()
    assert [item["id"] for item in fleet["items"]] == ["eu", "us"]
    assert fleet["items"][1] == {
        "id": "us",
        "name": "US East",
        "url": "https://us.test",
        "credentials": ["service_token"],
        "source": "config",
    }
    bound: Final = client.get("/api/v1/dataplanes", headers=_viewer(codec, "us")).json()
    assert [item["id"] for item in bound["items"]] == ["us"]


def test_unbound_dataplane_is_403_and_unknown_is_404() -> None:
    client, _, codec = _rig()
    assert client.get("/api/v1/dataplanes/eu", headers=_viewer(codec, "us")).status_code == 403
    missing: Final = client.get("/api/v1/dataplanes/apac", headers=_viewer(codec))
    assert missing.status_code == 404 and missing.json()["type"] == "urn:litellm:error:not-found"


def test_usage_viewer_lacks_the_dataplanes_read_scope() -> None:
    client, _, codec = _rig()
    denied: Final = client.get("/api/v1/dataplanes", headers=_viewer(codec, role=USAGE_VIEWER_ROLE))
    assert denied.status_code == 403 and "dataplanes:read" in denied.json()["detail"]


def test_whoami_forwards_the_callers_identity_and_reports_what_the_dataplane_saw() -> None:
    client, transport, codec = _rig(
        _response(200, {"user_id": "alice", "user_info": {"user_email": "alice@x.test", "user_role": "proxy_admin"}})
    )
    body: Final = client.get("/api/v1/dataplanes/us/whoami", headers=_viewer(codec, "us")).json()
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
    body: Final = client.get("/api/v1/dataplanes/eu/whoami", headers={"Authorization": f"Bearer {MASTER_KEY}"}).json()
    assert body == {"id": "eu", "credential": "service_token", "user_id": None, "user_email": None, "user_role": None}
    _, headers = transport.calls[0]
    assert headers is not None and "X-LiteLLM-Impersonate-User" not in headers


@pytest.mark.parametrize("status", [300, 301, 302, 303, 304, 307, 308])
def test_whoami_does_not_accept_redirects_as_identity_proof(status: int) -> None:
    client, _, codec = _rig(*(_response(status, {"user_id": "alice"}) for _ in range(3)))
    reply: Final = client.get("/api/v1/dataplanes/eu/whoami", headers=_viewer(codec))
    assert reply.status_code == 502
    assert str(status) in reply.json()["detail"]


def test_dataplane_rejection_and_outage_surface_as_502_errors() -> None:
    client, _, codec = _rig(_response(403, {"error": "lacks impersonate"}), *([httpx.ConnectError("down")] * 3))
    rejected: Final = client.get("/api/v1/dataplanes/eu/whoami", headers=_viewer(codec))
    assert rejected.status_code == 502 and "HTTP 403" in rejected.json()["detail"]
    down: Final = client.get("/api/v1/dataplanes/eu/whoami", headers=_viewer(codec))
    assert down.status_code == 502 and "3 attempt" in down.json()["detail"]


def test_transport_errors_do_not_expose_bearer_details_from_health_or_whoami() -> None:
    bearer: Final = "Bearer secret-service-token"
    error: Final = f"connection failed with {bearer}"
    client, _, codec = _rig(*(httpx.ConnectError(error) for _ in range(6)))
    health: Final = client.get("/api/v1/dataplanes/eu/health", headers=_viewer(codec))
    whoami: Final = client.get("/api/v1/dataplanes/eu/whoami", headers=_viewer(codec))
    assert health.status_code == 200 and bearer not in health.text
    assert whoami.status_code == 502 and bearer not in whoami.text


def test_health_reports_ok_degraded_and_unreachable_without_failing_the_request() -> None:
    client, _, codec = _rig(
        _response(200, {"status": "healthy"}), _response(500, "warming"), *([httpx.ReadTimeout("slow")] * 3)
    )
    headers: Final = _viewer(codec)
    assert client.get("/api/v1/dataplanes/eu/health", headers=headers).json()["status"] == "ok"
    degraded: Final = client.get("/api/v1/dataplanes/eu/health", headers=headers).json()
    assert (degraded["status"], degraded["http_status"]) == ("degraded", 500)
    unreachable: Final = client.get("/api/v1/dataplanes/eu/health", headers=headers).json()
    assert (unreachable["status"], unreachable["http_status"]) == (
        "unreachable",
        None,
    ) and "ReadTimeout" in unreachable["detail"]


def test_dataplanes_file_has_the_same_shape_as_the_env_json(tmp_path: Path) -> None:
    path: Final = tmp_path / "dataplanes.yaml"
    path.write_text("dataplanes:\n  eu:\n    url: https://eu.test\n    service_token:\n      token: sk-eu\n")
    specs: Final = read_dataplanes_file(str(path))
    assert list(specs) == ["eu"] and specs["eu"].service_token is not None
    assert specs["eu"].service_token.token.get_secret_value() == "sk-eu"


def test_a_redirect_or_not_modified_liveness_reply_is_degraded_not_ok() -> None:
    client, _, codec = _rig(_response(304), _response(204))
    not_modified: Final = client.get("/api/v1/dataplanes/eu/health", headers=_viewer(codec)).json()
    assert (not_modified["status"], not_modified["http_status"]) == ("degraded", 304)
    assert client.get("/api/v1/dataplanes/eu/health", headers=_viewer(codec)).json()["status"] == "ok"


def test_malformed_user_info_body_is_a_502_error_not_a_crash() -> None:
    client, _, codec = _rig(_response(200, {"user_info": []}), _response(200, ["alice"]))
    for _ in range(2):
        malformed: Final = client.get("/api/v1/dataplanes/eu/whoami", headers=_viewer(codec))
        assert malformed.status_code == 502 and "malformed" in malformed.json()["detail"]


@pytest.mark.asyncio
async def test_runtime_dataplane_is_secret_free_and_live_for_list_get_fanout_proxy_and_audit() -> None:
    dataplane_id: Final = "sk-" + "r" * 40
    store: Final = MemoryStore()
    stream: Final = MemoryStream()
    transport: Final = RoutedTransport(
        {"https://runtime.test/model/info": iter((response(200, {"data": []}), response(200, {"data": []})))}
    )
    rig: Final = runtime_container(store, stream, transport=transport)
    await rig.runtime_config.start()
    headers: Final = {"Authorization": f"Bearer {MASTER_KEY}"}
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(rig, Licensed())), base_url="http://cp"
        ) as client:
            put: Final = await client.put(
                f"/api/v1/dataplanes/{dataplane_id}",
                headers=headers,
                json={"url": "https://runtime.test/", "service_token": {"token": "private-service-credential"}},
            )
            assert put.status_code == 200, put.text
            assert put.json() == {
                "id": dataplane_id,
                "name": dataplane_id,
                "url": "https://runtime.test",
                "credentials": ["service_token"],
                "source": "runtime",
            }
            await stream.consumed(stream.entries[-1].revision)
            listed: Final = await client.get("/api/v1/dataplanes", headers=headers)
            fetched: Final = await client.get(f"/api/v1/dataplanes/{dataplane_id}", headers=headers)
            fleet: Final = await client.get("/api/v1/fleet/models", headers=headers)
            proxied: Final = await client.get(f"/api/v1/dataplanes/{dataplane_id}/proxy/model/info", headers=headers)
            assert listed.json() == {"items": [put.json()]}
            assert fetched.json() == put.json()
            assert fleet.status_code == proxied.status_code == 200
            assert len(transport.calls) == 2
            assert rig.audit_log.recent(1)[0].dataplane_id == dataplane_id
            persisted: Final = " ".join(row.model_dump_json() for row in store.rows)
            published: Final = " ".join(entry.model_dump_json() for entry in stream.entries)
            assert "private-service-credential" not in persisted + published + put.text + listed.text + fetched.text
            deleted: Final = await client.delete(f"/api/v1/dataplanes/{dataplane_id}", headers=headers)
            assert deleted.status_code == 204, deleted.text
            await stream.consumed(stream.entries[-1].revision)
            assert rig.registry.get(dataplane_id) is None
            assert (await client.get(f"/api/v1/dataplanes/{dataplane_id}", headers=headers)).status_code == 404
            assert store.rows == ()
    finally:
        await rig.runtime_config.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["PUT", "DELETE"])
@pytest.mark.parametrize("gate", ["scope", "binding"])
async def test_runtime_dataplane_write_scope_and_fleet_binding_are_required_and_audited(method: str, gate: str) -> None:
    stream: Final = MemoryStream()
    rig: Final = runtime_container(MemoryStore(), stream, roles={"writer": {"scopes": ["dataplanes:write"]}})
    rig.runtime_config.ready = True
    codec: Final = SessionTokenCodec(MASTER_KEY, rig.settings.session_issuer, timedelta(minutes=5), rig.policy)
    actor: Final = build_principal(
        rig.policy,
        "operator",
        None,
        frozenset({"viewer" if gate == "scope" else "writer"}),
        None,
        frozenset() if gate == "scope" else frozenset({"eu"}),
        "session",
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(rig, Licensed())), base_url="http://cp"
    ) as client:
        result: Final = await client.request(
            method,
            "/api/v1/dataplanes/new",
            headers={"Authorization": f"Bearer {codec.mint(actor)}"},
            **(
                {"json": {"url": "https://runtime.test", "service_token": {"token": "private"}}}
                if method == "PUT"
                else {}
            ),
        )
    assert result.status_code == 403, result.text
    assert stream.entries == ()
    event: Final = rig.audit_log.recent(1)[0]
    assert (event.actor, event.status, event.decision) == ("operator", 403, "denied")


@pytest.mark.asyncio
async def test_config_dataplanes_are_read_only_and_unconfigured_writes_are_unavailable() -> None:
    configured: Final = runtime_container(MemoryStore(), MemoryStream(), dataplanes=DATAPLANES)
    configured.runtime_config.ready = True
    unconfigured: Final = runtime_container()
    for rig, name, expected in ((configured, "eu", 409), (unconfigured, "new", 503)):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(rig, Licensed())), base_url="http://cp"
        ) as client:
            put: Final = await client.put(
                f"/api/v1/dataplanes/{name}",
                headers={"Authorization": f"Bearer {MASTER_KEY}"},
                json={"url": "https://runtime.test", "service_token": {"token": "private"}},
            )
            delete: Final = await client.delete(
                f"/api/v1/dataplanes/{name}", headers={"Authorization": f"Bearer {MASTER_KEY}"}
            )
        assert put.status_code == delete.status_code == expected
        assert len(rig.audit_log.recent(10)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("spec", "status"),
    [
        ({"url": "https://user:password@runtime.test", "service_token": {"token": "private"}}, 422),
        ({"url": "http://runtime.test", "service_token": {"token": "private"}}, 422),
        ({"url": "https://runtime.test"}, 400),
    ],
)
async def test_invalid_runtime_dataplanes_are_rejected_and_audited_before_publication(
    spec: dict[str, object], status: int
) -> None:
    stream: Final = MemoryStream()
    rig: Final = runtime_container(MemoryStore(), stream)
    rig.runtime_config.ready = True
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(rig, Licensed())), base_url="http://cp"
    ) as client:
        result: Final = await client.put(
            "/api/v1/dataplanes/new", headers={"Authorization": f"Bearer {MASTER_KEY}"}, json=spec
        )
    assert result.status_code == status, result.text
    assert stream.entries == ()
    assert rig.audit_log.recent(1)[0].status == status
