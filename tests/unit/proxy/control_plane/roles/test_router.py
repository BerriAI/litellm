from datetime import timedelta
from typing import Final

import httpx
import pytest

from litellm.proxy.control_plane.app import create_app
from litellm.proxy.control_plane.auth.jwt import SessionTokenCodec
from litellm.proxy.control_plane.auth.rbac import Principal, Scope, build_principal
from tests.unit.proxy.control_plane.store.support import MemoryStore, MemoryStream, container
from tests.unit.proxy.control_plane.support import MASTER_KEY, Licensed

HEADERS: Final = {"Authorization": f"Bearer {MASTER_KEY}"}


@pytest.mark.asyncio
async def test_roles_are_created_listed_propagated_to_existing_session_codec_and_deleted() -> None:
    store: Final = MemoryStore()
    stream: Final = MemoryStream()
    rig: Final = container(store, stream)
    codec: Final = SessionTokenCodec(
        MASTER_KEY,
        rig.settings.session_issuer,
        timedelta(minutes=5),
        rig.policy,
        audience=rig.settings.session_audience,
    )
    await rig.runtime_config.start()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(rig, Licensed())), base_url="http://cp"
        ) as client:
            created: Final = await client.put("/api/v1/roles/analyst", headers=HEADERS, json={"scopes": ["usage:read"]})
            assert created.status_code == 200, created.text
            assert created.json() == {"name": "analyst", "scopes": ["usage:read"], "inherits": [], "source": "runtime"}
            await stream.consumed(stream.entries[-1].revision)
            principal: Final = build_principal(
                rig.policy, "analyst-user", None, frozenset({"analyst"}), None, frozenset(), "session"
            )
            verified: Final = rig.token_verifier.verify(codec.mint(principal))
            assert isinstance(verified, Principal)
            assert verified.scopes == frozenset({Scope.USAGE_READ})
            listed: Final = await client.get("/api/v1/roles", headers=HEADERS)
            assert listed.status_code == 200, listed.text
            assert created.json() in listed.json()["items"]
            assert any(role["source"] == "config" for role in listed.json()["items"])
            deleted: Final = await client.delete("/api/v1/roles/analyst", headers=HEADERS)
            assert deleted.status_code == 204, deleted.text
            await stream.consumed(stream.entries[-1].revision)
            assert "analyst" not in rig.policy.names
            assert store.rows == ()
            assert tuple(event.decision for event in rig.audit_log.recent(2)) == ("allowed", "allowed")
    finally:
        await rig.runtime_config.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "PUT", "DELETE"])
async def test_role_routes_require_roles_write(method: str) -> None:
    rig: Final = container(MemoryStore(), MemoryStream())
    codec: Final = SessionTokenCodec(MASTER_KEY, rig.settings.session_issuer, timedelta(minutes=5), rig.policy)
    viewer: Final = build_principal(rig.policy, "viewer", None, frozenset({"viewer"}), None, frozenset(), "session")
    headers: Final = {"Authorization": f"Bearer {codec.mint(viewer)}"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(rig, Licensed())), base_url="http://cp"
    ) as client:
        result: Final = await client.request(
            method,
            "/api/v1/roles" if method == "GET" else "/api/v1/roles/new",
            headers=headers,
            **({"json": {"scopes": []}} if method == "PUT" else {}),
        )
    assert result.status_code == 403, result.text
    assert rig.audit_log.recent(1)[0].decision == "denied"
    assert rig.audit_log.recent(1)[0].status == 403


@pytest.mark.asyncio
async def test_role_escalation_is_rejected_even_through_an_inherited_role() -> None:
    stream: Final = MemoryStream()
    store: Final = MemoryStore()
    rig: Final = container(store, stream, roles={"delegator": {"scopes": ["roles:write"]}})
    rig.runtime_config.ready = True
    codec: Final = SessionTokenCodec(MASTER_KEY, rig.settings.session_issuer, timedelta(minutes=5), rig.policy)
    delegator: Final = build_principal(
        rig.policy, "delegator", None, frozenset({"delegator"}), None, frozenset(), "session"
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(rig, Licensed())), base_url="http://cp"
    ) as client:
        result: Final = await client.put(
            "/api/v1/roles/escalated",
            headers={"Authorization": f"Bearer {codec.mint(delegator)}"},
            json={"inherits": ["admin"]},
        )
    assert result.status_code == 403, result.text
    assert stream.entries == ()
    assert store.rows == ()
    assert rig.audit_log.recent(1)[0].status == 403


@pytest.mark.asyncio
async def test_default_and_config_roles_are_read_only_and_inherited_roles_cannot_be_deleted() -> None:
    store: Final = MemoryStore()
    stream: Final = MemoryStream()
    rig: Final = container(store, stream, roles={"configured": {"scopes": ["usage:read"]}})
    rig.runtime_config.ready = True
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(rig, Licensed())), base_url="http://cp"
    ) as client:
        for name in ("admin", "configured"):
            put: Final = await client.put(f"/api/v1/roles/{name}", headers=HEADERS, json={"scopes": []})
            delete: Final = await client.delete(f"/api/v1/roles/{name}", headers=HEADERS)
            assert (put.status_code, delete.status_code) == (409, 409)
        parent: Final = await client.put("/api/v1/roles/parent", headers=HEADERS, json={"scopes": ["usage:read"]})
        child: Final = await client.put("/api/v1/roles/child", headers=HEADERS, json={"inherits": ["parent"]})
        denied: Final = await client.delete("/api/v1/roles/parent", headers=HEADERS)
        missing: Final = await client.delete("/api/v1/roles/absent", headers=HEADERS)
    assert parent.status_code == child.status_code == 200
    assert child.json()["scopes"] == ["usage:read"]
    assert child.json()["inherits"] == ["parent"]
    assert denied.status_code == 409, denied.text
    assert missing.status_code == 404, missing.text
    assert len(store.rows) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{"scopes": ["unknown:scope"]}, {"inherits": ["unknown-parent"]}])
async def test_invalid_role_is_rejected_before_stream_publication(body: dict[str, list[str]]) -> None:
    stream: Final = MemoryStream()
    rig: Final = container(MemoryStore(), stream)
    rig.runtime_config.ready = True
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(rig, Licensed())), base_url="http://cp"
    ) as client:
        result: Final = await client.put("/api/v1/roles/invalid", headers=HEADERS, json=body)
    assert result.status_code == 400, result.text
    assert stream.entries == ()
    assert rig.audit_log.recent(1)[0].status == 400


@pytest.mark.asyncio
async def test_unconfigured_role_writes_are_unavailable_but_config_roles_can_be_read() -> None:
    rig: Final = container()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(rig, Licensed())), base_url="http://cp"
    ) as client:
        listed: Final = await client.get("/api/v1/roles", headers=HEADERS)
        put: Final = await client.put("/api/v1/roles/new", headers=HEADERS, json={"scopes": []})
        deleted: Final = await client.delete("/api/v1/roles/new", headers=HEADERS)
    assert listed.status_code == 200 and listed.json()["items"]
    assert put.status_code == deleted.status_code == 503
    assert put.json()["detail"] == deleted.json()["detail"]
    assert "not configured or not migrated" in put.json()["detail"]


@pytest.mark.asyncio
async def test_strict_audit_backpressure_prevents_any_runtime_write_before_publication() -> None:
    store: Final = MemoryStore()
    stream: Final = MemoryStream()
    rig: Final = container(store, stream, audit_sinks=("log",), audit_strict=True, audit_queue_size=2)
    rig.runtime_config.ready = True
    rig.audit_log.record("operator", None, "control_plane", "PUT", "/occupied", "allowed", 200)
    rig.audit_log.record("operator", None, "control_plane", "PUT", "/occupied", "allowed", 200)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(rig, Licensed())), base_url="http://cp"
    ) as client:
        role: Final = await client.put("/api/v1/roles/new", headers=HEADERS, json={"scopes": []})
        dataplane: Final = await client.put(
            "/api/v1/dataplanes/new",
            headers=HEADERS,
            json={"url": "https://runtime.test", "service_token": {"token": "private"}},
        )
    assert role.status_code == dataplane.status_code == 503
    assert "backpressure" in role.json()["detail"]
    assert store.loads == 0
    assert stream.entries == store.rows == ()
    assert tuple(event.status for event in rig.audit_log.recent(2)) == (503, 503)


@pytest.mark.asyncio
async def test_persistence_failure_is_audited_and_emits_resync() -> None:
    store: Final = MemoryStore()
    stream: Final = MemoryStream()
    store.fail_write = True
    rig: Final = container(store, stream)
    rig.runtime_config.ready = True
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(rig, Licensed())), base_url="http://cp"
    ) as client:
        result: Final = await client.put("/api/v1/roles/new", headers=HEADERS, json={"scopes": []})
    assert result.status_code == 503, result.text
    assert tuple(entry.change.op for entry in stream.entries) == ("upsert", "resync")
    assert rig.audit_log.recent(1)[0].status == 503
