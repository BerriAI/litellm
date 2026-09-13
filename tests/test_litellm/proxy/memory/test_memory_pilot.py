"""Bound unauthenticated upstream validation without replacing gateway authentication."""

import asyncio
import hashlib
import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.applications import Starlette


@pytest.mark.asyncio
async def test_unknown_keys_cannot_consume_registered_validation_capacity_and_slots_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UPSTREAM_LITELLM_BASE_URL", "https://upstream.example.invalid")
    filename = Path(__file__).resolve().parents[4] / "deploy" / "memory-pilot" / "pilot.py"
    spec = importlib.util.spec_from_file_location("memory_pilot_test", filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clock = MagicMock(return_value=0.0)
    gateway = module.PilotGateway(Starlette(), clock=clock)
    entered = asyncio.Event()
    release = asyncio.Event()
    count = 0

    async def upstream_get(*args: object, **kwargs: object) -> httpx.Response:
        if kwargs.get("headers") == {"Authorization": "Bearer sk-established"}:
            return httpx.Response(200, json={"data": [{"id": "model"}]})
        nonlocal count
        count += 1
        if count == 4:
            entered.set()
        await release.wait()
        raise httpx.ConnectError("unavailable")

    upstream = MagicMock(get=AsyncMock(side_effect=upstream_get))
    gateway.upstream = upstream
    database = MagicMock()

    async def key_lookup(*, where: dict[str, str]) -> dict[str, str] | None:
        digest = hashlib.sha256(b"sk-established").hexdigest()
        return {"token": digest} if where == {"token": digest} else None

    database.db.litellm_verificationtoken.find_unique = AsyncMock(side_effect=key_lookup)
    with patch.multiple(  # test-quality-ok: Inject external database/config; exercise real ASGI admission.
        "litellm.proxy.proxy_server", prisma_client=database, master_key="local-admin"
    ):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://pilot") as client:
            initial = await client.get("/v1/models", headers={"Authorization": "Bearer sk-established"})
            assert initial.status_code == 200
            pending = [
                asyncio.create_task(client.get("/v1/models", headers={"Authorization": f"Bearer sk-invalid-{i}"}))
                for i in range(4)
            ]
            await asyncio.wait_for(entered.wait(), timeout=2)
            refused = await client.get("/v1/models", headers={"Authorization": "Bearer sk-overload"})
            assert refused.status_code == 503 and refused.headers["retry-after"] == "1"
            assert upstream.get.await_count == 5
            clock.return_value = 59.0
            established = await client.get("/v1/models", headers={"Authorization": "Bearer sk-established"})
            assert established.status_code == 200 and established.json() == {"data": [{"id": "model"}]}
            clock.return_value = 61.0
            refreshed = await client.get("/v1/models", headers={"Authorization": "Bearer sk-established"})
            assert refreshed.status_code == 200
            clock.return_value = 122.0
            expired = await client.get("/v1/models", headers={"Authorization": "Bearer sk-established"})
            assert expired.status_code == 503 and expired.headers["retry-after"] == "1"
            release.set()
            assert all(response.status_code == 503 for response in await asyncio.gather(*pending))
            again = await client.get("/v1/models", headers={"Authorization": "Bearer sk-next"})
            assert again.status_code == 503 and "unavailable" in again.text
            assert upstream.get.await_count == 8


@pytest.mark.asyncio
@pytest.mark.parametrize("rejection_status", [401, 403])
async def test_rejected_enrolled_keys_lose_reserved_capacity_and_can_revalidate(
    monkeypatch: pytest.MonkeyPatch, rejection_status: int
) -> None:
    monkeypatch.setenv("UPSTREAM_LITELLM_BASE_URL", "https://upstream.example.invalid")
    filename = Path(__file__).resolve().parents[4] / "deploy" / "memory-pilot" / "pilot.py"
    spec = importlib.util.spec_from_file_location("memory_pilot_revoked_test", filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gateway = module.PilotGateway(Starlette())
    revoked = asyncio.Event()
    block = asyncio.Event()
    entered = asyncio.Event()
    release = asyncio.Event()
    count = 0

    async def upstream_get(*args: object, **kwargs: object) -> httpx.Response:
        if kwargs.get("headers") == {"Authorization": "Bearer sk-revoked"}:
            if block.is_set():
                nonlocal count
                count += 1
                if count == 4:
                    entered.set()
                await release.wait()
            if revoked.is_set():
                return httpx.Response(rejection_status, json={"error": "rejected"})
        return httpx.Response(200, json={"data": []})

    gateway.upstream = MagicMock(get=AsyncMock(side_effect=upstream_get))
    database = MagicMock()
    database.db.litellm_verificationtoken.find_unique = AsyncMock(return_value={"token": "enrolled"})
    with patch.multiple(  # test-quality-ok: Keep enrollment rows present while upstream revokes access.
        "litellm.proxy.proxy_server", prisma_client=database, master_key="local-admin"
    ):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://pilot") as client:
            for credential in ("sk-established", "sk-revoked"):
                initial = await client.get("/v1/models", headers={"Authorization": f"Bearer {credential}"})
                assert initial.status_code == 200
            revoked.set()
            rejected = await client.get("/v1/models", headers={"Authorization": "Bearer sk-revoked"})
            assert rejected.status_code == rejection_status
            block.set()
            pending = [
                asyncio.create_task(client.get("/v1/models", headers={"Authorization": "Bearer sk-revoked"}))
                for _ in range(4)
            ]
            await asyncio.wait_for(entered.wait(), timeout=2)
            refused = await client.get("/v1/models", headers={"Authorization": "Bearer sk-revoked"})
            assert refused.status_code == 503 and refused.headers["retry-after"] == "1"
            established = await client.get("/v1/models", headers={"Authorization": "Bearer sk-established"})
            assert established.status_code == 200
            release.set()
            assert all(response.status_code == rejection_status for response in await asyncio.gather(*pending))
            revoked.clear()
            restored = await client.get("/v1/models", headers={"Authorization": "Bearer sk-revoked"})
            assert restored.status_code == 200
            revoked.set()
            rechecked = await client.get("/v1/models", headers={"Authorization": "Bearer sk-revoked"})
            assert rechecked.status_code == rejection_status
