"""Bound unauthenticated upstream validation without replacing gateway authentication."""

import asyncio
import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.applications import Starlette


@pytest.mark.asyncio
async def test_upstream_validation_is_bounded_and_slots_release_after_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPSTREAM_LITELLM_BASE_URL", "https://upstream.example.invalid")
    filename = Path(__file__).resolve().parents[4] / "deploy" / "memory-pilot" / "pilot.py"
    spec = importlib.util.spec_from_file_location("memory_pilot_test", filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gateway = module.PilotGateway(Starlette())
    entered = asyncio.Event()
    release = asyncio.Event()
    count = 0

    async def upstream_get(*args: object, **kwargs: object) -> httpx.Response:
        nonlocal count
        count += 1
        if count == 16:
            entered.set()
        await release.wait()
        raise httpx.ConnectError("unavailable")

    upstream = MagicMock(get=AsyncMock(side_effect=upstream_get))
    gateway.upstream = upstream
    database = MagicMock()
    database.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=None)
    with patch.multiple(  # test-quality-ok: Inject external database/config; exercise real ASGI admission.
        "litellm.proxy.proxy_server", prisma_client=database, master_key="local-admin"
    ):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://pilot") as client:
            pending = [
                asyncio.create_task(client.get("/v1/models", headers={"Authorization": f"Bearer sk-invalid-{i}"}))
                for i in range(16)
            ]
            await asyncio.wait_for(entered.wait(), timeout=2)
            refused = await client.get("/v1/models", headers={"Authorization": "Bearer sk-overload"})
            assert refused.status_code == 503 and refused.headers["retry-after"] == "1"
            assert upstream.get.await_count == 16
            release.set()
            assert all(response.status_code == 503 for response in await asyncio.gather(*pending))
            again = await client.get("/v1/models", headers={"Authorization": "Bearer sk-next"})
            assert again.status_code == 503 and "unavailable" in again.text
            assert upstream.get.await_count == 17
