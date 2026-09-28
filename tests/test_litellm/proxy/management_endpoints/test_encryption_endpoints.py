"""Unit tests for the at-rest encryption-migration HTTP endpoints.

The endpoint bodies are exercised directly with the migration engine mocked, so
the admin guard, db-not-connected guard, and success path are all covered
without touching a live DB. The live ASGI/auth contract is covered separately in
``tests/proxy_behavior/management/test_credential_migration_endpoint.py``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.management_endpoints import credential_migration as cm
from litellm.proxy.management_endpoints.key_management_endpoints import (
    check_encryption_endpoint,
    migrate_encryption_endpoint,
)

ADMIN = SimpleNamespace(user_role=LitellmUserRoles.PROXY_ADMIN.value)
NONADMIN = SimpleNamespace(user_role=LitellmUserRoles.INTERNAL_USER.value)


def _sample_report() -> cm.MigrationReport:
    report = cm.MigrationReport()
    report.add(
        cm.LocationReport(location="model_table", scanned=2, migrated=1, legacy=0)
    )
    return report


# ------------------------------- check endpoint -------------------------------


@pytest.mark.asyncio
async def test_check_endpoint_success(monkeypatch):
    monkeypatch.setattr(proxy_server, "prisma_client", object())
    monkeypatch.setattr(
        cm, "check_encryption", AsyncMock(return_value=_sample_report())
    )

    out = await check_encryption_endpoint(user_api_key_dict=ADMIN)

    assert out["status"] == "success"
    assert out["report"]["residual_legacy"] == 0
    assert out["report"]["locations"]["model_table"]["scanned"] == 2


@pytest.mark.asyncio
async def test_check_endpoint_requires_admin(monkeypatch):
    monkeypatch.setattr(proxy_server, "prisma_client", object())
    with pytest.raises(HTTPException) as exc:
        await check_encryption_endpoint(user_api_key_dict=NONADMIN)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_check_endpoint_db_not_connected(monkeypatch):
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    with pytest.raises(HTTPException) as exc:
        await check_encryption_endpoint(user_api_key_dict=ADMIN)
    assert exc.value.status_code == 500


# ------------------------------ migrate endpoint ------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [False, True])
async def test_migrate_endpoint_success(monkeypatch, dry_run):
    monkeypatch.setattr(proxy_server, "prisma_client", object())
    fake = AsyncMock(return_value=_sample_report())
    monkeypatch.setattr(cm, "migrate_encryption", fake)

    out = await migrate_encryption_endpoint(user_api_key_dict=ADMIN, dry_run=dry_run)

    assert out["status"] == "success"
    assert out["dry_run"] is dry_run
    assert out["report"]["locations"]["model_table"]["migrated"] == 1
    # dry_run is threaded through to the engine unchanged.
    assert fake.await_args.kwargs["dry_run"] is dry_run


@pytest.mark.asyncio
async def test_migrate_endpoint_requires_admin(monkeypatch):
    monkeypatch.setattr(proxy_server, "prisma_client", object())
    with pytest.raises(HTTPException) as exc:
        await migrate_encryption_endpoint(user_api_key_dict=NONADMIN)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_migrate_endpoint_db_not_connected(monkeypatch):
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    with pytest.raises(HTTPException) as exc:
        await migrate_encryption_endpoint(user_api_key_dict=ADMIN)
    assert exc.value.status_code == 500
