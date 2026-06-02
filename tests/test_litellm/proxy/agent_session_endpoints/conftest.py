"""
Shared fixtures for `litellm/proxy/agent_session_endpoints/` tests.

Provides:
* ``fake_prisma_client``  — an in-memory stand-in for the proxy's Prisma
  client. It implements only the methods our endpoints actually call —
  no network, no schema. Tests assert against the data structures
  directly.
* ``client``               — FastAPI TestClient with all four routers
  mounted and ``user_api_key_auth`` overridden to a fixed proxy admin.
* ``other_tenant_client``  — TestClient where the auth dep returns a
  different (non-admin) caller, used for cross-tenant isolation tests.
"""

import os

import pytest

# Set a JWT secret BEFORE any module under test is imported.
os.environ.setdefault("LITELLM_AGENT_JWT_SECRET", "test-agent-jwt-secret")
os.environ.setdefault("LITELLM_MASTER_KEY", "sk-1234")

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


# ---------------------------------------------------------------------------
# In-memory Prisma stand-in
# ---------------------------------------------------------------------------


class _Row:
    """Plain object mimicking Prisma row attribute access.

    Missing attributes resolve to ``None`` to mirror Prisma's behavior
    of returning a row with optional columns left null.
    """

    def __init__(self, **fields: Any) -> None:
        for k, v in fields.items():
            setattr(self, k, v)

    def __getattr__(self, name: str) -> Any:
        # Only fires for attribute access that misses the instance dict;
        # underscore-prefixed names (e.g. dunders) should error normally.
        if name.startswith("_"):
            raise AttributeError(name)
        return None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _matches(row: _Row, where: Optional[Dict[str, Any]]) -> bool:
    if not where:
        return True
    for key, expected in where.items():
        actual = getattr(row, key, None)
        if isinstance(expected, dict):
            if "in" in expected:
                if actual not in expected["in"]:
                    return False
            elif "notIn" in expected:
                if actual in expected["notIn"]:
                    return False
            elif "lt" in expected:
                if actual is None or not (actual < expected["lt"]):
                    return False
            elif "gt" in expected:
                if actual is None or not (actual > expected["gt"]):
                    return False
            else:
                # Unknown operator dict — fall back to equality on raw dict.
                if actual != expected:
                    return False
        else:
            if actual != expected:
                return False
    return True


def _order_rows(rows: List[_Row], order: Any) -> List[_Row]:
    if not order:
        return rows
    if isinstance(order, dict):
        order = [order]
    for o in reversed(order):
        for k, direction in o.items():
            rows.sort(
                key=lambda r: getattr(r, k) or 0,
                reverse=(direction == "desc"),
            )
    return rows


class _Table:
    """In-memory table with the few async methods our endpoints use."""

    def __init__(self) -> None:
        self.rows: List[_Row] = []

    async def create(self, data: Dict[str, Any]) -> _Row:
        # Defaults that real Prisma would apply
        now = datetime.now(timezone.utc)
        defaults = {"created_at": now, "updated_at": now}
        merged = {**defaults, **data}
        row = _Row(**merged)
        self.rows.append(row)
        return row

    async def find_unique(self, where: Dict[str, Any]) -> Optional[_Row]:
        for row in self.rows:
            if _matches(row, where):
                return row
        return None

    async def find_first(
        self,
        where: Optional[Dict[str, Any]] = None,
        order: Any = None,
    ) -> Optional[_Row]:
        results = [r for r in self.rows if _matches(r, where)]
        results = _order_rows(results, order)
        return results[0] if results else None

    async def find_many(
        self,
        where: Optional[Dict[str, Any]] = None,
        order: Any = None,
        take: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> List[_Row]:
        results = [r for r in self.rows if _matches(r, where)]
        results = _order_rows(results, order)
        if skip:
            results = results[skip:]
        if take:
            results = results[:take]
        return results

    async def update(self, where: Dict[str, Any], data: Dict[str, Any]) -> _Row:
        for row in self.rows:
            if _matches(row, where):
                for k, v in data.items():
                    setattr(row, k, v)
                return row
        raise RuntimeError(f"No row to update for {where}")

    async def update_many(self, where: Dict[str, Any], data: Dict[str, Any]):
        count = 0
        for row in self.rows:
            if _matches(row, where):
                for k, v in data.items():
                    setattr(row, k, v)
                count += 1
        # Mimic Prisma's BatchPayload-ish object.
        return _Row(count=count)

    async def delete(self, where: Dict[str, Any]) -> _Row:
        for i, row in enumerate(self.rows):
            if _matches(row, where):
                return self.rows.pop(i)
        raise RuntimeError(f"No row to delete for {where}")


class FakePrismaClient:
    """Drop-in for ``prisma_client`` in ``litellm.proxy.proxy_server``."""

    def __init__(self) -> None:
        self.db = _DB()


class _DB:
    def __init__(self) -> None:
        self.litellm_agent = _Table()
        self.litellm_agentsession = _Table()
        self.litellm_agentrun = _AgentRunTable()
        self.litellm_agentrunevent = _AgentRunEventTable()
        # Warm-pool tables (LIT-2890). Same `_Table` semantics so the in-memory
        # client mirrors what the real Prisma client exposes.
        self.litellm_agentvm = _Table()
        self.litellm_agentvmconfig = _Table()
        self.litellm_agentsecret = _Table()


class _AgentRunTable(_Table):
    """Subclass that enforces the (session_id, idempotency_key) unique constraint."""

    async def create(self, data: Dict[str, Any]) -> _Row:
        sid = data.get("session_id")
        idem = data.get("idempotency_key")
        if idem is not None:
            for row in self.rows:
                if (
                    getattr(row, "session_id", None) == sid
                    and getattr(row, "idempotency_key", None) == idem
                ):
                    raise RuntimeError("idempotency_collision")
        return await super().create(data)


class _AgentRunEventTable(_Table):
    """Enforces the (run_id, seq) unique constraint."""

    async def create(self, data: Dict[str, Any]) -> _Row:
        rid = data.get("run_id")
        seq = data.get("seq")
        for row in self.rows:
            if getattr(row, "run_id", None) == rid and getattr(row, "seq", None) == seq:
                raise RuntimeError("event_seq_collision")
        return await super().create(data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_prisma_client(monkeypatch):
    """Patch ``litellm.proxy.proxy_server.prisma_client`` for the duration of
    the test with our in-memory stand-in.
    """
    from litellm.proxy import proxy_server

    fake = FakePrismaClient()
    monkeypatch.setattr(proxy_server, "prisma_client", fake)
    return fake


def _build_test_app(
    role: LitellmUserRoles, api_key: str = "sk-test-caller"
) -> TestClient:
    from litellm.proxy.agent_session_endpoints import (
        agent_router,
        internal_router,
        run_router,
        session_router,
    )
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    app = FastAPI()
    app.include_router(agent_router)
    app.include_router(session_router)
    app.include_router(run_router)
    app.include_router(internal_router)

    def _fake_auth() -> UserAPIKeyAuth:
        return UserAPIKeyAuth(
            user_id="test-user",
            user_role=role,
            api_key=api_key,
            team_id=None,
        )

    app.dependency_overrides[user_api_key_auth] = _fake_auth
    return TestClient(app)


@pytest.fixture
def client(fake_prisma_client):
    """TestClient where caller is a non-admin (regular tenant)."""
    return _build_test_app(LitellmUserRoles.INTERNAL_USER, api_key="sk-tenant-A")


@pytest.fixture
def admin_client(fake_prisma_client):
    """TestClient where caller is a proxy admin (sees everything)."""
    return _build_test_app(LitellmUserRoles.PROXY_ADMIN, api_key="sk-admin-key")


@pytest.fixture
def view_only_admin_client(fake_prisma_client):
    """TestClient where caller is a view-only proxy admin.

    View-only admins are allowed to READ across tenants (so the support
    UI can render any tenant's resources) but MUST NOT be allowed to
    mutate state on any tenant's resources — see ``assert_caller_can_mutate``.
    """
    return _build_test_app(
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, api_key="sk-view-only-admin"
    )


@pytest.fixture
def other_tenant_client(fake_prisma_client):
    """TestClient where caller is a different tenant. Used for
    cross-tenant isolation tests."""
    return _build_test_app(LitellmUserRoles.INTERNAL_USER, api_key="sk-tenant-B")


class _RecordingNoopProvider:
    """Test-only noop that tracks ``provision`` / ``terminate`` calls.

    A1's tests poke at ``provider.provision_calls`` / ``terminate_calls``
    while B1's ``NoopProvider`` only implements the ABC. Wrap B1's behavior
    here without polluting the production class.
    """

    name = "noop"

    def __init__(self) -> None:
        from litellm.proxy.agent_session_endpoints.vm_providers import NoopProvider

        self._inner = NoopProvider()
        self.provision_calls: list = []
        self.terminate_calls: list = []

    async def provision(self, ctx) -> Any:  # noqa: ANN001
        self.provision_calls.append(
            {
                "session_id": getattr(ctx, "session_id", None),
                "team_id": getattr(ctx, "team_id", None),
                "agent_id": getattr(ctx, "agent_id", None),
                "mode": getattr(ctx, "mode", None),
            }
        )
        return await self._inner.provision(ctx)

    async def terminate(self, vm) -> None:  # noqa: ANN001
        self.terminate_calls.append(
            {"session_id": vm.metadata.get("session_id"), "vm_id": vm.vm_id}
        )
        await self._inner.terminate(vm)

    async def status(self, vm):  # noqa: ANN001
        return await self._inner.status(vm)


@pytest.fixture
def noop_provider(monkeypatch):
    """Reset the VM provider registry to a fresh recording noop."""
    from litellm.proxy.agent_session_endpoints.vm_providers import (
        register_vm_provider,
    )
    from litellm.proxy.agent_session_endpoints.vm_providers.registry import (
        reset_vm_provider_registry,
    )

    reset_vm_provider_registry()
    provider = _RecordingNoopProvider()
    register_vm_provider(provider)
    yield provider
    reset_vm_provider_registry()
