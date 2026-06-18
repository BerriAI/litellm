"""Unit tests for ``POST /v2/agents`` (LIT-2922) and ``GET /v2/agents``.

Auth pattern: FastAPI ``app.dependency_overrides`` for ``user_api_key_auth``.

The router is included onto the proxy ``app`` inside a fixture so this test
file does not require Wave 3's wiring in ``proxy_server.py`` to be present
yet. The fixture is idempotent — re-including a router on the same app is
a no-op for FastAPI's matcher (it just registers another mount); we de-dup
by checking ``app.router.routes`` first.
"""

import os
import sys
import types
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import litellm.proxy.proxy_server as ps
from litellm.proxy.managed_agents_endpoints.agents import router as agents_router
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.proxy_server import app

sys.path.insert(0, os.path.abspath("../../../"))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _ensure_router_mounted() -> None:
    """Mount the agents router onto ``app`` exactly once.

    Wave 3 owns the real registration in ``proxy_server.py``. For tests we
    mount it here so we can hit ``POST /v2/agents`` via ``TestClient``.
    """
    paths = {getattr(r, "path", None) for r in app.router.routes}
    if "/v2/agents" not in paths:
        app.include_router(agents_router)


class _FakeAgentTable:
    """In-memory stand-in for ``prisma_client.db.litellm_managedagent``.

    Only the methods called by Wave 1's ``db.py`` helpers are implemented:
    ``create`` and ``find_first``. Both are ``AsyncMock`` so call assertions
    work as expected.
    """

    def __init__(self) -> None:
        self.rows: Dict[str, Dict[str, Any]] = {}
        self.create = AsyncMock(side_effect=self._create)
        self.find_first = AsyncMock(side_effect=self._find_first)
        self.find_many = AsyncMock(side_effect=self._find_many)

    async def _create(self, *, data: Dict[str, Any]) -> types.SimpleNamespace:
        # Mirror Prisma's behavior: store + return a row-like object.
        # ``prisma.Json(...)`` wraps the config dict — unwrap it for storage
        # so the test assertions can inspect raw dicts.
        config = data.get("config")
        if config is not None and hasattr(config, "data"):
            stored_config: Any = config.data
        elif isinstance(config, dict):
            stored_config = config
        else:
            # ``prisma.Json`` instances expose the original dict through
            # ``.data`` in real Prisma, but the stub used in CI may pass
            # through. Fall back to the raw value.
            stored_config = config

        row = {**data, "config": stored_config}
        self.rows[data["id"]] = row
        # Return a model_dump-able stand-in.
        return types.SimpleNamespace(model_dump=lambda r=row: dict(r))

    async def _find_first(
        self, *, where: Dict[str, Any]
    ) -> Optional[types.SimpleNamespace]:
        for row in self.rows.values():
            if all(row.get(k) == v for k, v in where.items()):
                return types.SimpleNamespace(model_dump=lambda r=row: dict(r))
        return None

    async def _find_many(
        self,
        *,
        where: Dict[str, Any],
        take: int,
        skip: int,
        order: Dict[str, str],
    ) -> List[types.SimpleNamespace]:
        # Filter on `where` (typically just created_by here).
        matched = [
            row
            for row in self.rows.values()
            if all(row.get(k) == v for k, v in where.items())
        ]

        # Order by created_at; default desc to match the handler.
        order_key, order_dir = next(iter(order.items()))
        matched.sort(
            key=lambda r: r.get(order_key) or datetime.min,
            reverse=(order_dir == "desc"),
        )

        page = matched[skip : skip + take]
        return [types.SimpleNamespace(model_dump=lambda r=row: dict(r)) for row in page]


@pytest.fixture
def client_and_mocks(monkeypatch):
    _ensure_router_mounted()

    # 1. Stub prisma client at the module ``ps`` (proxy_server) sees.
    fake_table = _FakeAgentTable()
    fake_db = types.SimpleNamespace(litellm_managedagent=fake_table)
    fake_prisma = MagicMock()
    fake_prisma.db = fake_db
    monkeypatch.setattr(ps, "prisma_client", fake_prisma)

    # 2. Stub ``prisma.Json`` so ``db.insert_agent`` can wrap the config.
    fake_prisma_module = types.ModuleType("prisma")

    class _Json:
        def __init__(self, data: Any) -> None:
            self.data = data

    fake_prisma_module.Json = _Json  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "prisma", fake_prisma_module)

    # 3. ``encrypt_value_helper`` uses ``master_key`` (or ``LITELLM_SALT_KEY``)
    # as the signing key for the secretbox. Tests run without either set,
    # so set a deterministic salt key here so encryption/decryption round-trips.
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-key-for-managed-agents")

    # 4. Override the auth dependency with a fixed user.
    fake_user = UserAPIKeyAuth(
        user_id="user_xyz",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: fake_user

    client = TestClient(app)

    yield client, fake_table, fake_user

    app.dependency_overrides.clear()


def _valid_payload(name: str = "code-reviewer") -> Dict[str, Any]:
    return {
        "name": name,
        "config": {
            "model": "anthropic/claude-opus-4",
            "system_prompt": "You are a senior engineer reviewing code.",
            "tools": ["read", "grep", "bash"],
            "litellm_api_key": "sk-supersecret123",
            "litellm_base_url": "http://localhost:4000",
        },
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_create_agent_success(client_and_mocks):
    client, fake_table, fake_user = client_and_mocks

    payload = _valid_payload()
    resp = client.post("/v2/agents", json=payload)

    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["name"] == payload["name"]
    assert body["created_by"] == fake_user.user_id
    assert body["id"].startswith("agt_")
    # Spec §6.1: id is "agt_" + 32-char uuid hex (no dashes) → length 36.
    assert len(body["id"]) == 36

    # Config: passthrough fields preserved, api_key masked.
    cfg = body["config"]
    assert cfg["model"] == payload["config"]["model"]
    assert cfg["system_prompt"] == payload["config"]["system_prompt"]
    assert cfg["tools"] == payload["config"]["tools"]
    assert cfg["litellm_base_url"] == payload["config"]["litellm_base_url"]
    # Mask: first 4 chars + "****" → "sk-s****"
    assert cfg["litellm_api_key"] == "sk-s****"
    assert payload["config"]["litellm_api_key"] not in resp.text

    # Timestamps are present and ISO-formatted.
    assert "created_at" in body and "updated_at" in body
    datetime.fromisoformat(body["created_at"].replace("Z", "+00:00"))
    datetime.fromisoformat(body["updated_at"].replace("Z", "+00:00"))

    # DB was called exactly once for create.
    fake_table.create.assert_awaited_once()
    create_kwargs = fake_table.create.await_args.kwargs
    inserted = create_kwargs["data"]
    # The persisted ``litellm_api_key`` MUST be encrypted — never the
    # caller's plaintext. Decrypting it must recover the original.
    from litellm.proxy.common_utils.encrypt_decrypt_utils import (
        decrypt_value_helper,
    )

    persisted_key = inserted["config"].data["litellm_api_key"]
    assert persisted_key != payload["config"]["litellm_api_key"]
    assert (
        decrypt_value_helper(value=persisted_key, key="litellm_api_key")
        == payload["config"]["litellm_api_key"]
    )
    assert inserted["created_by"] == fake_user.user_id
    assert inserted["name"] == payload["name"]


# ---------------------------------------------------------------------------
# Duplicate names
# ---------------------------------------------------------------------------


def test_create_agent_duplicate_name_same_user_returns_409(client_and_mocks):
    client, fake_table, _ = client_and_mocks

    payload = _valid_payload(name="code-reviewer")
    first = client.post("/v2/agents", json=payload)
    assert first.status_code == 200, first.text

    second = client.post("/v2/agents", json=payload)
    assert second.status_code == 409, second.text
    detail = second.json()["detail"]
    assert "code-reviewer" in detail
    assert "already exists" in detail.lower()

    # Only the first call should have hit ``create``.
    assert fake_table.create.await_count == 1


def test_create_agent_same_name_different_user_returns_200(client_and_mocks):
    client, fake_table, _ = client_and_mocks

    # First user creates "code-reviewer".
    first_payload = _valid_payload(name="code-reviewer")
    first = client.post("/v2/agents", json=first_payload)
    assert first.status_code == 200, first.text

    # Swap auth to a different user; same name must succeed.
    other_user = UserAPIKeyAuth(
        user_id="user_other",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: other_user

    second = client.post("/v2/agents", json=first_payload)
    assert second.status_code == 200, second.text
    assert second.json()["created_by"] == "user_other"

    # Two distinct rows should now be present.
    assert fake_table.create.await_count == 2
    created_bys = {row["created_by"] for row in fake_table.rows.values()}
    assert created_bys == {"user_xyz", "user_other"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_create_agent_missing_name_returns_422(client_and_mocks):
    client, fake_table, _ = client_and_mocks

    payload = _valid_payload()
    payload.pop("name")
    resp = client.post("/v2/agents", json=payload)

    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, list)
    assert any("name" in (err.get("loc") or []) for err in detail)
    fake_table.create.assert_not_awaited()


def test_create_agent_missing_config_model_returns_422(client_and_mocks):
    client, fake_table, _ = client_and_mocks

    payload = _valid_payload()
    payload["config"].pop("model")
    resp = client.post("/v2/agents", json=payload)

    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, list)
    assert any(
        "model" in (err.get("loc") or []) and "config" in (err.get("loc") or [])
        for err in detail
    )
    fake_table.create.assert_not_awaited()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_create_agent_no_auth_returns_401(client_and_mocks, monkeypatch):
    """Without an ``Authorization`` header and a master key set, real auth rejects.

    The proxy's ``user_api_key_auth`` is permissive when ``master_key is None``
    — it returns a default ``UserAPIKeyAuth`` so local dev doesn't need a key.
    To actually exercise the rejection path we set a master key, drop the
    dependency override, and send no header.
    """
    client, fake_table, _ = client_and_mocks

    # Set a master key so unauth'd requests get rejected by the real chain.
    monkeypatch.setattr(ps, "master_key", "sk-test-master")

    # Drop the override so the real user_api_key_auth runs on this request.
    app.dependency_overrides.pop(ps.user_api_key_auth, None)

    resp = client.post("/v2/agents", json=_valid_payload(name="no-auth"))

    # The real auth chain rejects unauthenticated calls. Proxy maps the raised
    # exception to ProxyException → 401 in most versions, but some 4xx is fine
    # — what matters is that it's an auth failure, NOT a successful 2xx, and
    # the DB was never touched.
    assert resp.status_code in (400, 401, 403), resp.text
    fake_table.create.assert_not_awaited()


# ---------------------------------------------------------------------------
# DB-not-connected guard
# ---------------------------------------------------------------------------


def test_create_agent_db_not_connected_returns_500(client_and_mocks, monkeypatch):
    client, _, _ = client_and_mocks

    monkeypatch.setattr(ps, "prisma_client", None)

    resp = client.post("/v2/agents", json=_valid_payload(name="no-db"))
    assert resp.status_code == 500, resp.text
    detail = resp.json()["detail"]
    assert "DB not connected" in detail["error"]


# ---------------------------------------------------------------------------
# GET /v2/agents — list
# ---------------------------------------------------------------------------


def _seed_agents(
    fake_table: "_FakeAgentTable",
    *,
    count: int,
    created_by: str,
    name_prefix: str = "agent",
    base_time: Optional[datetime] = None,
) -> None:
    """Seed `count` agents into the fake table for the given `created_by`.

    Each row is given a strictly increasing `created_at` so the desc-order
    list returns the highest index first.
    """
    if base_time is None:
        base_time = datetime(2026, 5, 1, tzinfo=timezone.utc)

    for i in range(count):
        agent_id = f"agt_{name_prefix}_{i:08d}"
        fake_table.rows[agent_id] = {
            "id": agent_id,
            "name": f"{name_prefix}-{i}",
            "config": {
                "model": "anthropic/claude-opus-4",
                "system_prompt": "test",
                "tools": ["read"],
                "litellm_api_key": f"sk-secret-{i}",
                "litellm_base_url": "http://localhost:4000",
            },
            "created_by": created_by,
            "created_at": base_time + timedelta(seconds=i),
            "updated_at": base_time + timedelta(seconds=i),
        }


def test_list_agents_returns_only_caller_rows(client_and_mocks):
    client, fake_table, fake_user = client_and_mocks

    _seed_agents(fake_table, count=2, created_by=fake_user.user_id, name_prefix="mine")
    _seed_agents(fake_table, count=3, created_by="user_other", name_prefix="theirs")

    resp = client.get("/v2/agents")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Only the caller's two rows should be returned.
    assert len(body["data"]) == 2
    for row in body["data"]:
        assert row["created_by"] == fake_user.user_id

    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_list_agents_masks_api_key(client_and_mocks):
    client, fake_table, fake_user = client_and_mocks

    _seed_agents(fake_table, count=1, created_by=fake_user.user_id)

    resp = client.get("/v2/agents")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert len(body["data"]) == 1
    cfg = body["data"][0]["config"]
    # Mask is "first 4 chars + ****"  → "sk-s****"
    assert cfg["litellm_api_key"] == "sk-s****"
    # Raw secret must never appear in response body.
    assert "sk-secret-0" not in resp.text


def test_list_agents_pagination_cursor_walks_pages(client_and_mocks):
    """Seed 4 records, page through with limit=2, walk via cursor."""
    client, fake_table, fake_user = client_and_mocks

    _seed_agents(fake_table, count=4, created_by=fake_user.user_id)

    # Page 1.
    resp1 = client.get("/v2/agents?limit=2")
    assert resp1.status_code == 200, resp1.text
    page1 = resp1.json()
    assert len(page1["data"]) == 2
    assert page1["has_more"] is True
    assert page1["next_cursor"] == "2"

    # Desc order: page 1 should contain the two newest (indices 3 and 2).
    page1_ids = [r["id"] for r in page1["data"]]

    # Page 2.
    resp2 = client.get(f"/v2/agents?limit=2&cursor={page1['next_cursor']}")
    assert resp2.status_code == 200, resp2.text
    page2 = resp2.json()
    assert len(page2["data"]) == 2
    # We've now seen all 4 records → no more pages.
    assert page2["has_more"] is False
    assert page2["next_cursor"] is None

    page2_ids = [r["id"] for r in page2["data"]]

    # Pages must not overlap and must collectively cover all 4 seeded rows.
    all_seen = set(page1_ids) | set(page2_ids)
    assert len(all_seen) == 4, "All 4 records should appear across the 2 pages"
    assert set(page1_ids).isdisjoint(set(page2_ids))


def test_list_agents_empty_returns_empty_data(client_and_mocks):
    client, _, _ = client_and_mocks

    resp = client.get("/v2/agents")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["data"] == []
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_list_agents_invalid_cursor_returns_422(client_and_mocks):
    client, _, _ = client_and_mocks

    resp = client.get("/v2/agents?cursor=not-a-number")
    assert resp.status_code == 422, resp.text


def test_list_agents_db_not_connected_returns_500(client_and_mocks, monkeypatch):
    client, _, _ = client_and_mocks
    monkeypatch.setattr(ps, "prisma_client", None)

    resp = client.get("/v2/agents")
    assert resp.status_code == 500, resp.text
    detail = resp.json()["detail"]
    assert "DB not connected" in detail["error"]
