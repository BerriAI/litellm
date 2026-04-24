"""
Unit tests for /v1/memory CRUD endpoints.

Uses FastAPI TestClient with an in-memory fake of the Prisma memory table.
Auth is overridden so we can simulate different callers (admin vs. scoped).
We patch the endpoint module's `_require_prisma` helper so we never need the
real proxy_server import chain (which pulls heavy optional deps).
"""

import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.memory.memory_endpoints import router


def _make_row(
    memory_id: str = "mem-1",
    key: str = "notes",
    value: str = "hello",
    user_id: Optional[str] = "user-a",
    team_id: Optional[str] = "team-a",
    metadata: Optional[Any] = None,
) -> MagicMock:
    """Build a Prisma-like row object."""
    now = datetime.now(timezone.utc)
    row = MagicMock()
    row.memory_id = memory_id
    row.key = key
    row.value = value
    row.metadata = metadata
    row.user_id = user_id
    row.team_id = team_id
    row.created_at = now
    row.created_by = user_id
    row.updated_at = now
    row.updated_by = user_id
    return row


class _InMemoryMemoryTable:
    """Tiny fake of prisma_client.db.litellm_memorytable used by the endpoints."""

    def __init__(self):
        self.rows: List[MagicMock] = []
        self._counter = 0

    def _matches(self, row: MagicMock, where: Dict[str, Any]) -> bool:
        for k, v in where.items():
            if k == "OR":
                if not any(self._matches(row, clause) for clause in v):
                    return False
                continue
            # Support Prisma-style filter dicts: {"startsWith": "..."}, etc.
            if isinstance(v, dict):
                actual = getattr(row, k, None)
                if "startsWith" in v:
                    if not isinstance(actual, str) or not actual.startswith(
                        v["startsWith"]
                    ):
                        return False
                    continue
                if "equals" in v:
                    if actual != v["equals"]:
                        return False
                    continue
                # Unknown filter — fall back to inequality (treat as no match).
                return False
            if getattr(row, k, None) != v:
                return False
        return True

    def _filter(self, where: Optional[Dict[str, Any]]) -> List[MagicMock]:
        if not where:
            return list(self.rows)
        return [r for r in self.rows if self._matches(r, where)]

    async def create(self, data: Dict[str, Any]) -> MagicMock:
        # Key is globally unique.
        for r in self.rows:
            if r.key == data["key"]:
                raise Exception("UniqueViolation: duplicate key")
        self._counter += 1
        row = _make_row(
            memory_id=f"mem-{self._counter}",
            key=data["key"],
            value=data["value"],
            user_id=data.get("user_id"),
            team_id=data.get("team_id"),
            metadata=data.get("metadata"),
        )
        row.created_by = data.get("created_by")
        row.updated_by = data.get("updated_by")
        self.rows.append(row)
        return row

    async def count(self, where: Optional[Dict[str, Any]] = None) -> int:
        return len(self._filter(where))

    async def find_many(
        self,
        where: Optional[Dict[str, Any]] = None,
        order: Optional[Dict[str, str]] = None,
        skip: int = 0,
        take: Optional[int] = None,
    ) -> List[MagicMock]:
        _ = order
        out = self._filter(where)
        if skip:
            out = out[skip:]
        if take is not None:
            out = out[:take]
        return out

    async def update(self, where: Dict[str, Any], data: Dict[str, Any]) -> MagicMock:
        for r in self.rows:
            if r.memory_id == where["memory_id"]:
                for k, v in data.items():
                    setattr(r, k, v)
                return r
        raise Exception("Not found")

    async def delete(self, where: Dict[str, Any]) -> MagicMock:
        for i, r in enumerate(self.rows):
            if r.memory_id == where["memory_id"]:
                return self.rows.pop(i)
        raise Exception("Not found")


def _make_prisma() -> MagicMock:
    client = MagicMock()
    client.db = MagicMock()
    client.db.litellm_memorytable = _InMemoryMemoryTable()
    return client


def _make_client(auth: UserAPIKeyAuth) -> TestClient:
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: auth
    return TestClient(app, raise_server_exceptions=True)


def _user_auth(user_id: str = "user-a", team_id: str = "team-a") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", user_id=user_id, team_id=team_id)


def _admin_auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-admin",
        user_id="admin",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )


def _patch_prisma(prisma: Any):
    """Patch the endpoint module's _require_prisma to return our fake."""
    return patch(
        "litellm.proxy.memory.memory_endpoints._require_prisma",
        return_value=prisma,
    )


class TestMemoryEndpoints:
    def setup_method(self):
        self.prisma = _make_prisma()

    def test_create_memory_defaults_scope_to_caller(self):
        client = _make_client(_user_auth("user-a", "team-a"))
        with _patch_prisma(self.prisma):
            resp = client.post("/v1/memory", json={"key": "notes", "value": "hello"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["key"] == "notes"
        assert body["value"] == "hello"
        assert body["user_id"] == "user-a"
        assert body["team_id"] == "team-a"

    def test_create_memory_duplicate_key_returns_409(self):
        client = _make_client(_user_auth("user-a", "team-a"))
        with _patch_prisma(self.prisma):
            r1 = client.post("/v1/memory", json={"key": "k", "value": "v1"})
            assert r1.status_code == 200
            r2 = client.post("/v1/memory", json={"key": "k", "value": "v2"})
        assert r2.status_code == 409

    def test_non_admin_cannot_set_foreign_user_id(self):
        client = _make_client(_user_auth("user-a", "team-a"))
        with _patch_prisma(self.prisma):
            resp = client.post(
                "/v1/memory",
                json={"key": "notes", "value": "x", "user_id": "user-b"},
            )
        assert resp.status_code == 403

    def test_create_memory_identity_less_caller_returns_400(self):
        """
        A non-admin caller with neither user_id nor team_id would produce an
        orphan row unreachable by the visibility filter. Reject up front.
        """
        client = _make_client(UserAPIKeyAuth(api_key="sk-anon"))
        with _patch_prisma(self.prisma):
            resp = client.post("/v1/memory", json={"key": "notes", "value": "x"})
        assert resp.status_code == 400

    def test_put_memory_admin_can_bootstrap_foreign_scope(self):
        """PUT-create should mirror POST's admin scope override."""
        client = _make_client(_admin_auth())
        with _patch_prisma(self.prisma):
            resp = client.put(
                "/v1/memory/notes",
                json={"value": "x", "user_id": "some-user", "team_id": "some-team"},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["user_id"] == "some-user"
        assert body["team_id"] == "some-team"

    def test_put_memory_race_returns_update_on_unique_violation(self):
        """
        Simulate the check-then-create race: _find_memory_for_caller says the
        row doesn't exist, then the create call gets a unique-constraint
        violation (a concurrent writer beat us). The handler should re-read
        and fall through to an update instead of surfacing a 500.
        """
        table = self.prisma.db.litellm_memorytable

        original_create = table.create
        original_find_many = table.find_many
        pre_create_calls = {"n": 0}

        async def racing_create(data):
            # On the very first create call we issue during the upsert, pretend
            # another writer inserted the row just before us.
            pre_create_calls["n"] += 1
            if pre_create_calls["n"] == 1:
                table.rows.append(
                    _make_row(
                        memory_id="m-race",
                        key=data["key"],
                        value="from-other-writer",
                        user_id="user-a",
                        team_id="team-a",
                    )
                )
                raise Exception("UniqueViolation: duplicate key (raced)")
            return await original_create(data)

        table.create = racing_create  # type: ignore[assignment]

        client = _make_client(_user_auth("user-a", "team-a"))
        with _patch_prisma(self.prisma):
            resp = client.put("/v1/memory/notes", json={"value": "mine"})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Update path kicked in: our value replaced the racer's.
        assert body["value"] == "mine"
        # Only one row exists (the one the racer inserted, now updated).
        assert len(table.rows) == 1

        # Restore the fake's methods.
        table.create = original_create  # type: ignore[assignment]
        table.find_many = original_find_many  # type: ignore[assignment]

    def test_admin_can_set_any_scope(self):
        client = _make_client(_admin_auth())
        with _patch_prisma(self.prisma):
            resp = client.post(
                "/v1/memory",
                json={
                    "key": "notes",
                    "value": "x",
                    "user_id": "some-user",
                    "team_id": "some-team",
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == "some-user"
        assert body["team_id"] == "some-team"

    def test_list_memory_scoped_to_caller(self):
        table = self.prisma.db.litellm_memorytable
        table.rows.extend(
            [
                _make_row(memory_id="m1", key="a", user_id="user-a", team_id=None),
                _make_row(memory_id="m2", key="b", user_id="user-b", team_id=None),
                _make_row(memory_id="m3", key="c", user_id=None, team_id="team-a"),
            ]
        )
        client = _make_client(_user_auth("user-a", "team-a"))
        with _patch_prisma(self.prisma):
            resp = client.get("/v1/memory")
        assert resp.status_code == 200
        body = resp.json()
        keys = {m["key"] for m in body["memories"]}
        assert keys == {"a", "c"}
        assert body["total"] == 2

    def test_list_memory_key_prefix_filter(self):
        """key_prefix should do a prefix match (Redis-style namespace scan)."""
        table = self.prisma.db.litellm_memorytable
        table.rows.extend(
            [
                _make_row(
                    memory_id="m1", key="user:profile", user_id="user-a", team_id=None
                ),
                _make_row(
                    memory_id="m2", key="user:prefs", user_id="user-a", team_id=None
                ),
                _make_row(
                    memory_id="m3",
                    key="project:context",
                    user_id="user-a",
                    team_id=None,
                ),
            ]
        )
        client = _make_client(_user_auth("user-a", "team-a"))
        with _patch_prisma(self.prisma):
            resp = client.get("/v1/memory?key_prefix=user:")
        assert resp.status_code == 200
        body = resp.json()
        keys = {m["key"] for m in body["memories"]}
        assert keys == {"user:profile", "user:prefs"}
        assert body["total"] == 2

    def test_list_memory_key_prefix_does_not_leak_across_scopes(self):
        """
        Even if a prefix would match another user's keys, the visibility
        filter must still scope results to the caller.
        """
        table = self.prisma.db.litellm_memorytable
        table.rows.extend(
            [
                # Caller's own "user:*" rows — should be visible.
                _make_row(
                    memory_id="m1", key="user:profile", user_id="user-a", team_id=None
                ),
                # Another user's "user:*" rows — must NOT leak.
                _make_row(
                    memory_id="m2", key="user:secret", user_id="user-b", team_id=None
                ),
                _make_row(
                    memory_id="m3", key="user:token", user_id="user-b", team_id=None
                ),
            ]
        )
        client = _make_client(_user_auth("user-a", "team-a"))
        with _patch_prisma(self.prisma):
            resp = client.get("/v1/memory?key_prefix=user:")
        assert resp.status_code == 200
        body = resp.json()
        keys = {m["key"] for m in body["memories"]}
        assert keys == {"user:profile"}
        assert body["total"] == 1

    def test_list_memory_admin_sees_all(self):
        table = self.prisma.db.litellm_memorytable
        table.rows.extend(
            [
                _make_row(memory_id="m1", key="a", user_id="user-a", team_id=None),
                _make_row(memory_id="m2", key="b", user_id="user-b", team_id=None),
            ]
        )
        client = _make_client(_admin_auth())
        with _patch_prisma(self.prisma):
            resp = client.get("/v1/memory")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_get_memory_by_key(self):
        table = self.prisma.db.litellm_memorytable
        table.rows.append(
            _make_row(memory_id="m1", key="notes", value="hi", user_id="user-a")
        )
        client = _make_client(_user_auth("user-a", "team-a"))
        with _patch_prisma(self.prisma):
            resp = client.get("/v1/memory/notes")
        assert resp.status_code == 200
        assert resp.json()["value"] == "hi"

    def test_get_memory_not_visible_returns_404(self):
        table = self.prisma.db.litellm_memorytable
        table.rows.append(
            _make_row(memory_id="m1", key="notes", user_id="user-b", team_id=None)
        )
        client = _make_client(_user_auth("user-a", "team-a"))
        with _patch_prisma(self.prisma):
            resp = client.get("/v1/memory/notes")
        assert resp.status_code == 404

    def test_put_memory_creates_when_missing(self):
        client = _make_client(_user_auth("user-a", "team-a"))
        with _patch_prisma(self.prisma):
            resp = client.put("/v1/memory/notes", json={"value": "new"})
        assert resp.status_code == 200
        assert resp.json()["value"] == "new"
        assert len(self.prisma.db.litellm_memorytable.rows) == 1

    def test_put_memory_updates_existing(self):
        table = self.prisma.db.litellm_memorytable
        table.rows.append(
            _make_row(
                memory_id="m1",
                key="notes",
                value="old",
                user_id="user-a",
                team_id="team-a",
            )
        )
        client = _make_client(_user_auth("user-a", "team-a"))
        with _patch_prisma(self.prisma):
            resp = client.put("/v1/memory/notes", json={"value": "new"})
        assert resp.status_code == 200
        assert resp.json()["value"] == "new"
        assert len(table.rows) == 1

    def test_put_memory_empty_body_returns_400(self):
        client = _make_client(_user_auth("user-a", "team-a"))
        with _patch_prisma(self.prisma):
            resp = client.put("/v1/memory/notes", json={})
        assert resp.status_code == 400

    def test_delete_memory(self):
        table = self.prisma.db.litellm_memorytable
        table.rows.append(
            _make_row(memory_id="m1", key="notes", user_id="user-a", team_id="team-a")
        )
        client = _make_client(_user_auth("user-a", "team-a"))
        with _patch_prisma(self.prisma):
            resp = client.delete("/v1/memory/notes")
        assert resp.status_code == 200
        assert resp.json() == {"key": "notes", "deleted": True}
        assert table.rows == []

    def test_delete_memory_not_visible_returns_404(self):
        table = self.prisma.db.litellm_memorytable
        table.rows.append(
            _make_row(memory_id="m1", key="notes", user_id="user-b", team_id=None)
        )
        client = _make_client(_user_auth("user-a", "team-a"))
        with _patch_prisma(self.prisma):
            resp = client.delete("/v1/memory/notes")
        assert resp.status_code == 404
