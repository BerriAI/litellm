"""
Endpoint tests for /v1/tasks. FastAPI TestClient + in-memory fake of the
prisma scheduled-tasks table. The /due test path uses a fake of query_raw +
tx() that exercises the same Python-side schedule advance logic.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy._types import UserAPIKeyAuth, hash_token
from litellm.proxy.scheduled_tasks.endpoints import router

# UserAPIKeyAuth's model_validator rewrites api_key + token to the hashed
# form when api_key starts with "sk-". The endpoint reads `.token` for the
# FK target, so we derive the same hash here for assertions/seeding.
_TEST_API_KEY = "sk-test"
_TEST_OWNER_TOKEN = hash_token(_TEST_API_KEY)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_row(**kwargs) -> MagicMock:
    """Build a Prisma-row-like object."""
    now = _now()
    defaults: Dict[str, Any] = {
        "task_id": "task-1",
        "owner_token": _TEST_OWNER_TOKEN,
        "user_id": "user-a",
        "team_id": "team-a",
        "agent_id": "agent-a",
        "title": "default",
        "action": "check",
        "action_args": None,
        "check_prompt": "is it done?",
        "format_prompt": None,
        "metadata": None,
        "schedule_kind": "interval",
        "schedule_spec": "5m",
        "schedule_tz": None,
        "next_run_at": now + timedelta(minutes=5),
        "expires_at": now + timedelta(days=1),
        "fire_once": True,
        "status": "pending",
        "last_fired_at": None,
        "consecutive_errors": 0,
        "last_error": None,
        "created_at": now,
        "updated_at": now,
    }
    defaults.update(kwargs)
    row = MagicMock()
    for k, v in defaults.items():
        setattr(row, k, v)
    return row


class _FakeScheduledTaskTable:
    """In-memory fake of prisma_client.db.litellm_scheduledtasktable."""

    def __init__(self):
        self.rows: List[MagicMock] = []
        self._counter = 0

    def _matches(self, row: MagicMock, where: Dict[str, Any]) -> bool:
        for k, v in where.items():
            if getattr(row, k, None) != v:
                return False
        return True

    def _filter(self, where: Optional[Dict[str, Any]]) -> List[MagicMock]:
        if not where:
            return list(self.rows)
        return [r for r in self.rows if self._matches(r, where)]

    async def create(self, data: Dict[str, Any]) -> MagicMock:
        # Mirror real Prisma write semantics:
        #  - relation field `owner_key: {connect: {token: X}}` becomes the
        #    scalar `owner_token` column on the row;
        #  - Json? columns (`action_args`, `metadata`) come in as JSON
        #    strings and are deserialised back to Python on read.
        normalized = dict(data)
        if "owner_key" in normalized:
            connect = normalized.pop("owner_key", {}).get("connect", {})
            if "token" in connect:
                normalized["owner_token"] = connect["token"]
        for json_field in ("action_args", "metadata"):
            if isinstance(normalized.get(json_field), str):
                try:
                    normalized[json_field] = json.loads(normalized[json_field])
                except ValueError:
                    pass
        self._counter += 1
        row = _make_row(task_id=f"task-{self._counter}", **normalized)
        self.rows.append(row)
        return row

    async def count(self, where: Optional[Dict[str, Any]] = None) -> int:
        return len(self._filter(where))

    async def find_first(
        self, where: Optional[Dict[str, Any]] = None
    ) -> Optional[MagicMock]:
        rows = self._filter(where)
        return rows[0] if rows else None

    async def find_many(
        self,
        where: Optional[Dict[str, Any]] = None,
        order: Optional[Dict[str, str]] = None,
    ) -> List[MagicMock]:
        _ = order
        return self._filter(where)

    async def update_many(self, where: Dict[str, Any], data: Dict[str, Any]) -> int:
        # Minimal Prisma-style filter: support {"lte": <dt>} for expires_at.
        count = 0
        for r in self.rows:
            ok = True
            for k, v in where.items():
                if isinstance(v, dict):
                    actual = getattr(r, k, None)
                    if "lte" in v:
                        if actual is None or actual > v["lte"]:
                            ok = False
                            break
                else:
                    if getattr(r, k, None) != v:
                        ok = False
                        break
            if not ok:
                continue
            for k, v in data.items():
                setattr(r, k, v)
            count += 1
        return count

    async def update(self, where: Dict[str, Any], data: Dict[str, Any]) -> MagicMock:
        for r in self.rows:
            if r.task_id == where["task_id"]:
                for k, v in data.items():
                    if k in ("action_args", "metadata") and isinstance(v, str):
                        try:
                            v = json.loads(v)
                        except ValueError:
                            pass
                    setattr(r, k, v)
                return r
        raise Exception("Not found")


class _FakeBatcher:
    """Capture batched updates and flush them on context exit."""

    def __init__(self, table: _FakeScheduledTaskTable):
        self._table = table
        self._ops: List = []

    @property
    def litellm_scheduledtasktable(self):
        outer = self

        class _Proxy:
            def update(self, where, data):
                outer._ops.append((where, data))

        return _Proxy()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        for where, data in self._ops:
            await self._table.update(where=where, data=data)


class _FakeTx:
    """Tx context — exposes query_raw + batch_ + the table."""

    def __init__(self, table: _FakeScheduledTaskTable):
        self._table = table
        self.litellm_scheduledtasktable = table

    async def query_raw(self, query: str, *args) -> List[Dict[str, Any]]:
        # Model the production claim query: rows where status='pending',
        # next_run_at <= now, optional agent_id and actions filters, ordered
        # by next_run_at, limited.
        agent_id, actions, limit = args
        now = _now()
        out: List[Dict[str, Any]] = []
        for r in sorted(self._table.rows, key=lambda x: x.next_run_at):
            if r.status != "pending":
                continue
            if r.next_run_at > now:
                continue
            if agent_id is not None and r.agent_id != agent_id:
                continue
            if actions is not None and r.action not in actions:
                continue
            out.append(
                {
                    "task_id": r.task_id,
                    "schedule_kind": r.schedule_kind,
                    "schedule_spec": r.schedule_spec,
                    "schedule_tz": r.schedule_tz,
                    "fire_once": r.fire_once,
                    "expires_at": r.expires_at,
                    "next_run_at": r.next_run_at,
                    "action": r.action,
                    "action_args": r.action_args,
                    "check_prompt": r.check_prompt,
                    "format_prompt": r.format_prompt,
                    "metadata": r.metadata,
                    "title": r.title,
                }
            )
            if len(out) >= limit:
                break
        return out

    def batch_(self) -> _FakeBatcher:
        return _FakeBatcher(self._table)


class _FakeTxFactory:
    def __init__(self, table: _FakeScheduledTaskTable):
        self._table = table

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return _FakeTx(self._table)

    async def __aexit__(self, *exc):
        return None


def _make_prisma() -> MagicMock:
    client = MagicMock()
    client.db = MagicMock()
    table = _FakeScheduledTaskTable()
    client.db.litellm_scheduledtasktable = table
    client.db.tx = _FakeTxFactory(table)
    return client


def _make_app(prisma: Any) -> TestClient:
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    def _auth_override():
        return UserAPIKeyAuth(
            api_key=_TEST_API_KEY,
            user_id="user-a",
            team_id="team-a",
            agent_id="agent-a",
        )

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = _auth_override
    return TestClient(app, raise_server_exceptions=True)


def _patch_prisma(prisma):
    return patch(
        "litellm.proxy.scheduled_tasks.endpoints._get_prisma_client",
        return_value=prisma,
    )


def _create_payload(**overrides) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "title": "watch PR 123",
        "action": "check",
        "check_prompt": "is PR 123 merged?",
        "schedule_kind": "interval",
        "schedule_spec": "5m",
        "schedule_tz": None,
        "expires_at": (_now() + timedelta(days=1)).isoformat(),
        "fire_once": True,
    }
    payload.update(overrides)
    return payload


class TestCreate:
    def setup_method(self):
        self.prisma = _make_prisma()
        self.client = _make_app(self.prisma)

    def test_happy_path_stamps_identity_from_auth(self):
        with _patch_prisma(self.prisma):
            r = self.client.post("/v1/tasks", json=_create_payload())
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["user_id"] == "user-a"
        assert body["team_id"] == "team-a"
        assert body["agent_id"] == "agent-a"
        # owner_token must match the auth-resolved hashed token (FK to
        # LiteLLM_VerificationToken.token), not the raw "sk-..." form.
        assert body["owner_token"]
        assert body["owner_token"] != "sk-test"
        assert body["status"] == "pending"

    def test_check_action_requires_check_prompt(self):
        payload = _create_payload(action="check", check_prompt=None)
        with _patch_prisma(self.prisma):
            r = self.client.post("/v1/tasks", json=payload)
        assert r.status_code == 400
        assert "check_prompt" in r.json()["detail"]

    def test_action_other_than_check_does_not_require_check_prompt(self):
        payload = _create_payload(
            action="pr_digest",
            check_prompt=None,
            action_args={"channel_id": "C456"},
            format_prompt="bullet list",
            schedule_kind="cron",
            schedule_spec="0 9 * * 1-5",
            schedule_tz="America/Los_Angeles",
            fire_once=False,
        )
        with _patch_prisma(self.prisma):
            r = self.client.post("/v1/tasks", json=payload)
        assert r.status_code == 200, r.text

    def test_invalid_cron_rejected(self):
        payload = _create_payload(
            schedule_kind="cron",
            schedule_spec="not a cron",
        )
        with _patch_prisma(self.prisma):
            r = self.client.post("/v1/tasks", json=payload)
        assert r.status_code == 400
        assert "cron" in r.json()["detail"].lower()

    def test_invalid_tz_rejected(self):
        payload = _create_payload(
            action="pr_digest",
            check_prompt=None,
            schedule_kind="cron",
            schedule_spec="0 9 * * *",
            schedule_tz="Europe/Atlantis",
        )
        with _patch_prisma(self.prisma):
            r = self.client.post("/v1/tasks", json=payload)
        assert r.status_code == 400

    def test_eleventh_task_rejected(self):
        with _patch_prisma(self.prisma):
            for i in range(10):
                r = self.client.post(
                    "/v1/tasks",
                    json=_create_payload(title=f"t{i}"),
                )
                assert r.status_code == 200, r.text
            r = self.client.post("/v1/tasks", json=_create_payload(title="overflow"))
        assert r.status_code == 429

    def test_metadata_roundtrips(self):
        payload = _create_payload(metadata={"agent_session": "abc", "tags": [1, 2, 3]})
        with _patch_prisma(self.prisma):
            r = self.client.post("/v1/tasks", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["metadata"] == {"agent_session": "abc", "tags": [1, 2, 3]}

    def test_metadata_omitted_when_not_supplied(self):
        with _patch_prisma(self.prisma):
            r = self.client.post("/v1/tasks", json=_create_payload())
        assert r.status_code == 200, r.text
        assert r.json()["metadata"] is None

    def test_expires_in_past_rejected(self):
        payload = _create_payload(
            expires_at=(_now() - timedelta(seconds=1)).isoformat(),
        )
        with _patch_prisma(self.prisma):
            r = self.client.post("/v1/tasks", json=payload)
        assert r.status_code == 400


class TestListAndGet:
    def setup_method(self):
        self.prisma = _make_prisma()
        self.client = _make_app(self.prisma)

    def test_list_excludes_terminal_by_default(self):
        with _patch_prisma(self.prisma):
            r1 = self.client.post("/v1/tasks", json=_create_payload(title="active"))
            assert r1.status_code == 200
            # Manually flip a row to 'fired' to simulate post-fire state
            self.prisma.db.litellm_scheduledtasktable.rows[0].status = "pending"
            self.prisma.db.litellm_scheduledtasktable.rows[0] = _make_row(
                task_id="task-1", title="active", status="pending"
            )
            self.prisma.db.litellm_scheduledtasktable.rows.append(
                _make_row(task_id="task-2", title="done", status="fired")
            )
            r = self.client.get("/v1/tasks")
            assert r.status_code == 200
            titles = [t["title"] for t in r.json()["tasks"]]
            assert "done" not in titles
            assert "active" in titles

    def test_list_with_terminal_includes_all(self):
        with _patch_prisma(self.prisma):
            self.prisma.db.litellm_scheduledtasktable.rows.append(
                _make_row(task_id="task-2", title="done", status="fired")
            )
            self.prisma.db.litellm_scheduledtasktable.rows.append(
                _make_row(task_id="task-3", title="active", status="pending")
            )
            r = self.client.get("/v1/tasks?include_terminal=true")
            assert r.status_code == 200
            titles = [t["title"] for t in r.json()["tasks"]]
            assert "done" in titles
            assert "active" in titles

    def test_get_owned(self):
        with _patch_prisma(self.prisma):
            self.client.post("/v1/tasks", json=_create_payload(title="mine"))
            tid = self.prisma.db.litellm_scheduledtasktable.rows[0].task_id
            r = self.client.get(f"/v1/tasks/{tid}")
            assert r.status_code == 200
            assert r.json()["title"] == "mine"

    def test_get_foreign_404(self):
        # Insert a row owned by another token.
        self.prisma.db.litellm_scheduledtasktable.rows.append(
            _make_row(task_id="foreign-1", owner_token="sk-other")
        )
        with _patch_prisma(self.prisma):
            r = self.client.get("/v1/tasks/foreign-1")
        assert r.status_code == 404


class TestUpdate:
    def setup_method(self):
        self.prisma = _make_prisma()
        self.client = _make_app(self.prisma)

    def test_update_pending(self):
        with _patch_prisma(self.prisma):
            self.client.post("/v1/tasks", json=_create_payload(title="orig"))
            tid = self.prisma.db.litellm_scheduledtasktable.rows[0].task_id
            r = self.client.patch(f"/v1/tasks/{tid}", json={"title": "renamed"})
            assert r.status_code == 200
            assert r.json()["title"] == "renamed"

    def test_update_invalid_field_rejected(self):
        # Pydantic strips unknown fields during model construction. The
        # whitelist guard inside store still bounces explicit attempts to
        # touch status/owner_token if they ever leak through.
        with _patch_prisma(self.prisma):
            self.client.post("/v1/tasks", json=_create_payload())
            tid = self.prisma.db.litellm_scheduledtasktable.rows[0].task_id
            # Empty body — no updatable fields → 400
            r = self.client.patch(f"/v1/tasks/{tid}", json={})
        assert r.status_code == 400

    def test_update_terminal_rejected(self):
        self.prisma.db.litellm_scheduledtasktable.rows.append(
            _make_row(task_id="t-fired", status="fired")
        )
        with _patch_prisma(self.prisma):
            r = self.client.patch("/v1/tasks/t-fired", json={"title": "nope"})
        assert r.status_code == 400

    def test_update_foreign_404(self):
        self.prisma.db.litellm_scheduledtasktable.rows.append(
            _make_row(task_id="foreign-1", owner_token="sk-other")
        )
        with _patch_prisma(self.prisma):
            r = self.client.patch("/v1/tasks/foreign-1", json={"title": "hijack"})
        assert r.status_code == 404

    def test_update_invalid_schedule_rejected(self):
        with _patch_prisma(self.prisma):
            self.client.post("/v1/tasks", json=_create_payload())
            tid = self.prisma.db.litellm_scheduledtasktable.rows[0].task_id
            r = self.client.patch(
                f"/v1/tasks/{tid}",
                json={"schedule_kind": "interval", "schedule_spec": "banana"},
            )
        assert r.status_code == 400


class TestCancel:
    def setup_method(self):
        self.prisma = _make_prisma()
        self.client = _make_app(self.prisma)

    def test_cancel_owned(self):
        with _patch_prisma(self.prisma):
            self.client.post("/v1/tasks", json=_create_payload())
            tid = self.prisma.db.litellm_scheduledtasktable.rows[0].task_id
            r = self.client.delete(f"/v1/tasks/{tid}")
            assert r.status_code == 200
            assert r.json()["status"] == "cancelled"

    def test_cancel_foreign_404(self):
        self.prisma.db.litellm_scheduledtasktable.rows.append(
            _make_row(task_id="foreign-1", owner_token="sk-other")
        )
        with _patch_prisma(self.prisma):
            r = self.client.delete("/v1/tasks/foreign-1")
        assert r.status_code == 404


class TestDue:
    def setup_method(self):
        self.prisma = _make_prisma()
        self.client = _make_app(self.prisma)

    def _seed_due_row(self, **kwargs):
        defaults: Dict[str, Any] = {
            "task_id": f"due-{len(self.prisma.db.litellm_scheduledtasktable.rows) + 1}",
            "next_run_at": _now() - timedelta(seconds=10),
            "status": "pending",
            "agent_id": "agent-a",
        }
        defaults.update(kwargs)
        self.prisma.db.litellm_scheduledtasktable.rows.append(_make_row(**defaults))

    def test_due_claims_pending_rows(self):
        self._seed_due_row(action="check")
        self._seed_due_row(action="pr_digest", fire_once=False, schedule_spec="5m")
        with _patch_prisma(self.prisma):
            r = self.client.get("/v1/tasks/due")
        assert r.status_code == 200
        body = r.json()
        assert len(body["tasks"]) == 2

    def test_due_advances_recurring_and_fires_once(self):
        self._seed_due_row(action="check", fire_once=True)
        self._seed_due_row(action="pr_digest", fire_once=False, schedule_spec="5m")
        with _patch_prisma(self.prisma):
            self.client.get("/v1/tasks/due")
        rows = self.prisma.db.litellm_scheduledtasktable.rows
        once_row = next(r for r in rows if r.fire_once)
        recurring_row = next(r for r in rows if not r.fire_once)
        assert once_row.status == "fired"
        assert recurring_row.status == "pending"
        assert recurring_row.next_run_at > _now()

    def test_due_second_call_empty(self):
        self._seed_due_row(action="check", fire_once=True)
        with _patch_prisma(self.prisma):
            r1 = self.client.get("/v1/tasks/due")
            assert len(r1.json()["tasks"]) == 1
            r2 = self.client.get("/v1/tasks/due")
            assert len(r2.json()["tasks"]) == 0

    def test_due_skips_other_agents(self):
        self._seed_due_row(action="check", agent_id="agent-other")
        self._seed_due_row(action="check", agent_id="agent-a")
        with _patch_prisma(self.prisma):
            r = self.client.get("/v1/tasks/due")
        assert len(r.json()["tasks"]) == 1
        assert r.json()["tasks"][0]["task_id"] != "due-1" or (
            self.prisma.db.litellm_scheduledtasktable.rows[0].agent_id == "agent-a"
        )

    def test_due_actions_filter(self):
        self._seed_due_row(action="check")
        self._seed_due_row(action="pr_digest")
        self._seed_due_row(action="other_thing")
        with _patch_prisma(self.prisma):
            r = self.client.get("/v1/tasks/due?actions=check,pr_digest")
        assert r.status_code == 200
        actions = sorted(t["action"] for t in r.json()["tasks"])
        assert actions == ["check", "pr_digest"]

    def test_due_expires_past_flips_to_expired(self):
        self._seed_due_row(
            action="check",
            fire_once=False,
            schedule_spec="5m",
            expires_at=_now() - timedelta(seconds=1),
        )
        with _patch_prisma(self.prisma):
            self.client.get("/v1/tasks/due")
        row = self.prisma.db.litellm_scheduledtasktable.rows[0]
        assert row.status == "expired"

    def test_due_skips_future_rows(self):
        self._seed_due_row(
            action="check",
            next_run_at=_now() + timedelta(minutes=5),
        )
        with _patch_prisma(self.prisma):
            r = self.client.get("/v1/tasks/due")
        assert len(r.json()["tasks"]) == 0


class TestReport:
    def setup_method(self):
        self.prisma = _make_prisma()
        self.client = _make_app(self.prisma)

    def _seed(self, **kwargs):
        defaults: Dict[str, Any] = {
            "task_id": "rep-1",
            "status": "pending",
        }
        defaults.update(kwargs)
        self.prisma.db.litellm_scheduledtasktable.rows.append(_make_row(**defaults))

    def test_success_clears_counter(self):
        self._seed(consecutive_errors=2, last_error="prev")
        with _patch_prisma(self.prisma):
            r = self.client.post(
                "/v1/tasks/rep-1/report",
                json={"result": "success"},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["consecutive_errors"] == 0
        assert body["last_error"] is None

    def test_first_error_bumps_counter_no_flip(self):
        self._seed()
        with _patch_prisma(self.prisma):
            r = self.client.post(
                "/v1/tasks/rep-1/report",
                json={"result": "error", "reason": "boom"},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["consecutive_errors"] == 1
        assert body["last_error"] == "boom"
        assert body["status"] == "pending"

    def test_third_error_flips_to_failed(self):
        self._seed(consecutive_errors=2)
        with _patch_prisma(self.prisma):
            r = self.client.post(
                "/v1/tasks/rep-1/report",
                json={"result": "error", "reason": "still broken"},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["consecutive_errors"] == 3
        assert body["status"] == "failed"
        assert body["last_error"] == "still broken"

    def test_report_foreign_404(self):
        self.prisma.db.litellm_scheduledtasktable.rows.append(
            _make_row(task_id="foreign-1", owner_token="sk-other")
        )
        with _patch_prisma(self.prisma):
            r = self.client.post(
                "/v1/tasks/foreign-1/report",
                json={"result": "error"},
            )
        assert r.status_code == 404

    def test_failed_task_not_returned_by_due(self):
        self._seed(
            status="failed",
            next_run_at=_now() - timedelta(seconds=10),
        )
        with _patch_prisma(self.prisma):
            r = self.client.get("/v1/tasks/due")
        assert r.json()["tasks"] == []


class TestLazyExpiry:
    def setup_method(self):
        self.prisma = _make_prisma()
        self.client = _make_app(self.prisma)

    def test_list_flips_expired_pending_rows(self):
        self.prisma.db.litellm_scheduledtasktable.rows.append(
            _make_row(
                task_id="exp-1",
                status="pending",
                next_run_at=_now() + timedelta(hours=1),  # not yet due
                expires_at=_now() - timedelta(seconds=1),  # already expired
            )
        )
        with _patch_prisma(self.prisma):
            r = self.client.get("/v1/tasks?include_terminal=true")
        assert r.status_code == 200
        body = r.json()
        statuses = {t["task_id"]: t["status"] for t in body["tasks"]}
        assert statuses["exp-1"] == "expired"
