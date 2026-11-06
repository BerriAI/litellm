import asyncio
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Final

from pydantic import BaseModel, ConfigDict

from litellm.models.workflow_definition import WorkflowBundle
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

NOW: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)


def bundle(name: str = "example", model: str = "gateway-model") -> WorkflowBundle:
    return WorkflowBundle.model_validate(
        {
            "contract_version": "1",
            "name": name,
            "timeout_seconds": 300.0,
            "input_schema": {"type": "array"},
            "output": {"from": "node_1", "output": "result"},
            "nodes": [
                {
                    "id": "node_1",
                    "type": "chat_completion",
                    "model": model,
                    "messages": {"from": "$input", "output": "messages"},
                    "timeout_seconds": 60.0,
                    "retry": {"max_attempts": 1, "backoff_seconds": 0.0},
                }
            ],
        }
    )


def caller(user_id: str | None = "author", team_id: str = "team") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id=user_id, team_id=team_id, user_role=LitellmUserRoles.INTERNAL_USER)


def _matches(row: Mapping[str, object], where: Mapping[str, object]) -> bool:
    for key, value in where.items():
        if key == "OR":
            if not any(_matches(row, condition) for condition in value):
                return False
        elif key == "AND":
            if not all(_matches(row, condition) for condition in value):
                return False
        elif isinstance(value, Mapping):
            actual = row.get(key)
            for operation, expected in value.items():
                if operation == "in" and actual not in expected:
                    return False
                if operation == "gt" and (actual is None or actual <= expected):
                    return False
                if operation == "lt" and (actual is None or actual >= expected):
                    return False
        elif row.get(key) != value:
            return False
    return True


class _Row(BaseModel):
    model_config = ConfigDict(extra="allow")


class Rows:
    def __init__(self, rows: tuple[Mapping[str, object], ...] = ()) -> None:
        self.rows = tuple({"updated_at": NOW, **row} for row in rows)

    async def find_first(
        self,
        skip: int | None = None,
        where: Mapping[str, object] | None = None,
        cursor: Mapping[str, object] | None = None,
        include: Mapping[str, object] | None = None,
        order: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
        distinct: Sequence[str] | None = None,
    ) -> object | None:
        rows = await self.find_many(take=1, where=where, order=order)
        return rows[0] if rows else None

    async def find_unique(
        self, where: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> object | None:
        return await self.find_first(where=where)

    async def find_many(
        self,
        take: int | None = None,
        skip: int | None = None,
        where: Mapping[str, object] | None = None,
        cursor: Mapping[str, object] | None = None,
        include: Mapping[str, object] | None = None,
        order: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
        distinct: Sequence[str] | None = None,
    ) -> Sequence[object]:
        matches = [row for row in self.rows if _matches(row, where or {})]
        orders = [order] if isinstance(order, Mapping) else list(order or [])
        for clause in reversed(orders):
            for key, direction in clause.items():
                matches.sort(key=lambda row: row[key], reverse=direction == "desc")
        return [_Row.model_validate(row) for row in matches[:take]]

    async def create(self, data: Mapping[str, object], include: Mapping[str, object] | None = None) -> object:
        if "etag" in data and any(
            row.get("name") == data["name"] and row.get("team_id") == data["team_id"] for row in self.rows
        ):
            from prisma.errors import UniqueViolationError

            raise UniqueViolationError({"is_panic": False, "message": "duplicate", "error_code": "P2002"})
        row: Final = {
            "created_at": NOW,
            "updated_at": NOW,
            "published_etag": None,
            "published_snapshot_id": None,
            **data,
        }
        self.rows = (*self.rows, row)
        return _Row.model_validate(row)

    async def count(self, where: Mapping[str, object] | None = None, take: int | None = None) -> int:
        return len(await self.find_many(where=where, take=take))

    async def update_many(self, data: Mapping[str, object], where: Mapping[str, object]) -> int:
        count = sum(_matches(row, where) for row in self.rows)
        self.rows = tuple({**row, **data} if _matches(row, where) else row for row in self.rows)
        return count

    async def update(
        self, data: Mapping[str, object], where: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> object | None:
        self.rows = tuple({**row, **data} if _matches(row, where) else row for row in self.rows)
        return await self.find_first(where=where)


class DefinitionRows:
    def __init__(self) -> None:
        self.litellm_workflowdefinition = Rows()
        self.litellm_workflowsnapshot = Rows()
        self.litellm_teamtable = Rows(
            (
                {
                    "team_id": "team",
                    "members_with_roles": [
                        {"user_id": "author", "role": "user"},
                        {"user_id": "reviewer", "role": "admin"},
                        {"user_id": "other", "role": "user"},
                    ],
                },
                {"team_id": "other-team", "members_with_roles": [{"user_id": "outsider", "role": "admin"}]},
            )
        )
        self.lock = asyncio.Lock()
        self.db = self
        self.writer_db = self

    @property
    def database(self) -> "DefinitionRows":
        return self

    @asynccontextmanager
    async def tx(self) -> AsyncGenerator["DefinitionRows"]:
        async with self.lock:
            definitions = self.litellm_workflowdefinition.rows
            snapshots = self.litellm_workflowsnapshot.rows
            try:
                yield self
            except BaseException:
                self.litellm_workflowdefinition.rows = definitions
                self.litellm_workflowsnapshot.rows = snapshots
                raise
