import asyncio
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Final, cast  # noqa: TID251  # the test database implements the delegates exercised by these routes

from liteflow import ChatCompletion, Input, Workflow, compile_workflow
from pydantic import TypeAdapter

from litellm.models.workflow_definition import WorkflowBundle
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.repositories.workflow_definition_repository import DefinitionDatabase

NOW: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
_COMPARISON: Final = TypeAdapter(Mapping[str, str])


def bundle(name: str = "example", model: str = "gateway-model") -> WorkflowBundle:
    return WorkflowBundle.model_validate(
        compile_workflow(Workflow(name=name, nodes=[ChatCompletion(model=model, messages=Input.messages)]))
    )


def caller(user_id: str | None = "author", team_id: str = "team") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id=user_id, team_id=team_id, user_role=LitellmUserRoles.INTERNAL_USER)


def _matches(row: Mapping[str, object], where: Mapping[str, object]) -> bool:
    return all(
        str(row.get(key)) > _COMPARISON.validate_python(value)["gt"]
        if isinstance(value, Mapping) and "gt" in value
        else row.get(key) == value
        for key, value in where.items()
    )


class Rows:
    def __init__(self, rows: tuple[Mapping[str, object], ...] = ()) -> None:
        self.rows = rows

    async def find_first(
        self,
        skip: int | None = None,
        where: Mapping[str, object] | None = None,
        cursor: Mapping[str, object] | None = None,
        include: Mapping[str, object] | None = None,
        order: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
        distinct: Sequence[str] | None = None,
    ) -> object | None:
        return next((row for row in self.rows if _matches(row, where or {})), None)

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
        matches: Final = tuple(row for row in self.rows if _matches(row, where or {}))
        return sorted(matches, key=lambda row: str(row.get("workflow_id")))[:take]

    async def create(self, data: Mapping[str, object], include: Mapping[str, object] | None = None) -> object:
        if "etag" in data and any(
            row.get("name") == data["name"] and row.get("team_id") == data["team_id"] for row in self.rows
        ):
            from prisma.errors import UniqueViolationError

            raise UniqueViolationError({"is_panic": False, "message": "duplicate", "error_code": "P2002"})
        row: Final = {
            "created_at": NOW,
            "published_etag": None,
            "published_snapshot_id": None,
            **data,
        }
        self.rows = (*self.rows, row)
        return row

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

    async def query_raw(self, query: str, *args: object) -> object:
        return ()

    @property
    def database(self) -> DefinitionDatabase:
        return cast(DefinitionDatabase, self)  # cast-ok: isolated Prisma boundary implements the used table operations

    @asynccontextmanager
    async def tx(self) -> AsyncGenerator["DefinitionRows"]:
        async with self.lock:
            yield self
