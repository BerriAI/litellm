import json
import sqlite3
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import reduce
from typing import Final

import pytest
from prisma import Json
from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.proxy.management_helpers.customer_hierarchy import effective_member_budget, related_budget
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget

NOW: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")


def matches_where(row: Mapping[str, object], where: Mapping[str, object]) -> bool:
    def condition(key: str, value: object) -> bool:
        if key in ("OR", "AND"):
            nested: Final = TypeAdapter(tuple[Mapping[str, object], ...]).validate_python(value)
            matched: Final = tuple(matches_where(row, part) for part in nested)
            return any(matched) if key == "OR" else all(matched)
        if isinstance(value, Mapping):
            if "in" in value:
                return row.get(key) in value["in"]
            if "not" in value:
                return not matches_where(row, {key: value["not"]})
            if "contains" in value:
                text: Final = str(value["contains"]).replace("\\%", "%").replace("\\_", "_").replace("\\\\", "\\")
                return text.lower() in str(row.get(key) or "").lower()
            return all(
                str(row.get(key)) >= str(bound) if op == "gte" else str(row.get(key)) <= str(bound)
                for op, bound in value.items()
            )
        return row.get(key) == value

    return all(condition(key, value) for key, value in where.items())


class Table:
    def __init__(self, id_field: str, database: "Database") -> None:
        self.id_field = id_field
        self.database = database
        self.rows: Mapping[str, Record] = {}

    async def include(self, row: Record, include: Mapping[str, object] | None) -> Record:
        if not include:
            return row
        relations: Final = {
            name: self.database.litellm_budgettable.rows.get(str(row.model_dump().get(name + "_id")))
            for name in include
        }
        return row.model_copy(
            update={name: budget.model_dump() if budget is not None else None for name, budget in relations.items()}
        )

    async def find_unique(
        self, where: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> Record | None:
        row: Final = self.rows.get(str(where[self.id_field]))
        return await self.include(row, include) if row else None

    async def create(self, data: Mapping[str, object], include: Mapping[str, object] | None = None) -> Record:
        from prisma.errors import UniqueViolationError

        if str(data[self.id_field]) in self.rows:
            raise UniqueViolationError({})
        row: Final = Record.model_validate(
            {
                "spend": 0.0,
                "budget_reset_at": None,
                "budget_ids": (),
                "budget_names": {},
                "created_at": NOW,
                "updated_at": NOW,
                **{key: value.data if isinstance(value, Json) else value for key, value in data.items()},
            }
        )
        self.rows = {**self.rows, str(data[self.id_field]): row}
        return await self.include(row, include)

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> Record | None:
        row_id: Final = str(where[self.id_field])
        if row_id not in self.rows:
            return None
        values: Final = {key: value.data if isinstance(value, Json) else value for key, value in data.items()}
        row: Final = self.rows[row_id].model_copy(update=values)
        self.rows = {**self.rows, row_id: row}
        return row

    async def find_many(
        self,
        where: Mapping[str, object] | None = None,
        order: Sequence[Mapping[str, object]] | None = None,
        include: Mapping[str, object] | None = None,
        skip: int = 0,
        take: int | None = None,
    ) -> tuple[Record, ...]:
        matches: Final = tuple(row for row in self.rows.values() if matches_where(row.model_dump(), where or {}))
        ordered: Final = reduce(
            lambda rows, key: tuple(
                sorted(
                    rows,
                    key=lambda row: str(row.model_dump().get(next(iter(key)))),
                    reverse=next(iter(key.values())) == "desc",
                )
            ),
            reversed(order or ()),
            matches,
        )
        selected: Final = ordered[skip : skip + take] if take is not None else ordered[skip:]
        return tuple([await self.include(row, include) for row in selected])

    async def count(self, where: Mapping[str, object] | None = None) -> int:
        return len(await self.find_many(where=where))

    async def delete(self, where: Mapping[str, object]) -> Record | None:
        row_id: Final = TypeAdapter(str).validate_python(where[self.id_field])
        row: Final = self.rows.get(row_id)
        self.rows = {key: value for key, value in self.rows.items() if key != row_id}
        return row

    async def delete_many(self, where: Mapping[str, object]) -> int:
        matched: Final = await self.find_many(where=where)
        self.rows = {key: row for key, row in self.rows.items() if row not in matched}
        return len(matched)


class Database:
    def __init__(self) -> None:
        self.litellm_customerorganizationtable = Table("customer_org_id", self)
        self.litellm_customerteamtable = Table("customer_team_id", self)
        self.litellm_customermembertable = Table("customer_member_id", self)
        self.litellm_budgettable = Table("budget_id", self)
        self.locked_organization_ids: tuple[str, ...] = ()

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        if "FOR UPDATE" in query:
            self.locked_organization_ids = (*self.locked_organization_ids, *(str(arg) for arg in args))
            return ()
        with sqlite3.connect(":memory:") as connection:
            connection.row_factory = sqlite3.Row
            connection.executescript(
                'CREATE TABLE "LiteLLM_CustomerTeamTable" (customer_team_id TEXT, parent_org_id TEXT);'
                'CREATE TABLE "LiteLLM_CustomerMemberTable" (customer_member_id TEXT, parent_team_id TEXT);'
            )
            connection.executemany(
                'INSERT INTO "LiteLLM_CustomerTeamTable" VALUES (?, ?)',
                tuple(
                    (row.model_dump()["customer_team_id"], row.model_dump()["parent_org_id"])
                    for row in self.litellm_customerteamtable.rows.values()
                ),
            )
            connection.executemany(
                'INSERT INTO "LiteLLM_CustomerMemberTable" VALUES (?, ?)',
                tuple(
                    (row.model_dump()["customer_member_id"], row.model_dump()["parent_team_id"])
                    for row in self.litellm_customermembertable.rows.values()
                ),
            )
            return tuple(
                dict(row)
                for row in connection.execute(
                    query.replace("= ANY($1::text[])", "IN (SELECT value FROM json_each($1))"),
                    tuple(json.dumps(arg) for arg in args),
                )
            )

    @asynccontextmanager
    async def tx(self) -> AsyncIterator["Database"]:
        tables: Final = (
            self.litellm_customerorganizationtable,
            self.litellm_customerteamtable,
            self.litellm_customermembertable,
            self.litellm_budgettable,
        )
        snapshots: Final = tuple(table.rows for table in tables)
        try:
            yield self
        except Exception:
            for table, rows in zip(tables, snapshots):
                table.rows = rows
            raise


@pytest.mark.parametrize("expires_at", (None, datetime(2027, 1, 1, tzinfo=timezone.utc)))
def test_effective_budget_merges_non_null_fields_and_empty_models_inherit(expires_at: datetime | None) -> None:
    template: Final = Budget(max_budget=100, rpm_limit=10, allowed_models=("model-a",))
    override: Final = Budget(max_budget=0, rpm_limit=None, allowed_models=())
    assert effective_member_budget(template, override, expires_at, NOW) == Budget(
        max_budget=0,
        rpm_limit=10,
        allowed_models=("model-a",),
    )


@pytest.mark.parametrize("expires_at", (NOW, datetime(2025, 1, 1), datetime(2025, 1, 1, tzinfo=timezone.utc)))
def test_expired_override_is_ignored_at_the_expiration_boundary(expires_at: datetime) -> None:
    template: Final = Budget(max_budget=100)
    assert effective_member_budget(template, Budget(max_budget=1), expires_at, NOW) == template


def test_custom_budget_and_missing_sources_resolve_without_a_template() -> None:
    assert effective_member_budget(None, Budget(max_budget=4), None, NOW) == Budget(max_budget=4, allowed_models=())
    assert effective_member_budget(None, None, None, NOW) == Budget(allowed_models=())


@pytest.mark.parametrize(
    "expiry, expected",
    (
        (datetime(2027, 1, 1, tzinfo=timezone.utc), 15),
        (NOW, 10),
    ),
)
def test_existing_budget_model_applies_temporary_increase_using_injected_now(expiry: datetime, expected: float) -> None:
    resolved: Final = related_budget(
        {
            "max_budget": 10,
            "temp_budget_increase": 5,
            "temp_budget_expiry": expiry,
        },
        NOW,
    )
    assert resolved == Budget(max_budget=expected)


@pytest.mark.parametrize("expiry, expected", ((datetime(2027, 1, 1), 105), (NOW, 100)))
def test_member_temporary_increase_is_added_to_its_inherited_maximum(expiry: datetime, expected: float) -> None:
    resolved: Final = related_budget({"temp_budget_increase": 5, "temp_budget_expiry": expiry}, NOW, 100)
    assert resolved == Budget(max_budget=expected)
