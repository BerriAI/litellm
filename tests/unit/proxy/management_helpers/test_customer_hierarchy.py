import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from functools import reduce
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

import pytest
from prisma import Json
from prisma.errors import DataError, RawQueryError, TransactionExpiredError
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.models.budget import BudgetLimit, LiteLLM_BudgetTable
from litellm.proxy._types import BudgetWriteRequest
from litellm.proxy.management_helpers.customer_hierarchy import (
    CustomerHierarchy,
    CustomerHierarchyStore,
    ErrorCode,
    Failure,
    budget_row_values,
    effective_member_budget,
    related_budget,
)
from litellm.proxy.spend_tracking.budget_reservation import customer_window_counters
from litellm.types.proxy.management_endpoints.customer_hierarchy import (
    Budget,
    CustomerMember,
    MemberCreate,
    OrganizationCreate,
    OrganizationPatch,
    Team,
    TeamCreate,
)

if TYPE_CHECKING:
    from prisma import Prisma

NOW: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize("windows", (None, []))
def test_scalar_budget_accepts_absent_windows(windows: object) -> None:
    budget: Final = LiteLLM_BudgetTable.model_validate({"max_budget": 0, "budget_limits": windows})
    assert budget.max_budget == 0


def test_scalar_budget_rejects_concurrent_windows() -> None:
    row: Final = {"budget_limits": [{"budget_duration": "1mo", "max_budget": 0}]}
    with pytest.raises(ValidationError, match="customer hierarchies"):
        LiteLLM_BudgetTable.model_validate(row)
    assert Budget.model_validate(row).budget_limits == (BudgetLimit(budget_duration="1mo", max_budget=0),)


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")


def matches_where(row: Mapping[str, object], where: Mapping[str, object]) -> bool:
    def condition(key: str, value: object) -> bool:
        if key in ("OR", "AND"):
            nested: Final = TypeAdapter(tuple[Mapping[str, object], ...]).validate_python(value)
            matched: Final = tuple(matches_where(row, part) for part in nested)
            return any(matched) if key == "OR" else all(matched)
        if isinstance(value, Mapping):
            if "has" in value:
                items: Final = TypeAdapter(tuple[str, ...]).validate_python(row.get(key, ()))
                return value["has"] in items
            if "in" in value:
                return row.get(key) in value["in"]
            if "has" in value:
                return value["has"] in TypeAdapter(tuple[str, ...]).validate_python(row.get(key) or ())
            if "has_some" in value:
                return bool(
                    set(value["has_some"]) & set(TypeAdapter(tuple[str, ...]).validate_python(row.get(key) or ()))
                )
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
    def __init__(
        self,
        id_field: str,
        database: "Database",
        *,
        create_error: DataError | None = None,
        hydration_limit: int | None = None,
        read_limit: int | None = None,
        delete_error: Exception | None = None,
    ) -> None:
        self.id_field = id_field
        self.database = database
        self.create_error = create_error
        self.rows: Mapping[str, Record] = {}
        self.hydration_limit = hydration_limit
        self.read_limit = read_limit
        self.read_count = 0
        self.delete_error = delete_error

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

        if self.create_error is not None:
            raise self.create_error
        if str(data[self.id_field]) in self.rows:
            raise UniqueViolationError({})
        row: Final = Record.model_validate(
            {
                "spend": 0.0,
                "budget_template_ids": (),
                "created_at": NOW,
                "updated_at": NOW,
                **{key: value.data if isinstance(value, Json) else value for key, value in data.items()},
            }
        )
        self.rows = {**self.rows, str(data[self.id_field]): row}
        return await self.include(row, include)

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> Record | None:
        from prisma import Json

        row_id: Final = str(where[self.id_field])
        if row_id not in self.rows:
            return None
        values: Final = {key: value.data if isinstance(value, Json) else value for key, value in data.items()}
        row: Final = self.rows[row_id].model_copy(update=values)
        self.rows = {**self.rows, row_id: row}
        return row

    async def upsert(self, where: Mapping[str, object], data: Mapping[str, object]) -> Record:
        existing: Final = await self.find_unique(where)
        values: Final = TypeAdapter(Mapping[str, object]).validate_python(data["update" if existing else "create"])
        if existing is None:
            return await self.create(values)
        updated: Final = await self.update(where, values)
        assert updated is not None
        return updated

    def matches(self, row: Record, where: Mapping[str, object]) -> bool:
        values: Final = row.model_dump()
        for key, value in where.items():
            if key == "AND":
                clauses: Final = TypeAdapter(tuple[Mapping[str, object], ...]).validate_python(value)
                if not all(self.matches(row, clause) for clause in clauses):
                    return False
            elif isinstance(value, Mapping) and "none" in value:
                if self.database.has_budget_reference(str(values[self.id_field]), key):
                    return False
            elif isinstance(value, Mapping) and "is" in value:
                related: Final = self.database.litellm_customerteamtable.rows.get(str(values.get("parent_team_id")))
                if related is None or not self.database.litellm_customerteamtable.matches(related, value["is"]):
                    return False
            elif not matches_where(values, {key: value}):
                return False
        return True

    async def find_many(
        self,
        where: Mapping[str, object] | None = None,
        order: Sequence[Mapping[str, object]] | None = None,
        include: Mapping[str, object] | None = None,
        skip: int = 0,
        take: int | None = None,
    ) -> tuple[Record, ...]:
        self.read_count += 1
        if self.read_limit is not None and self.read_count > self.read_limit:
            raise TransactionExpiredError("Transaction exhausted its query budget")
        matches: Final = tuple(row for row in self.rows.values() if self.matches(row, where or {}))
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
        if include and self.hydration_limit is not None:
            assert len(selected) <= self.hydration_limit
        return tuple([await self.include(row, include) for row in selected])

    async def count(self, where: Mapping[str, object] | None = None) -> int:
        return len(await self.find_many(where=where))

    async def delete(self, where: Mapping[str, object]) -> Record | None:
        if self.delete_error is not None:
            raise self.delete_error
        row_id: Final = TypeAdapter(str).validate_python(where[self.id_field])
        row: Final = self.rows.get(row_id)
        self.rows = {key: value for key, value in self.rows.items() if key != row_id}
        return row

    async def delete_many(self, where: Mapping[str, object]) -> int:
        matched: Final = await self.find_many(where=where)
        self.rows = {key: row for key, row in self.rows.items() if row not in matched}
        return len(matched)

    async def update_many(self, where: Mapping[str, object], data: Mapping[str, object]) -> int:
        matched: Final = await self.find_many(where=where)
        values: Final = {key: value.data if isinstance(value, Json) else value for key, value in data.items()}
        self.rows = {key: (row.model_copy(update=values) if row in matched else row) for key, row in self.rows.items()}
        return len(matched)


class BoundedChildrenTable(Table):
    def __init__(self, id_field: str, database: "Database") -> None:
        super().__init__(id_field, database)
        self.batch_sizes: tuple[int, ...] = ()

    async def find_many(
        self,
        where: Mapping[str, object] | None = None,
        order: Sequence[Mapping[str, object]] | None = None,
        include: Mapping[str, object] | None = None,
        skip: int = 0,
        take: int | None = None,
    ) -> tuple[Record, ...]:
        rows: Final = await super().find_many(where=where, order=order, include=include, skip=skip, take=take)
        if where is not None and any(name in where for name in ("parent_org_id", "parent_team_id", "parent_team_ids")):
            assert take == 100
            self.batch_sizes = (*self.batch_sizes, len(rows))
        return rows


class DeleteConflictTable(Table):
    def __init__(self, id_field: str, database: "Database") -> None:
        super().__init__(id_field, database)
        self.members_at_delete: int | None = None

    async def delete(self, where: Mapping[str, object]) -> Record | None:
        from prisma.errors import ForeignKeyViolationError

        self.members_at_delete = len(self.database.litellm_customermembertable.rows)
        raise ForeignKeyViolationError({})


class AdvisoryLocks:
    def __init__(self) -> None:
        self.locks: Mapping[tuple[str, ...], asyncio.Lock] = MappingProxyType({})

    def lock_for(self, query: str, args: Sequence[object]) -> asyncio.Lock:
        key: Final = (query, *(str(arg) for arg in args))
        existing: Final = self.locks.get(key)
        if existing is not None:
            return existing
        lock: Final = asyncio.Lock()
        self.locks = MappingProxyType({**self.locks, key: lock})
        return lock


class PausedCreateTable(Table):
    def __init__(self, id_field: str, database: "Database", entered: asyncio.Event, release: asyncio.Event) -> None:
        super().__init__(id_field, database)
        self.entered = entered
        self.release = release

    async def create(self, data: Mapping[str, object], include: Mapping[str, object] | None = None) -> Record:
        if data.get(self.id_field) == "blocked":
            self.entered.set()
            await self.release.wait()
        return await super().create(data, include)


class Database:
    def __init__(
        self,
        advisory_locks: AdvisoryLocks | None = None,
        source: "Database | None" = None,
        *,
        team_create_error: DataError | None = None,
        hydration_limit: int | None = None,
        member_read_limit: int | None = None,
        organization_delete_error: Exception | None = None,
    ) -> None:
        self.litellm_customerorganizationtable = (
            Table("customer_org_id", self, delete_error=organization_delete_error)
            if source is None
            else source.litellm_customerorganizationtable
        )
        self.litellm_customerteamtable = (
            Table(
                "customer_team_id",
                self,
                create_error=team_create_error,
                hydration_limit=hydration_limit,
            )
            if source is None
            else source.litellm_customerteamtable
        )
        self.litellm_customermembertable = (
            Table(
                "customer_member_id",
                self,
                hydration_limit=hydration_limit,
                read_limit=member_read_limit,
            )
            if source is None
            else source.litellm_customermembertable
        )
        self.litellm_budgettable = Table("budget_id", self) if source is None else source.litellm_budgettable
        self.litellm_config = Table("param_name", self) if source is None else source.litellm_config
        self.litellm_budgetwindowspend = (
            Table("entity_id", self) if source is None else source.litellm_budgetwindowspend
        )
        self.litellm_verificationtoken = Table("token", self) if source is None else source.litellm_verificationtoken
        self.litellm_projecttable = Table("project_id", self) if source is None else source.litellm_projecttable
        self.litellm_customerbudgetincreaserequest = (
            Table("request_id", self) if source is None else source.litellm_customerbudgetincreaserequest
        )
        self.advisory_locks = advisory_locks
        self.held_locks: tuple[asyncio.Lock, ...] = ()
        self.locked_organization_ids: tuple[str, ...] = ()

    def has_budget_reference(self, budget_id: str, relation: str) -> bool:
        references: Final = {
            "customer_organization_defaults": (self.litellm_customerorganizationtable, "default_budget_template_id"),
            "customer_organizations": (self.litellm_customerorganizationtable, "budget_id"),
            "customer_teams": (self.litellm_customerteamtable, "budget_id"),
            "customer_members": (self.litellm_customermembertable, "budget_id"),
            "customer_member_overrides": (self.litellm_customermembertable, "budget_override_id"),
            "keys": (self.litellm_verificationtoken, "budget_id"),
            "projects": (self.litellm_projecttable, "budget_id"),
        }
        reference: Final = references.get(relation)
        if reference is None:
            return False
        table, field = reference
        return any(row.model_dump().get(field) == budget_id for row in table.rows.values())

    async def acquire_advisory_lock(self, query: str, args: Sequence[object]) -> tuple[Mapping[str, object], ...]:
        if self.advisory_locks is None:
            return ()
        lock: Final = self.advisory_locks.lock_for(query, args)
        if lock in self.held_locks:
            return ()
        await lock.acquire()
        self.held_locks = (*self.held_locks, lock)
        return ()

    def budget_sort_rows(self, query: str, args: Sequence[object]) -> Sequence[Mapping[str, object]]:
        table: Final = (
            self.litellm_customerteamtable
            if '"LiteLLM_CustomerTeamTable"' in query
            else self.litellm_customermembertable
        )
        fields: Final = (
            table.id_field,
            "parent_org_id",
            "alias",
            "created_at",
            "budget_id",
            "budget_override_id",
            "budget_override_expires_at",
            "parent_team_ids",
        )
        with sqlite3.connect(":memory:") as connection:
            connection.row_factory = sqlite3.Row
            name: Final = (
                "LiteLLM_CustomerTeamTable"
                if table is self.litellm_customerteamtable
                else "LiteLLM_CustomerMemberTable"
            )
            connection.execute(f'CREATE TABLE "{name}" ({", ".join(fields)})')
            connection.executemany(
                f'INSERT INTO "{name}" VALUES ({", ".join("?" for _ in fields)})',
                tuple(
                    tuple(
                        json.dumps(row.model_dump().get(field))
                        if field == "parent_team_ids"
                        else None
                        if row.model_dump().get(field) is None
                        else str(row.model_dump()[field])
                        for field in fields
                    )
                    for row in table.rows.values()
                ),
            )
            sql: Final = (
                query.replace(
                    "parent_team_ids @> ARRAY[$2]::text[]",
                    "EXISTS (SELECT 1 FROM json_each(parent_team_ids) WHERE value = $2)",
                )
                .replace("::text", "")
                .replace("ILIKE $3", "LIKE $3 ESCAPE '\\' COLLATE NOCASE")
            )
            parameters: Final = {str(index): None if arg is None else str(arg) for index, arg in enumerate(args, 1)}
            return tuple(dict(row) for row in connection.execute(sql, parameters))

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        if " AS entity_id" in query:
            return self.budget_sort_rows(query, args)
        if "set_config('lock_timeout'" in query:
            return ()
        if "pg_advisory_xact_lock" in query:
            return await self.acquire_advisory_lock(query, args)
        if "SELECT EXISTS" in query:
            return ({"referenced": False},)
        if "FOR UPDATE" in query:
            self.locked_organization_ids = (*self.locked_organization_ids, *(str(arg) for arg in args))
            return ()
        with sqlite3.connect(":memory:") as connection:
            connection.row_factory = sqlite3.Row
            connection.executescript(
                'CREATE TABLE "LiteLLM_CustomerTeamTable" (customer_team_id TEXT, parent_org_id TEXT);'
                'CREATE TABLE "LiteLLM_CustomerMemberTable" (customer_member_id TEXT, parent_org_id TEXT);'
                "CREATE TABLE member_teams (team_id TEXT);"
            )
            connection.executemany(
                'INSERT INTO "LiteLLM_CustomerTeamTable" VALUES (?, ?)',
                tuple(
                    (row.model_dump()["customer_team_id"], row.model_dump()["parent_org_id"])
                    for row in self.litellm_customerteamtable.rows.values()
                ),
            )
            connection.executemany(
                "INSERT INTO member_teams VALUES (?)",
                tuple(
                    (team_id,)
                    for team_id in chain.from_iterable(
                        TypeAdapter(tuple[str, ...]).validate_python(row.model_dump()["parent_team_ids"])
                        for row in self.litellm_customermembertable.rows.values()
                    )
                ),
            )
            connection.executemany(
                'INSERT INTO "LiteLLM_CustomerMemberTable" VALUES (?, ?)',
                tuple(
                    (row.model_dump()["customer_member_id"], row.model_dump()["parent_org_id"])
                    for row in self.litellm_customermembertable.rows.values()
                ),
            )
            return tuple(
                dict(row)
                for row in connection.execute(
                    query.replace("= ANY($1::text[])", "IN (SELECT value FROM json_each($1))").replace(
                        '"LiteLLM_CustomerMemberTable", unnest(parent_team_ids) AS team_id', "member_teams"
                    ),
                    {
                        str(index): json.dumps(arg) if not isinstance(arg, int) else arg
                        for index, arg in enumerate(args, 1)
                    },
                )
            )

    @asynccontextmanager
    async def tx(self) -> AsyncIterator["Database"]:
        tables: Final = (
            self.litellm_customerorganizationtable,
            self.litellm_customerteamtable,
            self.litellm_customermembertable,
            self.litellm_budgettable,
            self.litellm_config,
            self.litellm_verificationtoken,
            self.litellm_projecttable,
        )
        snapshots: Final = tuple(table.rows for table in tables)
        try:
            yield self
        except Exception:
            for table, rows in zip(tables, snapshots):
                table.rows = rows
            raise
        finally:
            for lock in self.held_locks:
                lock.release()
            self.held_locks = ()


class CapacityErrorDatabase(Database):
    def __init__(self, error_code: str, scope: str) -> None:
        super().__init__()
        self.error_code = error_code
        self.scope = scope

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        if "pg_advisory_xact_lock" in query and args and args[0] == self.scope:
            raise RawQueryError(
                {"user_facing_error": {"meta": {"code": self.error_code, "message": "advisory lock failed"}}}
            )
        return await super().query_raw(query, *args)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("team", "member"))
@pytest.mark.parametrize("same_parent", (False, True))
async def test_capacity_contention_is_isolated_to_each_parent(kind: str, same_parent: bool) -> None:
    locks: Final = AdvisoryLocks()
    database: Final = Database(locks)
    hierarchy: Final = CustomerHierarchy(
        cast("Prisma", database), premium_user=False, clock=lambda: NOW
    )  # cast-ok: in-memory Prisma edge
    for org_id in ("a", "b"):
        assert not isinstance(
            await hierarchy.create_organization(OrganizationCreate(customer_org_id=org_id), "admin"), Failure
        )
        if kind == "member":
            assert not isinstance(
                await hierarchy.create_team(TeamCreate(customer_team_id=org_id, parent_org_id=org_id), "admin"), Failure
            )
    entered: Final = asyncio.Event()
    release: Final = asyncio.Event()
    if kind == "team":
        database.litellm_customerteamtable = PausedCreateTable("customer_team_id", database, entered, release)
    else:
        database.litellm_customermembertable = PausedCreateTable("customer_member_id", database, entered, release)
    other: Final = CustomerHierarchy(
        cast("Prisma", Database(locks, source=database)), premium_user=False, clock=lambda: NOW
    )  # cast-ok: independent connection to the shared in-memory tables

    async def create(store: CustomerHierarchy, parent: str, identity: str) -> Team | CustomerMember | Failure:
        if kind == "team":
            return await store.create_team(TeamCreate(customer_team_id=identity, parent_org_id=parent), "admin")
        return await store.create_member(
            MemberCreate(customer_member_id=identity, parent_team_ids=(parent,), budget_override=Budget(max_budget=10)),
            "admin",
        )

    for index in range(4):
        assert not isinstance(await create(hierarchy, "a", f"seed-{index}"), Failure)
    blocked: Final = asyncio.create_task(create(hierarchy, "a", "blocked"))
    await entered.wait()
    independent: Final = asyncio.create_task(create(other, "a" if same_parent else "b", "independent"))
    try:
        if same_parent:
            await asyncio.sleep(0)
            release.set()
        result: Final = await asyncio.wait_for(independent, timeout=10)
        if same_parent:
            assert result == Failure(ErrorCode.FREE_TIER_LIMIT)
        else:
            assert not isinstance(result, Failure)
            assert result == (
                await other.get_team("independent") if kind == "team" else await other.get_member("independent")
            )
    finally:
        release.set()
        assert not isinstance(await blocked, Failure)


@pytest.mark.asyncio
@pytest.mark.parametrize("premium", (False, True))
async def test_customer_portal_capacity_caps_every_parent(premium: bool) -> None:
    database: Final = Database()
    hierarchy: Final = CustomerHierarchy(
        cast("Prisma", database), premium_user=premium, clock=lambda: NOW
    )  # cast-ok: in-memory Prisma edge
    for index in range(5):
        assert not isinstance(
            await hierarchy.create_organization(OrganizationCreate(customer_org_id=f"org-{index}"), "admin"), Failure
        )
    sixth_org: Final = await hierarchy.create_organization(OrganizationCreate(customer_org_id="sixth"), "admin")
    assert isinstance(sixth_org, Failure) is not premium
    if not premium:
        assert sixth_org == Failure(ErrorCode.FREE_TIER_LIMIT)
    for index in range(5):
        assert not isinstance(
            await hierarchy.create_team(TeamCreate(customer_team_id=f"team-{index}", parent_org_id="org-0"), "admin"),
            Failure,
        )
    sixth_team: Final = await hierarchy.create_team(
        TeamCreate(customer_team_id="sixth", parent_org_id="org-0"), "admin"
    )
    assert isinstance(sixth_team, Failure) is not premium
    if not premium:
        assert sixth_team == Failure(ErrorCode.FREE_TIER_LIMIT)
    for index in range(5):
        assert not isinstance(
            await hierarchy.create_member(
                MemberCreate(
                    customer_member_id=f"member-{index}",
                    parent_team_ids=("team-0", "team-1"),
                    budget_override=Budget(max_budget=10),
                ),
                "admin",
            ),
            Failure,
        )
    sixth_member: Final = await hierarchy.create_member(
        MemberCreate(customer_member_id="sixth", parent_team_ids=("team-1",), budget_override=Budget(max_budget=10)),
        "admin",
    )
    assert isinstance(sixth_member, Failure) is not premium
    if not premium:
        assert sixth_member == Failure(ErrorCode.FREE_TIER_LIMIT)


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


@pytest.mark.parametrize("request_type", (Budget, BudgetWriteRequest))
@pytest.mark.parametrize("maximum", (-1.0, float("nan"), float("inf")))
def test_budget_windows_reject_invalid_caps(request_type: type[BaseModel], maximum: float) -> None:
    with pytest.raises(ValidationError):
        request_type.model_validate({"budget_limits": [{"budget_duration": "1mo", "max_budget": maximum}]})


@pytest.mark.parametrize("request_type", (Budget, BudgetWriteRequest))
@pytest.mark.parametrize("first,second", (("1mo", "1mo"), ("2d", "2d")))
def test_budget_windows_reject_duplicate_durations(request_type: type[BaseModel], first: str, second: str) -> None:
    with pytest.raises(ValidationError):
        request_type.model_validate(
            {
                "budget_limits": [
                    {"budget_duration": first, "max_budget": 0},
                    {"budget_duration": second, "max_budget": 10},
                ]
            }
        )


@pytest.mark.parametrize("request_type", (Budget, BudgetWriteRequest))
@pytest.mark.parametrize("duration", ("", "0s", "-1d", "1y", "1.5h",
                                    "999999999999999999999d", "999999999999999999999h", "999999999999999999999s"))
def test_budget_windows_reject_noncanonical_durations(request_type: type[BaseModel], duration: str) -> None:
    with pytest.raises(ValidationError):
        request_type.model_validate({"budget_limits": [{"budget_duration": duration, "max_budget": 0}]})


@pytest.mark.parametrize("duration", ("30s", "15m", "2h", "6d", "2w", "30d", "daily"))
def test_additional_budget_windows_keep_the_submitted_duration_without_changing_the_scalar(duration: str) -> None:
    budget: Final = Budget(max_budget=50, budget_duration="daily",
                          budget_limits=(BudgetLimit(budget_duration=duration, max_budget=0),))
    assert (budget.max_budget, budget.budget_duration) == (50, "daily")
    assert tuple(limit.budget_duration for limit in budget.budget_limits or ()) == (duration,)

@pytest.mark.asyncio
@pytest.mark.parametrize("elapsed", (1, 3))
@pytest.mark.parametrize("shared", (False, True))
async def test_editing_a_custom_cap_retains_its_original_reset_anchor(elapsed: int, shared: bool) -> None:
    database: Final = Database()
    store: Final = CustomerHierarchyStore(cast("Prisma", database), NOW, premium_user=True)
    policy: Final = Budget(budget_limits=(BudgetLimit(budget_duration="2d", max_budget=10),))
    budget_id: Final = await store.create_budget(policy, "admin")
    original: Final = database.litellm_budgettable.rows[budget_id].model_dump()["budget_limits"]
    later: Final = CustomerHierarchyStore(cast("Prisma", database), NOW + timedelta(days=elapsed), premium_user=True)
    patch: Final = Budget(budget_limits=(BudgetLimit(budget_duration="2d", max_budget=20),))
    updated_id: Final = await later.private_budget_patch(
        budget_id, OrganizationPatch(budget=patch), "admin", shared_ids=(budget_id,) if shared else ()
    )
    assert updated_id is not None and (updated_id != budget_id) is shared
    assert database.litellm_budgettable.rows[updated_id].model_dump()["budget_limits"] == [
        {**limit, "max_budget": 20} for limit in original
    ]
    current: Final = NOW + timedelta(days=elapsed)
    before: Final = customer_window_counters(
        "customer_member", "member", tuple(BudgetLimit.model_validate(limit) for limit in original), now=current
    )[0]
    after: Final = customer_window_counters(
        "customer_member",
        "member",
        tuple(
            BudgetLimit.model_validate(limit)
            for limit in database.litellm_budgettable.rows[updated_id].model_dump()["budget_limits"]
        ),
        now=current,
    )[0]
    assert after.counter_key == before.counter_key
    assert after.window_start == before.window_start
    assert after.reset_at == before.reset_at


class ResetAdvancingBudgetTable(Table):
    def __init__(self, database: Database) -> None:
        super().__init__("budget_id", database)
        self.locked = False

    def advance_reset(self, budget_id: str) -> None:
        row: Final = self.rows[budget_id]
        windows: Final = TypeAdapter(tuple[Mapping[str, object], ...]).validate_python(
            row.model_dump()["budget_limits"]
        )
        self.rows = {
            **self.rows,
            budget_id: row.model_copy(
                update={"budget_limits": [{**window, "reset_at": (NOW + timedelta(days=2)).isoformat()}
                                          for window in windows]}
            ),
        }

    async def find_unique(
        self, where: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> Record | None:
        snapshot: Final = await super().find_unique(where, include)
        if snapshot is not None and not self.locked:
            self.advance_reset(str(where["budget_id"]))
        return snapshot


class ResetAdvancingDatabase(Database):
    def __init__(self) -> None:
        super().__init__()
        self.budget_rows: Final = ResetAdvancingBudgetTable(self)
        self.litellm_budgettable = self.budget_rows
        self.in_transaction = False

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        if query.startswith('SELECT budget_limits FROM "LiteLLM_BudgetTable"') and "FOR UPDATE" in query:
            assert self.in_transaction, "Budget row locks must cover the edit transaction"
            budget_id: Final = str(args[0])
            self.budget_rows.advance_reset(budget_id)
            self.budget_rows.locked = True
            return ({"budget_limits": self.budget_rows.rows[budget_id].model_dump()["budget_limits"]},)
        return await super().query_raw(query, *args)

    @asynccontextmanager
    async def tx(self) -> AsyncIterator["ResetAdvancingDatabase"]:
        async with super().tx():
            self.in_transaction = True
            try:
                yield self
            finally:
                self.budget_rows.locked = False
                self.in_transaction = False


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ("edit", "clear-null", "clear-empty", "scalar"))
async def test_customer_budget_edit_preserves_a_racing_reset(change: str) -> None:
    database: Final = ResetAdvancingDatabase()
    hierarchy: Final = CustomerHierarchy(
        cast("Prisma", database), premium_user=True, clock=lambda: NOW
    )  # cast-ok: deterministic Prisma protocol fixture
    organization: Final = await hierarchy.create_organization(
        OrganizationCreate(
            customer_org_id="org",
            budget=Budget(
                max_budget=50,
                budget_limits=(BudgetLimit(budget_duration="2d", max_budget=10, reset_at=NOW),),
            ),
        ),
        "admin",
    )
    assert not isinstance(organization, Failure)
    assert organization.budget_id is not None
    edits: Final = {
        "edit": Budget(budget_limits=(BudgetLimit(budget_duration="2d", max_budget=20),)),
        "clear-null": Budget(budget_limits=None),
        "clear-empty": Budget(budget_limits=()),
        "scalar": Budget(max_budget=75),
    }
    result: Final = await hierarchy.update_organization(
        "org", OrganizationPatch(budget=edits[change]), "admin"
    )
    assert not isinstance(result, Failure)
    stored: Final = database.budget_rows.rows[organization.budget_id].model_dump()
    expected_windows: Final = (
        None if change == "clear-null" else [] if change == "clear-empty" else [
            {"budget_duration": "2d", "max_budget": 20 if change == "edit" else 10,
             "reset_at": (NOW + timedelta(days=2)).isoformat()}
        ]
    )
    assert (stored["max_budget"], stored["budget_limits"]) == (
        75 if change == "scalar" else 50, expected_windows
    )


def test_window_override_preserves_other_period_and_scalar_fields() -> None:
    template: Final = Budget(
        max_budget=500,
        budget_duration="30d",
        budget_limits=(
            BudgetLimit(budget_duration="1mo", max_budget=100),
            BudgetLimit(budget_duration="12mo", max_budget=1000),
        ),
    )
    override: Final = Budget(budget_limits=(BudgetLimit(budget_duration="1mo", max_budget=0),))
    effective: Final = effective_member_budget(template, override, None, NOW)
    assert effective.budget_limits == (
        BudgetLimit(budget_duration="1mo", max_budget=0),
        BudgetLimit(budget_duration="12mo", max_budget=1000),
    )
    assert effective.max_budget == template.max_budget
    assert effective.budget_duration == template.budget_duration
    assert effective_member_budget(template, override, NOW, NOW) == template
    assert effective_member_budget(template, Budget(budget_limits=()), None, NOW) == template


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


@pytest.mark.parametrize("column", ("budget_limits", "model_tpm_limit", "model_rpm_limit"))
def test_null_budget_json_fields_are_encoded_as_prisma_json_null(column: str) -> None:
    values: Final = budget_row_values({"max_budget": 1.0, column: None})
    assert values["max_budget"] == 1.0
    encoded: Final = values[column]
    assert isinstance(encoded, Json)
    assert encoded.data is None
