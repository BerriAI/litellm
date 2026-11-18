import asyncio
import copy
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from types import TracebackType
from typing import Final

from prisma.errors import UniqueViolationError
from pydantic import BaseModel, TypeAdapter

from litellm.proxy.mcp_server.db import LOADABLE_MCP_SERVER_FILTER


class FakeRow(BaseModel, extra="allow", frozen=True):
    pass


_SERVER_DEFAULTS: Final[Mapping[str, object]] = {
    "transport": "http",
    "credentials": "{}",
    "mcp_access_groups": [],
    "allowed_tools": [],
    "extra_headers": [],
    "args": [],
    "env": {},
    "env_vars": [],
    "approval_status": "active",
    **dict.fromkeys(
        (
            "auth_type",
            "url",
            "spec_path",
            "description",
            "issuer",
            "authorization_url",
            "token_url",
            "registration_url",
            "oauth2_flow",
            "token_exchange_endpoint",
            "audience",
            "subject_token_type",
            "token_exchange_profile",
            "alias",
            "mcp_info",
        )
    ),
}


_CREATED_ROWS: Final = TypeAdapter(tuple[Mapping[str, object], ...])
_IDENTIFIER_WHERE: Final = TypeAdapter(tuple[dict[str, list[dict[str, dict[str, str] | str | None]]], ...])
_JSON_COLUMNS: Final = ("credentials", "mcp_info", "static_headers", "env", "env_vars")


def _as_read(row: Mapping[str, object]) -> dict[str, object]:
    return {k: json.loads(v) if k in _JSON_COLUMNS and isinstance(v, str) else v for k, v in row.items()}


def _violation(fields: str) -> UniqueViolationError:
    return UniqueViolationError(
        {"user_facing_error": {"error_code": "P2002", "message": f"Unique constraint failed on {fields}"}}
    )


class _ProfileTable:
    def __init__(self, store: "FakeMcpPrisma") -> None:
        self._store = store

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> FakeRow:
        row: Final = next(row for row in self._store.profiles if row["profile_id"] == where["profile_id"])
        row.update(data)
        return FakeRow(**_as_read(row))

    async def find_many(
        self, where: Mapping[str, object] | None = None, include: Mapping[str, object] | None = None
    ) -> Sequence[FakeRow]:
        if where is None:
            return [self._with_server(row, include) for row in self._store.profiles]
        if "server" in where:
            assert where == {"server": {"is": LOADABLE_MCP_SERVER_FILTER}}, where
            return [
                self._with_server(row, include)
                for row in self._store.profiles
                if next(s for s in self._store.servers if s["server_id"] == row["server_id"])["approval_status"]
                in (None, "active", "approved")
            ]
        assert len(where) == 1, "profile lookups filter on one column"
        column, condition = next(iter(where.items()))
        assert column in ("server_id", "profile_id_normalized"), where
        wanted: Final = condition if isinstance(condition, Mapping) else {"in": [condition]}
        assert set(wanted) == {"in"}, f"inexact match requested: {where}"
        found: Final = [FakeRow(**_as_read(row)) for row in self._store.profiles if row[column] in wanted["in"]]
        if column in self._store.held_lookup_columns:
            await self._store.wait_for_concurrent_lookups()
        return found

    async def delete_many(self, where: Mapping[str, object]) -> int:
        kept: Final = [row for row in self._store.profiles if row["server_id"] != where["server_id"]]
        deleted: Final = len(self._store.profiles) - len(kept)
        self._store.profiles[:] = kept
        return deleted

    async def create_many(self, data: Sequence[Mapping[str, object]]) -> int:
        if self._store.vanishing_profile_conflicts:
            self._store.vanishing_profile_conflicts -= 1
            raise _violation("profile_id_normalized")
        for created in data:
            if any(
                row["profile_id"] == created["profile_id"]
                or row["profile_id_normalized"] == created["profile_id_normalized"]
                for row in self._store.profiles
            ):
                raise _violation("profile_id_normalized")
            self._store.profiles.append({"created_at": self._store.next_created_at(), **created})
        return len(data)

    def _with_server(self, row: Mapping[str, object], include: Mapping[str, object] | None) -> FakeRow:
        if not (include or {}).get("server"):
            return FakeRow(**_as_read(row))
        server: Final = next((s for s in self._store.servers if s["server_id"] == row["server_id"]), None)
        return FakeRow(**_as_read(row), server=None if server is None else FakeRow(**_as_read(server)))


class _ServerTable:
    def __init__(self, store: "FakeMcpPrisma") -> None:
        self._store = store

    def _as_row(self, row: Mapping[str, object]) -> FakeRow:
        return FakeRow(**_as_read(row))

    def _apply_nested(self, server_id: str, nested: Mapping[str, object]) -> None:
        for created in _CREATED_ROWS.validate_python(nested.get("create", ())):
            if any(
                p["profile_id"] == created["profile_id"]
                or p["profile_id_normalized"] == created["profile_id_normalized"]
                for p in self._store.profiles
            ):
                raise _violation("profile_id_normalized")
            self._store.profiles.append(
                {"created_at": self._store.next_created_at(), **created, "server_id": server_id}
            )

    def _find(self, server_id: object) -> dict[str, object] | None:
        return next((row for row in self._store.servers if row["server_id"] == server_id), None)

    async def create(self, data: Mapping[str, object], include: Mapping[str, object] | None = None) -> FakeRow:
        if self._find(data["server_id"]) is not None:
            raise _violation("server_id")
        row: Final = {
            **_SERVER_DEFAULTS,
            "updated_at": self._store.next_updated_at(),
            **{k: v for k, v in data.items() if k != "auth_profiles"},
        }
        self._store.servers.append(row)
        nested: Final = data.get("auth_profiles")
        if isinstance(nested, Mapping):
            self._apply_nested(str(data["server_id"]), nested)
        return self._as_row(row)

    async def update(
        self, data: Mapping[str, object], where: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> FakeRow | None:
        assert "auth_profiles" not in data, (
            "Prisma runs nested writes in the order sent, so profiles are replaced by ordered statements"
        )
        row: Final = self._find(where["server_id"])
        if row is None:
            return None
        row.update(data)
        return self._as_row(row)

    async def find_unique(
        self, where: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> FakeRow | None:
        row: Final = self._find(where["server_id"])
        return None if row is None else self._as_row(row)

    async def find_first(self, where: Mapping[str, object] | None = None, **_: object) -> FakeRow | None:
        if not self._store.identifier_checks or where is None:
            return None
        name_match, _approval, *own_row_guards = _IDENTIFIER_WHERE.validate_python(where["AND"])
        wanted: Final = str(name_match["OR"][0]["server_name"]["equals"]).lower()
        excluded: Final = frozenset(guard["NOT"][0]["server_id"] for guard in own_row_guards)
        return next(
            (
                self._as_row(row)
                for row in self._store.servers
                if row["server_id"] not in excluded
                and wanted in {str(row.get("server_name") or "").lower(), str(row.get("alias") or "").lower()}
            ),
            None,
        )

    async def find_many(
        self, where: Mapping[str, object] | None = None, include: Mapping[str, object] | None = None, **_: object
    ) -> Sequence[FakeRow]:
        return [self._as_row(row) for row in self._store.servers]


class _ConfigTable:
    async def find_unique(self, **_: object) -> None:
        return None


class _Transaction:
    def __init__(self, store: "FakeMcpPrisma") -> None:
        self._store = store
        self.litellm_mcpservertable = store.db.litellm_mcpservertable
        self.litellm_mcpauthprofiletable = store.db.litellm_mcpauthprofiletable
        self._snapshot: tuple[list[dict[str, object]], list[dict[str, object]]] | None = None

    async def execute_raw(self, query: str, *args: object) -> int:
        return 0

    async def __aenter__(self) -> "_Transaction":
        self._snapshot = (copy.deepcopy(self._store.servers), copy.deepcopy(self._store.profiles))
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> bool:
        if exc_type is not None and self._snapshot is not None:
            self._store.servers[:], self._store.profiles[:] = self._snapshot
        return False


class _Db:
    def __init__(self, store: "FakeMcpPrisma") -> None:
        self.litellm_mcpservertable = _ServerTable(store)
        self.litellm_mcpauthprofiletable = _ProfileTable(store)
        self.litellm_config = _ConfigTable()
        self._store = store

    def tx(self) -> _Transaction:
        return _Transaction(self._store)


class FakeMcpPrisma:
    def __init__(self) -> None:
        self.servers: list[dict[str, object]] = []
        self.profiles: list[dict[str, object]] = []
        self.db = _Db(self)
        self.writer_db = self.db
        self._hold_until: int | None = None
        self.held_lookup_columns: frozenset[str] = frozenset()
        self._arrived = 0
        self._release: asyncio.Event | None = None
        self._writes = 0
        self._profile_creates = 0
        self.identifier_checks = False
        self.vanishing_profile_conflicts = 0

    def next_updated_at(self) -> datetime:
        self._writes += 1
        return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=self._writes)

    def next_created_at(self) -> datetime:
        self._profile_creates += 1
        return datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=self._profile_creates)

    def hold_profile_lookups_until(
        self, concurrent_lookups: int, columns: frozenset[str] = frozenset({"profile_id_normalized"})
    ) -> None:
        self._hold_until = concurrent_lookups
        self.held_lookup_columns = columns
        self._release = asyncio.Event()

    async def wait_for_concurrent_lookups(self) -> None:
        if self._hold_until is None or self._release is None:
            return
        self._arrived += 1
        if self._arrived >= self._hold_until:
            self._release.set()
        await self._release.wait()

    def lag_profile_reads(self) -> None:
        self.db = _LaggingReplica(self.writer_db)


class _UnreplicatedProfileTable:
    async def find_many(self, **_: object) -> Sequence[FakeRow]:
        return []


class _LaggingReplica:
    def __init__(self, writer: _Db) -> None:
        self._writer = writer
        self.litellm_mcpauthprofiletable = _UnreplicatedProfileTable()

    def __getattr__(self, name: str) -> object:
        return getattr(self._writer, name)

    def tx(self) -> _Transaction:
        transaction: Final = self._writer.tx()
        transaction.litellm_mcpauthprofiletable = self._writer.litellm_mcpauthprofiletable
        return transaction
