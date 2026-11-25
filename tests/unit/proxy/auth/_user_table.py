"""An in-memory LiteLLM_UserTable that answers the queries the sign-in user lookup makes."""

import re
from collections.abc import Iterator, Mapping, MutableMapping
from types import SimpleNamespace
from typing import Final

from litellm.proxy._types import LiteLLM_UserTable
from litellm.repositories import rust_database
from litellm.rust_bridge._native import RustUserRepository
from tests.unit.proxy._prisma import prisma_with_empty_config
from tests.unit.proxy.rust_auth_stores import FakeUserStore

_ILIKE_TOKEN: Final = re.compile(r"\\(.)|(%)|(_)|(.)", re.DOTALL)


def _ilike_piece(escaped: str, percent: str, underscore: str, char: str) -> str:
    if escaped:
        return re.escape(escaped)
    if percent:
        return ".*"
    if underscore:
        return "."
    return re.escape(char)


def _ilike(value: str, pattern: str) -> bool:
    """Postgres ILIKE, which Prisma uses for an insensitive `equals`: `%` and `_` are wildcards, `\\` escapes."""
    regex: Final = "".join(_ilike_piece(*token) for token in _ILIKE_TOKEN.findall(pattern))
    return re.fullmatch(regex, value, flags=re.IGNORECASE | re.DOTALL) is not None


def _holds(value: object, condition: object) -> bool:
    if isinstance(condition, Mapping) and "equals" in condition:
        wanted: Final = str(condition["equals"])
        if not isinstance(value, str):
            return False
        return _ilike(value, wanted) if condition.get("mode") == "insensitive" else value == wanted
    if isinstance(condition, Mapping) and "not" in condition:
        return value != condition["not"]
    return value == condition


def _matches(row: LiteLLM_UserTable, where: Mapping[str, object]) -> bool:
    return all(_holds(getattr(row, field), condition) for field, condition in where.items())


class UserTable:
    def __init__(self, *rows: LiteLLM_UserTable) -> None:
        self.rows = list(rows)
        self.first_lookups: list[Mapping[str, object]] = []
        self.email_checks: list[tuple[str, str]] = []

    def row(self, user_id: str) -> LiteLLM_UserTable:
        return next(r for r in self.rows if r.user_id == user_id)

    def with_email(self, email: str) -> list[str]:
        return [r.user_id for r in self.rows if (r.user_email or "").casefold() == email.casefold()]

    async def find_unique(self, where: Mapping[str, object], include: object = None) -> LiteLLM_UserTable | None:
        return next((r for r in self.rows if _matches(r, where)), None)

    async def find_first(self, where: Mapping[str, object], include: object = None) -> LiteLLM_UserTable | None:
        self.first_lookups.append(where)
        return next((r for r in self.rows if _matches(r, where)), None)

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> LiteLLM_UserTable | None:
        row: Final = await self.find_unique(where)
        if row is None:
            return None
        updated: Final = row.model_copy(update=dict(data))
        self.rows[self.rows.index(row)] = updated
        return updated

    async def update_many(self, where: Mapping[str, object], data: Mapping[str, object]) -> int:
        matched: Final = [r for r in self.rows if _matches(r, where)]
        for row in matched:
            self.rows[self.rows.index(row)] = row.model_copy(update=dict(data))
        return len(matched)

    async def count(self, where: Mapping[str, object] | None = None) -> int:
        return sum(1 for r in self.rows if _matches(r, where or {}))

    async def upsert(self, where: Mapping[str, object], data: Mapping[str, Mapping[str, object]]) -> LiteLLM_UserTable:
        existing: Final = await self.find_unique(where)
        return existing if existing is not None else await self.create(data["create"])

    async def create(self, data: Mapping[str, object], include: object = None) -> LiteLLM_UserTable:
        columns: Final = ("user_id", "user_email", "sso_user_id", "user_role")
        row: Final = LiteLLM_UserTable(organization_memberships=[], **{k: data[k] for k in columns if k in data})
        self.rows.append(row)
        return row


def user_row(user_id: str, email: str | None, sso_user_id: str | None = None) -> LiteLLM_UserTable:
    return LiteLLM_UserTable(
        user_id=user_id,
        user_email=email,
        sso_user_id=sso_user_id,
        user_role="internal_user",
        organization_memberships=[],
    )


def prisma_over(table: UserTable) -> SimpleNamespace:
    async def insert_data(data: Mapping[str, object], table_name: str, **_: object) -> LiteLLM_UserTable:
        assert table_name == "user"
        return await table.create(data)

    _follow(table)
    database: Final = prisma_with_empty_config().db
    database.litellm_usertable = table
    return SimpleNamespace(db=database, writer_db=database, insert_data=insert_data, jsonify_object=dict)


class _RowsById(MutableMapping[str, Mapping[str, object] | Exception]):
    """``UserTable``'s rows keyed by ``user_id``, as the Rust user store fake reads and writes them."""

    def __init__(self, table: UserTable) -> None:
        self._table: Final = table

    def __getitem__(self, user_id: str) -> Mapping[str, object]:
        row: Final = next((r for r in self._table.rows if r.user_id == user_id), None)
        if row is None:
            raise KeyError(user_id)
        return row.model_dump()

    def __setitem__(self, user_id: str, value: Mapping[str, object] | Exception) -> None:
        assert not isinstance(value, Exception)
        updated: Final = LiteLLM_UserTable(**value)
        rows: Final = self._table.rows
        index: Final = next((i for i, r in enumerate(rows) if r.user_id == user_id), None)
        if index is None:
            rows.append(updated)
        else:
            rows[index] = updated

    def __delitem__(self, user_id: str) -> None:
        self._table.rows.remove(self._table.row(user_id))

    def __iter__(self) -> Iterator[str]:
        return iter([r.user_id for r in self._table.rows])

    def __len__(self) -> int:
        return len(self._table.rows)


def rows_by_id(table: UserTable) -> MutableMapping[str, Mapping[str, object] | Exception]:
    return _RowsById(table)


def user_store_over(table: UserTable, store: type[FakeUserStore] = FakeUserStore) -> FakeUserStore:
    """The Rust user store fake over ``table``, so a sign-in the store resolves and a row Prisma creates meet."""
    return store(rows=rows_by_id(table))


def _follow(table: UserTable) -> None:
    """Points an installed Rust user store fake at ``table``, so lookups and Prisma writes share its rows."""
    installed: Final = rust_database.repository(RustUserRepository)
    if isinstance(installed, FakeUserStore):
        installed.rows = rows_by_id(table)
        installed.email_checks = table.email_checks
