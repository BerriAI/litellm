"""An in-memory LiteLLM_UserTable that answers the queries the sign-in user lookup makes."""

import re
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Final

from litellm.proxy._types import LiteLLM_UserTable

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

    database: Final = SimpleNamespace(litellm_usertable=table)
    return SimpleNamespace(db=database, writer_db=database, insert_data=insert_data, jsonify_object=dict)
