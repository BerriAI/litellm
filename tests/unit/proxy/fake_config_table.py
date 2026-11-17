import asyncio
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Final


@dataclass(frozen=True, slots=True)
class StoredConfigRow:
    param_name: str
    param_value: dict[str, object]


class ConfigRows:

    def __init__(self, rows: Mapping[str, Mapping[str, object]]) -> None:
        self.serialized_rows: dict[str, str] = {name: json.dumps(dict(value)) for name, value in rows.items()}

    async def find_first(self, *, where: Mapping[str, str]) -> StoredConfigRow | None:
        name: Final = where["param_name"]
        serialized: Final = self.serialized_rows.get(name)
        await asyncio.sleep(0)
        return None if serialized is None else StoredConfigRow(param_name=name, param_value=json.loads(serialized))

    async def upsert(self, *, where: Mapping[str, str], data: Mapping[str, Mapping[str, str]]) -> None:
        self.serialized_rows[where["param_name"]] = data["update"]["param_value"]

    def stored(self, name: str) -> dict[str, object]:
        return json.loads(self.serialized_rows[name])


def _status_matches(status: str | None, condition: object) -> bool:
    if isinstance(condition, Mapping) and "in" in condition:
        return status in condition["in"]
    if isinstance(condition, Mapping) and "not" in condition:
        return status is not None and status != condition["not"]
    return status == condition


def _any_status_matches(status: str | None, conditions: object) -> bool:
    assert isinstance(conditions, Sequence)
    return any(_status_matches(status, condition["approval_status"]) for condition in conditions)


class RowsWithIds:

    def __init__(
        self, id_field: str, ids: Iterable[str], *, approval_statuses: Mapping[str, str] = MappingProxyType({})
    ) -> None:
        self._id_field: Final = id_field
        self._ids: Final = frozenset(ids)
        self._approval_statuses: Final = approval_statuses

    def _row_matches(self, row_id: str, where: Mapping[str, object]) -> bool:
        status: Final = self._approval_statuses.get(row_id)
        return (
            row_id in self._ids
            and ("OR" not in where or _any_status_matches(status, where["OR"]))
            and ("approval_status" not in where or _status_matches(status, where["approval_status"]))
        )

    async def find_many(self, *, where: Mapping[str, object]) -> list[SimpleNamespace]:
        id_filter: Final = where[self._id_field]
        assert isinstance(id_filter, Mapping)
        requested: Final = id_filter["in"]
        if not all(isinstance(row_id, str) for row_id in requested):
            raise ValueError(f"Invalid argument type. `{self._id_field}.in` should be a list of String")
        return [SimpleNamespace(**{self._id_field: row_id}) for row_id in requested if self._row_matches(row_id, where)]


class SectionLockedTransaction:

    def __init__(self, litellm_config: object, section_lock: asyncio.Lock) -> None:
        self.litellm_config: Final = litellm_config
        self._section_lock: Final = section_lock
        self._holds_section_lock = False

    async def __aenter__(self) -> "SectionLockedTransaction":
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._holds_section_lock:
            self._section_lock.release()

    async def query_raw(self, query: str, section_name: str) -> list[dict[str, int]]:
        await self._section_lock.acquire()
        self._holds_section_lock = True
        return [{"locked": 1}]


def section_locked_transactions(litellm_config: object) -> Callable[[], SectionLockedTransaction]:
    section_lock: Final = asyncio.Lock()
    return lambda: SectionLockedTransaction(litellm_config, section_lock)


class ReplicaPrisma:

    def __init__(
        self,
        rows: ConfigRows,
        cached_rows: Mapping[str, Mapping[str, object]] | None = None,
        tables: Mapping[str, object] | None = None,
    ) -> None:
        self.rows: Final = rows
        self._cached_rows: Final = cached_rows
        self.tx: Final = section_locked_transactions(rows)
        self.db: Final = SimpleNamespace(litellm_config=rows, **(tables or {}))
        self.writer_db: Final = self.db

    async def get_generic_data(self, key: str, value: str, table_name: str) -> SimpleNamespace | None:
        cached: Final = (
            self._cached_rows.get(value)
            if self._cached_rows is not None
            else (self.rows.stored(value) if value in self.rows.serialized_rows else None)
        )
        return None if cached is None else SimpleNamespace(param_name=value, param_value=dict(cached))
