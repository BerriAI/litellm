from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import pytest
from prisma import models as prisma_models
from prisma.builder import QueryBuilder

from litellm.repositories.bounded_in import (
    IN_LIST_CHUNK_SIZE,
    SameFieldFilterError,
    count_in,
    delete_many_in,
    find_many_in,
    update_many_in,
)

SIZES: Final = (0, 1, 5_000, 5_001, 12_345)


def _matches(row: Mapping[str, object], where: Mapping[str, object]) -> bool:
    def clause(key: str, condition: object) -> bool:
        if key == "AND":
            return all(_matches(row, part) for part in condition)
        if isinstance(condition, Mapping):
            return row[key] in condition["in"]
        return row[key] == condition

    return all(clause(key, condition) for key, condition in where.items())


@dataclass
class FakeTable:
    """Evaluates the filters it is sent against in-memory rows, and records each one."""

    rows: list[dict[str, object]]
    filters: list[Mapping[str, object]] = field(default_factory=list)

    def _select(self, where: Mapping[str, object]) -> list[dict[str, object]]:
        self.filters.append(where)
        return [row for row in self.rows if _matches(row, where)]

    async def find_many(self, *, where: Mapping[str, object]) -> Sequence[dict[str, object]]:
        return self._select(where)

    async def count(self, *, where: Mapping[str, object]) -> int:
        return len(self._select(where))

    async def update_many(self, *, data: Mapping[str, object], where: Mapping[str, object]) -> int:
        selected = self._select(where)
        for row in selected:
            row.update(data)
        return len(selected)

    async def delete_many(self, *, where: Mapping[str, object]) -> int:
        selected = self._select(where)
        self.rows = [row for row in self.rows if row not in selected]
        return len(selected)

    def in_list_sizes(self) -> list[int]:
        return [len(_membership(where)["in"]) for where in self.filters]


def _membership(where: Mapping[str, object]) -> Mapping[str, Sequence[object]]:
    inner = where["AND"][1] if "AND" in where else where
    ((_, condition),) = inner.items()
    return condition


def _table(size: int) -> FakeTable:
    return FakeTable(rows=[{"id": f"id-{n}", "team": "even" if n % 2 == 0 else "odd"} for n in range(size + 10)])


def _ids(size: int) -> list[str]:
    return [f"id-{n}" for n in range(size)]


def _expected_chunks(size: int) -> list[int]:
    return [min(IN_LIST_CHUNK_SIZE, size - start) for start in range(0, size, IN_LIST_CHUNK_SIZE)]


@pytest.mark.parametrize("size", SIZES)
async def test_find_many_in_returns_every_matching_row_in_bounded_chunks(size: int) -> None:
    table = _table(size)
    rows = await find_many_in(table, "id", _ids(size))
    assert [row["id"] for row in rows] == _ids(size)
    assert table.in_list_sizes() == _expected_chunks(size)


@pytest.mark.parametrize("size", SIZES)
async def test_count_in_sums_the_chunk_counts(size: int) -> None:
    table = _table(size)
    assert await count_in(table, "id", _ids(size)) == size
    assert table.in_list_sizes() == _expected_chunks(size)


@pytest.mark.parametrize("size", SIZES)
async def test_update_many_in_updates_every_row_and_sums_counts(size: int) -> None:
    table = _table(size)
    updated = await update_many_in(table, "id", _ids(size), data={"team": "moved"}, atomicity="per_chunk_ok")
    assert updated == size
    assert [row["id"] for row in table.rows if row["team"] == "moved"] == _ids(size)
    assert table.in_list_sizes() == _expected_chunks(size)


@pytest.mark.parametrize("size", SIZES)
async def test_delete_many_in_deletes_every_row_and_sums_counts(size: int) -> None:
    table = _table(size)
    deleted = await delete_many_in(table, "id", _ids(size), atomicity="caller_transaction")
    assert deleted == size
    assert [row["id"] for row in table.rows] == [f"id-{n}" for n in range(size, size + 10)]
    assert table.in_list_sizes() == _expected_chunks(size)


async def test_an_empty_list_sends_no_query() -> None:
    table = _table(0)
    assert await find_many_in(table, "id", []) == ()
    assert await count_in(table, "id", []) == 0
    assert await update_many_in(table, "id", [], data={"team": "x"}, atomicity="per_chunk_ok") == 0
    assert await delete_many_in(table, "id", [], atomicity="per_chunk_ok") == 0
    assert table.filters == []


async def test_duplicate_values_are_sent_once_in_first_seen_order() -> None:
    table = _table(IN_LIST_CHUNK_SIZE + 1)
    values = [*reversed(_ids(IN_LIST_CHUNK_SIZE + 1)), *_ids(IN_LIST_CHUNK_SIZE + 1)]
    assert await count_in(table, "id", values) == IN_LIST_CHUNK_SIZE + 1
    sent = [value for where in table.filters for value in _membership(where)["in"]]
    assert sent == list(reversed(_ids(IN_LIST_CHUNK_SIZE + 1)))


async def test_where_is_anded_with_each_chunk() -> None:
    table = _table(12_345)
    where = {"team": "even"}
    rows = await find_many_in(table, "id", _ids(12_345), where=where)
    assert [row["id"] for row in rows] == [f"id-{n}" for n in range(0, 12_345, 2)]
    assert [set(where_sent) for where_sent in table.filters] == [{"AND"}] * 3
    assert all(where_sent["AND"][0] == where for where_sent in table.filters)
    assert table.in_list_sizes() == _expected_chunks(12_345)


@pytest.mark.parametrize(
    "where",
    [
        {"id": "id-1"},
        {"id": {"not": "id-1"}},
        {"AND": [{"team": "even"}, {"id": {"in": ["id-1"]}}]},
        {"OR": ({"id": "id-1"},)},
        {"NOT": {"id": "id-1"}},
        {"AND": [{"OR": [{"NOT": {"id": "id-1"}}]}]},
    ],
)
async def test_where_filtering_the_chunked_field_is_refused_before_any_query(where: Mapping[str, object]) -> None:
    table = _table(3)
    with pytest.raises(SameFieldFilterError, match="`id`"):
        await count_in(table, "id", _ids(3), where=where)
    assert table.filters == []


async def test_writes_require_an_atomicity_decision() -> None:
    table = _table(1)
    with pytest.raises(TypeError, match="atomicity"):
        await update_many_in(table, "id", _ids(1), data={"team": "x"})  # pyright: ignore[reportCallIssue]  # the missing argument is the test
    with pytest.raises(TypeError, match="atomicity"):
        await delete_many_in(table, "id", _ids(1))  # pyright: ignore[reportCallIssue]  # the missing argument is the test
    assert table.filters == []


def _find_many_query(where: Mapping[str, object]) -> str:
    return QueryBuilder(
        method="find_many", model=prisma_models.LiteLLM_Config, arguments={"where": where}
    ).build_query()


async def test_the_composed_filter_renders_like_a_hand_written_prisma_filter() -> None:
    table = FakeTable(rows=[{"param_name": "a", "param_value": 1}])
    await find_many_in(table, "param_name", ["a", "b", "a"], where={"param_value": 1})
    hand_written = {"AND": [{"param_value": 1}, {"param_name": {"in": ["a", "b"]}}]}
    assert _find_many_query(table.filters[0]) == _find_many_query(hand_written)
