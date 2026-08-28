from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType

import pytest

from litellm.proxy.list_api.in_memory import Cells, InMemoryListExecutor
from litellm.proxy.list_api.list_framework import (
    AnyOf,
    Compare,
    IsNull,
    QueryPlan,
    SortKey,
    Within,
)


@dataclass(frozen=True, slots=True)
class Row:
    name: str
    size: float | None = None
    tags: tuple[str | None, ...] = ()
    seen_at: datetime | None = None


def _cells(row: Row) -> Cells:
    return MappingProxyType({"name": row.name, "size": row.size, "tags": row.tags, "seen_at": row.seen_at})


def _executor(*rows: Row, **kwargs) -> InMemoryListExecutor[Row]:
    return InMemoryListExecutor(rows=rows, cells=_cells, **kwargs)


def _plan(where=(), order=(SortKey(field="name", descending=False),), skip=0, take=50) -> QueryPlan:
    return QueryPlan(where=where, order=order, skip=skip, take=take)


async def _names(executor: InMemoryListExecutor[Row], plan: QueryPlan) -> list[str]:
    return [row.name for row in await executor.find_many(plan)]


@pytest.mark.asyncio
async def test_the_page_is_sliced_after_the_sort_not_before():
    executor = _executor(Row("c"), Row("a"), Row("b"), Row("d"))

    assert await _names(executor, _plan(skip=1, take=2)) == ["b", "c"]


@pytest.mark.asyncio
async def test_count_ignores_the_page_and_counts_the_match_set():
    executor = _executor(*(Row(f"r{index}") for index in range(7)))

    assert await executor.count(()) == 7
    assert len(await executor.find_many(_plan(take=3))) == 3


@pytest.mark.asyncio
async def test_nulls_sort_last_in_both_directions():
    """`order_by_sql` renders NULLS LAST both ways; an in-memory plan has to agree."""
    executor = _executor(Row("small", size=1.0), Row("unsized"), Row("big", size=9.0))

    ascending = SortKey(field="size", descending=False)
    descending = SortKey(field="size", descending=True)
    assert await _names(executor, _plan(order=(ascending,))) == ["small", "big", "unsized"]
    assert await _names(executor, _plan(order=(descending,))) == ["big", "small", "unsized"]


@pytest.mark.asyncio
async def test_the_last_sort_key_breaks_ties_in_the_first():
    executor = _executor(Row("b", size=1.0), Row("a", size=1.0), Row("c", size=0.0))

    order = (SortKey(field="size", descending=False), SortKey(field="name", descending=False))

    assert await _names(executor, _plan(order=order)) == ["c", "a", "b"]


@pytest.mark.asyncio
async def test_a_predicate_holds_when_any_element_of_a_repeated_field_matches():
    executor = _executor(Row("azure", tags=("azure", "bedrock")), Row("openai", tags=("openai",)))

    where = (Compare(field="tags", op="contains", value="bedrock"),)

    assert await _names(executor, _plan(where=where)) == ["azure"]


@pytest.mark.asyncio
async def test_a_repeated_field_with_no_elements_matches_nothing():
    executor = _executor(Row("untagged"))

    where = (Compare(field="tags", op="contains", value="anything"),)

    assert await _names(executor, _plan(where=where)) == []


@pytest.mark.asyncio
async def test_a_repeated_field_is_matched_element_by_element_not_as_one_string():
    """Without the per-element lift the tuple stringifies, and its punctuation becomes matchable."""
    executor = _executor(Row("azure", tags=("azure", "bedrock")))

    where = (Compare(field="tags", op="contains", value="e', 'b"),)

    assert await _names(executor, _plan(where=where)) == []


@pytest.mark.asyncio
async def test_within_matches_an_element_of_a_repeated_field():
    executor = _executor(Row("azure", tags=("azure", "bedrock")), Row("openai", tags=("openai",)))

    where = (Within(field="tags", values=("bedrock",)),)

    assert await _names(executor, _plan(where=where)) == ["azure"]


@pytest.mark.asyncio
async def test_contains_is_case_insensitive_like_ilike():
    executor = _executor(Row("GPT-5"), Row("claude-opus"))

    where = (Compare(field="name", op="contains", value="gpt"),)

    assert await _names(executor, _plan(where=where)) == ["GPT-5"]


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["eq", "not", "gt", "gte", "lt", "lte", "contains"])
async def test_a_null_cell_satisfies_no_comparison(op: str):
    """SQL's three-valued logic: `col <> 1` does not return NULL rows, so neither does this."""
    executor = _executor(Row("unsized"))

    where = (Compare(field="size", op=op, value=1.0),)

    assert await _names(executor, _plan(where=where)) == []


@pytest.mark.asyncio
async def test_is_null_is_the_way_to_ask_for_the_null_rows():
    executor = _executor(Row("unsized"), Row("sized", size=2.0))

    assert await _names(executor, _plan(where=(IsNull(field="size", negated=False),))) == ["unsized"]
    assert await _names(executor, _plan(where=(IsNull(field="size", negated=True),))) == ["sized"]


@pytest.mark.asyncio
async def test_is_null_reads_a_repeated_field_element_by_element_too():
    """Every other predicate lifts over a repeated field; `is_null` reading the container
    instead would make a field holding only nulls indistinguishable from a populated one."""
    executor = _executor(Row("only_nulls", tags=(None,)), Row("populated", tags=("openai",)))

    assert await _names(executor, _plan(where=(IsNull(field="tags", negated=False),))) == ["only_nulls"]
    assert await _names(executor, _plan(where=(IsNull(field="tags", negated=True),))) == ["populated"]


@pytest.mark.asyncio
async def test_ordering_comparisons_work_across_the_cell_types():
    when = datetime(2026, 8, 1, tzinfo=timezone.utc)
    executor = _executor(Row("early", seen_at=when), Row("late", seen_at=datetime(2026, 9, 1, tzinfo=timezone.utc)))

    where = (Compare(field="seen_at", op="gt", value=when),)

    assert await _names(executor, _plan(where=where)) == ["late"]


@pytest.mark.asyncio
async def test_a_value_of_the_wrong_type_matches_nothing_rather_than_raising():
    executor = _executor(Row("a", size=1.0))

    where = (Compare(field="size", op="gt", value="not-a-number"),)

    assert await _names(executor, _plan(where=where)) == []


@pytest.mark.asyncio
async def test_within_matches_any_of_its_values():
    executor = _executor(Row("a"), Row("b"), Row("c"))

    where = (Within(field="name", values=("a", "c")),)

    assert await _names(executor, _plan(where=where)) == ["a", "c"]


@pytest.mark.asyncio
async def test_any_of_is_a_disjunction_and_the_plan_is_a_conjunction():
    executor = _executor(Row("alpha", size=1.0), Row("beta", size=1.0), Row("alpha-2", size=9.0))

    where = (
        Compare(field="size", op="lte", value=5.0),
        AnyOf(clauses=(Compare(field="name", op="contains", value="alpha"),)),
    )

    assert await _names(executor, _plan(where=where)) == ["alpha"]


@pytest.mark.asyncio
async def test_enrich_page_sees_the_page_and_only_the_page():
    seen: list[tuple[str, ...]] = []

    async def _record(rows: Sequence[Row]) -> Sequence[Row]:
        seen.append(tuple(row.name for row in rows))
        return rows

    executor = _executor(*(Row(f"r{index:02d}") for index in range(20)), enrich_page=_record)

    await executor.find_many(_plan(skip=5, take=3))

    assert seen == [("r05", "r06", "r07")]


@pytest.mark.asyncio
async def test_enrich_page_can_replace_the_rows_it_is_given():
    async def _rename(rows: Sequence[Row]) -> Sequence[Row]:
        return tuple(Row(f"{row.name}!") for row in rows)

    executor = _executor(Row("a"), Row("b"), enrich_page=_rename)

    assert await _names(executor, _plan()) == ["a!", "b!"]


@pytest.mark.asyncio
async def test_counting_never_enriches():
    async def _explode(rows: Sequence[Row]) -> Sequence[Row]:
        raise AssertionError("count must not resolve anything a row does not already carry")

    executor = _executor(Row("a"), Row("b"), enrich_page=_explode)

    assert await executor.count(()) == 2


@pytest.mark.asyncio
async def test_rows_pass_through_untouched_without_an_enricher():
    executor = _executor(Row("a"), Row("b"))

    assert await _names(executor, _plan()) == ["a", "b"]
