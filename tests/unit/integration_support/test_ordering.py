from __future__ import annotations

from itertools import groupby
from types import MappingProxyType
from typing import Final

import pytest

from tests.integration._support.ordering import (
    CollectedCase,
    ScopedParameter,
    case_shards,
    fixture_groups,
    parse_timings,
)
from tests.integration.conftest import _order_key


def test_seeded_order_reuses_each_scoped_configuration_without_dropping_cases() -> None:
    values: Final = (
        "D1",
        "D1",
        "D1",
        "D1",
        "D1",
        "D2",
        "D2",
        "D2",
        "D2",
        "D2",
        "D3",
        "D3",
        "D3",
        "D3",
        "D4",
        "D4",
        "D4",
        "D4",
    )
    cases: Final = tuple(
        CollectedCase(
            f"tests/integration/security/test_slots.py::test_route[{value}-{index}]",
            (ScopedParameter("module:slots", "rig", value),),
        )
        for index, value in enumerate(values)
    )
    groups: Final = fixture_groups(cases)
    ordered: Final = tuple(sorted(cases, key=lambda case: _order_key(1848224111, case.nodeid, groups[case.nodeid])))
    configurations: Final = tuple(value for value, _ in groupby(case.parameters[0].value for case in ordered))
    assert len(configurations) == len(set(values))
    assert set(configurations) == set(values)
    assert frozenset(case.nodeid for case in ordered) == frozenset(case.nodeid for case in cases)
    assert len(ordered) == len(cases)
    assert ordered == tuple(sorted(cases, key=lambda case: _order_key(1848224111, case.nodeid, groups[case.nodeid])))
    assert ordered != tuple(sorted(cases, key=lambda case: _order_key(984872314, case.nodeid, groups[case.nodeid])))


def test_equal_unhashable_parameters_share_a_group_only_within_their_fixture_owner() -> None:
    first: Final = CollectedCase("test_slots.py::test_first", (ScopedParameter("class:first", "rig", {"slot": "D1"}),))
    equal: Final = CollectedCase("test_slots.py::test_equal", (ScopedParameter("class:first", "rig", {"slot": "D1"}),))
    other_owner: Final = CollectedCase(
        "test_slots.py::test_other", (ScopedParameter("class:second", "rig", {"slot": "D1"}),)
    )
    other_file: Final = CollectedCase("test_other.py::test_first", first.parameters)
    plain: Final = CollectedCase("test_slots.py::test_plain")
    groups: Final = fixture_groups((first, equal, other_owner, other_file, plain))
    assert groups[first.nodeid] == groups[equal.nodeid]
    assert groups[first.nodeid] != groups[other_owner.nodeid]
    assert groups[first.nodeid] != groups[other_file.nodeid]
    assert groups[plain.nodeid] == ""


def test_shards_balance_cases_without_splitting_a_scoped_configuration() -> None:
    cases: Final = tuple(
        CollectedCase(
            f"test_slots.py::test_route[{index}]",
            (ScopedParameter("module:slots", "rig", index // 4),),
            ("module:slots:extra",) if index % 2 else (),
        )
        for index in range(20)
    )
    plain: Final = tuple(CollectedCase(f"test_slots.py::test_plain[{index}]") for index in range(7))
    shards: Final = case_shards(cases + plain, 3)
    assert frozenset.union(*shards) == frozenset(case.nodeid for case in cases + plain)
    assert sum(len(shard) for shard in shards) == len(cases + plain)
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
    for _, group in groupby(cases, key=lambda case: case.parameters[0].value):
        configuration: Final = frozenset(case.nodeid for case in group)
        assert sum(configuration <= shard for shard in shards) == 1
    assert shards == case_shards(tuple(reversed(cases + plain)), 3)


def test_shards_reuse_unparametrized_fixtures_and_keep_different_results_separate() -> None:
    shared: Final = tuple(
        CollectedCase(f"test_sockets.py::test_override[{index}]", fixtures=("module:test_sockets.py:override",))
        for index in range(4)
    )
    other: Final = CollectedCase("test_sockets.py::test_default", fixtures=("module:test_sockets.py:default",))
    plain: Final = tuple(CollectedCase(f"test_other.py::test_plain[{index}]") for index in range(7))
    shards: Final = case_shards((*shared, other, *plain), 3)
    configuration: Final = frozenset(case.nodeid for case in shared)
    assert sum(configuration <= shard for shard in shards) == 1
    assert sorted(map(len, shards)) == [4, 4, 4]
    assert frozenset.union(*shards) == frozenset(case.nodeid for case in (*shared, other, *plain))
    groups: Final = fixture_groups((*shared, other))
    assert len({groups[case.nodeid] for case in shared}) == 1
    assert groups[other.nodeid] != groups[shared[0].nodeid]
    assert shards == case_shards(tuple(reversed((*shared, other, *plain))), 3)


def test_an_oversized_unparametrized_fixture_is_split_into_bounded_groups() -> None:
    cases: Final = tuple(
        CollectedCase(f"test_sockets.py::test_route[{index:02}]", fixtures=("module:test_sockets.py:owned",))
        for index in range(20)
    )
    shards: Final = case_shards(cases, 4)
    assert tuple(map(len, shards)) == (5, 5, 5, 5)
    assert frozenset.union(*shards) == frozenset(case.nodeid for case in cases)
    assert sum(len(shard) for shard in shards) == len(cases)


def test_measured_cost_splits_an_expensive_shared_fixture_without_losing_cases() -> None:
    expensive: Final = tuple(
        CollectedCase(f"test_tracing.py::test_route[{index}]", fixtures=("module:test_tracing.py:gateway",))
        for index in range(4)
    )
    cheap: Final = tuple(CollectedCase(f"test_other.py::test_route[{index}]") for index in range(8))
    cases: Final = (*expensive, *cheap)
    durations: Final = MappingProxyType({case.nodeid: 40.0 if case in expensive else 1.0 for case in cases})
    shards: Final = case_shards(cases, 4, durations)
    assert sorted(sum(durations[nodeid] for nodeid in shard) for shard in shards) == [42.0] * 4
    assert frozenset.union(*shards) == frozenset(case.nodeid for case in cases)
    assert sum(len(shard) for shard in shards) == len(cases)
    assert shards == case_shards(tuple(reversed(cases)), 4, durations)


@pytest.mark.parametrize(
    "content",
    (
        "invalid",
        "{}",
        '[{"nodeid":"case","seconds":-1}]',
        '[{"nodeid":"case","seconds":"1"}]',
        '[{"nodeid":"case","seconds":NaN}]',
        '[{"nodeid":"case","seconds":Infinity}]',
        '[{"nodeid":"case","seconds":1},{"nodeid":"case","seconds":2}]',
    ),
)
def test_invalid_timing_cache_preserves_the_complete_cold_partition(content: str) -> None:
    cases: Final = tuple(CollectedCase(f"test_owned.py::test_route[{index}]") for index in range(9))
    durations: Final = parse_timings(content)
    assert not durations
    assert case_shards(cases, 3, durations) == case_shards(cases, 3)


def test_timing_cache_keeps_zero_duration_skips_and_assigns_new_cases() -> None:
    durations: Final = parse_timings('[{"nodeid":"skip","seconds":0},{"nodeid":"old","seconds":12}]')
    cases: Final = (CollectedCase("skip"), CollectedCase("new"))
    shards: Final = case_shards(cases, 2, durations)
    assert durations["skip"] > 0
    assert frozenset.union(*shards) == frozenset({"skip", "new"})
    assert tuple(map(len, shards)) == (1, 1)


def test_measured_cost_keeps_parametrized_fixture_configurations_atomic() -> None:
    shared: Final = tuple(
        CollectedCase(
            f"test_tracing.py::test_route[{index}]",
            (ScopedParameter("module:test_tracing.py", "configuration", "shared"),),
        )
        for index in range(3)
    )
    plain: Final = tuple(CollectedCase(f"test_other.py::test_route[{index}]") for index in range(5))
    cases: Final = (*shared, *plain)
    durations: Final = MappingProxyType({case.nodeid: 40.0 if case in shared else 1.0 for case in cases})
    shards: Final = case_shards(cases, 3, durations)
    configuration: Final = frozenset(case.nodeid for case in shared)
    assert sum(configuration <= shard for shard in shards) == 1
    assert frozenset.union(*shards) == frozenset(case.nodeid for case in cases)
    assert sum(len(shard) for shard in shards) == len(cases)
