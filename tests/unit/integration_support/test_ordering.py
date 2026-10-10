from __future__ import annotations

from itertools import groupby
from typing import Final

from tests.integration._support.ordering import CollectedCase, ScopedParameter, case_shards, fixture_groups
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
        CollectedCase(f"test_slots.py::test_route[{index}]", (ScopedParameter("module:slots", "rig", index // 4),))
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
