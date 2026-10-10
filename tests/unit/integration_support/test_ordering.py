from __future__ import annotations

from itertools import groupby
from typing import Final

from tests.integration._support.ordering import (
    CollectedCase,
    ScopedParameter,
    fixture_groups,
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
