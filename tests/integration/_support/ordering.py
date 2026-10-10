from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import reduce
from itertools import chain, groupby
from types import MappingProxyType
from typing import Final

import pytest
from _pytest.python import get_direct_param_fixture_func


@dataclass(frozen=True, slots=True)
class ScopedParameter:
    owner: str
    name: str
    value: object


@dataclass(frozen=True, slots=True)
class CollectedCase:
    nodeid: str
    parameters: tuple[ScopedParameter, ...] = ()
    fixtures: tuple[str, ...] = ()


def _parameter(item: pytest.Function, name: str, value: object) -> ScopedParameter | None:
    definitions: Final = item._fixtureinfo.name2fixturedefs.get(name)
    scope: Final = item.callspec._arg2scope[name].value
    if not definitions or scope == "function" or definitions[-1].func is get_direct_param_fixture_func:
        return None
    parent: Final = item.getparent(pytest.Class) if scope == "class" else None
    owner: Final = parent.nodeid if parent is not None else definitions[-1].baseid
    return ScopedParameter(f"{scope}:{owner}", name, value)


def collected_case(item: pytest.Item) -> CollectedCase:
    if not isinstance(item, pytest.Function):
        return CollectedCase(item.nodeid)
    fixtures: Final = tuple(
        f"{definitions[-1].scope}:{definitions[-1].baseid}:{name}"
        for name, definitions in item._fixtureinfo.name2fixturedefs.items()
        if definitions
        and definitions[-1].scope in ("module", "class")
        and definitions[-1].baseid
        and definitions[-1].func is not get_direct_param_fixture_func
    )
    parameters: Final = (
        tuple(_parameter(item, name, value) for name, value in item.callspec.params.items())
        if hasattr(item, "callspec")
        else ()
    )
    scoped: Final = tuple(parameter for parameter in parameters if parameter is not None)
    return CollectedCase(
        item.nodeid,
        tuple(sorted(scoped, key=lambda parameter: (parameter.owner, parameter.name))),
        tuple(sorted(fixtures)),
    )


def _equal(first: object, second: object) -> bool:
    try:
        return bool(first == second)
    except (ValueError, RuntimeError):
        return first is second


def _same_parameters(first: CollectedCase, second: CollectedCase) -> bool:
    if len(first.parameters) != len(second.parameters):
        return False
    if not first.parameters and first.fixtures != second.fixtures:
        return False
    return all(
        left.owner == right.owner and left.name == right.name and _equal(left.value, right.value)
        for left, right in zip(first.parameters, second.parameters, strict=True)
    )


def _file(node: CollectedCase) -> str:
    return node.nodeid.split("::", 1)[0]


def _file_groups(cases: Iterable[CollectedCase]) -> tuple[tuple[str, str], ...]:
    members: Final = tuple(cases)
    return tuple(
        (case.nodeid, min(other.nodeid for other in members if _same_parameters(case, other)))
        if case.parameters or case.fixtures
        else (case.nodeid, "")
        for case in members
    )


def fixture_groups(cases: tuple[CollectedCase, ...]) -> Mapping[str, str]:
    files: Final = groupby(sorted(cases, key=_file), key=_file)
    return MappingProxyType(dict(chain.from_iterable(_file_groups(members) for _, members in files)))


def _allocate(shards: tuple[tuple[str, ...], ...], group: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    index: Final = min(range(len(shards)), key=lambda candidate: len(shards[candidate]))
    return tuple((*shard, *group) if candidate == index else shard for candidate, shard in enumerate(shards))


def _partition_group(cases: Iterable[CollectedCase], maximum: int) -> tuple[tuple[str, ...], ...]:
    members: Final = tuple(sorted(cases, key=lambda case: case.nodeid))
    nodeids: Final = tuple(case.nodeid for case in members)
    if members[0].parameters:
        return (nodeids,)
    return tuple(nodeids[start : start + maximum] for start in range(0, len(nodeids), maximum))


def case_shards(cases: tuple[CollectedCase, ...], count: int) -> tuple[frozenset[str], ...]:
    if count < 1:
        raise ValueError("Integration shard count must be positive")
    groups: Final = fixture_groups(cases)

    def key(case: CollectedCase) -> str:
        return groups[case.nodeid] or case.nodeid

    maximum: Final = max(1, (len(cases) + count - 1) // count)
    members: Final = tuple(
        chain.from_iterable(_partition_group(group, maximum) for _, group in groupby(sorted(cases, key=key), key=key))
    )
    ordered: Final = tuple(sorted(members, key=lambda group: (-len(group), group[0])))
    allocation: Final = reduce(_allocate, ordered, tuple(() for _ in range(count)))
    return tuple(frozenset(shard) for shard in allocation)
