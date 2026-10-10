from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import reduce
from itertools import chain, groupby
from math import ceil, isfinite
from statistics import median
from types import MappingProxyType
from typing import Final

import pytest
from _pytest.python import get_direct_param_fixture_func
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, TypeAdapter, ValidationError

EMPTY_DURATIONS: Final[Mapping[str, float]] = MappingProxyType({})


class CaseTiming(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    nodeid: str = Field(min_length=1)
    seconds: FiniteFloat = Field(ge=0)


TIMINGS: Final = TypeAdapter(tuple[CaseTiming, ...])


def parse_timings(content: str) -> Mapping[str, float]:
    try:
        values: Final = TIMINGS.validate_json(content)
    except ValidationError:
        return EMPTY_DURATIONS
    durations: Final = {value.nodeid: max(value.seconds, 0.001) for value in values}
    if len(durations) != len(values) or not isfinite(sum(durations.values())):
        return EMPTY_DURATIONS
    return MappingProxyType(durations)


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


def _allocate(
    shards: tuple[tuple[str, ...], ...], group: tuple[str, ...], durations: Mapping[str, float]
) -> tuple[tuple[str, ...], ...]:
    index: Final = min(
        range(len(shards)),
        key=lambda candidate: (sum(durations[nodeid] for nodeid in shards[candidate]), len(shards[candidate])),
    )
    return tuple((*shard, *group) if candidate == index else shard for candidate, shard in enumerate(shards))


def _add_case(
    groups: tuple[tuple[str, ...], ...], case: CollectedCase, maximum: float, durations: Mapping[str, float]
) -> tuple[tuple[str, ...], ...]:
    if not groups:
        return ((case.nodeid,),)
    if sum(durations[nodeid] for nodeid in groups[-1]) + durations[case.nodeid] > maximum:
        return (*groups, (case.nodeid,))
    return (*groups[:-1], (*groups[-1], case.nodeid))


def _partition_group(
    cases: Iterable[CollectedCase], maximum: float, durations: Mapping[str, float]
) -> tuple[tuple[str, ...], ...]:
    members: Final = tuple(sorted(cases, key=lambda case: case.nodeid))
    nodeids: Final = tuple(case.nodeid for case in members)
    if members[0].parameters:
        return (nodeids,)
    return reduce(lambda groups, case: _add_case(groups, case, maximum, durations), members, ())


def case_shards(
    cases: tuple[CollectedCase, ...], count: int, durations: Mapping[str, float] = EMPTY_DURATIONS
) -> tuple[frozenset[str], ...]:
    if count < 1:
        raise ValueError("Integration shard count must be positive")
    groups: Final = fixture_groups(cases)
    known: Final = tuple(durations[case.nodeid] for case in cases if case.nodeid in durations)
    default: Final = max(float(median(known)), 0.001) if known else 1.0
    weights: Final = MappingProxyType({case.nodeid: max(durations.get(case.nodeid, default), 0.001) for case in cases})

    def key(case: CollectedCase) -> str:
        return groups[case.nodeid] or case.nodeid

    maximum: Final = sum(weights.values()) / count if known else float(max(1, ceil(len(cases) / count)))
    members: Final = tuple(
        chain.from_iterable(
            _partition_group(group, maximum, weights) for _, group in groupby(sorted(cases, key=key), key=key)
        )
    )
    ordered: Final = tuple(sorted(members, key=lambda group: (-sum(weights[nodeid] for nodeid in group), group[0])))
    allocation: Final = reduce(
        lambda shards, group: _allocate(shards, group, weights), ordered, tuple(() for _ in range(count))
    )
    return tuple(frozenset(shard) for shard in allocation)
