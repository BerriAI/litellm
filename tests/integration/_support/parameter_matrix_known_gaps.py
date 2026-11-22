import functools
import itertools
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, assert_never

from integration._support.parameter_matrix import Cell, Endpoint, Group, cells
from integration._support.parameter_matrix_rig import GapKind
from pydantic import BaseModel, ConfigDict

EVERY_PROVIDER: Final = None


@dataclass(frozen=True, slots=True)
class KnownGap:
    endpoint: Endpoint
    native: bool
    kind: GapKind
    attribute: str
    provider: str | None
    model: str | None = None

    @property
    def id(self) -> str:
        path: Final = "native" if self.native else "bridged"
        scope: Final = self.model or self.provider or "every_provider"
        return "-".join((self.endpoint.name.lower(), path, self.kind.value, self.attribute, scope))


@dataclass(frozen=True, slots=True)
class KnownPlainFailure:
    endpoint: Endpoint
    drop_params: bool
    model: str

    @property
    def id(self) -> str:
        setting: Final = "on" if self.drop_params else "off"
        return f"{self.endpoint.name.lower()}-{self.model}-drop_params_{setting}"


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Known(_Record):
    endpoint: Endpoint
    native: bool
    kind: GapKind
    attribute: str
    providers: tuple[str, ...] | None = ()
    models: tuple[str, ...] = ()

    def gaps(self) -> Iterator[KnownGap]:
        for provider in (EVERY_PROVIDER,) if self.providers is EVERY_PROVIDER else self.providers:
            yield KnownGap(self.endpoint, self.native, self.kind, self.attribute, provider)
        for model in self.models:
            yield KnownGap(self.endpoint, self.native, self.kind, self.attribute, EVERY_PROVIDER, model)


class _KnownPlain(_Record):
    endpoint: Endpoint
    drop_params: bool
    models: tuple[str, ...]

    def failures(self) -> Iterator[KnownPlainFailure]:
        for model in self.models:
            yield KnownPlainFailure(self.endpoint, self.drop_params, model)


class _Ledger(_Record):
    gaps: tuple[_Known, ...]
    plain_failures: tuple[_KnownPlain, ...]
    unroutable: frozenset[str]


_LEDGER: Final = _Ledger.model_validate_json(Path(__file__).with_suffix(".json").read_bytes())

KNOWN_UNROUTABLE: Final = _LEDGER.unroutable
KNOWN_GAPS: Final = frozenset(itertools.chain.from_iterable(known.gaps() for known in _LEDGER.gaps))
KNOWN_PLAIN_FAILURES: Final = frozenset(
    itertools.chain.from_iterable(known.failures() for known in _LEDGER.plain_failures)
)


def _slot(cell: Cell) -> tuple[str, bool, str]:
    return cell.group.endpoint.value, cell.group.native, cell.attribute


def _known_slot(known: KnownGap) -> tuple[str, bool, str]:
    return known.endpoint.value, known.native, known.attribute


@functools.cache
def _known_by_slot() -> Mapping[tuple[str, bool, str], tuple[KnownGap, ...]]:
    ordered: Final = sorted(KNOWN_GAPS, key=_known_slot)
    return MappingProxyType({slot: tuple(found) for slot, found in itertools.groupby(ordered, key=_known_slot)})


@functools.cache
def _listed(group: Group) -> frozenset[str]:
    return frozenset(model.key for model in group.models)


def _covers(known: KnownGap, group: Group) -> bool:
    if known.model is not None:
        return known.model in _listed(group)
    return known.provider in (EVERY_PROVIDER, group.provider)


def known_gap(cell: Cell, kind: GapKind) -> bool:
    return any(known.kind is kind and _covers(known, cell.group) for known in _known_by_slot().get(_slot(cell), ()))


@functools.cache
def _by_slot() -> Mapping[tuple[str, bool, str], tuple[Cell, ...]]:
    return MappingProxyType(
        {slot: tuple(found) for slot, found in itertools.groupby(sorted(cells(), key=_slot), key=_slot)}
    )


def _could_show(known: KnownGap, cell: Cell) -> bool:
    listed: Final = cell.attribute in cell.group.supported
    match known.kind:
        case GapKind.LISTED_NOT_SENT:
            return listed
        case GapKind.LISTED_REFUSED:
            return listed and not cell.drop_params
        case GapKind.UNLISTED_VANISHES:
            return not listed and not cell.drop_params
        case GapKind.REFUSED_WITH_DROP:
            return cell.drop_params
        case GapKind.BROKEN | GapKind.UNSAMPLED:
            return True
        case _:
            assert_never(known.kind)


def candidates(known: KnownGap) -> Iterator[Cell]:
    for cell in _by_slot().get(_known_slot(known), ()):
        if _covers(known, cell.group) and _could_show(known, cell):
            yield cell
