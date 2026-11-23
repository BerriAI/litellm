"""The live ``litellm.model_cost`` table, every write to it, and its version."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Generic, ParamSpec, TypeVar

from litellm.constants import DEFAULT_MAX_LRU_CACHE_SIZE

P = ParamSpec("P")
R = TypeVar("R")


class AliasedCostEntry(dict[str, object]):
    """A ``litellm.model_cost`` row, the one object every alias name of a model is served under."""

    __slots__ = ("names", "registered_fields")

    def __init__(
        self, entry: Mapping[str, object], names: tuple[str, ...], registered_fields: frozenset[str] = frozenset()
    ) -> None:
        super().__init__(entry)
        self.names: tuple[str, ...] = names
        self.registered_fields: frozenset[str] = registered_fields


@dataclass(frozen=True, slots=True)
class ModelCostVersion:
    table: int
    size: int
    writes: int


class _Writes:
    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count: int = 0

    def record(self) -> None:
        self.count += 1


_WRITES: Final = _Writes()


def model_cost_version() -> ModelCostVersion:
    import litellm

    table: Final = litellm.model_cost
    return ModelCostVersion(table=id(table), size=len(table), writes=_WRITES.count)


def publish(table: dict[str, object]) -> None:
    """Makes ``table`` the live ``litellm.model_cost``."""
    import litellm

    litellm.model_cost = table
    _WRITES.record()


def write(rows: dict[str, object], written: Mapping[str, object]) -> None:
    """Writes one registration's rows into ``rows``, the published table."""
    rows.update(written)
    _WRITES.record()


def changed_in_place() -> None:
    """Moves the version and drops the name index after an in-place edit of ``litellm.model_cost``."""
    from litellm.litellm_core_utils.get_model_cost_map import model_cost_inputs

    model_cost_inputs.invalidate()
    _WRITES.record()


@dataclass(frozen=True, slots=True)
class Uncached(Generic[R]):
    """A result ``VersionCached`` hands back without caching, so the next call computes it again."""

    value: R


class VersionCached(Generic[P, R]):
    """Decorates ``function`` with an ``lru_cache`` keyed on the ``litellm.model_cost`` version each call read, and
    exposes that cache's ``cache_clear`` and ``cache_info``. A result ``function`` wraps in ``Uncached`` is returned
    unwrapped and left out of the cache."""

    __slots__ = ("_call", "cache_clear", "cache_info")

    def __init__(self, function: Callable[P, R | Uncached[R]]) -> None:
        class NotCached(Exception):
            def __init__(self, value: R) -> None:
                super().__init__()
                self.value: Final = value

        def at(
            _version: ModelCostVersion,
            *args: P.args,
            **kwargs: P.kwargs,  # kwargs-ok: forwards a ParamSpec unchanged
        ) -> R:
            result: Final = function(*args, **kwargs)
            if isinstance(result, Uncached):
                raise NotCached(result.value)
            return result

        cached: Final = lru_cache(maxsize=DEFAULT_MAX_LRU_CACHE_SIZE)(at)

        def call(*args: P.args, **kwargs: P.kwargs) -> R:  # kwargs-ok: forwards a ParamSpec unchanged
            try:
                return cached(model_cost_version(), *args, **kwargs)
            except NotCached as not_cached:
                return not_cached.value

        self._call: Final = call
        self.cache_clear: Final = cached.cache_clear
        self.cache_info: Final = cached.cache_info

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:  # kwargs-ok: forwards a ParamSpec unchanged
        return self._call(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class Landing:
    key: str
    base: Mapping[str, object]
    matched_builtin: bool


def published() -> dict[str, object]:
    """The table published as ``litellm.model_cost`` now."""
    import litellm

    return litellm.model_cost
