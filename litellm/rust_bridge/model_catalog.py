"""The model catalog the process serves, backed by the native ``ModelCatalog``.

A ``ModelCatalog`` is immutable. Registering, unregistering or adopting a newly fetched registry yields a new one,
and ``ModelCatalogSlot`` swaps it in whole, so a reader never sees a half-applied change. Every registration belongs
to an owner and keeps only the fields that owner set, so a new registry shows through every field nobody overrode.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Final, NoReturn

from pydantic import JsonValue

from litellm.rust_bridge._native import ModelCatalog, bundled_model_cost_map


@dataclass(frozen=True, slots=True)
class RegistryRejected:
    reason: str


def parse_registry(
    body: bytes, *, source: str, revision: str | None = None, etag: str | None = None
) -> ModelCatalog | RegistryRejected:
    try:
        return ModelCatalog.parse(body, source, revision, etag)
    except ValueError as error:
        return RegistryRejected(str(error))


def bundled_registry_bytes() -> bytes:
    return bundled_model_cost_map()


class ModelCatalogSlot:
    """Holds the catalog the process serves and serializes the changes that replace it."""

    def __init__(self, registry: ModelCatalog) -> None:
        self._registry: ModelCatalog = registry
        self._current: ModelCatalog = registry
        self._lock: Final = threading.RLock()

    @property
    def current(self) -> ModelCatalog:
        return self._current

    @property
    def registry(self) -> ModelCatalog:
        """The registry served underneath the registrations, without any of them."""
        return self._registry

    def register(self, owner: str, entries: Mapping[str, Mapping[str, object]]) -> ModelCatalog:
        with self._lock:
            self._current = self._current.register(owner, entries)
            return self._current

    def unregister(self, owner: str) -> ModelCatalog:
        """Take back every entry ``owner`` registered. Call it as soon as the owner goes away."""
        with self._lock:
            self._current = self._current.unregister(owner)
            return self._current

    def adopt(self, registry: ModelCatalog) -> None:
        """Serve ``registry`` underneath every registration made so far.

        Raises ``ValueError`` and keeps serving the current registry when a registration no longer reads cleanly on
        top of ``registry``.
        """
        with self._lock:
            current: Final = self._current.with_registry(registry)
            self._registry = registry
            self._current = current

    def fork(self) -> "ModelCatalogSlot":
        """A new slot serving what this one serves now, so changes to either stay out of the other."""
        forked: Final = ModelCatalogSlot(self._registry)
        forked._current = self._current
        return forked


_READ_ONLY: Final = "litellm.model_cost entries are read-only; register changes with litellm.register_model"


class CatalogEntry(dict[str, JsonValue]):
    """One served catalog entry: a ``dict`` for every reader, refusing writes. Copies are plain dicts."""

    __slots__ = ()

    def __setitem__(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise TypeError(_READ_ONLY)

    def __delitem__(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise TypeError(_READ_ONLY)

    def update(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise TypeError(_READ_ONLY)

    def pop(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise TypeError(_READ_ONLY)

    def popitem(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise TypeError(_READ_ONLY)

    def setdefault(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise TypeError(_READ_ONLY)

    def clear(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise TypeError(_READ_ONLY)

    def __ior__(self, _other: object) -> NoReturn:
        raise TypeError(_READ_ONLY)

    def copy(self) -> dict[str, JsonValue]:
        return dict(self)

    def __copy__(self) -> dict[str, JsonValue]:
        return dict(self)

    def __deepcopy__(self, memo: dict[int, object]) -> dict[str, JsonValue]:
        from copy import deepcopy

        return {key: deepcopy(value, memo) for key, value in self.items()}

    def __reduce__(self) -> tuple[type[dict[str, JsonValue]], tuple[dict[str, JsonValue]]]:
        return dict, (dict(self),)


class ModelCostView(Mapping[str, CatalogEntry]):
    """A read-only ``litellm.model_cost`` over whichever catalog ``catalog`` returns at read time.

    Entries are converted once per catalog and served as ``CatalogEntry``, so repeated reads stay dict-cheap and a
    caller writing into one fails loudly instead of changing a copy nobody else sees.
    """

    def __init__(self, catalog: Callable[[], ModelCatalog]) -> None:
        self._catalog: Final = catalog
        self._converted: tuple[ModelCatalog, dict[str, CatalogEntry]] | None = None

    def _entries(self) -> tuple[ModelCatalog, dict[str, CatalogEntry]]:
        catalog: Final = self._catalog()
        converted: Final = self._converted
        if converted is not None and converted[0] is catalog:
            return converted
        fresh: Final[tuple[ModelCatalog, dict[str, CatalogEntry]]] = (
            catalog,
            {},
        )  # mutable-ok: per-catalog conversion cache
        self._converted = fresh
        return fresh

    def __getitem__(self, key: str) -> CatalogEntry:
        catalog, converted = self._entries()
        cached: Final = converted.get(key)
        if cached is not None:
            return cached
        try:
            entry: Final = catalog.get(key)
        except TypeError:
            raise KeyError(key) from None
        if entry is None:
            raise KeyError(key)
        served: Final = CatalogEntry(entry)
        converted[key] = served
        return served

    def __iter__(self) -> Iterator[str]:
        return iter(self._catalog().keys_list())

    def __len__(self) -> int:
        return len(self._catalog())

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._catalog()

    def __repr__(self) -> str:
        return f"ModelCostView({len(self)} entries)"
