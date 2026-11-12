"""
``HostedModelList``: the Rust core's live ``model_list``, read back from router-core.

router-core holds the deployments. This list is what ``registry().all()`` and the deprecated
``model_list`` attribute hand out: a ``list`` (so callers that check for one, serialize it or
validate it keep working) whose contents are router-core's deployment records, read through the
handle after every write. Every write to it is a changeset instead of a Python-side mutation:

- ``append`` / ``extend`` / an ``insert`` at the end add at the end (``append``)
- removing one deployment whose id no other entry shares is a ``delete``
- any other write (an assignment, a slice, a removal of one of several entries sharing an id,
  ``clear``, ``sort``, ...) hands router-core the resulting list (``replace``)

A build serves the deployments it added once it ends (or fails), and the list reads them back
then. After writes that only appended, it reads just the appended records
(``RouterCoreHandle.deployments_from``), so adding deployments one at a time costs each one its own
record rather than the whole list. These writes are the raw ones a caller makes: like the Python
router's list, they reindex nothing.

An entry read back from router-core is a dict whose writes (its own, or a nested dict's) hand
router-core the list again, so an edit in place reaches it as the Python router's list would show
it. Each deployment keeps one Python object while router-core's record for it is unchanged, keyed by
its id and its place among entries sharing that id, so identity checks on stored entries (the
model info caches) mean what they did: the same object until a write changes that deployment.
"""

import copy
import datetime
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    SupportsIndex,
    TypeAlias,
    TypeVar,
    cast,  # noqa: TID251  # any value a caller wrote into the list
    overload,
)

from pydantic import TypeAdapter, ValidationError

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.types.router import ModelAccessWindow

if TYPE_CHECKING:
    from litellm.rust_bridge._native import RouterCoreHandle

Entry: TypeAlias = dict[str, object]
# (id, its place among the entries sharing it) -> (the record it was read as, the entry object)
_Stable: TypeAlias = dict[tuple[str, int], tuple[object, Entry]]


_DATETIME: Final = TypeAdapter(datetime.datetime)
_Field = TypeVar("_Field")  # rebind-ok: a TypeVar cannot be Final


def _dumped_as(adapter: TypeAdapter[_Field]) -> Callable[[object], object]:
    """A value validated by ``adapter``, then dumped back to its Python form."""

    def dump(value: object) -> object:
        return adapter.dump_python(adapter.validate_python(value), exclude_none=True)

    return dump


# The ``ModelInfo`` fields whose Python value is no JSON value, which router-core holds as strings.
_TYPED_MODEL_INFO: Final[Mapping[str, Callable[[object], object]]] = MappingProxyType(
    {
        "created_at": _dumped_as(_DATETIME),
        "updated_at": _dumped_as(_DATETIME),
        "ptu_effective_from": _dumped_as(_DATETIME),
        "ptu_effective_to": _dumped_as(_DATETIME),
        "access_windows": _dumped_as(TypeAdapter(tuple[ModelAccessWindow, ...])),
    }
)


def typed_deployment(record: Mapping[str, object]) -> Mapping[str, object]:
    """A deployment record as ``Deployment.to_json`` dumped it: its ``model_info`` typed again. One
    with none of those fields comes back as it is, and so does a value a raw write left unreadable."""
    info: Final = as_str_mapping(record.get("model_info"))
    if info is None or info.keys().isdisjoint(_TYPED_MODEL_INFO):
        return record
    typed: Final[dict[str, object]] = {}  # mutable-ok: filled here
    for key, dumped in _TYPED_MODEL_INFO.items():
        value = info.get(key)
        if value is None:
            continue
        try:
            typed[key] = dumped(value)
        except ValidationError:
            continue
    return {**record, "model_info": {**info, **typed}}


def _record(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value)  # cast-ok: a JSON object router-core wrote, or a dict inside one


def _entry_id(entry: object) -> str | None:
    info: Final = as_str_mapping((as_str_mapping(entry) or {}).get("model_info")) or {}
    value: Final = info.get("id")
    return None if value is None else str(value)


def _keys(
    entries: Iterable[object], seen: Mapping[str, int] = MappingProxyType({})
) -> tuple[list[tuple[str, int] | None], Mapping[str, int]]:  # mutable-ok: a fresh list
    """Each entry's identity key (its id and how many earlier entries share it, ``seen`` counting
    the ids of entries before these), and the counts once past them."""
    counts: Final[dict[str, int]] = dict(seen)  # mutable-ok: counted here
    keys: Final[list[tuple[str, int] | None]] = []  # mutable-ok: filled here
    for entry in entries:
        entry_id = _entry_id(entry)
        if entry_id is None:
            keys.append(None)
            continue
        nth = counts.get(entry_id, 0)
        counts[entry_id] = nth + 1
        keys.append((entry_id, nth))
    return keys, counts


class _HostedEntry(dict[str, object]):
    """A deployment entry read back from router-core: a write to it, or to a dict inside it, hands
    router-core the list again (``replace``), so it learns what the entry now says. A copy is a
    plain dict."""

    def __init__(self, record: Mapping[str, object], written: Callable[[], None]) -> None:
        super().__init__(
            (key, value if (nested := as_str_mapping(value)) is None else _HostedEntry(nested, written))
            for key, value in record.items()
        )
        self._written: Final = written

    def __setitem__(self, key: str, value: object) -> None:
        super().__setitem__(key, value)
        self._written()

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        self._written()

    def pop(self, key: str, *default: object) -> object:
        value: Final = super().pop(key, *default)
        self._written()
        return value

    def popitem(self) -> tuple[str, object]:
        item: Final = super().popitem()
        self._written()
        return item

    def clear(self) -> None:
        super().clear()
        self._written()

    def update(self, *args: object, **kwargs: object) -> None:  # kwargs-ok: dict.update's own arguments
        super().update(*args, **kwargs)  # pyright: ignore[reportCallIssue, reportArgumentType]  # dict.update's own arguments
        self._written()

    def setdefault(self, key: str, default: object = None) -> object:
        if key in self:
            return self[key]
        self[key] = default
        return default

    def __ior__(self, other: object) -> "_HostedEntry":  # pyright: ignore[reportIncompatibleMethodOverride]  # dict's in-place merge, as a write
        self.update(other)
        return self

    def __copy__(self) -> dict[str, object]:  # mutable-ok: a plain copy
        return dict(self)

    def __deepcopy__(self, memo: dict[int, object]) -> dict[str, object]:  # mutable-ok: a plain copy
        return copy.deepcopy(dict(self), memo)

    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[object, ...]:
        return (dict, (copy.deepcopy(dict(self)),))


class HostedModelList(list[Entry]):
    """The live ``model_list``: router-core's deployments, every write a changeset."""

    def __init__(
        self,
        handle: Callable[[], "RouterCoreHandle | None"],
        apply: Callable[[Mapping[str, object]], None],
        live: Callable[[object], Mapping[str, object]] = lambda _: {},
        live_info: Callable[[object], Mapping[str, object]] = lambda _: {},
        *,
        lock: AbstractContextManager[object] | None = None,
    ) -> None:
        super().__init__()
        self._handle: Final = handle
        self._lock: Final[AbstractContextManager[object]] = threading.RLock() if lock is None else lock
        self._apply: Final = apply
        # A deployment's litellm_params values its record carries as text, by id (``live_params``).
        self._live: Final = live
        # A deployment's model_info values its record reads back changed (non-string keys), by id.
        self._live_info: Final = live_info
        # The handle and registry version the contents were last read at.
        self._read_at: tuple[object, int] | None = None
        self._stable: _Stable = {}
        # How many of the shown entries hold each id, for the keys of entries appended next.
        self._id_counts: Mapping[str, int] = {}
        # While a build runs: the length, id counts and read to go back to when it ends.
        self._building: tuple[int, Mapping[str, int], tuple[object, int] | None] | None = None

    # -- reading ----------------------------------------------------------------------------------

    def refresh(self) -> None:
        """Read router-core's deployments again when its registry moved since the last read."""
        with self._lock:
            self._refresh()

    def _refresh(self) -> None:
        handle: Final = self._handle()
        if handle is None or self._building is not None:
            # While a build runs, the list shows what it accepted until router-core serves it.
            return
        version: Final = handle.registry_version()
        if self._read_at is not None and self._read_at[0] is handle and self._read_at[1] == version:
            return
        read_at: Final = self._read_at
        self._read_at = (handle, version)
        if read_at is not None and read_at[0] is handle and handle.registry_rewritten() <= read_at[1]:
            # Only appends since the last read: the shown entries stand, and the new ones follow.
            super().extend(self._kept(handle.deployments_from(len(self)), appended=True))
            return
        self._show(self._kept(handle.deployments()))

    def _kept(self, records: Sequence[object], *, appended: bool = False) -> list[Entry]:  # mutable-ok: a fresh list
        """``records`` as entries, each the object it was while its record is unchanged; when
        ``appended``, they follow the shown entries, which keep theirs."""
        stable: Final[_Stable] = self._stable if appended else {}
        keys, counts = _keys(records, self._id_counts if appended else {})
        entries: Final[list[Entry]] = []  # mutable-ok: filled here
        for key, record in zip(keys, records, strict=True):
            typed = self._with_live(typed_deployment(_record(record)))
            held = self._stable.get(key) if key is not None else None
            kept = held is not None and held[0] == typed
            entry = held[1] if held is not None and kept else _HostedEntry(typed, self._entry_written)
            if key is not None:
                stable[key] = (held[0] if held is not None and kept else copy.deepcopy(typed), entry)
            entries.append(entry)
        self._stable = stable
        self._id_counts = counts
        return entries

    def _with_live(self, record: Mapping[str, object]) -> Mapping[str, object]:
        params: Final = as_str_mapping(record.get("litellm_params"))
        info: Final = as_str_mapping(record.get("model_info"))
        model_id: Final = (info or {}).get("id")
        live: Final = self._live(model_id) if params is not None else {}
        live_info: Final = self._live_info(model_id) if info is not None else {}
        if live:
            record = {**record, "litellm_params": {**(params or {}), **live}}  # rebind-ok: given its live values
        if live_info:
            record = {**record, "model_info": {**(info or {}), **live_info}}  # rebind-ok: see above
        return record

    # -- a build ----------------------------------------------------------------------------------

    def begin_build(self, *, replacing: bool) -> None:
        """A build starts: router-core serves what it adds once it ends, so until then the list
        shows each deployment it accepts here (``built``), after the entries it keeps, or alone
        when it replaces them, as the Python router's list grows while it adds them."""
        with self._lock:
            self._refresh()
            if replacing:
                self._show([])
                self._id_counts = {}
            self._building = (len(self), self._id_counts, self._read_at)

    def built(self, entry: Mapping[str, object]) -> None:
        """A deployment the build accepted, shown until the build ends."""
        with self._lock:
            if self._building is not None:
                super().extend(self._kept([entry], appended=True))

    def end_build(self) -> None:
        """The build ended (or failed): router-core serves what it added, read back in place of
        the entries shown meanwhile, each kept as the object it was shown as while unchanged."""
        with self._lock:
            if self._building is None:
                return
            start, counts, read_at = self._building
            self._building = None
            super().__delitem__(slice(start, None))
            self._id_counts = counts
            self._read_at = read_at
            self._refresh()

    def _seed(self, entries: Sequence[object]) -> None:
        """Keep ``entries``, the list router-core now holds, as the objects its records for them
        read back as."""
        for key, entry in zip(_keys(entries)[0], entries, strict=True):
            self._seed_one(key, entry)

    def _seed_appended(self, entries: Sequence[object]) -> None:
        """Keep ``entries``, appended after the shown ones, as the objects they read back as."""
        for key, entry in zip(_keys(entries, self._id_counts)[0], entries, strict=True):
            self._seed_one(key, entry)

    def _seed_one(self, key: tuple[str, int] | None, entry: object) -> None:
        if key is not None and isinstance(entry, dict):
            stored: Final = cast(Entry, entry)  # cast-ok: a model_list entry is a dict
            self._stable[key] = (copy.deepcopy(stored), stored)

    def _entry_written(self) -> None:
        """A write inside an entry: router-core takes the list as it now reads."""
        self._replace(list(self))

    def _show(self, entries: list[Entry]) -> None:  # mutable-ok: the entries shown
        super().__setitem__(slice(None), entries)

    # -- writing ----------------------------------------------------------------------------------

    def _replace(self, entries: list[Entry]) -> None:  # mutable-ok: the list router-core takes
        # Seeded first: router-core's records for them read back as these objects however soon
        # the list is read again.
        with self._lock:
            self._seed(entries)
            self._apply({"replace": entries})
            self._refresh()

    def replace(self, entries: Iterable[object]) -> None:
        """Every deployment at once (``router.model_list = [...]``)."""
        self._replace(cast(list[Entry], list(entries)))  # cast-ok: the entries as assigned, which router-core validates

    def append(self, value: Entry) -> None:
        with self._lock:
            self._refresh()
            self._seed_appended([value])
            self._apply({"append": [value]})
            self._refresh()

    def extend(self, values: Iterable[Entry]) -> None:
        for value in tuple(values):
            self.append(value)

    def __iadd__(self, values: Iterable[Entry]) -> "HostedModelList":  # pyright: ignore[reportIncompatibleMethodOverride]  # list's in-place add, as a changeset
        self.extend(values)
        return self

    def insert(self, index: SupportsIndex, value: Entry) -> None:
        if int(index) >= len(self):
            self.append(value)
            return
        changed: Final = list(self)
        changed.insert(index, value)
        self._replace(changed)

    @overload
    def __setitem__(self, index: SupportsIndex, value: Entry) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[Entry]) -> None: ...
    def __setitem__(self, index: SupportsIndex | slice, value: object) -> None:
        changed: Final = list(self)
        changed[index] = value  # pyright: ignore[reportArgumentType, reportCallIssue]  # list's own item and slice assignment
        self._replace(changed)

    def __delitem__(self, index: SupportsIndex | slice) -> None:
        changed: Final = list(self)
        if isinstance(index, slice):
            del changed[index]
            self._replace(changed)
            return
        removed: Final = changed.pop(index)
        removed_id: Final = _entry_id(removed)
        if removed_id is None or any(_entry_id(entry) == removed_id for entry in changed):
            # A delete removes every deployment with the id; the list removes only this one.
            self._replace(changed)
            return
        with self._lock:
            self._apply({"delete": [removed_id]})
            self._refresh()

    def pop(self, index: SupportsIndex = -1) -> Entry:
        entry: Final = self[index]
        del self[index]
        return entry

    def remove(self, value: Entry) -> None:
        del self[self.index(value)]

    def clear(self) -> None:
        self._replace([])

    def sort(self, *args: object, **kwargs: object) -> None:  # kwargs-ok: list.sort's own arguments
        changed: Final = list(self)
        changed.sort(*args, **kwargs)  # pyright: ignore[reportCallIssue, reportArgumentType]  # list.sort's own arguments
        self._replace(changed)

    def reverse(self) -> None:
        self._replace(list(reversed(self)))

    def __imul__(self, count: SupportsIndex) -> "HostedModelList":  # pyright: ignore[reportIncompatibleMethodOverride]  # list's in-place repeat, as a changeset
        self._replace(list(self) * int(count))
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[object, ...]:
        # A copy or pickle of the live list is a plain list of its entries.
        return (list, (list(self),))
