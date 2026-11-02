# pyright: reportIncompatibleMethodOverride=false
# (each view overrides its builtin's mutators with the same behaviour, as a changeset)
"""
``HostedIndexes``: the registry indexes the Rust core hands out, read back from router-core.

router-core keeps the indexes the Python router keeps by history (``host_index.rs``): the id,
group and team position maps, the group names a ``set_model_list`` was given, the deployment
names that only ever grow, the team public names, the provider default ids and the wildcard
pattern routers. ``registry().index()``, ``names(kind=...)``, ``patterns()`` and the deprecated
attributes return the views here: a ``dict``, ``set`` or ``list`` (so ``isinstance`` and ``json``
keep working), one object per router, whose contents are router-core's, read again when an entry
point hands a view out (or before a write to it) once the registry moved, not on every registry
write. A write to any of them, a nested one included (``index("model_name")["gpt"].append(3)``),
is a ``host_index`` changeset that replaces that index in router-core, and the view reads it back.
The lookups over these indexes run in router-core (``lookup.rs``), not over the views.
"""

import copy
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Final, Literal, SupportsIndex, TypeAlias, cast

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.pattern_match_deployments import PatternMatchRouter

if TYPE_CHECKING:
    from litellm.rust_bridge._native import RouterCoreHandle

IndexField: TypeAlias = Literal[
    "id_map",
    "model_name",
    "team_model",
    "model_names",
    "deployment_names",
    "team_public_model_names",
    "provider_default_ids",
    "patterns",
    "team_patterns",
]
_Commit: TypeAlias = Callable[[object], None]
# Deployment copies a pattern router routes to, fresh for the caller.
Routed: TypeAlias = list[dict[str, object]]
_Pairs: TypeAlias = list[tuple[str, list[object]]]


def _stay() -> None:
    """A view with no contents of its own to read again before a write."""


def _as_set(value: object) -> set[object]:  # mutable-ok: set's own in-place operand
    return cast(set[object], value)  # cast-ok: an in-place set operand, checked by set itself


def _as_mapping(value: object) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        return {}
    return cast(Mapping[object, object], value)  # cast-ok: any mapping a caller wrote


def _json(value: object) -> object:
    """A view's contents as router-core takes them: plain JSON, a set as a list."""
    mapping: Final = _as_mapping(value)
    if mapping or isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in mapping.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_json(item) for item in cast(Iterable[object], value)]  # cast-ok: any collection a caller wrote
    return value


class _Positions(list[object]):
    """One index's position list (or a pattern's deployments): a write commits the whole list."""

    def __init__(self, items: Iterable[object], commit: _Commit, fresh: Callable[[], None] = _stay) -> None:
        super().__init__(items)
        self._commit: Final = commit
        self._fresh: Final = fresh

    def _load(self, items: Iterable[object]) -> None:
        """Show router-core's contents, without a write."""
        super().__setitem__(slice(None), list(items))

    def _write(self, change: Callable[[list[object]], object]) -> object:  # mutable-ok: the copy it changes
        self._fresh()  # a view held across a write elsewhere changes router-core's contents, not stale ones
        changed: Final = list(self)
        result: Final = change(changed)
        self._commit(changed)
        return result

    def append(self, value: object) -> None:
        _ = self._write(lambda items: items.append(value))

    def extend(self, values: Iterable[object]) -> None:
        added: Final = tuple(values)
        _ = self._write(lambda items: items.extend(added))

    def insert(self, index: SupportsIndex, value: object) -> None:
        _ = self._write(lambda items: items.insert(index, value))

    def remove(self, value: object) -> None:
        _ = self._write(lambda items: items.remove(value))

    def pop(self, index: SupportsIndex = -1) -> object:
        return self._write(lambda items: items.pop(index))

    def clear(self) -> None:
        _ = self._write(lambda items: items.clear())

    def sort(self, *args: object, **kwargs: object) -> None:  # kwargs-ok: list.sort's own arguments
        _ = self._write(lambda items: items.sort(*args, **kwargs))  # pyright: ignore[reportCallIssue, reportArgumentType]  # list.sort's own arguments

    def reverse(self) -> None:
        _ = self._write(lambda items: items.reverse())  # pyright: ignore[reportUnknownLambdaType]  # list.reverse returns None

    def __setitem__(self, index: object, value: object) -> None:
        _ = self._write(lambda items: items.__setitem__(index, value))  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownLambdaType]  # list's own item and slice assignment

    def __delitem__(self, index: object) -> None:
        _ = self._write(lambda items: items.__delitem__(index))  # pyright: ignore[reportArgumentType]  # list's own item and slice deletion

    def __iadd__(self, values: Iterable[object]) -> "_Positions":
        self.extend(values)
        return self

    def __imul__(self, count: SupportsIndex) -> "_Positions":
        _ = self._write(lambda items: items.__imul__(count))
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[object, ...]:
        return (list, (list(self),))


class _IndexDict(dict[object, object]):
    """A key-to-positions index: a write commits the whole index."""

    def __init__(self, commit: _Commit, fresh: Callable[[], None] = _stay) -> None:
        super().__init__()
        self._commit: Final = commit
        self._fresh: Final = fresh

    def _load(self, items: Mapping[object, object]) -> None:
        """Show router-core's contents, without a write."""
        super().clear()
        super().update(items)

    def _write(self, change: Callable[[dict[object, object]], object]) -> object:  # mutable-ok: the copy it changes
        self._fresh()  # a view held across a write elsewhere changes router-core's contents, not stale ones
        changed: Final = dict(self)
        result: Final = change(changed)
        self._commit(changed)
        return result

    def __setitem__(self, key: object, value: object) -> None:
        _ = self._write(lambda items: items.__setitem__(key, value))

    def __delitem__(self, key: object) -> None:
        _ = self._write(lambda items: items.__delitem__(key))

    def pop(self, key: object, *default: object) -> object:
        return self._write(lambda items: items.pop(key, *default))

    def popitem(self) -> tuple[object, object]:
        return cast(tuple[object, object], self._write(lambda items: items.popitem()))  # cast-ok: dict.popitem's pair

    def clear(self) -> None:
        _ = self._write(lambda items: items.clear())

    def update(self, *args: object, **kwargs: object) -> None:  # kwargs-ok: dict.update's own arguments
        _ = self._write(lambda items: items.update(*args, **kwargs))  # pyright: ignore[reportCallIssue, reportArgumentType]  # dict.update's own arguments

    def setdefault(self, key: object, default: object = None) -> object:
        if key in self:
            return self[key]
        self[key] = default
        return self[key]

    def __ior__(self, other: object) -> "_IndexDict":
        self.update(other)
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[object, ...]:
        return (dict, (dict(self),))


class _NameSet(set[object]):
    """``model_names``: a write commits the whole set."""

    def __init__(self, commit: _Commit, fresh: Callable[[], None] = _stay) -> None:
        super().__init__()
        self._commit: Final = commit
        self._fresh: Final = fresh

    def _load(self, items: Iterable[object]) -> None:
        """Show router-core's contents, without a write."""
        super().clear()
        super().update(items)

    def _write(self, change: Callable[[set[object]], object]) -> object:  # mutable-ok: the copy it changes
        self._fresh()  # a view held across a write elsewhere changes router-core's contents, not stale ones
        changed: Final = set(self)
        result: Final = change(changed)
        self._commit(changed)
        return result

    def add(self, value: object) -> None:
        _ = self._write(lambda items: items.add(value))

    def discard(self, value: object) -> None:
        _ = self._write(lambda items: items.discard(value))

    def remove(self, value: object) -> None:
        _ = self._write(lambda items: items.remove(value))

    def pop(self) -> object:
        return self._write(lambda items: items.pop())

    def clear(self) -> None:
        _ = self._write(lambda items: items.clear())

    def update(self, *others: Iterable[object]) -> None:
        _ = self._write(lambda items: items.update(*others))

    def difference_update(self, *others: Iterable[object]) -> None:
        _ = self._write(lambda items: items.difference_update(*others))

    def intersection_update(self, *others: Iterable[object]) -> None:
        _ = self._write(lambda items: items.intersection_update(*others))

    def symmetric_difference_update(self, other: Iterable[object]) -> None:
        _ = self._write(lambda items: items.symmetric_difference_update(other))

    def __ior__(self, other: object) -> "_NameSet":
        _ = self._write(lambda items: items.__ior__(_as_set(other)))
        return self

    def __iand__(self, other: object) -> "_NameSet":
        _ = self._write(lambda items: items.__iand__(_as_set(other)))
        return self

    def __isub__(self, other: object) -> "_NameSet":
        _ = self._write(lambda items: items.__isub__(_as_set(other)))
        return self

    def __ixor__(self, other: object) -> "_NameSet":
        _ = self._write(lambda items: items.__ixor__(_as_set(other)))
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[object, ...]:
        return (set, (set(self),))


class HostedPatternRouter(PatternMatchRouter):
    """A wildcard pattern router whose patterns are router-core's (``pattern_router``, or one team's
    router): routing runs in router-core, and ``add_pattern`` / ``remove_deployment`` / a write to
    ``patterns`` are changesets."""

    def __init__(self, owner: "HostedIndexes", team: str | None) -> None:
        super().__init__()  # its empty `patterns` write is ignored: router-core holds them
        self._owner: Final = owner
        self._team: Final = team
        self._view: Final = _IndexDict(self._commit, owner.refresh)

    @property
    def patterns(self) -> dict[str, list[object]]:  # pyright: ignore[reportIncompatibleVariableOverride]  # the base class's attribute, read from router-core  # mutable-ok: the live patterns, writes are changesets
        self._owner.refresh()
        return cast(dict[str, list[object]], self._view)  # cast-ok: regex keys, deployment lists

    @patterns.setter
    def patterns(self, value: Mapping[str, object]) -> None:  # pyright: ignore[reportIncompatibleVariableOverride]  # see the getter
        if not hasattr(self, "_owner"):
            return  # the base constructor's empty dict: router-core holds this router's patterns
        self._commit(value)

    def _show(self, pairs: Iterable[tuple[str, Sequence[object]]]) -> None:
        shown: Final = {
            regex: _Positions(deployments, partial(self._write_pattern, regex)) for regex, deployments in pairs
        }
        self._view._load(cast(Mapping[object, object], shown))  # pyright: ignore[reportPrivateUsage]  # cast-ok: regex to deployments

    def _write_pattern(self, regex: str, deployments: object) -> None:
        self._commit({**self.patterns, regex: deployments})

    def _commit(self, value: object) -> None:
        pairs: Final = [
            [str(regex), _json(deployments)] for regex, deployments in (as_str_mapping(value) or {}).items()
        ]
        self._owner.write_router(self._team, pairs)

    def route(self, request: str | None, filtered_model_names: Sequence[str] | None = None) -> Routed | None:  # pyright: ignore[reportIncompatibleMethodOverride]  # the base's list types, read only here
        if request is None:
            return None
        return self._owner.route(self._team, request, filtered_model_names)

    def get_models_by_pattern(self, model: str, custom_llm_provider: str | None = None) -> frozenset[str]:
        from litellm.litellm_core_utils.get_llm_provider_logic import inferred_provider

        provider: Final = custom_llm_provider or inferred_provider(model)
        return self._owner.models_by_pattern(self._team, [model, f"{provider}/{model}"])


class _TeamRouters(dict[str, HostedPatternRouter]):
    """``team_pattern_routers``: team id to its router; removing a team's router is a changeset."""

    def __init__(self, owner: "HostedIndexes") -> None:
        super().__init__()
        self._owner: Final = owner

    def _load(self, routers: Mapping[str, HostedPatternRouter]) -> None:
        super().clear()
        super().update(routers)

    def __delitem__(self, team: str) -> None:
        self._owner.drop_team_router(team)

    def pop(self, team: str, *default: object) -> object:
        if team not in self:
            if default:
                return default[0]
            raise KeyError(team)
        router: Final = self[team]
        self._owner.drop_team_router(team)
        return router

    def clear(self) -> None:
        self._owner.write({"team_patterns": []})

    def setdefault(self, team: str, default: HostedPatternRouter | None = None) -> HostedPatternRouter:
        if team not in self:
            self._owner.write_router(team, [])
        return self[team]


class HostedIndexes:
    """Router-core's host indexes, as the live objects the registry hands out."""

    def __init__(
        self,
        handle: Callable[[], "RouterCoreHandle | None"],
        apply: Callable[[Mapping[str, object]], None],
    ) -> None:
        self._handle: Final = handle
        self._apply: Final = apply
        self._read_at: tuple[object, int] | None = None
        self._document: Mapping[str, object] = {}
        # The views are read again when an entry point hands them out, and before a write to them.
        self.id_map: Final = _IndexDict(lambda value: self.write({"id_map": value}), self.refresh)
        self.model_name: Final = _IndexDict(lambda value: self.write({"model_name": value}), self.refresh)
        self.team_model: Final = _IndexDict(lambda value: self._write_team_model(value), self.refresh)
        self.model_names: Final = _NameSet(lambda value: self.write({"model_names": value}), self.refresh)
        self.deployment_names: Final = _Positions(
            (), lambda value: self.write({"deployment_names": value}), self.refresh
        )
        self.provider_default_ids: Final = _Positions(
            (), lambda value: self.write({"provider_default_ids": value}), self.refresh
        )
        self.pattern_router: Final = HostedPatternRouter(self, None)
        self.team_pattern_routers: Final = _TeamRouters(self)
        self._team_routers: dict[str, HostedPatternRouter] = {}  # mutable-ok: one router object per team
        self._team_public: frozenset[str] = frozenset()

    # -- reading ----------------------------------------------------------------------------------

    def refresh(self) -> None:
        """Read router-core's indexes again when its registry moved since the last read."""
        handle: Final = self._handle()
        if handle is None:
            return
        version: Final = handle.registry_version()
        if self._read_at is not None and self._read_at[0] is handle and self._read_at[1] == version:
            return
        self._read_at = (handle, version)
        self._document = as_str_mapping(handle.host_indexes()) or {}
        self._show()

    def _field(self, name: str) -> object:
        return self._document.get(name)

    def _show(self) -> None:
        ids: Final = _as_mapping(self._field("id_map"))
        self.id_map._load(ids)  # pyright: ignore[reportPrivateUsage]  # a view this object owns
        named: Final[dict[object, object]] = {}  # mutable-ok: built here
        for name, positions in (as_str_mapping(self._field("model_name")) or {}).items():
            named[name] = self._positions(positions, partial(self._write_named, name))
        self.model_name._load(named)  # pyright: ignore[reportPrivateUsage]  # a view this object owns
        teams: Final[dict[object, object]] = {}  # mutable-ok: built here
        for team, public, positions in _triples(self._field("team_model")):
            teams[(team, public)] = self._positions(positions, partial(self._write_team_key, (team, public)))
        self.team_model._load(teams)  # pyright: ignore[reportPrivateUsage]  # a view this object owns
        self.model_names._load(_items(self._field("model_names")))  # pyright: ignore[reportPrivateUsage]  # a view this object owns
        self.deployment_names._load(_items(self._field("deployment_names")))  # pyright: ignore[reportPrivateUsage]  # a view this object owns
        self.provider_default_ids._load(_items(self._field("provider_default_ids")))  # pyright: ignore[reportPrivateUsage]  # a view this object owns
        self._team_public = frozenset(str(name) for name in _items(self._field("team_public_model_names")))
        self.pattern_router._show(_pattern_pairs(self._field("patterns")))  # pyright: ignore[reportPrivateUsage]  # the router this object owns
        routers: Final[dict[str, HostedPatternRouter]] = {}  # mutable-ok: rebuilt each read
        for team, pairs in _team_pairs(self._field("team_patterns")):
            router = self._team_routers.get(team) or HostedPatternRouter(self, team)
            router._show(pairs)  # pyright: ignore[reportPrivateUsage]  # the router this object owns
            routers[team] = router
        self._team_routers = routers
        self.team_pattern_routers._load(routers)  # pyright: ignore[reportPrivateUsage]  # a view this object owns

    def _write_named(self, name: str, items: object) -> None:
        self.write({"model_name": {**self.model_name, name: items}})

    def _write_team_key(self, key: tuple[str, str], items: object) -> None:
        self._write_team_model({**self.team_model, key: items})

    def _positions(self, positions: object, commit: _Commit) -> object:
        """A position list as a write-through list; anything else a caller wrote, as written."""
        items: Final = _items(positions)
        return _Positions(items, commit) if isinstance(positions, list) else positions

    @property
    def team_public_model_names(self) -> frozenset[str]:
        self.refresh()
        return self._team_public

    # -- writing ----------------------------------------------------------------------------------

    def write(self, fields: Mapping[str, object]) -> None:
        """Replace the named indexes in router-core, then read them back."""
        self._apply({"host_index": {name: _json(value) for name, value in fields.items()}})
        self.refresh()

    def assign(self, field: IndexField, value: object) -> None:
        """A deprecated attribute assignment of one index (``router.model_names = {...}``)."""
        if field == "team_model":
            self._write_team_model(value)
        else:
            self.write({field: value})

    def _write_team_model(self, value: object) -> None:
        mapping: Final = _as_mapping(value)
        triples: Final[list[list[object]]] = []  # mutable-ok: JSON triples
        for key, positions in mapping.items():
            parts = cast(tuple[object, ...], key) if isinstance(key, tuple) else ()  # cast-ok: a key a caller wrote
            if len(parts) == 2:
                triples.append([str(parts[0]), str(parts[1]), _json(positions)])
        self.write({"team_model": triples})

    def write_router(self, team: str | None, pairs: list[list[object]]) -> None:  # mutable-ok: JSON pairs
        """Replace one pattern router's patterns."""
        if team is None:
            self.write({"patterns": pairs})
            return
        teams: Final = [[t, _json(_pairs_of(router))] for t, router in self._team_routers.items() if t != team]
        self.write({"team_patterns": [*teams, [team, pairs]]})

    def drop_team_router(self, team: str) -> None:
        if team not in self._team_routers:
            raise KeyError(team)
        self.write({"team_patterns": [[t, _json(_pairs_of(r))] for t, r in self._team_routers.items() if t != team]})

    # -- routing ----------------------------------------------------------------------------------

    def route(self, team: str | None, request: str, filtered: Sequence[str] | None) -> Routed | None:
        handle: Final = self._handle()
        routed: Final = (
            None
            if handle is None
            else handle.pattern_route(team, request, None if filtered is None else list(filtered))
        )
        return None if routed is None else [dict(deployment) for deployment in routed]

    def models_by_pattern(self, team: str | None, requests: list[str]) -> frozenset[str]:  # mutable-ok: read only
        handle: Final = self._handle()
        found: Final = None if handle is None else handle.models_by_pattern(team, requests)
        return frozenset(found or ())


def _items(value: object) -> list[object]:  # mutable-ok: a fresh list
    return list(cast(list[object], value)) if isinstance(value, list) else []  # cast-ok: a JSON list


def _triples(value: object) -> Iterator[tuple[str, str, object]]:
    for item in _items(value):
        parts = _items(item)
        if len(parts) == 3:
            yield str(parts[0]), str(parts[1]), parts[2]


def _pattern_pairs(value: object) -> _Pairs:
    pairs: Final[_Pairs] = []
    for item in _items(value):
        parts = _items(item)
        if len(parts) == 2:
            pairs.append((str(parts[0]), _items(parts[1])))
    return pairs


def _team_pairs(value: object) -> Iterator[tuple[str, _Pairs]]:
    for item in _items(value):
        parts = _items(item)
        if len(parts) == 2:
            yield str(parts[0]), _pattern_pairs(parts[1])


def _pairs_of(router: HostedPatternRouter) -> list[list[object]]:  # mutable-ok: JSON pairs
    view: Final[Mapping[object, object]] = router._view  # pyright: ignore[reportPrivateUsage]  # a router this module owns
    return [[regex, copy.deepcopy(deployments)] for regex, deployments in view.items()]
