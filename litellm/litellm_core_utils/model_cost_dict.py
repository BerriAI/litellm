from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Protocol, TypeAlias, TypeVar, cast, overload

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.rust_bridge._native import ModelCostLayers as NativeModelCostLayers
from litellm.rust_bridge._native import ModelRegistry

_PlainRow: TypeAlias = dict[str, object]
_PlainCatalog: TypeAlias = dict[str, _PlainRow]
_DefaultT = TypeVar("_DefaultT")
_MISSING: Final = object()
_JsonRow: TypeAlias = dict[str, JsonValue]
_JsonCatalog: TypeAlias = dict[str, _JsonRow]
_DeepCopyMemo: TypeAlias = dict[int, object]
_ComposedRows: TypeAlias = list[tuple[str, _JsonRow, frozenset[str]]]
_ComposedKeyRows: TypeAlias = list[tuple[str, _JsonRow | None, frozenset[str]]]
_JSON_VALUE_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


def _is_native_json_value(value: object) -> bool:
    try:
        _JSON_VALUE_ADAPTER.validate_python(value, strict=True)
    except ValidationError:
        return False
    return True


def to_native_json_fields(fields: Mapping[str, object]) -> _JsonRow:
    return {
        name: _JSON_VALUE_ADAPTER.validate_python(value, strict=True)
        for name, value in fields.items()
        if _is_native_json_value(value)
    }


class ServedRow(dict[str, object]):
    def __init__(
        self,
        values: Mapping[str, object] | None = None,
        names: tuple[str, ...] = (),
        registered_fields: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__(values or {})
        self.names: Final = names
        self.registered_fields: Final = registered_fields

    def __copy__(self) -> _PlainRow:
        return dict(self)

    def __deepcopy__(self, memo: _DeepCopyMemo) -> _PlainRow:
        return deepcopy(dict(self), memo)

    def __reduce__(self) -> tuple[type[_PlainRow], tuple[_PlainRow]]:
        return dict, (dict(self),)


class WriteSink(Protocol):
    def wrote(self, key: str, row: Mapping[str, object]) -> None: ...

    def removed(self, key: str) -> None: ...

    def cleared(self, keys: Iterable[str]) -> None: ...


class _DictUpdateSource(Protocol):
    def keys(self, /) -> Iterable[str]: ...

    def __getitem__(self, key: str, /) -> ServedRow: ...


class ModelCostDirectWriteWarning(DeprecationWarning):
    pass


class ModelCostDict(dict[str, ServedRow]):
    def __init__(
        self,
        values: Mapping[str, Mapping[str, object]] | None = None,
        *,
        sink: WriteSink | None = None,
    ) -> None:
        self._sink: Final = sink
        super().__init__(
            (key, value if isinstance(value, ServedRow) else ServedRow(value)) for key, value in (values or {}).items()
        )

    def _warn_direct_write(self, stacklevel: int = 3) -> None:
        warnings.warn(
            "Direct writes to litellm.model_cost are deprecated; use set_model_cost_overrides instead",
            ModelCostDirectWriteWarning,
            stacklevel=stacklevel,
        )

    def __setitem__(self, key: str, value: ServedRow) -> None:
        super().__setitem__(key, value)
        if self._sink is not None:
            self._sink.wrote(key, cast(Mapping[str, object], value))
        self._warn_direct_write()

    def _update(
        self,
        other: _DictUpdateSource | Iterable[tuple[str, ServedRow]],
        warning_stacklevel: int,
        /,
        **kwargs: ServedRow,
    ) -> None:
        values: Final = dict(other, **kwargs)
        super().update(values)
        if self._sink is not None:
            for key, row in values.items():
                self._sink.wrote(key, cast(Mapping[str, object], row))
        if values:
            self._warn_direct_write(warning_stacklevel)

    def update(
        self,
        other: _DictUpdateSource | Iterable[tuple[str, ServedRow]] = (),
        /,
        **kwargs: ServedRow,
    ) -> None:
        self._update(other, 4, **kwargs)

    @overload
    def setdefault(self, key: str, default: ServedRow, /) -> ServedRow: ...

    @overload
    def setdefault(self, key: str, default: None = None, /) -> ServedRow | None: ...

    def setdefault(self, key: str, default: ServedRow | None = None, /) -> ServedRow | None:
        if key in self:
            return super().__getitem__(key)
        value: Final = cast(ServedRow | None, super().setdefault(key, cast(ServedRow, default)))
        if self._sink is not None and value is not None:
            self._sink.wrote(key, cast(Mapping[str, object], value))
        self._warn_direct_write()
        return value

    def __ior__(self, other: object, /) -> ModelCostDict:
        self._update(
            cast(_DictUpdateSource | Iterable[tuple[str, ServedRow]], other),
            4,
        )
        return self

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        if self._sink is not None:
            self._sink.removed(key)
        self._warn_direct_write()

    @overload
    def pop(self, key: str, /) -> ServedRow: ...

    @overload
    def pop(self, key: str, default: ServedRow, /) -> ServedRow: ...

    @overload
    def pop(self, key: str, default: None = None, /) -> ServedRow | None: ...

    @overload
    def pop(self, key: str, default: _DefaultT, /) -> ServedRow | _DefaultT: ...

    def pop(self, key: str, default: object = _MISSING, /) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]  # retain dict's omitted-default KeyError behavior
        if key not in self:
            return super().pop(key) if default is _MISSING else super().pop(key, default)
        value: Final = super().pop(key)
        if self._sink is not None:
            self._sink.removed(key)
        self._warn_direct_write()
        return value

    def popitem(self) -> tuple[str, ServedRow]:
        key, value = super().popitem()
        if self._sink is not None:
            self._sink.removed(key)
        self._warn_direct_write()
        return key, value

    def clear(self) -> None:
        keys: Final = tuple(self)
        super().clear()
        if keys:
            if self._sink is not None:
                self._sink.cleared(keys)
            self._warn_direct_write()

    def _plain_snapshot(self) -> _PlainCatalog:
        rows: Final = {id(row): dict(row) for row in self.values()}
        return {key: rows[id(row)] for key, row in self.items()}

    def copy(self) -> _PlainCatalog:  # pyright: ignore[reportIncompatibleMethodOverride]  # plain rows are required by the serialization contract
        return self._plain_snapshot()

    def __copy__(self) -> _PlainCatalog:
        return self._plain_snapshot()

    def __deepcopy__(self, memo: _DeepCopyMemo) -> _PlainCatalog:
        return deepcopy(self._plain_snapshot(), memo)

    def __reduce__(self) -> tuple[type[_PlainCatalog], tuple[_PlainCatalog]]:
        return dict, (self._plain_snapshot(),)


@dataclass(frozen=True, slots=True)
class RegistrationIdentity:
    original_key: str
    custom_llm_provider: str | None
    handle: int
    key: str
    alias_keys: tuple[str, ...]
    fields: Mapping[str, object]
    defaults: Mapping[str, object]
    reset_fields: frozenset[str]
    cleanup_overrides: tuple[str, ...]
    fallback: bool
    names_the_row: bool


_IdentityIndex: TypeAlias = dict[int, RegistrationIdentity]


class ModelCostLayers:
    def __init__(self) -> None:
        self._native: NativeModelCostLayers = NativeModelCostLayers()
        self._identities: _IdentityIndex = {}

    def record_registration(
        self,
        key: str,
        inherits_from: str | None,
        fields: Mapping[str, object],
        *,
        original_key: str | None = None,
        custom_llm_provider: str | None = None,
        alias_keys: tuple[str, ...] = (),
        defaults: Mapping[str, object] | None = None,
        reset_fields: frozenset[str] = frozenset(),
        cleanup_overrides: tuple[str, ...] = (),
        fallback: bool = False,
        names_the_row: bool = False,
    ) -> int:
        identity_fields: Final = MappingProxyType(dict(fields))
        identity_defaults: Final = MappingProxyType(dict(defaults or {}))
        native_fields: Final = to_native_json_fields(identity_fields)
        native_defaults: Final = to_native_json_fields(identity_defaults)
        handle: Final = self._native.record_registration_with_options(
            key,
            inherits_from,
            native_fields,
            {
                "alias_keys": alias_keys,
                "defaults": native_defaults,
                "reset_fields": reset_fields,
                "cleanup_overrides": cleanup_overrides,
                "fallback": fallback,
                "names_the_row": names_the_row,
                "ignore_none": True,
            },
        )
        self._identities[handle] = RegistrationIdentity(
            original_key=original_key or key,
            custom_llm_provider=custom_llm_provider,
            handle=handle,
            key=key,
            alias_keys=alias_keys,
            fields=identity_fields,
            defaults=identity_defaults,
            reset_fields=reset_fields,
            cleanup_overrides=cleanup_overrides,
            fallback=fallback,
            names_the_row=names_the_row,
        )
        return handle

    def record_direct(self, key: str, row: Mapping[str, object]) -> None:
        self._native.record_direct(key, to_native_json_fields(row))

    def rebind_registration(
        self,
        identity: RegistrationIdentity,
        key: str,
        inherits_from: str | None,
        *,
        alias_keys: tuple[str, ...] = (),
        defaults: Mapping[str, object] | None = None,
        cleanup_overrides: tuple[str, ...] | None = None,
        fallback: bool | None = None,
        names_the_row: bool | None = None,
    ) -> bool:
        identity_defaults: Final = defaults if defaults is not None else identity.defaults
        identity_cleanup_overrides: Final = (
            cleanup_overrides if cleanup_overrides is not None else identity.cleanup_overrides
        )
        identity_names_the_row: Final = names_the_row if names_the_row is not None else identity.names_the_row
        identity_fallback: Final = fallback if fallback is not None else identity.fallback
        rebound: Final = self._native.rebind_registration(
            identity.handle,
            key,
            inherits_from,
            to_native_json_fields(identity.fields),
            {
                "alias_keys": alias_keys,
                "defaults": to_native_json_fields(identity_defaults),
                "reset_fields": identity.reset_fields,
                "cleanup_overrides": identity_cleanup_overrides,
                "fallback": identity_fallback,
                "names_the_row": identity_names_the_row,
                "ignore_none": True,
            },
        )
        if not rebound:
            return False
        self._identities[identity.handle] = RegistrationIdentity(
            original_key=identity.original_key,
            custom_llm_provider=identity.custom_llm_provider,
            handle=identity.handle,
            key=key,
            alias_keys=alias_keys,
            fields=identity.fields,
            defaults=MappingProxyType(dict(identity_defaults)),
            reset_fields=identity.reset_fields,
            cleanup_overrides=identity_cleanup_overrides,
            fallback=identity_fallback,
            names_the_row=identity_names_the_row,
        )
        return rebound

    def record_tombstone(self, key: str) -> None:
        self._native.record_tombstone(key)

    def set_owner(
        self,
        name: str,
        precedence: Literal["override", "fallback"],
        rows: Mapping[str, Mapping[str, object]],
    ) -> None:
        native_rows: Final = {key: to_native_json_fields(row) for key, row in rows.items()}
        self._native.set_owner(name, precedence, native_rows)

    def drop_owner(self, name: str) -> None:
        self._native.drop_owner(name)

    def forget_writes(self) -> None:
        self._native.forget_writes()
        self._identities.clear()

    def snapshot(self) -> ModelCostLayers:
        snapshot: Final = ModelCostLayers()
        snapshot._native = self._native.snapshot()
        snapshot._identities = dict(self._identities)
        return snapshot

    def restore(self, snapshot: ModelCostLayers) -> None:
        self._native.restore(snapshot._native)
        self._identities = dict(snapshot._identities)

    def compose(self, base: ModelRegistry) -> _ComposedRows:
        return self._native.compose(base)

    def compose_keys(self, base: ModelRegistry, keys: frozenset[str]) -> _ComposedKeyRows:
        return self._native.compose_keys(base, keys)

    def compose_prefix(self, base: ModelRegistry, handle: int) -> _ComposedRows:
        return self._native.compose_prefix(base, handle)

    @property
    def identities(self) -> tuple[RegistrationIdentity, ...]:
        return tuple(self._identities.values())


class LayersSink:
    def __init__(self, layers: ModelCostLayers) -> None:
        self._layers: Final = layers

    @staticmethod
    def _remove_known_model(key: str) -> None:
        import litellm

        module_globals: Final = cast(dict[str, object], vars(litellm))
        for name, value in module_globals.items():
            if name.endswith("_models") and isinstance(value, set):
                cast(set[str], value).discard(key)

    def _refresh(self, keys: Iterable[str], rows: Mapping[str, Mapping[str, object]] | None = None) -> None:
        import litellm
        from litellm import utils

        for key in keys:
            self._remove_known_model(key)
        utils.invalidate_model_cost_caches()
        provider_rows: Final[dict[str, dict[str, object]]] = (
            {key: dict(row) for key, row in rows.items()} if rows is not None else {}
        )
        litellm.add_known_models(provider_rows)  # pyright: ignore[reportUnknownMemberType]  # legacy helper has unknown dict value types

    def wrote(self, key: str, row: Mapping[str, object]) -> None:
        self._layers.record_direct(key, row)
        self._refresh((key,), {key: row})

    def removed(self, key: str) -> None:
        self._layers.record_tombstone(key)
        self._refresh((key,))

    def cleared(self, keys: Iterable[str]) -> None:
        key_list: Final = tuple(keys)
        for key in key_list:
            self._layers.record_tombstone(key)
        self._refresh(key_list)
