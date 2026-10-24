from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.rust_bridge._native import ModelCostLayers as NativeModelCostLayers
from litellm.rust_bridge._native import ModelRegistry

_PlainRow: TypeAlias = dict[str, object]
_PlainCatalog: TypeAlias = dict[str, _PlainRow]
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


class ModelCostDict(dict[str, ServedRow]):
    def __init__(self, values: Mapping[str, Mapping[str, object]] | None = None) -> None:
        super().__init__(
            (key, value if isinstance(value, ServedRow) else ServedRow(value)) for key, value in (values or {}).items()
        )

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
