from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from types import MappingProxyType
from typing import Final

from litellm.proxy.config_resolvers._descriptors import FieldSource
from litellm.proxy.config_resolvers.settings_rules import (
    ABSENT,
    Absent,
    DbRow,
    JsonValue,
    Resolved,
    Section,
    SettingValue,
    resolve,
    rule_for,
)

_EMPTY_VALUES: Final[Mapping[str, JsonValue]] = MappingProxyType({})
_EMPTY_ROWS: Final[Mapping[DbRow, Mapping[str, JsonValue]]] = MappingProxyType({})


class SettingsStore(MutableMapping[str, JsonValue]):
    def __init__(self, section: Section) -> None:
        self._section: Final = section
        self._yaml_values: Mapping[str, JsonValue] = _EMPTY_VALUES
        self._database_rows: Mapping[DbRow, Mapping[str, JsonValue]] = _EMPTY_ROWS
        self._runtime_values: Mapping[str, JsonValue] = _EMPTY_VALUES
        self._deleted_runtime_keys: frozenset[str] = frozenset()

    def load_yaml(self, mapping: Mapping[str, JsonValue]) -> None:
        self._yaml_values = MappingProxyType(dict(mapping))
        self._clear_runtime()

    def apply_db_row(self, row: DbRow, db_row: Mapping[str, JsonValue]) -> None:
        self._database_rows = MappingProxyType({**self._database_rows, row: MappingProxyType(dict(db_row))})
        self._clear_runtime()

    def resolved(self) -> Mapping[str, JsonValue]:
        return MappingProxyType(dict(self))

    def source(self, key: str) -> FieldSource:
        return self._resolution_for(key).source

    def __getitem__(self, key: str) -> JsonValue:
        if key in self._deleted_runtime_keys:
            raise KeyError(key)
        if key in self._runtime_values:
            return self._runtime_values[key]
        resolved: Final = self._resolution_for(key)
        if isinstance(resolved.value, Absent):
            raise KeyError(key)
        return resolved.value

    def __setitem__(self, key: str, value: JsonValue) -> None:
        self._runtime_values = MappingProxyType({**self._runtime_values, key: value})
        self._deleted_runtime_keys = self._deleted_runtime_keys - {key}

    def __delitem__(self, key: str) -> None:
        if key not in self:
            raise KeyError(key)
        self._runtime_values = MappingProxyType(
            {key_: value for key_, value in self._runtime_values.items() if key_ != key}
        )
        self._deleted_runtime_keys = self._deleted_runtime_keys | {key}

    def __iter__(self) -> Iterator[str]:
        return iter(key for key in self._keys() if key not in self._deleted_runtime_keys)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def _clear_runtime(self) -> None:
        self._runtime_values = _EMPTY_VALUES
        self._deleted_runtime_keys = frozenset()

    def _keys(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *self._yaml_values,
                    *(key for row in self._database_rows.values() for key in row),
                    *self._runtime_values,
                )
            )
        )

    def _resolution_for(self, key: str) -> Resolved:
        rule: Final = rule_for(self._section, key)
        yaml_value: Final[SettingValue] = self._yaml_values.get(key, ABSENT)
        db_value: Final[SettingValue] = self._database_rows.get(rule.db_row, _EMPTY_VALUES).get(key, ABSENT)
        return resolve(rule, yaml_value, db_value)
