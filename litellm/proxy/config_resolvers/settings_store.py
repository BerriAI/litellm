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

    def config_value(self, key: str) -> JsonValue:
        return self._yaml_values.get(key)

    def owned_by_config(self, key: str) -> bool:
        return key in self._yaml_values

    def rejected_writes(self, incoming: Mapping[str, JsonValue]) -> tuple[str, ...]:
        return tuple(
            sorted(
                key for key, value in incoming.items() if self.owned_by_config(key) and value != self._yaml_values[key]
            )
        )

    def apply_db_row(self, row: DbRow, db_row: Mapping[str, JsonValue]) -> None:
        previous_row: Final = self._database_rows.get(row, _EMPTY_VALUES)
        self._database_rows = MappingProxyType({**self._database_rows, row: MappingProxyType(dict(db_row))})
        self._clear_runtime_keys(frozenset((*previous_row, *db_row)))

    def resolved(self) -> Mapping[str, JsonValue]:
        return MappingProxyType(dict(self))

    def apply_runtime_values(self, values: Mapping[str, JsonValue]) -> None:
        self._runtime_values = MappingProxyType(dict(values))
        self._deleted_runtime_keys = frozenset()

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
        if self.owned_by_config(key):
            return
        self._runtime_values = MappingProxyType({**self._runtime_values, key: value})
        self._deleted_runtime_keys = self._deleted_runtime_keys - frozenset((key,))

    def __delitem__(self, key: str) -> None:
        if key not in self:
            raise KeyError(key)
        if self.owned_by_config(key):
            return
        self._runtime_values = MappingProxyType(
            {key_: value for key_, value in self._runtime_values.items() if key_ != key}
        )
        self._deleted_runtime_keys = self._deleted_runtime_keys | frozenset((key,))

    def __iter__(self) -> Iterator[str]:
        return iter(
            key
            for key in self._keys()
            if key not in self._deleted_runtime_keys
            and (key in self._runtime_values or not isinstance(self._resolution_for(key).value, Absent))
        )

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def clear(self) -> None:
        self._runtime_values = _EMPTY_VALUES
        self._deleted_runtime_keys = frozenset(key for key in self._keys() if not self.owned_by_config(key))

    def _clear_runtime(self) -> None:
        self._runtime_values = _EMPTY_VALUES
        self._deleted_runtime_keys = frozenset()

    def _clear_runtime_keys(self, keys: frozenset[str]) -> None:
        if not keys:
            return
        self._runtime_values = MappingProxyType(
            {key: value for key, value in self._runtime_values.items() if key not in keys}
        )
        self._deleted_runtime_keys = self._deleted_runtime_keys - keys

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
        return resolve(yaml_value, db_value)
