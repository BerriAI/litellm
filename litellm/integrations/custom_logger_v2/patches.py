from abc import abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from functools import reduce
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

from litellm.integrations.custom_logger_v2.enums import Field
from litellm.integrations.custom_logger_v2.outcomes import InvalidPatchError, MergeField, Patch, PatchOp, SetField

_STRING_KEYED: Final = TypeAdapter(Mapping[str, object])


class LiveFields(Mapping[Field, object]):
    __slots__ = ()

    @abstractmethod
    def apply(self, patch: Patch) -> Mapping[Field, object]: ...


class PatchedFields(Mapping[Field, object]):
    __slots__ = ("_base", "_changed")

    def __init__(self, base: Mapping[Field, object], changed: Mapping[Field, object]) -> None:
        self._base = base
        self._changed = changed

    def with_value(self, field: Field, value: object) -> "PatchedFields":
        return PatchedFields(self._base, MappingProxyType({**self._changed, field: value}))

    def __getitem__(self, field: Field) -> object:
        return self._changed[field] if field in self._changed else self._base[field]

    def __iter__(self) -> Iterator[Field]:
        return iter(self._changed.keys() | self._base.keys())

    def __len__(self) -> int:
        return len(self._changed.keys() | self._base.keys())


def _with_value(fields: Mapping[Field, object], field: Field, value: object) -> PatchedFields:
    patched: Final = fields if isinstance(fields, PatchedFields) else PatchedFields(fields, MappingProxyType({}))
    return patched.with_value(field, value)


def _merged(current: object, field: Field, keys: Mapping[str, object]) -> Mapping[str, object]:
    if current is None:
        return MappingProxyType(dict(keys))
    if not isinstance(current, Mapping):
        raise InvalidPatchError(f"{field.value} holds no mapping, so keys cannot be merged into it")
    return MappingProxyType({**_STRING_KEYED.validate_python(current), **keys})


def _apply_op(fields: Mapping[Field, object], op: PatchOp) -> Mapping[Field, object]:
    match op:
        case SetField(field=field, value=value):
            return _with_value(fields, field, value)
        case MergeField(field=field, keys=keys):
            return _with_value(fields, field, _merged(fields.get(field), field, keys))


def apply_patch(fields: Mapping[Field, object], patch: Patch) -> Mapping[Field, object]:
    if isinstance(fields, LiveFields):
        return fields.apply(patch)
    return reduce(_apply_op, patch.ops, fields)


def apply_patches(fields: Mapping[Field, object], patches: Iterable[Patch]) -> Mapping[Field, object]:
    return reduce(apply_patch, patches, fields)
