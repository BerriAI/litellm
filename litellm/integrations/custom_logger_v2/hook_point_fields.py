from collections.abc import Iterator, Mapping
from functools import reduce
from typing import Final

from litellm.integrations.custom_logger_v2.enums import Field
from litellm.integrations.custom_logger_v2.legacy import KwargsFields
from litellm.integrations.custom_logger_v2.outcomes import InvalidPatchError, Patch, PatchOp, SetField
from litellm.integrations.custom_logger_v2.patches import LiveFields

_CANDIDATE_FIELDS: Final = (Field.deprecated_kwargs, Field.candidate_deployments, Field.model)


class CandidateFields(LiveFields):
    __slots__ = ("_deployments", "_model")

    def __init__(self, deployments: object, model: str) -> None:
        self._deployments: Final = deployments
        self._model: Final = model

    @property
    def deployments(self) -> object:
        return self._deployments

    def __getitem__(self, field: Field) -> object:
        if field is Field.deprecated_kwargs:
            return self._deployments
        if field is Field.model:
            return self._model
        if field is Field.candidate_deployments and isinstance(self._deployments, list | tuple):
            return tuple(self._deployments)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportUnknownVariableType]  # untyped deployment dicts
        raise KeyError(field)

    def __iter__(self) -> Iterator[Field]:
        return iter(tuple(field for field in _CANDIDATE_FIELDS if self._has(field)))

    def __len__(self) -> int:
        return sum(1 for field in _CANDIDATE_FIELDS if self._has(field))

    def _has(self, field: Field) -> bool:
        return field is not Field.candidate_deployments or isinstance(self._deployments, list | tuple)

    def apply(self, patch: Patch) -> Mapping[Field, object]:
        return reduce(_narrow, patch.ops, self)


def _narrow(fields: Mapping[Field, object], op: PatchOp) -> Mapping[Field, object]:
    if not isinstance(fields, CandidateFields) or not isinstance(op, SetField):
        raise InvalidPatchError("candidate deployments are replaced whole, never merged")
    if op.field is Field.deprecated_kwargs:
        return CandidateFields(op.value, str(fields[Field.model]))
    if op.field is not Field.candidate_deployments:
        raise InvalidPatchError(f"{op.field.value} has no place in filter_deployments")
    return CandidateFields(_narrowed(fields.deployments, op.value), str(fields[Field.model]))


def _narrowed(current: object, proposed: object) -> list[object]:  # mutable-ok: legacy filters take a list
    if not isinstance(current, list | tuple) or not isinstance(proposed, list | tuple):
        raise InvalidPatchError("candidate_deployments must stay a sequence of the current candidates")
    allowed: Final = frozenset(map(id, current))  # pyright: ignore[reportUnknownArgumentType]  # untyped deployment dicts
    picked: Final = tuple(map(id, proposed))  # pyright: ignore[reportUnknownArgumentType]  # see above
    if not allowed.issuperset(picked) or len(frozenset(picked)) != len(picked):
        raise InvalidPatchError("filter_deployments may only narrow the candidates, never add or repeat one")
    return list(proposed)  # pyright: ignore[reportUnknownArgumentType]  # see above


class ResultFields(LiveFields):
    __slots__ = ("_request", "_slot", "value")

    def __init__(self, request: object, keys: Mapping[Field, str], slot: Field, value: object) -> None:
        self._request: Final = KwargsFields(request, keys)
        self._slot: Final = slot
        self.value = value

    def __getitem__(self, field: Field) -> object:
        if field is self._slot or field is Field.deprecated_response_obj:
            return self.value
        return self._request[field]

    def __iter__(self) -> Iterator[Field]:
        return iter((*self._request, self._slot, Field.deprecated_response_obj))

    def __len__(self) -> int:
        return len(self._request) + 2

    def apply(self, patch: Patch) -> Mapping[Field, object]:
        values: Final = tuple(_slot_value(op, self._slot) for op in patch.ops)
        if values:
            self.value = values[-1]
        return self


def _slot_value(op: PatchOp, slot: Field) -> object:
    if not isinstance(op, SetField) or op.field not in (slot, Field.deprecated_response_obj):
        raise InvalidPatchError(f"{op.field.value} cannot be patched here, only {slot.value}")
    return op.value


__all__ = ["CandidateFields", "ResultFields"]
