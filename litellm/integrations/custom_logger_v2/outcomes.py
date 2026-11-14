from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Final, TypeAlias

from pydantic import BaseModel

from litellm.integrations.custom_logger_v2.enums import Field

JsonValue: TypeAlias = "str | int | float | bool | None | Sequence[JsonValue] | Mapping[str, JsonValue]"

MERGEABLE_FIELDS: Final = frozenset(
    {
        Field.dynamic_params,
        Field.guardrail_info,
        Field.metadata,
        Field.optional_params,
        Field.request_headers,
        Field.response_headers,
        Field.provider_request_body,
        Field.standard_logging_payload,
    }
)
OUTCOME_PATCH_FIELDS: Final = frozenset({Field.guardrail_info, Field.metadata})


class InvalidPatchError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SetField:
    field: Field
    value: object


@dataclass(frozen=True, slots=True)
class MergeField:
    field: Field
    keys: Mapping[str, object]


PatchOp: TypeAlias = SetField | MergeField


@dataclass(frozen=True, slots=True)
class Patch:
    ops: tuple[PatchOp, ...] = ()

    def set(self, field: Field, value: object) -> "Patch":
        return Patch((*self.ops, SetField(field, value)))

    def merge(self, field: Field, keys: Mapping[str, object]) -> "Patch":
        if field not in MERGEABLE_FIELDS:
            raise InvalidPatchError(f"{field.value} is not a mapping field, so it cannot be merged by key")
        return Patch((*self.ops, MergeField(field, MappingProxyType(dict(keys)))))

    @property
    def fields(self) -> frozenset[Field]:
        return frozenset(op.field for op in self.ops)


def _require_outcome_patch(patch: Patch | None) -> None:
    if patch is None:
        return
    outside: Final = patch.fields - OUTCOME_PATCH_FIELDS
    if outside:
        names: Final = ", ".join(sorted(field.value for field in outside))
        raise InvalidPatchError(f"an outcome may only patch guardrail_info and metadata, not {names}")


@dataclass(frozen=True, slots=True)
class Abort:
    terminal: ClassVar[bool] = True

    status: int
    message: str
    code: str | None = None
    headers: Mapping[str, str] | None = None
    body: JsonValue = None
    patch: Patch | None = None
    exception: Exception | None = None

    def __post_init__(self) -> None:
        _require_outcome_patch(self.patch)


@dataclass(frozen=True, slots=True)
class Respond:
    terminal: ClassVar[bool] = True

    response: "JsonValue | BaseModel"
    status: int = 200
    patch: Patch | None = None

    def __post_init__(self) -> None:
        _require_outcome_patch(self.patch)


@dataclass(frozen=True, slots=True)
class Reroute:
    terminal: ClassVar[bool] = False

    model: str


Outcome: TypeAlias = Abort | Respond | Reroute
