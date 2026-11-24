from collections.abc import Sequence

from typing_extensions import ReadOnly, TypedDict


class ValidationErrorDetail(TypedDict):
    type: ReadOnly[str]
    loc: ReadOnly[tuple[int | str, ...]]
    msg: ReadOnly[str]


def public_validation_errors(errors: Sequence[ValidationErrorDetail]) -> tuple[ValidationErrorDetail, ...]:
    return tuple(ValidationErrorDetail(type=error["type"], loc=error["loc"], msg=error["msg"]) for error in errors)
