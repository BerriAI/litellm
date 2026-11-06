from typing import Final

import pytest
from pydantic import ValidationError

from litellm.proxy.workflows.monty.models import MontyLimits, PythonProgram


def _artifact() -> PythonProgram:
    return PythonProgram(
        contract_version="1",
        source="def transform(value): return value",
        function_name="transform",
        parameter_names=("value",),
        input_schema={},
        output_schema={},
        source_sha256="a" * 64,
    )


def test_program_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        PythonProgram.model_validate({**_artifact().model_dump(), "credentials": "not-allowed"})


def test_limits_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        MontyLimits.model_validate({"mount": "/"})


@pytest.mark.parametrize("field", tuple(MontyLimits.model_fields))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1, 0, 10**20])
def test_every_limit_is_positive_finite_and_bounded(field: str, value: float | int) -> None:
    with pytest.raises(ValidationError):
        MontyLimits.model_validate({field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("function_name", "not valid"),
        ("function_name", "class"),
        ("function_name", "x" * 129),
        ("parameter_names", ("bad-name",)),
        ("parameter_names", ("x" * 129,)),
        ("parameter_names", tuple(f"p{i}" for i in range(33))),
        ("source_sha256", "A" * 64),
        ("source_sha256", "a" * 63),
        ("source_sha256", "g" * 64),
    ],
)
def test_artifact_names_count_and_digest_are_bounded(field: str, value: object) -> None:
    payload: Final = {**_artifact().model_dump(), field: value}
    with pytest.raises(ValidationError):
        PythonProgram.model_validate(payload)
