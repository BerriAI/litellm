from __future__ import annotations

from typing import Final, cast

from pydantic import JsonValue

from tests.route_parity.models import CapturedRequest, Execution


def validate_harness(python: Execution, accelerated: Execution, python_user_agent: str) -> None:
    if python.request.user_agent != python_user_agent:
        raise AssertionError(
            f"Python provider request did not carry fallback sentinel user-agent {python_user_agent!r}: "
            f"{python.request.user_agent!r}"
        )
    if accelerated.request.user_agent == python_user_agent:
        raise AssertionError("accelerated route fell back to the Python HTTP implementation")


def _request_after_transformation(request: CapturedRequest) -> CapturedRequest:
    return request.model_copy(update={"user_agent": None})


def json_values_equal(left: JsonValue, right: JsonValue) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(json_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            json_values_equal(left_value, right_value) for left_value, right_value in zip(left, right, strict=True)
        )
    return left == right


def assert_request_parity(python: Execution, accelerated: Execution) -> None:
    python_request: Final = cast(JsonValue, _request_after_transformation(python.request).model_dump(mode="json"))
    accelerated_request: Final = cast(
        JsonValue,
        _request_after_transformation(accelerated.request).model_dump(mode="json"),
    )
    assert json_values_equal(python_request, accelerated_request)


def assert_response_parity(python: Execution, accelerated: Execution) -> None:
    assert json_values_equal(python.report.response, accelerated.report.response)


def assert_parity(python: Execution, accelerated: Execution, python_user_agent: str) -> None:
    validate_harness(python, accelerated, python_user_agent)
    assert_request_parity(python, accelerated)
    assert_response_parity(python, accelerated)
