from __future__ import annotations

from typing import Final

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


def assert_parity(python: Execution, accelerated: Execution, python_user_agent: str) -> None:
    validate_harness(python, accelerated, python_user_agent)
    python_request: Final = _request_after_transformation(python.request)
    accelerated_request: Final = _request_after_transformation(accelerated.request)
    assert python_request == accelerated_request
    assert python.report.response == accelerated.report.response
