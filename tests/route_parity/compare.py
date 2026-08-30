from __future__ import annotations

from typing import Final, TypeVar

from pydantic import BaseModel

from tests.route_parity.models import CapturedRequest, Execution

ModelT = TypeVar("ModelT", bound=BaseModel)


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


def assert_request_parity(python: CapturedRequest, accelerated: CapturedRequest) -> None:
    python_request: Final = _request_after_transformation(python)
    accelerated_request: Final = _request_after_transformation(accelerated)
    assert python_request == accelerated_request


def public_model_copy(model: ModelT) -> ModelT:
    copied: Final = model.model_copy(deep=True)
    object.__setattr__(copied, "__pydantic_private__", None)
    return copied


def assert_model_parity(python: BaseModel, accelerated: BaseModel) -> None:
    assert type(python) is type(accelerated)
    assert public_model_copy(python) == public_model_copy(accelerated)


def assert_parity(python: Execution, accelerated: Execution, python_user_agent: str) -> None:
    validate_harness(python, accelerated, python_user_agent)
    assert_request_parity(python.request, accelerated.request)
    assert python.report.response == accelerated.report.response
