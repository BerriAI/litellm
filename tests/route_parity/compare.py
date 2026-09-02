from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final, TypeVar, cast

from pydantic import BaseModel

from tests.route_parity.models import CapturedRequest, Execution

ModelT = TypeVar("ModelT", bound=BaseModel)


def validate_harness(baseline: Execution, candidate: Execution, baseline_user_agent: str) -> None:
    for request in baseline.requests:
        if request.user_agent != baseline_user_agent:
            raise AssertionError(
                f"baseline provider request did not carry sentinel user-agent {baseline_user_agent!r}: "
                f"{request.user_agent!r}"
            )
    for request in candidate.requests:
        if request.user_agent == baseline_user_agent:
            raise AssertionError("candidate route fell back to the baseline HTTP implementation")


def _request_after_transformation(request: CapturedRequest) -> CapturedRequest:
    return request.model_copy(update={"user_agent": None})


def assert_request_parity(baseline: tuple[CapturedRequest, ...], candidate: tuple[CapturedRequest, ...]) -> None:
    baseline_requests: Final = tuple(_request_after_transformation(request) for request in baseline)
    candidate_requests: Final = tuple(_request_after_transformation(request) for request in candidate)
    assert baseline_requests == candidate_requests


def public_model_copy(model: ModelT) -> ModelT:
    copied: Final = model.model_copy(deep=True)
    object.__setattr__(copied, "__pydantic_private__", None)
    return copied


def assert_model_parity(baseline: BaseModel, candidate: BaseModel) -> None:
    assert type(baseline) is type(candidate)
    _assert_value_parity(
        public_model_copy(baseline).model_dump(mode="python"),
        public_model_copy(candidate).model_dump(mode="python"),
        path="$",
    )


def _assert_value_parity(baseline: object, candidate: object, *, path: str) -> None:
    if isinstance(baseline, Mapping) and isinstance(candidate, Mapping):
        baseline_mapping: Final = cast(Mapping[object, object], baseline)
        candidate_mapping: Final = cast(Mapping[object, object], candidate)
        assert baseline_mapping.keys() == candidate_mapping.keys(), f"mapping keys differ at {path}"
        for key in baseline_mapping:
            _assert_value_parity(baseline_mapping[key], candidate_mapping[key], path=f"{path}.{key}")
        return
    if (
        isinstance(baseline, Sequence)
        and not isinstance(baseline, (str, bytes))
        and isinstance(candidate, Sequence)
        and not isinstance(candidate, (str, bytes))
    ):
        baseline_sequence: Final = cast(Sequence[object], baseline)
        candidate_sequence: Final = cast(Sequence[object], candidate)
        assert len(baseline_sequence) == len(candidate_sequence), f"sequence lengths differ at {path}"
        for index, (baseline_item, candidate_item) in enumerate(zip(baseline_sequence, candidate_sequence)):
            _assert_value_parity(baseline_item, candidate_item, path=f"{path}[{index}]")
        return
    assert baseline == candidate, f"value mismatch at {path}: {baseline!r} != {candidate!r}"


def assert_parity(baseline: Execution, candidate: Execution, baseline_user_agent: str) -> None:
    validate_harness(baseline, candidate, baseline_user_agent)
    assert_request_parity(baseline.requests, candidate.requests)
    assert baseline.report == candidate.report
