from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping, Sequence
from typing import Final, cast

from pydantic import BaseModel

from .models import CapturedRequest, Execution


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


def _multipart_boundary(content_type: str) -> str | None:
    parameters: Final = (parameter.strip() for parameter in content_type.split(";")[1:])
    boundary: Final = next(
        (
            value.strip().strip('"')
            for parameter in parameters
            for name, separator, value in (parameter.partition("="),)
            if separator and name.lower() == "boundary"
        ),
        None,
    )
    return boundary or None


def _request_after_transformation(request: CapturedRequest) -> CapturedRequest:
    content_type: Final = next(
        (value for name, value in request.headers if name.lower() == "content-type"),
        "",
    )
    boundary: Final = _multipart_boundary(content_type)
    if boundary is None or not isinstance(request.body, str):
        return request.model_copy(update={"user_agent": None})
    try:
        raw_body: Final = base64.b64decode(request.body, validate=True)
    except (binascii.Error, ValueError):
        return request.model_copy(update={"user_agent": None})
    canonical_boundary: Final = "litellm-parity-boundary"
    canonical_headers: Final = tuple(
        (name, f"multipart/form-data; boundary={canonical_boundary}")
        if name.lower() == "content-type"
        else (name, value)
        for name, value in request.headers
    )
    delimiter: Final = f"--{boundary}".encode()
    canonical_delimiter: Final = f"--{canonical_boundary}".encode()
    canonical_body: Final = base64.b64encode(
        raw_body.replace(delimiter, canonical_delimiter)
    ).decode("ascii")
    return request.model_copy(
        update={
            "headers": canonical_headers,
            "body": canonical_body,
            "user_agent": None,
        }
    )


def assert_request_parity(baseline: tuple[CapturedRequest, ...], candidate: tuple[CapturedRequest, ...]) -> None:
    baseline_requests: Final = tuple(_request_after_transformation(request) for request in baseline)
    candidate_requests: Final = tuple(_request_after_transformation(request) for request in candidate)
    assert_value_parity(baseline_requests, candidate_requests)


def _public_model_values(model: BaseModel) -> dict[str, object]:
    fields: Final = (*type(model).model_fields, *type(model).model_computed_fields)
    extras: Final = cast(Mapping[str, object], model.model_extra or {})
    return {
        **{name: cast(object, getattr(model, name)) for name in fields if not name.startswith("_")},
        **{name: value for name, value in extras.items() if not name.startswith("_")},
    }


def assert_model_parity(baseline: BaseModel, candidate: BaseModel) -> None:
    assert_value_parity(baseline, candidate)


def assert_value_parity(baseline: object, candidate: object, *, path: str = "$") -> None:
    assert type(baseline) is type(candidate), f"type mismatch at {path}: {type(baseline)} != {type(candidate)}"
    if isinstance(baseline, BaseModel) and isinstance(candidate, BaseModel):
        assert_value_parity(_public_model_values(baseline), _public_model_values(candidate), path=path)
        return
    if isinstance(baseline, Mapping) and isinstance(candidate, Mapping):
        baseline_mapping: Final = cast(Mapping[object, object], baseline)
        candidate_mapping: Final = cast(Mapping[object, object], candidate)
        assert frozenset((type(key), key) for key in baseline_mapping) == frozenset(
            (type(key), key) for key in candidate_mapping
        ), f"mapping keys differ at {path}"
        for key in baseline_mapping:
            assert_value_parity(baseline_mapping[key], candidate_mapping[key], path=f"{path}.{key}")
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
        for index, (baseline_item, candidate_item) in enumerate(
            zip(baseline_sequence, candidate_sequence, strict=True)
        ):
            assert_value_parity(baseline_item, candidate_item, path=f"{path}[{index}]")
        return
    assert baseline == candidate, f"value mismatch at {path}: {baseline!r} != {candidate!r}"


def assert_parity(baseline: Execution, candidate: Execution, baseline_user_agent: str) -> None:
    validate_harness(baseline, candidate, baseline_user_agent)
    assert_request_parity(baseline.requests, candidate.requests)
    assert_value_parity(baseline.report, candidate.report)
