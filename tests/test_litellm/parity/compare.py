from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final, cast

from pydantic import BaseModel

from tests.test_litellm.parity.models import CapturedRequest, Execution, ParityTrace


def _stable_request(request: CapturedRequest) -> dict[str, object]:
    return request.model_dump(mode="json", exclude={"user_agent"})


def assert_parity(python: Execution, rust: Execution, python_user_agent: str) -> None:
    assert python.report.native.rust_enabled is False
    assert rust.report.native.rust_enabled is True
    assert rust.report.native.native_callable_loaded is True
    assert rust.report.native.native_handled_case is True
    assert tuple(request.user_agent for request in python.requests) == (python_user_agent,) * len(python.requests)
    assert all(request.user_agent != python_user_agent for request in rust.requests)
    assert tuple(_stable_request(request) for request in rust.requests) == tuple(
        _stable_request(request) for request in python.requests
    )
    assert rust.report.trace == python.report.trace
    assert python.report.trace.exception is None
    assert python.report.trace.outputs


def _short(value: object) -> str:
    rendered: Final = repr(value)
    return rendered if len(rendered) <= 240 else f"{rendered[:237]}..."


def _diff(left: object, right: object, path: str = "$") -> tuple[str, ...]:
    if isinstance(left, BaseModel) and isinstance(right, BaseModel):
        return _diff(left.model_dump(mode="json"), right.model_dump(mode="json"), path)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_mapping: Final = cast(Mapping[str, object], left)
        right_mapping: Final = cast(Mapping[str, object], right)
        left_keys: Final = frozenset(left_mapping)
        right_keys: Final = frozenset(right_mapping)
        missing: Final = tuple(f"{path}.{key}: missing from left" for key in sorted(right_keys - left_keys))
        extra: Final = tuple(f"{path}.{key}: missing from right" for key in sorted(left_keys - right_keys))
        shared: Final = tuple(
            line
            for key in sorted(left_keys & right_keys)
            for line in _diff(left_mapping[key], right_mapping[key], f"{path}.{key}")
        )
        return (*missing, *extra, *shared)
    if (
        isinstance(left, Sequence)
        and not isinstance(left, (str, bytes))
        and isinstance(right, Sequence)
        and not isinstance(right, (str, bytes))
    ):
        left_sequence: Final = cast(Sequence[object], left)
        right_sequence: Final = cast(Sequence[object], right)
        length_diff: Final = (
            (f"{path}: lengths differ ({len(left_sequence)} != {len(right_sequence)})",)
            if len(left_sequence) != len(right_sequence)
            else ()
        )
        item_diff: Final = tuple(
            line
            for index, (left_item, right_item) in enumerate(zip(left_sequence, right_sequence))
            for line in _diff(left_item, right_item, f"{path}[{index}]")
        )
        return (*length_diff, *item_diff)
    left_value: Final = cast(object, left)
    right_value: Final = cast(object, right)
    return () if left_value == right_value else (f"{path}: {_short(left_value)} != {_short(right_value)}",)


def parity_comparison(left: object, right: object) -> list[str] | None:
    if not isinstance(left, (ParityTrace, CapturedRequest)) or type(left) is not type(right):
        return None
    differences: Final = _diff(left, right)
    return [f"Comparing {type(left).__name__} values:", *(f"  {line}" for line in differences)]
