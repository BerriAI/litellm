from __future__ import annotations

from typing import Final

import pytest
from pydantic import JsonValue

from tests.route_parity.compare import assert_parity, json_values_equal
from tests.route_parity.models import CapturedRequest, Execution, SDKReport

SENTINEL: Final = "python-parity-fallback"


def _execution(*, body: JsonValue = None, markdown: str = "same", user_agent: str | None = None) -> Execution:
    return Execution(
        request=CapturedRequest(
            method="POST",
            path="/v1/test-route?mode=test",
            headers=(("authorization", "Bearer test-key"), ("content-type", "application/json")),
            body={"model": "test-model"} if body is None else body,
            user_agent=user_agent,
        ),
        report=SDKReport(response={"items": [{"text": markdown}], "model": "test-model"}),
    )


def test_parity_rejects_request_difference() -> None:
    python: Final = _execution(user_agent=SENTINEL)
    rust: Final = _execution(body={"model": "different"}, user_agent="litellm-rust")

    with pytest.raises(AssertionError):
        assert_parity(python, rust, SENTINEL)


def test_parity_rejects_response_difference() -> None:
    python: Final = _execution(user_agent=SENTINEL)
    rust: Final = _execution(markdown="different", user_agent="litellm-rust")

    with pytest.raises(AssertionError):
        assert_parity(python, rust, SENTINEL)


def test_parity_rejects_rust_fallback() -> None:
    python: Final = _execution(user_agent=SENTINEL)
    rust: Final = _execution(user_agent=SENTINEL)

    with pytest.raises(AssertionError, match="fell back"):
        assert_parity(python, rust, SENTINEL)


def test_json_comparison_preserves_scalar_types() -> None:
    assert json_values_equal(False, 0) is False
