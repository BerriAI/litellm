from __future__ import annotations

from typing import Final

import pytest
from pydantic import BaseModel, JsonValue, PrivateAttr

from tests.route_parity.compare import assert_model_parity, assert_parity
from tests.route_parity.models import CapturedRequest, Execution, SDKReport

SENTINEL: Final = "python-parity-fallback"


class _ComparableResponse(BaseModel):
    value: str
    _hidden_params: dict[str, object] = PrivateAttr(default_factory=dict)

    def set_hidden_param(self, key: str, value: object) -> None:
        self._hidden_params[key] = value


class _DifferentResponse(BaseModel):
    value: str


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


def test_model_parity_compares_public_values_and_ignores_private_attrs() -> None:
    python: Final = _ComparableResponse(value="same")
    rust: Final = _ComparableResponse(value="same")
    python.set_hidden_param("litellm_call_id", "python-id")
    rust.set_hidden_param("litellm_call_id", "rust-id")

    assert_model_parity(python, rust)


def test_model_parity_rejects_public_value_difference() -> None:
    python: Final = _ComparableResponse(value="python")
    rust: Final = _ComparableResponse(value="rust")

    with pytest.raises(AssertionError):
        assert_model_parity(python, rust)


def test_model_parity_rejects_type_difference() -> None:
    with pytest.raises(AssertionError):
        assert_model_parity(_ComparableResponse(value="same"), _DifferentResponse(value="same"))
