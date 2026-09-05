from __future__ import annotations

from types import MappingProxyType
from typing import Final

import pytest
from pydantic import BaseModel, ConfigDict, JsonValue, PrivateAttr

from .compare import assert_model_parity, assert_parity
from .models import CapturedRequest, Execution, SDKError, SDKSuccess, sdk_error_report

SENTINEL: Final = "python-parity-fallback"


class _ComparableResponse(BaseModel):
    value: str
    _hidden_params: dict[str, object] = PrivateAttr(default_factory=dict)

    def set_hidden_param(self, key: str, value: object) -> None:
        self._hidden_params[key] = value


class _DifferentResponse(BaseModel):
    value: str


class _FloatResponse(BaseModel):
    values: list[float]


class _PublicValue(BaseModel):
    model_config = ConfigDict(extra="allow")

    value: object


class _PublicError(ValueError):
    status_code: Final = 400


def _execution(*, body: JsonValue = None, markdown: str = "same", user_agent: str | None = None) -> Execution:
    return Execution(
        requests=(
            CapturedRequest(
                method="POST",
                path="/v1/test-route?mode=test",
                headers=(("authorization", "Bearer test-key"), ("content-type", "application/json")),
                body={"model": "test-model"} if body is None else body,
                user_agent=user_agent,
            ),
        ),
        report=SDKSuccess(response={"items": [{"text": markdown}], "model": "test-model"}),
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


def test_parity_rejects_error_difference() -> None:
    python: Final = Execution(
        requests=(),
        report=SDKError(
            exception_type="litellm.exceptions.BadRequestError",
            message="bad request",
            status_code=400,
            code=None,
            error_type=None,
            param=None,
            model="test-model",
            llm_provider="test-provider",
        ),
    )
    rust: Final = python.model_copy(update={"report": python.report.model_copy(update={"status_code": 500})})

    with pytest.raises(AssertionError):
        assert_parity(python, rust, SENTINEL)


def test_sdk_error_report_removes_traceback_but_keeps_public_fields() -> None:
    error: Final = _PublicError("invalid input\nTraceback (most recent call last):\n  unstable")

    report: Final = sdk_error_report(error)

    assert report.exception_type.endswith("._PublicError")
    assert report.message == "invalid input"
    assert report.status_code == 400


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


def test_model_parity_rejects_wire_float_rounding_difference() -> None:
    with pytest.raises(AssertionError, match=r"\$\.values\[0\]"):
        assert_model_parity(
            _FloatResponse(values=[0.22590550796036835]),
            _FloatResponse(values=[0.22590550796036837]),
        )


def test_model_parity_rejects_meaningful_float_difference() -> None:
    with pytest.raises(AssertionError, match=r"\$\.values\[0\]"):
        assert_model_parity(
            _FloatResponse(values=[0.22590550796036835]),
            _FloatResponse(values=[0.2259]),
        )


@pytest.mark.parametrize(
    ("baseline", "candidate"),
    (
        (_ComparableResponse(value="same"), {"value": "same"}),
        (_ComparableResponse(value="same"), _DifferentResponse(value="same")),
        (True, 1),
        (1, 1.0),
        (["same"], ("same",)),
        ({"value": "same"}, MappingProxyType({"value": "same"})),
        ({True: "same"}, {1: "same"}),
    ),
    ids=("model-dict", "model-class", "bool-int", "int-float", "list-tuple", "mapping-class", "key-type"),
)
def test_model_parity_rejects_nested_type_changes(baseline: object, candidate: object) -> None:
    with pytest.raises(AssertionError, match=r"\$\.value\[0\]"):
        assert_model_parity(_PublicValue(value=[baseline]), _PublicValue(value=[candidate]))


def test_model_parity_ignores_nested_private_attributes() -> None:
    baseline: Final = _ComparableResponse(value="same")
    candidate: Final = _ComparableResponse(value="same")
    baseline.set_hidden_param("request_id", "baseline")
    candidate.set_hidden_param("request_id", "candidate")

    assert_model_parity(_PublicValue(value={"nested": [baseline]}), _PublicValue(value={"nested": [candidate]}))


@pytest.mark.parametrize("extras", ({"provider_value": "changed"}, {}, {"provider_value": {"value": "same"}}))
def test_model_parity_compares_public_extras(extras: dict[str, object]) -> None:
    baseline: Final = _PublicValue.model_validate({"value": None, "provider_value": _ComparableResponse(value="same")})
    candidate: Final = _PublicValue.model_validate({"value": None, **extras})

    with pytest.raises(AssertionError):
        assert_model_parity(baseline, candidate)


def test_serialized_parity_rejects_boolean_integer_substitution() -> None:
    with pytest.raises(AssertionError, match="type mismatch"):
        assert_parity(
            _execution(body={"enabled": True}, user_agent=SENTINEL),
            _execution(body={"enabled": 1}, user_agent="candidate"),
            SENTINEL,
        )
