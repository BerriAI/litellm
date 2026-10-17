# What is this?
## Unit Tests for guardrails config
import importlib
import os
import time
from typing import Any, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

import litellm
import litellm.litellm_core_utils
import litellm.litellm_core_utils.litellm_logging
from litellm import completion
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.guardrails import BaseLitellmParams, LitellmParams
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


class CustomLoggingIntegration(CustomLogger):
    def __init__(self) -> None:
        super().__init__()

    def logging_hook(self, kwargs: dict, result: Any, call_type: str) -> Tuple[dict, Any]:
        input: Optional[Any] = kwargs.get("input", None)
        messages: Optional[List] = kwargs.get("messages", None)
        if call_type == "completion":
            # assume input is of type messages
            if input is not None and isinstance(input, list):
                input[0]["content"] = "Hey, my name is [NAME]."
            if messages is not None and isinstance(messages, List):
                messages[0]["content"] = "Hey, my name is [NAME]."

            kwargs["input"] = input
            kwargs["messages"] = messages
        return kwargs, result


def test_guardrail_masking_logging_only():
    """
    Assert response is unmasked.

    Assert logged response is masked.
    """
    callback = CustomLoggingIntegration()

    with patch.object(callback, "log_success_event", new=MagicMock()) as mock_call:
        litellm.callbacks = [callback]
        messages = [{"role": "user", "content": "Hey, my name is Peter."}]
        response = completion(model="gpt-5-mini", messages=messages, mock_response="Hi Peter!")

        assert response.choices[0].message.content == "Hi Peter!"  # type: ignore

        time.sleep(3)
        mock_call.assert_called_once()

        assert mock_call.call_args.kwargs["kwargs"]["messages"][0]["content"] == "Hey, my name is [NAME]."


def test_guardrail_list_of_event_hooks():
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.guardrails import GuardrailEventHooks

    cg = CustomGuardrail(guardrail_name="custom-guard", event_hook=["pre_call", "post_call"])

    data = {"model": "gpt-5-mini", "metadata": {"guardrails": ["custom-guard"]}}
    assert cg.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call)

    assert cg.should_run_guardrail(data=data, event_type=GuardrailEventHooks.post_call)

    assert not cg.should_run_guardrail(data=data, event_type=GuardrailEventHooks.during_call)


def test_guardrail_info_response():
    from litellm.types.guardrails import (
        GuardrailInfoResponse,
        LitellmParams,
    )

    guardrail_info = GuardrailInfoResponse(
        guardrail_name="aporia-pre-guard",
        litellm_params=LitellmParams(
            guardrail="aporia",
            mode="pre_call",
        ),
        guardrail_info={
            "guardrail_name": "aporia-pre-guard",
            "litellm_params": {
                "guardrail": "aporia",
                "mode": "always_on",
            },
        },
    )

    assert guardrail_info.litellm_params.default_on == False


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="function", autouse=True)
def isolate_litellm_state():
    """
    Per-function isolation fixture.

    Saves and restores litellm callback/global state so tests don't leak
    side effects. Works safely under pytest-xdist parallel execution.
    """
    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in ("set_verbose", "cache", "num_retries"):
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr in ("success_callback", "failure_callback", "_async_success_callback", "_async_failure_callback"):
        if hasattr(litellm, attr):
            setattr(litellm, attr, [])
    yield
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    """
    Module-scoped setup. Reloads litellm only in single-process mode
    (skipped under xdist to avoid cross-worker interference).
    """
    import litellm

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", None)
    if worker_id is None:
        importlib.reload(litellm)
        try:
            if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
                import litellm.proxy.proxy_server

                importlib.reload(litellm.proxy.proxy_server)
        except Exception as e:
            print(f"Error reloading litellm.proxy.proxy_server: {e}")
        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.flush_cache()
    yield


@pytest.mark.parametrize("blank", ["", "   "])
def test_litellm_params_treats_blank_values_as_unset(blank: str) -> None:
    """
    Regression: `os.environ/X` with X exported blank resolves to "", which pydantic rejected on
    bool fields (default_on, mask_request_content, fail_on_error), so the guardrail was dropped at
    boot. Blank now means unset, so the field keeps the same default as when it is omitted.
    """
    with_blank_values = LitellmParams(
        guardrail="presidio",
        mode="pre_call",
        default_on=blank,
        mask_request_content=blank,
        api_key=blank,
    )
    omitted = LitellmParams(guardrail="presidio", mode="pre_call")
    assert with_blank_values.model_dump() == omitted.model_dump()

    assert BaseLitellmParams(fail_on_error=blank).model_dump() == BaseLitellmParams().model_dump()


@pytest.mark.parametrize("value", [0, 1, 0.5])
def test_litellm_params_accepts_valid_sample_rate(value: float) -> None:
    params = LitellmParams(guardrail="llm_as_a_judge", mode="pre_call", sample_rate=value, on_failure="log")
    assert params.sample_rate == float(value)


@pytest.mark.parametrize("value", [-0.01, 1.01, "0.5", True])
def test_litellm_params_rejects_invalid_sample_rate(value: object) -> None:
    with pytest.raises(ValidationError):
        LitellmParams(guardrail="llm_as_a_judge", mode="pre_call", sample_rate=value)


@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_litellm_params_rejects_invalid_max_input_chars(value: object) -> None:
    with pytest.raises(ValidationError):
        LitellmParams(guardrail="llm_as_a_judge", mode="pre_call", max_input_chars=value)


def test_litellm_params_accepts_max_input_chars() -> None:
    params = LitellmParams(guardrail="llm_as_a_judge", mode="pre_call", max_input_chars=500, on_failure="log")
    assert params.max_input_chars == 500


@pytest.mark.parametrize(
    "litellm_params",
    [
        {"sample_rate": 0.5, "on_failure": "block", "mode": "pre_call"},
        {"sample_rate": 0.5, "mode": "pre_call"},
        {"max_input_chars": 5000, "on_failure": "block", "mode": "pre_call"},
        {"max_input_chars": 5000, "on_failure": "block", "mode": "during_call"},
    ],
)
def test_llm_judge_sampling_and_cap_rejected_when_judge_can_block(litellm_params: dict) -> None:
    with pytest.raises(ValidationError, match="on_failure: log"):
        LitellmParams(guardrail="llm_as_a_judge", **litellm_params)


@pytest.mark.parametrize(
    "litellm_params",
    [
        {"sample_rate": 1, "on_failure": "block", "mode": "pre_call"},
        {"sample_rate": 1, "mode": "post_call"},
        {"sample_rate": 0.05, "max_input_chars": 5000, "on_failure": "log", "mode": "pre_call"},
        {"sample_rate": 0.5, "on_failure": "block", "mode": "logging_only"},
        {"sample_rate": 0.5, "max_input_chars": 500, "mode": ["logging_only"]},
    ],
)
def test_llm_judge_sampling_and_cap_allowed_when_judge_cannot_block(litellm_params: dict) -> None:
    params = LitellmParams(guardrail="llm_as_a_judge", **litellm_params)
    assert params.guardrail == "llm_as_a_judge"


@pytest.mark.parametrize("value", [True, 5])
def test_non_judge_max_input_chars_keeps_base_coercion(value: object) -> None:
    params = LitellmParams(guardrail="decision_model", mode="pre_call", max_input_chars=value)
    assert params.max_input_chars == int(value)


def test_non_judge_max_input_chars_zero_still_rejected() -> None:
    with pytest.raises(ValidationError):
        LitellmParams(guardrail="decision_model", mode="pre_call", max_input_chars=0)


@pytest.mark.parametrize("value", [1.5, "0.5", True, -2])
def test_non_judge_sample_rate_passes_through_unvalidated(value: object) -> None:
    params = LitellmParams(guardrail="decision_model", mode="pre_call", sample_rate=value)
    assert params.sample_rate == value


def test_non_judge_sampling_and_cap_with_block_unaffected() -> None:
    params = LitellmParams(
        guardrail="decision_model", mode="pre_call", sample_rate=0.5, max_input_chars=500, on_failure="block"
    )
    assert params.sample_rate == 0.5
    assert params.max_input_chars == 500
