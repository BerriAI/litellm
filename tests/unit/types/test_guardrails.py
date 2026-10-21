# What is this?
## Unit Tests for guardrails config
import importlib
import os
import time
from typing import Any, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

import litellm
import litellm.litellm_core_utils
import litellm.litellm_core_utils.litellm_logging
from litellm import completion
from litellm.integrations.custom_logger import CustomLogger
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
