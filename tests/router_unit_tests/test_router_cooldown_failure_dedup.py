"""
Regression for #32574:

Router retries/fallbacks reuse the same Logging object. Observability failure
callbacks are correctly deduplicated via has_logged_sync_failure, but
deployment_callback_on_failure must still run for every failed deployment so
cooldown state stays correct.
"""

import os
import sys
import time
from unittest.mock import MagicMock

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.router_utils.cooldown_handlers import _async_get_cooldown_deployments


def _make_logging_obj() -> LitellmLogging:
    return LitellmLogging(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="cooldown-dedup-test",
        function_id="cooldown-dedup-test",
    )


def test_failure_handler_runs_cooldown_callback_on_each_attempt():
    """Cooldown callback runs per attempt; observability callbacks stay deduped."""
    logging_obj = _make_logging_obj()

    cooldown_calls = []
    observability_calls = []

    def deployment_callback_on_failure(kwargs, completion_response, start_time, end_time):
        cooldown_calls.append(kwargs.get("litellm_params", {}).get("model_info", {}).get("id"))

    def observability_failure_callback(kwargs, completion_response, start_time, end_time):
        observability_calls.append(True)

    original_failure_callback = litellm.failure_callback
    try:
        litellm.failure_callback = [
            deployment_callback_on_failure,
            observability_failure_callback,
        ]

        # Attempt 1 — deployment A
        logging_obj.model_call_details["litellm_params"] = {
            "model_info": {"id": "deployment-a"},
            "metadata": {"model_group": "test-group"},
        }
        logging_obj.failure_handler(
            exception=Exception("A failed"),
            traceback_exception="traceback-a",
        )

        # Attempt 2 — same Logging object, deployment B (would be skipped by dedup)
        logging_obj.model_call_details["litellm_params"] = {
            "model_info": {"id": "deployment-b"},
            "metadata": {"model_group": "test-group"},
        }
        logging_obj.failure_handler(
            exception=Exception("B failed"),
            traceback_exception="traceback-b",
        )

        assert cooldown_calls == ["deployment-a", "deployment-b"]
        assert len(observability_calls) == 1
        assert logging_obj.should_run_logging(event_type="sync_failure") is False
    finally:
        litellm.failure_callback = original_failure_callback


def test_deployment_callback_on_failure_is_internal_callback():
    logging_obj = _make_logging_obj()
    mock_cb = MagicMock()
    mock_cb.__name__ = "deployment_callback_on_failure"
    assert logging_obj._is_internal_litellm_proxy_callback(mock_cb) is True


@pytest.mark.asyncio
async def test_router_cools_down_each_failed_deployment_across_retries():
    """End-to-end: each failed deployment in one request is added to cooldown."""
    deployment_a = "deployment-a"
    deployment_b = "deployment-b"

    router = litellm.Router(
        model_list=[
            {
                "model_name": "test-group",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "mock-key-a",
                },
                "model_info": {"id": deployment_a},
            },
            {
                "model_name": "test-group",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "mock-key-b",
                },
                "model_info": {"id": deployment_b},
            },
        ],
        num_retries=1,
        allowed_fails=0,
        cooldown_time=60,
    )

    rate_limit_error = litellm.RateLimitError(
        message="upstream throttled",
        llm_provider="openai",
        model="openai/gpt-4o-mini",
        response=httpx.Response(
            status_code=429,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        ),
    )

    from unittest.mock import patch

    with patch(
        "litellm.llms.openai.openai.OpenAIChatCompletion.acompletion",
        side_effect=rate_limit_error,
    ):
        with pytest.raises(litellm.RateLimitError):
            await router.acompletion(
                model="test-group",
                messages=[{"role": "user", "content": "hi"}],
            )

    cooldown_ids = await _async_get_cooldown_deployments(
        litellm_router_instance=router, parent_otel_span=None
    )
    assert deployment_a in cooldown_ids, (
        f"Expected {deployment_a!r} in cooldown; got {cooldown_ids}"
    )
    assert deployment_b in cooldown_ids, (
        f"Expected {deployment_b!r} in cooldown; got {cooldown_ids}"
    )
