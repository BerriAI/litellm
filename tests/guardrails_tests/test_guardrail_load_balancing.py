"""
Test guardrail load balancing through the Router and ProxyLogging.
"""

import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock

sys.path.insert(0, os.path.abspath("../.."))

import litellm
import pytest
from litellm import Router
from litellm.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks


class MockGuardrail(CustomGuardrail):
    """Mock guardrail that tracks calls."""

    call_count = 0

    def __init__(self, guardrail_name: str, guardrail_id: str):
        super().__init__(guardrail_name=guardrail_name)
        self.guardrail_id = guardrail_id
        self.calls = 0

    def should_run_guardrail(self, data, event_type) -> bool:
        return True

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        self.calls += 1
        MockGuardrail.call_count += 1
        return None


@pytest.mark.asyncio
async def test_proxy_logging_pre_call_hook_load_balancing():
    """Test that async_pre_call_hook load balances across multiple guardrails."""
    # Reset call count
    MockGuardrail.call_count = 0

    # Create two mock guardrails with same name
    guardrail_1 = MockGuardrail(guardrail_name="content-filter", guardrail_id="g1")
    guardrail_2 = MockGuardrail(guardrail_name="content-filter", guardrail_id="g2")

    # Create router with multiple guardrails of same name
    guardrail_list = [
        {
            "guardrail_name": "content-filter",
            "litellm_params": {"guardrail": "custom", "mode": "pre_call"},
            "callback": guardrail_1,
            "id": "guardrail-1",
        },
        {
            "guardrail_name": "content-filter",
            "litellm_params": {"guardrail": "custom", "mode": "pre_call"},
            "callback": guardrail_2,
            "id": "guardrail-2",
        },
    ]

    router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "fake-key"},
            }
        ],
        guardrail_list=guardrail_list,
    )

    # Create ProxyLogging instance
    proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

    # Add guardrail to litellm.callbacks so it gets picked up
    original_callbacks = litellm.callbacks.copy()
    litellm.callbacks = [guardrail_1]

    try:
        with patch("litellm.proxy.proxy_server.llm_router", router):
            # Call pre_call_hook 50 times
            for _ in range(50):
                await proxy_logging.pre_call_hook(
                    user_api_key_dict=MagicMock(),
                    data={"messages": [{"role": "user", "content": "test"}]},
                    call_type="completion",
                )

            # Both guardrails should have been called (load balanced)
            assert guardrail_1.calls > 0, "Guardrail 1 should have been called"
            assert guardrail_2.calls > 0, "Guardrail 2 should have been called"

            # Total calls should be 50
            total = guardrail_1.calls + guardrail_2.calls
            assert total == 50, f"Expected 50 total calls, got {total}"

            # Verify reasonable distribution (not all to one)
            min_calls = min(guardrail_1.calls, guardrail_2.calls)
            assert min_calls >= 10, f"Expected at least 10 calls to each guardrail, got min={min_calls}"

    finally:
        litellm.callbacks = original_callbacks
