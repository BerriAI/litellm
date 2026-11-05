"""
Test guardrail load balancing through the Router and ProxyLogging.
"""

from unittest.mock import MagicMock, patch, AsyncMock
from typing import Final


import litellm
import pytest
from litellm import Router
from litellm.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.utils import ProxyLogging


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

    async def async_moderation_hook(self, data, user_api_key_dict, call_type):
        self.calls += 1

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        self.calls += 1
        return response

    async def async_post_call_streaming_hook(self, user_api_key_dict, response):
        self.calls += 1

    async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data):
        self.calls += 1
        async for chunk in response:
            yield chunk


class InheritedIteratorGuardrail(MockGuardrail):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("hook", ("pre_call", "during_call", "post_call", "streaming", "iterator"))
@pytest.mark.parametrize("deployment_count", (0, 1, 2))
async def test_proxy_logging_hooks_load_balance_once_per_request(hook: str, deployment_count: int):
    """Test that async_pre_call_hook load balances across multiple guardrails."""
    MockGuardrail.call_count = 0

    guardrail_1 = MockGuardrail(guardrail_name="content-filter", guardrail_id="g1")
    second_guardrail_class: Final = (
        InheritedIteratorGuardrail if hook == "iterator" and deployment_count == 2 else MockGuardrail
    )
    guardrail_2 = second_guardrail_class(guardrail_name="content-filter", guardrail_id="g2")

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
                "model_name": "gpt-5.5",
                "litellm_params": {"model": "gpt-5.5", "api_key": "fake-key"},
            }
        ],
        guardrail_list=guardrail_list[:deployment_count],
    )

    proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

    original_callbacks = litellm.callbacks.copy()
    litellm.callbacks = [guardrail_1, guardrail_2]

    try:
        with patch("litellm.proxy.proxy_server.llm_router", router if deployment_count else None):
            for _ in range(50):
                previous_calls: Final = guardrail_1.calls + guardrail_2.calls
                data: Final = {"messages": [{"role": "user", "content": "test"}]}
                auth: Final = MagicMock()
                response: Final = litellm.ModelResponse(choices=[{"message": {"content": "test", "role": "assistant"}}])
                if hook == "pre_call":
                    await proxy_logging.pre_call_hook(user_api_key_dict=auth, data=data, call_type="completion")
                elif hook == "during_call":
                    await proxy_logging.during_call_hook(user_api_key_dict=auth, data=data, call_type="completion")
                elif hook == "post_call":
                    await proxy_logging.post_call_success_hook(user_api_key_dict=auth, data=data, response=response)
                elif hook == "streaming":
                    await proxy_logging.async_post_call_streaming_hook(user_api_key_dict=auth, data=data, response=response)
                else:
                    async def chunks(chunk=response):
                        yield chunk

                    async for _chunk in proxy_logging.async_post_call_streaming_iterator_hook(
                        user_api_key_dict=auth, request_data=data, response=chunks()
                    ):
                        pass
                expected_calls: Final = 1 if deployment_count == 2 else 2
                assert guardrail_1.calls + guardrail_2.calls == previous_calls + expected_calls

            assert guardrail_1.calls > 0, "Guardrail 1 should have been called"
            assert guardrail_2.calls > 0, "Guardrail 2 should have been called"

            total = guardrail_1.calls + guardrail_2.calls
            expected_total: Final = 50 if deployment_count == 2 else 100
            assert total == expected_total, f"Expected {expected_total} total calls, got {total}"

    finally:
        litellm.callbacks = original_callbacks
