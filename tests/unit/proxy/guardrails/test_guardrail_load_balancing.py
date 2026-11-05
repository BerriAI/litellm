"""
Test guardrail load balancing through the Router and ProxyLogging.
"""

from unittest.mock import MagicMock, patch, AsyncMock
from typing import Final
from collections.abc import AsyncIterator
from queue import SimpleQueue


import httpx
import litellm
import pytest
from litellm import Router
from litellm.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.utils import ProxyLogging
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails import guardrail_registry
from litellm.proxy.guardrails.guardrail_registry import InMemoryGuardrailHandler
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api import GenericGuardrailAPI
from litellm.types.utils import ModelResponseStream


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
                    await proxy_logging.async_post_call_streaming_hook(
                        user_api_key_dict=auth, data=data, response=response
                    )
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


async def _output_chunks() -> AsyncIterator[ModelResponseStream]:
    yield ModelResponseStream(
        model="test-model",
        created=1700000000,
        choices=[{"index": 0, "delta": {"role": "assistant", "content": "raw"}, "finish_reason": "stop"}],
    )


def _chunk_text(chunk: object) -> str:
    assert isinstance(chunk, ModelResponseStream)
    return chunk.choices[0].delta.content or ""


async def _guarded_output(logging: ProxyLogging, streaming: bool) -> str:
    data: Final = {"messages": [{"role": "user", "content": "raw"}]}
    auth: Final = UserAPIKeyAuth(request_route="/v1/chat/completions")
    if streaming:
        return "".join(
            [
                _chunk_text(chunk)
                async for chunk in logging.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=auth, request_data=data, response=_output_chunks()
                )
            ]
        )
    response: Final = await logging.post_call_success_hook(
        user_api_key_dict=auth,
        data=data,
        response=litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "raw"}}]),
    )
    assert isinstance(response, litellm.ModelResponse)
    choice: Final = response.choices[0]
    assert isinstance(choice, litellm.Choices)
    content: Final = choice.message.content
    assert isinstance(content, str)
    return content


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
async def test_presidio_load_balancing_keeps_selected_output_sibling(
    streaming: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    router: Final = Router(model_list=[])
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(guardrail_registry, "IN_MEMORY_GUARDRAIL_HANDLER", InMemoryGuardrailHandler())
    monkeypatch.setattr(litellm, "callbacks", [])
    init_guardrails_v2(
        all_guardrails=[
            {
                "guardrail_name": "presidio-balanced",
                "litellm_params": {
                    "guardrail": "presidio",
                    "mode": "pre_call",
                    "default_on": True,
                    "presidio_filter_scope": "both",
                    "presidio_analyzer_api_base": "https://presidio.invalid/analyze",
                    "presidio_anonymizer_api_base": "https://presidio.invalid/anonymize",
                    "mock_redacted_text": {"text": masked, "items": []},
                },
            }
            for masked in ("masked-east", "masked-west")
        ],
        llm_router=router,
    )
    logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    outputs: Final[SimpleQueue[str]] = SimpleQueue()
    for _ in range(50):
        content: Final = await _guarded_output(logging, streaming)
        assert content in ("masked-east", "masked-west")
        outputs.put(content)
    assert frozenset(outputs.get() for _ in range(outputs.qsize())) == {"masked-east", "masked-west"}


@pytest.mark.asyncio
async def test_iterator_uses_selected_generic_deployments_cached_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    native: Final = MockGuardrail("mixed-iterator", "native")
    requests: Final[SimpleQueue[bytes]] = SimpleQueue()

    async def answer(request: httpx.Request) -> httpx.Response:
        requests.put(request.content)
        return httpx.Response(200, json={"action": "BLOCKED", "blocked_reason": "test policy"})

    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(answer))
    generic: Final = GenericGuardrailAPI(
        guardrail_name="mixed-iterator",
        api_base="https://guardrail.invalid",
        event_hook="post_call",
        default_on=True,
        async_handler=handler,
        streaming_sampling_rate=1,
    )
    router: Final = Router(
        model_list=[],
        guardrail_list=[
            {
                "guardrail_name": "mixed-iterator",
                "callback": cb,
                "id": deployment_id,
                "litellm_params": {"guardrail": "custom", "mode": "post_call"},
            }
            for cb, deployment_id in ((native, "native"), (generic, "generic"))
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(litellm, "callbacks", [native, generic])
    logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    blocked: Final[SimpleQueue[litellm.exceptions.GuardrailRaisedException]] = SimpleQueue()
    try:
        for _ in range(50):
            try:
                async for _chunk in logging.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=UserAPIKeyAuth(request_route="/v1/chat/completions"),
                    request_data={"messages": [{"role": "user", "content": "hi"}]},
                    response=_output_chunks(),
                ):
                    pass
            except litellm.exceptions.GuardrailRaisedException as exc:
                blocked.put(exc)
        assert blocked.qsize() > 0
        assert requests.qsize() == blocked.qsize()
        assert native.calls + blocked.qsize() == 50
        assert native.calls > 0
    finally:
        await handler.close()
