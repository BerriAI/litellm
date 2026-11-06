"""
Test guardrail load balancing through the Router and ProxyLogging.
"""

from collections.abc import AsyncIterator, Mapping
from contextlib import nullcontext
from queue import SimpleQueue
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from mcp.types import CallToolResult, TextContent

import litellm
from litellm import Router
from litellm.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai.chat.guardrail_translation.handler import OpenAIChatCompletionsHandler
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.guardrails import guardrail_registry
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api import GenericGuardrailAPI
from litellm.proxy.guardrails.guardrail_registry import InMemoryGuardrailHandler
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2
from litellm.proxy.policy_engine.pipeline_executor import PipelineExecutor
from litellm.proxy.utils import ProxyLogging
from litellm.types.proxy.policy_engine.pipeline_types import GuardrailPipeline, PipelineStep
from litellm.types.guardrails import GuardrailEventHooks, GuardrailStreamScope
from litellm.types.utils import CallTypesLiteral, ModelResponseStream


class MockGuardrail(CustomGuardrail):
    """Mock guardrail that tracks calls."""

    call_count = 0

    def __init__(
        self,
        guardrail_name: str,
        guardrail_id: str,
        event_hook: GuardrailEventHooks | None = None,
        default_on: bool = False,
        stream_scope: GuardrailStreamScope = "both",
        run_in_parallel: bool = False,
        scan_raw_request: bool = False,
        fail: bool = False,
    ):
        super().__init__(
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
            stream_scope=stream_scope,
            run_in_parallel=run_in_parallel,
            scan_raw_request=scan_raw_request,
        )
        self.guardrail_id = guardrail_id
        self.calls = 0
        self.fail = fail

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        self.calls += 1
        MockGuardrail.call_count += 1
        if self.fail:
            raise RuntimeError("guardrail offline")
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


class NonIteratorGuardrail(CustomGuardrail):
    pass


class RelayGuardrail(CustomGuardrail):
    def __init__(self, guardrail_id: str, fail: bool = False):
        super().__init__(guardrail_name="relay-lb", event_hook="post_call", default_on=True)
        self.guardrail_id = guardrail_id
        self.calls = 0
        self.fail = fail
        self.streaming_end_of_stream_only = True

    async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("guardrail offline")
        return inputs


class LegacyPostCallGuardrail(CustomGuardrail):
    def __init__(self, fail: bool):
        super().__init__(guardrail_name="relay-lb", event_hook="post_call", default_on=True)
        self.calls = 0
        self.response: object = None
        self.fail = fail

    async def async_post_call_success_hook(
        self,
        data: Mapping[str, object],
        user_api_key_dict: UserAPIKeyAuth,
        response: object,
    ) -> object:
        self.calls += 1
        self.response = response
        if response is not None and self.fail:
            raise RuntimeError("guardrail offline")
        return response


class FallbackIteratorGuardrail(CustomGuardrail):
    def __init__(self, fail: bool):
        super().__init__(guardrail_name="relay-lb", event_hook="post_call", default_on=True)
        self.calls = 0
        self.fail = fail

    async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data):
        self.calls += 1
        if self.fail:
            raise RuntimeError("guardrail offline")
        async for chunk in response:
            yield chunk


class McpResultGuardrail(CustomGuardrail):
    def __init__(self) -> None:
        super().__init__(guardrail_name="relay-lb", event_hook=GuardrailEventHooks.post_mcp_call, default_on=True)
        self.calls = 0

    async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
        self.calls += 1
        return inputs


class GuardrailActivator(CustomLogger):
    def __init__(self, action: str):
        super().__init__()
        self.action = action

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: Mapping[str, object],
        call_type: CallTypesLiteral,
    ) -> Mapping[str, object]:
        if self.action == "raise":
            raise ValueError("activation blocked")
        return {**data, "guardrails": ["dynamic-lb"]} if self.action == "enable" else data


class GuardrailRemover(CustomLogger):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: Mapping[str, object],
        call_type: CallTypesLiteral,
    ) -> Mapping[str, object]:
        return {key: value for key, value in data.items() if key != "guardrails"}


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
@pytest.mark.parametrize("pipeline", (False, True))
async def test_presidio_load_balancing_keeps_selected_output_sibling(
    streaming: bool, pipeline: bool, monkeypatch: pytest.MonkeyPatch
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
        if pipeline:
            chunks: Final = [chunk async for chunk in _output_chunks()] if streaming else None
            result: Final = await PipelineExecutor.execute_steps(
                steps=[PipelineStep(guardrail="presidio-balanced")],
                mode="post_call",
                data={
                    "messages": [{"role": "user", "content": "raw"}],
                    "response": litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "raw"}}]),
                },
                user_api_key_dict=UserAPIKeyAuth(request_route="/v1/chat/completions"),
                call_type="completion",
                policy_name="presidio-pipeline",
                streaming_chunks=chunks,
                endpoint_translation=OpenAIChatCompletionsHandler() if streaming else None,
            )
            assert result.step_results[0].outcome == "pass"
            if streaming:
                assert chunks is not None
                content = "".join(_chunk_text(chunk) for chunk in chunks)
            else:
                assert result.modified_data is not None
                content = result.modified_data["response"].choices[0].message.content
        else:
            content = await _guarded_output(logging, streaming)
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


async def _dispatch_scoped_hook(logging: ProxyLogging, hook: str) -> None:
    auth: Final = UserAPIKeyAuth(request_route="/v1/chat/completions")
    data: Final = {"messages": [{"role": "user", "content": "raw"}], "stream": hook in ("streaming", "iterator")}
    response: Final = litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "raw"}}])
    if hook == "pre_call":
        await logging.pre_call_hook(user_api_key_dict=auth, data=data, call_type="completion")
    elif hook == "during_call":
        await logging.during_call_hook(user_api_key_dict=auth, data=data, call_type="completion")
    elif hook == "post_call":
        await logging.post_call_success_hook(user_api_key_dict=auth, data=data, response=response)
    elif hook == "streaming":
        await logging.async_post_call_streaming_hook(user_api_key_dict=auth, data=data, response=response)
    else:
        async for _chunk in logging.async_post_call_streaming_iterator_hook(
            user_api_key_dict=auth, request_data=data, response=_output_chunks()
        ):
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize("hook", ("pre_call", "during_call", "post_call", "streaming", "iterator"))
@pytest.mark.parametrize("mismatch", ("mode", "default_on", "stream_scope", "none_eligible", "weights"))
async def test_load_balancing_selects_only_hook_eligible_deployments(
    hook: str, mismatch: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    event: Final = GuardrailEventHooks(hook if hook in ("pre_call", "during_call") else "post_call")
    other_event: Final = (
        GuardrailEventHooks.post_call if event == GuardrailEventHooks.pre_call else GuardrailEventHooks.pre_call
    )
    eligible: Final = MockGuardrail("scoped-lb", "eligible", event_hook=event, default_on=mismatch != "none_eligible")
    ineligible: Final = MockGuardrail(
        "scoped-lb",
        "other",
        event_hook=other_event if mismatch == "mode" else event,
        default_on=mismatch not in ("default_on", "none_eligible"),
        stream_scope=("non_streaming" if hook in ("streaming", "iterator") else "streaming")
        if mismatch == "stream_scope"
        else "both",
    )
    router: Final = Router(
        model_list=[],
        guardrail_list=[
            {
                "guardrail_name": "scoped-lb",
                "id": str(index),
                "callback": cb,
                "litellm_params": {"guardrail": "custom", "mode": "post_call", "weight": weight},
            }
            for index, (cb, weight) in enumerate(
                ((eligible, 1 if mismatch == "weights" else 0), (ineligible, 0 if mismatch == "weights" else 1))
            )
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(litellm, "callbacks", [ineligible, eligible])
    logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    for request_index in range(20):
        await _dispatch_scoped_hook(logging, hook)
        assert eligible.calls == (0 if mismatch == "none_eligible" else request_index + 1)
        assert ineligible.calls == 0


@pytest.mark.asyncio
async def test_streaming_selects_iterator_capable_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    eligible: Final = MockGuardrail(
        "iterator-lb", "iterator", event_hook=GuardrailEventHooks.post_call, default_on=True
    )
    unsupported: Final = NonIteratorGuardrail(
        guardrail_name="iterator-lb", event_hook=GuardrailEventHooks.post_call, default_on=True
    )
    router: Final = Router(
        model_list=[],
        guardrail_list=[
            {
                "guardrail_name": "iterator-lb",
                "id": str(index),
                "callback": cb,
                "litellm_params": {"guardrail": "custom", "mode": "post_call", "weight": weight},
            }
            for index, (cb, weight) in enumerate(((eligible, 0), (unsupported, 1)))
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(litellm, "callbacks", [unsupported, eligible])
    logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    for request_index in range(20):
        await _dispatch_scoped_hook(logging, "iterator")
        assert eligible.calls == request_index + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ("enable", "noop", "raise"))
@pytest.mark.parametrize("run_in_parallel", (False, True))
@pytest.mark.parametrize("call_type", ("completion", "call_mcp_tool"))
@pytest.mark.parametrize("activator_first", (False, True))
async def test_pre_call_load_balancing_observes_earlier_callback_activation(
    action: str, run_in_parallel: bool, call_type: CallTypesLiteral, activator_first: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event: Final = GuardrailEventHooks.pre_mcp_call if call_type == "call_mcp_tool" else GuardrailEventHooks.pre_call
    replicas: Final = tuple(
        MockGuardrail("dynamic-lb", region, event_hook=event, default_on=False, run_in_parallel=run_in_parallel)
        for region in ("east", "west")
    )
    router: Final = Router(
        model_list=[],
        guardrail_list=[
            {
                "guardrail_name": "dynamic-lb",
                "id": cb.guardrail_id,
                "callback": cb,
                "litellm_params": {"guardrail": "custom", "mode": event.value},
            }
            for cb in replicas
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    activator: Final = GuardrailActivator(action)
    monkeypatch.setattr(litellm, "callbacks", [activator, *replicas] if activator_first else [*replicas, activator])
    logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    for request_index in range(30):
        if action == "raise":
            with pytest.raises(ValueError, match="activation blocked"):
                await logging.pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(), data={"model": "test", "messages": []}, call_type=call_type
                )
        else:
            await logging.pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(), data={"model": "test", "messages": []}, call_type=call_type
            )
        assert sum(cb.calls for cb in replicas) == (
            request_index + 1 if action == "enable" and (activator_first or run_in_parallel) else 0
        )
    assert all(cb.calls > 0 for cb in replicas) is (action == "enable" and (activator_first or run_in_parallel))


@pytest.mark.asyncio
@pytest.mark.parametrize("scan_raw_request", (False, True))
@pytest.mark.parametrize("requested", (False, True))
@pytest.mark.parametrize("fail", (False, True))
async def test_late_parallel_activation_rechecks_final_request(
    scan_raw_request: bool, requested: bool, fail: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    replicas: Final = tuple(
        MockGuardrail(
            "dynamic-lb", region, event_hook=GuardrailEventHooks.pre_call, default_on=False,
            run_in_parallel=True, scan_raw_request=scan_raw_request, fail=fail,
        )
        for region in ("east", "west")
    )
    router: Final = Router(
        model_list=[],
        guardrail_list=[
            {
                "guardrail_name": "dynamic-lb", "id": cb.guardrail_id, "callback": cb,
                "litellm_params": {"guardrail": "custom", "mode": "pre_call"},
            }
            for cb in replicas
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(litellm, "callbacks", [*replicas, GuardrailActivator("enable")])
    logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    should_run: Final = requested or not scan_raw_request
    with pytest.raises(RuntimeError, match="guardrail offline") if fail and should_run else nullcontext():
        await logging.pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            data={"model": "test", "messages": [], "guardrails": ["dynamic-lb"] if requested else []},
            call_type="completion",
        )
    assert any(cb.calls > 0 for cb in replicas) is should_run


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel_weight", (0, 1))
async def test_mixed_parallel_and_sequential_replicas_keep_one_weighted_selection(
    parallel_weight: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequential: Final = MockGuardrail("dynamic-lb", "sequential", default_on=True)
    parallel: Final = MockGuardrail("dynamic-lb", "parallel", default_on=True, run_in_parallel=True)
    router: Final = Router(
        model_list=[],
        guardrail_list=[
            {
                "guardrail_name": "dynamic-lb", "callback": cb,
                "litellm_params": {"guardrail": "custom", "mode": "pre_call", "weight": weight},
            }
            for cb, weight in ((sequential, 1 - parallel_weight), (parallel, parallel_weight))
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(litellm, "callbacks", [sequential, parallel])
    logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    for request_index in range(20):
        await logging.pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(), data={"model": "test", "messages": []}, call_type="completion",
        )
        assert sequential.calls == (request_index + 1) * (1 - parallel_weight)
        assert parallel.calls == (request_index + 1) * parallel_weight


@pytest.mark.asyncio
@pytest.mark.parametrize("requested", (False, True))
@pytest.mark.parametrize("run_in_parallel", (False, True))
async def test_raw_request_scans_survive_earlier_callback_removing_guardrail_names(
    requested: bool, run_in_parallel: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    single: Final = MockGuardrail(
        "single-raw",
        "single",
        event_hook=GuardrailEventHooks.pre_call,
        scan_raw_request=True,
        run_in_parallel=run_in_parallel,
    )
    replicas: Final = tuple(
        MockGuardrail(
            "balanced-raw",
            region,
            event_hook=GuardrailEventHooks.pre_call,
            scan_raw_request=True,
            run_in_parallel=run_in_parallel,
        )
        for region in ("east", "west")
    )
    router: Final = Router(
        model_list=[],
        guardrail_list=[
            {
                "guardrail_name": cb.guardrail_name,
                "id": cb.guardrail_id,
                "callback": cb,
                "litellm_params": {"guardrail": "custom", "mode": "pre_call"},
            }
            for cb in (single, *replicas)
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(litellm, "callbacks", [GuardrailRemover(), single, *replicas])
    logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    await logging.pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        data={"model": "test", "messages": [], "guardrails": ["single-raw", "balanced-raw"] if requested else []},
        call_type="completion",
    )
    assert single.calls == int(requested)
    assert (sum(cb.calls for cb in replicas) > 0) is requested


@pytest.mark.asyncio
@pytest.mark.parametrize("hook", ("pre_call", "during_call", "post_call", "streaming", "iterator"))
@pytest.mark.parametrize("healthy", (False, True))
async def test_missing_callback_does_not_poison_healthy_replica(
    hook: str, healthy: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    guardrail: Final = MockGuardrail("partial-lb", "healthy", default_on=True)
    router: Final = Router(
        model_list=[],
        guardrail_list=[
            {"guardrail_name": "partial-lb", "id": "missing", "litellm_params": {"guardrail": "custom", "weight": 0}},
            {
                "guardrail_name": "partial-lb",
                "id": "healthy",
                "callback": guardrail if healthy else None,
                "litellm_params": {"guardrail": "custom", "weight": 1},
            },
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(litellm, "callbacks", [guardrail])
    logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    await _dispatch_scoped_hook(logging, hook)
    assert guardrail.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("pre_call", "post_call"))
async def test_pipeline_balanced_step_splits_traffic_once_per_request(mode, monkeypatch):
    copies: Final = tuple(
        MockGuardrail("pipeline-lb", region, event_hook=GuardrailEventHooks(mode))
        for region in ("east", "west")
    )
    router: Final = Router(
        model_list=[],
        guardrail_list=[
            {
                "guardrail_name": "pipeline-lb",
                "callback": cb,
                "litellm_params": {"guardrail": "custom", "mode": mode},
            }
            for cb in copies
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(litellm, "callbacks", list(copies))
    for index in range(50):
        result = await PipelineExecutor.execute_steps(
            steps=[PipelineStep(guardrail="pipeline-lb")],
            mode=mode,
            data={"messages": [{"role": "user", "content": "raw"}]},
            user_api_key_dict=UserAPIKeyAuth(),
            call_type="completion",
            policy_name="balanced-pipeline",
        )
        assert result.step_results[0].outcome == "pass"
        assert sum(cb.calls for cb in copies) == index + 1
    assert all(cb.calls > 0 for cb in copies)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ("stream_scope", "none", "weights", "single"))
async def test_pipeline_balancing_respects_eligibility_weights_and_single_copy(scenario, monkeypatch):
    first: Final = MockGuardrail(
        "pipeline-scoped", "east",
        event_hook=GuardrailEventHooks.post_call,
        stream_scope="streaming" if scenario in ("stream_scope", "none") else "both",
    )
    second: Final = MockGuardrail(
        "pipeline-scoped", "west",
        event_hook=GuardrailEventHooks.post_call,
        stream_scope="streaming" if scenario == "none" else "both",
    )
    copies: Final = (first, second)
    router: Final = Router(
        model_list=[],
        guardrail_list=[
            {
                "guardrail_name": "pipeline-scoped",
                "callback": cb,
                "litellm_params": {"guardrail": "custom", "mode": "post_call", "weight": weight},
            }
            for cb, weight in ((first, 0), (second, 1))[:1 if scenario == "single" else 2]
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(litellm, "callbacks", list(copies))
    response: Final = object()
    for _ in range(20):
        result = await PipelineExecutor.execute_steps(
            steps=[PipelineStep(guardrail="pipeline-scoped")],
            mode="post_call",
            data={"stream": False, "response": response},
            user_api_key_dict=UserAPIKeyAuth(),
            call_type="completion",
            policy_name="balanced-pipeline",
        )
        assert result.step_results[0].outcome == ("skip" if scenario == "none" else "pass")
    assert first.calls == (20 if scenario == "single" else 0)
    assert second.calls == (0 if scenario in ("single", "none") else 20)


@pytest.mark.asyncio
@pytest.mark.parametrize("deployment_count", (1, 2))
async def test_balanced_pipeline_step_runs_a_copy_whatever_its_own_mode(deployment_count, monkeypatch):
    copies: Final = tuple(
        MockGuardrail("mcp-guard", region, event_hook=GuardrailEventHooks.pre_mcp_call, fail=True)
        for region in ("east", "west")
    )
    router: Final = Router(
        model_list=[],
        guardrail_list=[
            {"guardrail_name": "mcp-guard", "callback": cb, "litellm_params": {"guardrail": "custom", "mode": "pre_mcp_call"}}
            for cb in copies[:deployment_count]
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(litellm, "callbacks", list(copies[:deployment_count]))
    for index in range(20):
        result = await PipelineExecutor.execute_steps(
            steps=[PipelineStep(guardrail="mcp-guard")],
            mode="pre_call",
            data={"name": "echo", "arguments": {"text": "raw"}},
            user_api_key_dict=UserAPIKeyAuth(),
            call_type="call_mcp_tool",
            policy_name="mcp-pipeline",
        )
        assert result.terminal_action == "block"
        assert sum(cb.calls for cb in copies) == index + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("outside_position", (0, 2))
async def test_same_name_guardrail_outside_the_router_pool_still_runs(outside_position, monkeypatch):
    pooled: Final = tuple(MockGuardrail("pii", region, default_on=True) for region in ("east", "west"))
    outside: Final = MockGuardrail("pii", "db", default_on=True)
    router: Final = Router(
        model_list=[],
        guardrail_list=[
            {"guardrail_name": "pii", "callback": cb, "litellm_params": {"guardrail": "custom", "mode": "pre_call"}}
            for cb in pooled
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    callbacks: Final = list(pooled)
    callbacks.insert(outside_position, outside)
    monkeypatch.setattr(litellm, "callbacks", callbacks)
    logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    for index in range(20):
        await _dispatch_scoped_hook(logging, "pre_call")
        assert outside.calls == index + 1
        assert sum(cb.calls for cb in pooled) == index + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("weights", ((0, 1), ("heavy", "light")))
async def test_config_weights_steer_the_pick(weights, monkeypatch: pytest.MonkeyPatch) -> None:
    router: Final = Router(model_list=[])
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(guardrail_registry, "IN_MEMORY_GUARDRAIL_HANDLER", InMemoryGuardrailHandler())
    monkeypatch.setattr(litellm, "callbacks", [])
    init_guardrails_v2(
        all_guardrails=[
            {
                "guardrail_name": "presidio-weighted",
                "litellm_params": {
                    "guardrail": "presidio",
                    "mode": "pre_call",
                    "default_on": True,
                    "presidio_filter_scope": "both",
                    "presidio_analyzer_api_base": "https://presidio.invalid/analyze",
                    "presidio_anonymizer_api_base": "https://presidio.invalid/anonymize",
                    "mock_redacted_text": {"text": masked, "items": []},
                    "weight": weight,
                },
            }
            for masked, weight in zip(("masked-east", "masked-west"), weights)
        ],
        llm_router=router,
    )
    logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    outputs: Final = frozenset([await _guarded_output(logging, streaming=False) for _ in range(40)])
    assert outputs == ({"masked-west"} if weights == (0, 1) else {"masked-east", "masked-west"})


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", (False, True))
async def test_streamed_pipeline_selects_supported_replica_and_preserves_errors(fail, monkeypatch):
    unsupported: Final = MockGuardrail("relay-lb", "native", event_hook=GuardrailEventHooks.post_call)
    supported: Final = RelayGuardrail("unified", fail=fail)
    supported.default_on = False
    router: Final = Router(
        model_list=[],
        guardrail_list=[
            {
                "guardrail_name": "relay-lb",
                "callback": cb,
                "litellm_params": {"guardrail": "custom", "mode": "post_call", "weight": weight},
            }
            for cb, weight in ((unsupported, 1), (supported, 0))
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(litellm, "callbacks", [unsupported, supported])
    logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    for index in range(20):
        with pytest.raises(RuntimeError, match="guardrail offline") if fail else nullcontext():
            chunks = [
                chunk
                async for chunk in logging.async_post_call_streaming_iterator_hook(
                    response=_output_chunks(),
                    user_api_key_dict=UserAPIKeyAuth(request_route="/v1/chat/completions"),
                    request_data={
                        "model": "test-model",
                        "stream": True,
                        "metadata": {
                            "_guardrail_pipelines": [("streamed-pipeline", GuardrailPipeline(
                                mode="post_call", steps=[PipelineStep(guardrail="relay-lb")],
                            ))],
                        },
                    },
                )
            ]
            assert "".join(_chunk_text(chunk) for chunk in chunks) == "raw"
        assert supported.calls == index + 1
        assert unsupported.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", (False, True))
@pytest.mark.parametrize("empty", (False, True))
@pytest.mark.parametrize("route", ("/v1/chat/completions", "/v1beta/models/gemini-2.5-flash:streamGenerateContent"))
async def test_streamed_pipeline_selects_replica_compatible_with_endpoint(route, empty, fail, monkeypatch):
    legacy: Final = LegacyPostCallGuardrail(fail=fail)
    unified: Final = RelayGuardrail("unified", fail=fail)
    router: Final = Router(
        model_list=[],
        guardrail_list=[
            {
                "guardrail_name": "relay-lb",
                "callback": cb,
                "litellm_params": {"guardrail": "custom", "mode": "post_call", "weight": weight},
            }
            for cb, weight in ((legacy, 1), (unified, 0))
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(litellm, "callbacks", [legacy, unified])
    logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    gemini: Final = route.endswith(":streamGenerateContent")
    payload: Final = (
        ({"candidates": [{"content": {"parts": [{"text": "raw"}]}, "finishReason": "STOP"}]},)
        if gemini
        else tuple([chunk async for chunk in _output_chunks()])
    )
    source: Final = () if empty else payload

    async def response_chunks() -> AsyncIterator[object]:
        for chunk in source:
            yield chunk

    for index in range(3):
        with pytest.raises(RuntimeError, match="guardrail offline") if fail and not empty else nullcontext():
            delivered = [
                chunk
                async for chunk in logging.async_post_call_streaming_iterator_hook(
                    response=response_chunks(),
                    user_api_key_dict=UserAPIKeyAuth(request_route=route),
                    request_data={
                        "model": "test-model",
                        "stream": True,
                        "metadata": {
                            "_guardrail_pipelines": [
                                (
                                    "streamed-pipeline",
                                    GuardrailPipeline(
                                        mode="post_call",
                                        steps=[PipelineStep(guardrail="relay-lb")],
                                    ),
                                )
                            ],
                        },
                    },
                )
            ]
            assert delivered == list(source)
        assert unified.calls == (index + 1 if gemini and not empty else 0)
        assert legacy.calls == (index + 1 if not gemini and not empty else 0)
        assert isinstance(legacy.response, litellm.ModelResponse) is (not gemini and not empty)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ("stream_scope", "overlap", "none"))
@pytest.mark.parametrize("output", ("normal", "empty", "error"))
async def test_streaming_pipeline_falls_back_when_eligible_and_capable_replicas_do_not_overlap(
    scenario, output, monkeypatch,
):
    unified: Final = RelayGuardrail("unified", fail=output == "error")
    unified.apply_stream_scope("both" if scenario == "overlap" else "non_streaming")
    native: Final = FallbackIteratorGuardrail(fail=output == "error")
    native.apply_stream_scope("non_streaming" if scenario == "none" else "both")
    router: Final = Router(
        model_list=[],
        guardrail_list=[
            {
                "guardrail_name": "relay-lb", "callback": cb,
                "litellm_params": {"guardrail": "custom", "mode": "post_call", "weight": weight},
            }
            for cb, weight in ((native, 1), (unified, 0))
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(litellm, "callbacks", [native, unified])
    logging: Final = ProxyLogging(user_api_key_cache=DualCache())

    async def response_chunks() -> AsyncIterator[ModelResponseStream]:
        if output != "empty":
            async for chunk in _output_chunks():
                yield chunk

    with pytest.raises(RuntimeError, match="guardrail offline") if output == "error" and scenario != "none" else nullcontext():
        delivered: Final = [
            chunk
            async for chunk in logging.async_post_call_streaming_iterator_hook(
                response=response_chunks(),
                user_api_key_dict=UserAPIKeyAuth(request_route="/v1/chat/completions"),
                request_data={
                    "model": "test-model", "stream": True,
                    "metadata": {"_guardrail_pipelines": [("streamed-pipeline", GuardrailPipeline(
                        mode="post_call", steps=[PipelineStep(guardrail="relay-lb")],
                    ))]},
                },
            )
        ]
        assert "".join(_chunk_text(chunk) for chunk in delivered) == ("" if output == "empty" else "raw")
    assert native.calls == (1 if scenario == "stream_scope" else 0)
    assert unified.calls == (1 if scenario == "overlap" and output != "empty" else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("first_balanced", (False, True))
@pytest.mark.parametrize("output", ("normal", "empty", "error"))
async def test_skipped_streaming_step_does_not_drop_later_policy_only_guardrail(
    first_balanced, output, monkeypatch,
):
    skipped: Final = tuple(RelayGuardrail(region) for region in (("east", "west") if first_balanced else ("east",)))
    for callback in skipped:
        callback.guardrail_name = "skipped-step"
        callback.apply_stream_scope("non_streaming")
    checked: Final = RelayGuardrail("checked", fail=output == "error")
    checked.guardrail_name = "policy-only"
    checked.default_on = False
    callbacks: Final = (*skipped, checked)
    router: Final = Router(
        model_list=[],
        guardrail_list=[
            {
                "guardrail_name": cb.guardrail_name, "callback": cb,
                "litellm_params": {"guardrail": "custom", "mode": "post_call"},
            }
            for cb in callbacks
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(litellm, "callbacks", list(callbacks))
    logging: Final = ProxyLogging(user_api_key_cache=DualCache())

    async def response_chunks() -> AsyncIterator[ModelResponseStream]:
        if output != "empty":
            async for chunk in _output_chunks():
                yield chunk

    with pytest.raises(RuntimeError, match="guardrail offline") if output == "error" else nullcontext():
        delivered: Final = [
            chunk
            async for chunk in logging.async_post_call_streaming_iterator_hook(
                response=response_chunks(),
                user_api_key_dict=UserAPIKeyAuth(request_route="/v1/chat/completions"),
                request_data={
                    "model": "test-model", "stream": True,
                    "metadata": {"_guardrail_pipelines": [("streamed-pipeline", GuardrailPipeline(
                        mode="post_call", steps=[PipelineStep(guardrail="skipped-step"), PipelineStep(guardrail="policy-only")],
                    ))]},
                },
            )
        ]
        assert "".join(_chunk_text(chunk) for chunk in delivered) == ("" if output == "empty" else "raw")
    assert checked.calls == (0 if output == "empty" else 1)
    assert all(cb.calls == 0 for cb in skipped)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", (False, True))
async def test_passthrough_balanced_relay_splits_traffic_once_per_request(fail, monkeypatch):
    copies: Final = tuple(RelayGuardrail(region, fail=fail) for region in ("east", "west"))
    router: Final = Router(
        model_list=[],
        guardrail_list=[
            {
                "guardrail_name": "relay-lb",
                "callback": cb,
                "litellm_params": {"guardrail": "custom", "mode": "post_call"},
            }
            for cb in copies
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(litellm, "callbacks", list(copies))
    logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    for index in range(50):
        with pytest.raises(RuntimeError, match="guardrail offline") if fail else nullcontext():
            chunks = [
                chunk
                async for chunk in logging.async_post_call_event_stream_iterator_hook(
                    response=_output_chunks(),
                    user_api_key_dict=UserAPIKeyAuth(request_route="/v1/chat/completions"),
                    request_data={"model": "test-model", "stream": True},
                    guardrails=tuple((cb, False) for cb in copies),
                )
            ]
            assert "".join(_chunk_text(chunk) for chunk in chunks) == "raw"
        assert sum(cb.calls for cb in copies) == index + 1
    assert all(cb.calls > 0 for cb in copies)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ("mode", "stream_scope", "default_off", "none", "weights", "single"))
async def test_passthrough_balancing_respects_eligible_layers_and_weights(scenario, monkeypatch):
    first: Final = RelayGuardrail("east")
    second: Final = RelayGuardrail("west")
    if scenario == "mode":
        first.event_hook = GuardrailEventHooks.pre_call
    if scenario == "stream_scope":
        first.stream_scope_default = "non_streaming"
    if scenario in ("default_off", "none"):
        first.default_on = False
    if scenario == "none":
        second.default_on = False
    copies: Final = (first,) if scenario == "single" else (first, second)
    router: Final = Router(
        model_list=[],
        guardrail_list=[
            {
                "guardrail_name": "relay-lb",
                "callback": cb,
                "litellm_params": {
                    "guardrail": "custom", "mode": "post_call",
                    "weight": 0 if cb is first and scenario == "weights" else 1,
                },
            }
            for cb in copies
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(litellm, "callbacks", list(copies))
    logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    for _ in range(20):
        chunks = [
            chunk
            async for chunk in logging.async_post_call_event_stream_iterator_hook(
                response=_output_chunks(),
                user_api_key_dict=UserAPIKeyAuth(request_route="/v1/chat/completions"),
                request_data={"model": "test-model", "stream": True},
                guardrails=tuple((cb, cb is second) for cb in copies),
            )
        ]
        assert "".join(_chunk_text(chunk) for chunk in chunks) == "raw"
    assert first.calls == (20 if scenario == "single" else 0)
    assert second.calls == (0 if scenario in ("none", "single") else 20)


@pytest.mark.asyncio
@pytest.mark.parametrize("hook", ("post_mcp_call", "deferred_stream"))
async def test_result_hooks_run_one_balanced_copy_per_request(hook, monkeypatch):
    copies: Final = tuple(
        McpResultGuardrail() if hook == "post_mcp_call" else LegacyPostCallGuardrail(fail=False) for _ in range(2)
    )
    router: Final = Router(
        model_list=[],
        guardrail_list=[
            {
                "guardrail_name": "relay-lb",
                "callback": cb,
                "litellm_params": {"guardrail": "custom", "mode": "post_call"},
            }
            for cb in copies
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(litellm, "callbacks", list(copies))
    logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    for index in range(50):
        if hook == "post_mcp_call":
            await logging.post_mcp_call_hook(
                response=CallToolResult(content=[TextContent(type="text", text="result")], isError=False),
                request_data={"mcp_tool_name": "echo"},
            )
        else:
            await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
                captured_data={"messages": [{"role": "user", "content": "hi"}]},
                captured_user_api_key_dict=UserAPIKeyAuth(),
                captured_logging_obj=MagicMock(),
                assembled_response=litellm.ModelResponse(
                    choices=[{"message": {"content": "hello", "role": "assistant"}}]
                ),
                cache_hit=False,
            )
        assert sum(cb.calls for cb in copies) == index + 1
    assert all(cb.calls > 0 for cb in copies)
