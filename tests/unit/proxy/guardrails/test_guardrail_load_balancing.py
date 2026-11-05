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

import litellm
from litellm import Router
from litellm.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai.chat.guardrail_translation.handler import OpenAIChatCompletionsHandler
from litellm.proxy._types import UserAPIKeyAuth
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
async def test_pre_call_load_balancing_observes_earlier_callback_activation(
    action: str, run_in_parallel: bool, call_type: CallTypesLiteral, monkeypatch: pytest.MonkeyPatch
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
    monkeypatch.setattr(litellm, "callbacks", [GuardrailActivator(action), *replicas])
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
        assert sum(cb.calls for cb in replicas) == (request_index + 1 if action == "enable" else 0)
    assert all(cb.calls > 0 for cb in replicas) is (action == "enable")


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
    if healthy:
        await _dispatch_scoped_hook(logging, hook)
        assert guardrail.calls == 1
    else:
        with pytest.raises(ValueError, match="No callback found for guardrail: partial-lb"):
            await _dispatch_scoped_hook(logging, hook)
        assert guardrail.calls == 0


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
@pytest.mark.parametrize("scenario", ("mode", "stream_scope", "none", "weights", "single"))
async def test_pipeline_balancing_respects_eligibility_weights_and_single_copy(scenario, monkeypatch):
    first: Final = MockGuardrail(
        "pipeline-scoped", "east",
        event_hook=GuardrailEventHooks.pre_call if scenario in ("mode", "single", "none") else GuardrailEventHooks.post_call,
        stream_scope="streaming" if scenario == "stream_scope" else "both",
    )
    second: Final = MockGuardrail(
        "pipeline-scoped", "west",
        event_hook=GuardrailEventHooks.pre_call if scenario == "none" else GuardrailEventHooks.post_call,
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
