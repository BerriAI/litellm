import asyncio
import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Final, Literal

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.initialize_dynamic_callback_params import initialize_standard_callback_dynamic_params
from litellm.router_strategy.complexity_router.config import ComplexityRouterConfig, ContextCompactionConfig
from litellm.router_strategy.complexity_router.context_compaction import (
    CompactionState,
    _history,
    compact_to_fit,
    compaction_executor,
)
from litellm.router_utils.auto_router_model_naming import strategy_router_dependencies
from litellm.types.router import Deployment

pytestmark: Final = pytest.mark.usefixtures("local_model_cost_map")


@pytest.fixture(autouse=True)
def isolated_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    monkeypatch.setenv("LITELLM_LICENSE", "")


def make_router(
    *, mode: Literal["configured", "default", "empty", "disabled"] = "configured",
    compactor_window: int = 8192, target_window: int | None = 512, pin_output: bool = True,
    compactor_model: str = "anthropic/claude-sonnet-5",
    compactor_api_base: str | None = None,
    target_model: str = "anthropic/claude-haiku-4-5-20251001",
    target_output: int | None = 128,
) -> litellm.Router:
    compaction: Final = {"configured": {"model": "large", "max_tokens": 512}, "empty": {}, "disabled": False}
    config: Final = {
        "tiers": {"SIMPLE": "small", "MEDIUM": "large", "COMPLEX": "large", "REASONING": "large"},
        "keyword_tier_rules": [{"keywords": ["answer"], "tier": "SIMPLE"}],
        **({"context_compaction": compaction[mode]} if mode != "default" else {}),
        "enable_context_window_escalation": mode != "disabled",
        "max_tokens_from_tier_model": pin_output,
    }
    return litellm.Router(
        model_list=[
            {"model_name": "auto", "litellm_params": {"model": "auto_router/complexity_router", "complexity_router_config": config}},
            {"model_name": "small", "litellm_params": {"model": target_model, "api_key": "test"}, "model_info": {"id": "pinned-small", "max_input_tokens": target_window, "max_output_tokens": target_output}},
            {"model_name": "large", "litellm_params": {"model": compactor_model, "api_key": "test", **({"api_base": compactor_api_base} if compactor_api_base else {})}, "model_info": {"id": "native-large", "max_input_tokens": compactor_window, "max_output_tokens": 4096}},
        ],
        enable_pre_call_checks=True,
        num_retries=0,
        disable_cooldowns=True,
    )


def history() -> list[dict[str, object]]:
    return [
        {"role": "user", "content": "Project code MAPLE-47. Background detail. " * 150},
        {"role": "assistant", "content": "Recorded"},
        {"role": "user", "content": "Answer with the project code"},
    ]


def provider_reply(payload: Mapping[str, object], summary: str = "Project code MAPLE-47") -> httpx.Response:
    compact: Final = "compaction" in payload
    return httpx.Response(200, json={
        "id": "msg_test", "type": "message", "role": "assistant", "model": payload["model"],
        "content": [{"type": "compaction", "content": summary, "signature": "native-signature"}] if compact else [{"type": "text", "text": "MAPLE-47"}],
        "stop_reason": "compaction" if compact else "end_turn",
        "usage": {"input_tokens": 0, "output_tokens": 0, "iterations": [{"type": "compaction", "input_tokens": 1200, "output_tokens": 20}]} if compact else {"input_tokens": 60, "output_tokens": 8},
    })


@pytest.mark.asyncio
@pytest.mark.parametrize("messages_api", [False, True])
@pytest.mark.parametrize("target_window", [512, 1640])
@pytest.mark.parametrize(("mode", "gateway"), [("configured", False), ("default", False), ("empty", False), ("default", True)])
async def test_router_compacts_on_native_model_and_answers_on_selected_deployment(
    monkeypatch: pytest.MonkeyPatch, messages_api: bool, target_window: int,
    mode: Literal["configured", "default", "empty"], gateway: bool,
) -> None:
    monkeypatch.setattr(litellm, "use_chat_completions_url_for_anthropic_messages", False)
    router: Final = make_router(
        target_window=target_window, mode=mode,
        compactor_model="openai/anthropic/claude-sonnet-5" if gateway else "anthropic/claude-sonnet-5",
        compactor_api_base="https://gateway.test/v1" if gateway else None,
    )
    messages: Final = history()
    original: Final = deepcopy(messages)
    expected: Final = messages if messages_api else [{**message, "content": [{"type": "text", "text": message["content"]}]} for message in messages]

    def respond(request: httpx.Request) -> httpx.Response:
        payload: Final = json.loads(request.content)
        assert "_context_compaction_state" not in payload
        if "compaction" in payload:
            assert payload["model"] == ("anthropic/claude-sonnet-5" if gateway else "claude-sonnet-5")
            assert payload["compaction"] == {"type": "summarize"}
            assert "compact-2026-09-04" in request.headers.get("anthropic-beta", "").split(",")
            assert payload["messages"] == (messages if gateway else expected)[:-1]
            assert payload["max_tokens"] == (512 if mode == "configured" else 4096)
            if gateway:
                return httpx.Response(200, json={
                    "id": "summary", "model": payload["model"], "object": "chat.completion", "created": 0,
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {
                        "role": "assistant", "content": "", "provider_specific_fields": {
                            "compaction_blocks": [{"type": "compaction", "content": "Project code MAPLE-47", "signature": "native-signature"}],
                        },
                    }}],
                    "usage": {"prompt_tokens": 1200, "completion_tokens": 20, "total_tokens": 1220},
                })
        else:
            assert payload["model"] == "claude-haiku-4-5-20251001"
            assert payload["messages"][-1] == expected[-1]
            assert "MAPLE-47" in str(payload["messages"][0])
            assert "signature" not in str(payload["messages"])
        return provider_reply(payload)

    with respx.mock(assert_all_called=False) as transport:
        route: Final = transport.post("https://api.anthropic.com/v1/messages").mock(side_effect=respond)
        gateway_route: Final = transport.post("https://gateway.test/v1/chat/completions").mock(side_effect=respond)
        call: Final = router.aanthropic_messages if messages_api else router.acompletion
        response: Final = await call(model="auto", messages=messages, max_tokens=64)
        assert route.call_count == (1 if gateway else 2)
        assert gateway_route.call_count == int(gateway)
        assert response is not None
    assert messages == original


@pytest.mark.asyncio
@pytest.mark.parametrize("messages_api", [False, True])
@pytest.mark.parametrize("capability", [True, False, None])
async def test_catalog_compactor_support_is_independent_of_deepseek_answering_model(
    monkeypatch: pytest.MonkeyPatch, messages_api: bool, capability: bool | None,
) -> None:
    monkeypatch.setitem(litellm.model_cost, "native-summary-fixture", {
        "litellm_provider": "anthropic", "mode": "chat", "max_input_tokens": 8192,
        "max_output_tokens": 4096, "supports_anthropic_compaction": capability,
    })
    router: Final = make_router(
        mode="default", compactor_model="anthropic/native-summary-fixture",
        target_model="deepseek/deepseek-v4-flash",
    )

    def summarize(request: httpx.Request) -> httpx.Response:
        payload: Final = json.loads(request.content)
        assert payload["model"] == "native-summary-fixture"
        assert payload["compaction"] == {"type": "summarize"}
        return provider_reply(payload)

    def answer(request: httpx.Request) -> httpx.Response:
        payload: Final = json.loads(request.content)
        assert payload["model"] == "deepseek-v4-flash"
        assert "Project code MAPLE-47" in str(payload["messages"][0])
        assert payload["messages"][-1] == history()[-1]
        assert "compaction" not in payload and "signature" not in str(payload["messages"])
        if messages_api:
            return provider_reply(payload)
        return httpx.Response(200, json={
            "id": "answer", "model": payload["model"], "object": "chat.completion", "created": 0,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "MAPLE-47"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 60, "completion_tokens": 8, "total_tokens": 68},
        })

    with respx.mock(assert_all_called=False) as transport:
        native: Final = transport.post("https://api.anthropic.com/v1/messages").mock(side_effect=summarize)
        target: Final = transport.post("https://api.deepseek.com/anthropic/v1/messages" if messages_api else "https://api.deepseek.com/beta/chat/completions").mock(side_effect=answer)
        call: Final = router.aanthropic_messages if messages_api else router.acompletion
        if capability:
            await call(model="auto", messages=history(), max_tokens=64)
        else:
            with pytest.raises(litellm.BadRequestError, match="No configured tier model"):
                await call(model="auto", messages=history(), max_tokens=64)
        assert native.call_count == target.call_count == int(capability is True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["no_history", "compactor_too_small", "summary_too_large", "retained_too_large", "not_native", "empty_summary", "invalid_window"])
async def test_overflow_fails_without_sending_an_oversized_answer(failure: str) -> None:
    router: Final = make_router(compactor_window=800 if failure == "compactor_too_small" else 8192, target_window=0 if failure == "invalid_window" else 512)
    messages: Final = history()[:1] if failure == "no_history" else [*history()[:-1], {"role": "user", "content": "Answer " * 1500}] if failure == "retained_too_large" else history()

    def respond(request: httpx.Request) -> httpx.Response:
        payload: Final = json.loads(request.content)
        assert "compaction" in payload
        return provider_reply(
            {"model": payload["model"]} if failure == "not_native" else payload,
            summary="" if failure == "empty_summary" else "oversized " * 1500,
        )

    with respx.mock(assert_all_called=False) as transport:
        route: Final = transport.post("https://api.anthropic.com/v1/messages").mock(side_effect=respond)
        with pytest.raises((litellm.BadRequestError, litellm.ContextWindowExceededError)):
            await router.acompletion(model="auto", messages=messages, max_tokens=64)
        assert route.call_count == (1 if failure in ("summary_too_large", "not_native", "empty_summary") else 0)


@pytest.mark.asyncio
async def test_explicit_opt_out_disables_compaction_without_changing_existing_routing() -> None:
    router: Final = make_router(mode="disabled")
    with respx.mock(assert_all_called=False) as transport:
        route: Final = transport.post("https://api.anthropic.com/v1/messages").mock(return_value=provider_reply({"model": "claude-sonnet-5"}))
        await router.acompletion(model="auto", messages=history(), max_tokens=64)
        assert route.call_count == 1
        assert "compaction" not in json.loads(route.calls[0].request.content)


@pytest.mark.parametrize("disabled", [False, None])
def test_compaction_opt_out_survives_management_config_serialization(disabled: Literal[False] | None) -> None:
    config: Final = ComplexityRouterConfig(context_compaction=disabled)
    restored: Final = ComplexityRouterConfig.model_validate(config.model_dump(exclude_none=True))
    assert restored.context_compaction is False


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "window", "mixed"), (
    ("anthropic/claude-haiku-4-5-20251001", 8192, False),
    ("anthropic/claude-sonnet-5", 800, False),
    ("anthropic/claude-sonnet-5", 8192, True),
))
async def test_default_compaction_rejects_ineligible_tier_compactors_before_spending(model: str, window: int, mixed: bool) -> None:
    router: Final = make_router(mode="default", compactor_model=model, compactor_window=window)
    if mixed:
        router.add_deployment(Deployment.model_validate({
            "model_name": "large", "litellm_params": {"model": model, "api_key": "test"},
            "model_info": {"id": "undersized-member", "max_input_tokens": 800, "max_output_tokens": 512},
        }))
    with respx.mock(assert_all_called=False) as transport:
        route: Final = transport.post("https://api.anthropic.com/v1/messages")
        with pytest.raises(litellm.BadRequestError, match="No configured tier model"):
            await router.acompletion(model="auto", messages=history(), max_tokens=64)
        assert route.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("oversized", [False, True])
@pytest.mark.parametrize(("messages_api", "output"), (
    (False, None), (False, 460), (False, 512), (True, 460), (True, 512),
))
async def test_unusable_reserved_budget_does_not_bypass_known_input_window(
    oversized: bool, messages_api: bool, output: int | None,
) -> None:
    router: Final = make_router(
        mode="default", pin_output=False, target_model="anthropic/unknown-output-fixture", target_output=None,
    )
    messages: Final = history() if oversized else [{"role": "user", "content": "Answer hello"}]
    with respx.mock(assert_all_called=False) as transport:
        route: Final = transport.post("https://api.anthropic.com/v1/messages").mock(return_value=provider_reply({"model": "unknown-output-fixture"}))
        call: Final = router.aanthropic_messages if messages_api else router.acompletion
        if oversized:
            with pytest.raises(litellm.BadRequestError, match="known input window"):
                await call(model="auto", messages=messages, max_tokens=output)
        else:
            await call(model="auto", messages=messages, max_tokens=output)
        assert route.call_count == int(not oversized)


@pytest.mark.asyncio
@pytest.mark.parametrize("target_window", [None, 512])
@pytest.mark.parametrize(("messages_api", "output"), (
    (False, {"max_tokens": 64}),
    (False, {"max_tokens": 64, "max_completion_tokens": None}),
    (False, {"max_tokens": None}),
    (False, {"max_tokens": 460}),
    (False, {"max_tokens": 512}),
    (True, {"max_tokens": 64}),
    (True, {"max_tokens": 64, "max_completion_tokens": None}),
    (True, {"max_tokens": 460}),
    (True, {"max_tokens": 512}),
))
async def test_fitting_or_unknown_budget_requests_do_not_call_compactor(messages_api: bool, output: Mapping[str, int | None], target_window: int | None) -> None:
    router: Final = make_router(mode="default", pin_output=False, target_window=target_window, target_model="anthropic/unknown-window-fixture", compactor_model="anthropic/claude-haiku-4-5-20251001")
    with respx.mock(assert_all_called=False) as transport:
        route: Final = transport.post("https://api.anthropic.com/v1/messages").mock(return_value=provider_reply({"model": "claude-haiku-4-5-20251001"}))
        call: Final = router.aanthropic_messages if messages_api else router.acompletion
        await call(model="auto", messages=history() if target_window is None else [{"role": "user", "content": "Answer hello"}], **output)
        assert route.call_count == 1
        assert "compaction" not in json.loads(route.calls[0].request.content)


@pytest.mark.asyncio
@pytest.mark.parametrize(("privacy", "settings", "global_privacy"), (
    (True, {"turn_off_message_logging": True}, False),
    (True, {"metadata": {"turn_off_message_logging": True}}, False),
    (True, {}, True),
    (True, {"metadata": {"headers": {"x-litellm-enable-message-redaction": "true"}}}, False),
    (False, {"turn_off_message_logging": False}, False),
    (False, {"metadata": {"headers": {"litellm-disable-message-redaction": "true"}}}, True),
))
async def test_summary_reused_on_retry_and_includes_system_and_tools_in_budget(
    monkeypatch: pytest.MonkeyPatch, privacy: bool, settings: Mapping[str, object], global_privacy: bool,
) -> None:
    router: Final = make_router()
    deployment: Final = router.get_deployment(model_id="pinned-small").model_dump()
    state: Final = CompactionState(config=ContextCompactionConfig(model="large", max_tokens=512))
    monkeypatch.setattr(litellm, "turn_off_message_logging", global_privacy)
    payload: Final = {
        "model": "small", "messages": history(), "max_tokens": 64, "_context_compaction_state": state, **settings,
    }
    calls: Final = asyncio.Queue[Mapping[str, object]]()

    async def execute(protocol: str, request: Mapping[str, object]) -> Mapping[str, object]:
        assert "turn_off_message_logging" not in request and "turn_off_message_logging" not in request["metadata"]
        assert initialize_standard_callback_dynamic_params().get("turn_off_message_logging", False) is privacy
        calls.put_nowait(request)
        return {"choices": [{"message": {"provider_specific_fields": {"compaction_blocks": [{"type": "compaction", "content": "MAPLE-47", "signature": "signed"}]}}}]}

    token: Final = compaction_executor.set(execute)
    try:
        first: Final = await compact_to_fit(router, deployment, payload, "chat")
        second: Final = await compact_to_fit(router, deployment, payload, "chat")
        assert first == second and calls.qsize() == 1
        with pytest.raises(litellm.BadRequestError, match="leave no room"):
            await compact_to_fit(router, deployment, {**payload, "max_tokens": 450}, "chat")
        with pytest.raises(litellm.BadRequestError, match="leave no room"):
            await compact_to_fit(router, deployment, {**payload, "system": "instructions " * 600}, "chat")
    finally:
        compaction_executor.reset(token)
    assert initialize_standard_callback_dynamic_params().get("turn_off_message_logging") is None


@pytest.mark.asyncio
async def test_compaction_timeout_is_replayed_without_cancelling_retry_or_spending_again() -> None:
    router: Final = make_router()
    deployment: Final = router.get_deployment(model_id="pinned-small").model_dump()
    state: Final = CompactionState(config=ContextCompactionConfig(model="large", timeout_seconds=0.01))
    payload: Final = {"model": "small", "messages": history(), "max_tokens": 64, "_context_compaction_state": state}
    calls: Final = asyncio.Queue[None]()
    stopped: Final = asyncio.Event()

    async def execute(protocol: str, request: Mapping[str, object]) -> Mapping[str, object]:
        calls.put_nowait(None)
        try:
            return await asyncio.Future[Mapping[str, object]]()
        finally:
            stopped.set()

    token: Final = compaction_executor.set(execute)
    try:
        for _attempt in range(2):
            with pytest.raises(asyncio.TimeoutError):
                await compact_to_fit(router, deployment, payload, "chat")
            assert calls.qsize() == 1 and stopped.is_set()
    finally:
        compaction_executor.reset(token)


def test_native_tool_result_tail_is_retained_and_compactor_is_an_authorized_dependency() -> None:
    messages: Final = [*history(), {"role": "assistant", "content": [{"type": "tool_use", "id": "call", "name": "lookup", "input": {}}]}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call", "content": "value"}]}]
    _, prefix, tail = _history(messages, "small")
    assert list(prefix) == messages[:2]
    assert list(tail) == messages[2:]
    dependencies: Final = strategy_router_dependencies({"model": "auto_router/complexity_router", "complexity_router_config": {"context_compaction": {"model": "large"}}})
    assert [(dependency.model_name, dependency.role) for dependency in dependencies] == [("large", "compactor")]
