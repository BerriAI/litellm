import asyncio
import json
from collections.abc import Iterator, Mapping
from copy import deepcopy
from functools import partial
from typing import Final, Literal

import httpx
import pytest
import respx

import litellm
from litellm.llms import compaction as native
from litellm.router_strategy.complexity_router.config import ContextCompactionConfig
from litellm.router_strategy.complexity_router.context_compaction import (
    CompactionState,
    Surface,
    arm_compaction,
    compact_to_fit,
    compaction_executor,
)
from litellm.types.llms.anthropic import ANTHROPIC_BETA_HEADER_VALUES
from litellm.types.router import Deployment

pytestmark: Final = [pytest.mark.asyncio, pytest.mark.usefixtures("local_model_cost_map")]
SCHEMA: Final = {"type": "object", "properties": {"code": {"type": "string"}}}


@pytest.fixture(autouse=True)
def native_catalog(monkeypatch: pytest.MonkeyPatch, local_model_cost_map: None) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    monkeypatch.setenv("LITELLM_LICENSE", "")
    monkeypatch.setattr(litellm, "use_chat_completions_url_for_anthropic_messages", False)
    monkeypatch.setitem(litellm.model_cost, "summary-fixture", {
        "litellm_provider": "anthropic", "mode": "chat", "max_input_tokens": 32000,
        "max_output_tokens": 4096, "supports_anthropic_compaction": True,
    })


def make_router(
    window: int | None = 512, settings: Mapping[str, object] | None = None, *, compactor_window: int = 32000,
    conflict: bool = False, output: int | None = 64,
    answer_defaults: Mapping[str, object] | None = None,
    context_fallback: bool = False,
) -> litellm.Router:
    config: Final = {
        "tiers": {"SIMPLE": "small", "MEDIUM": "large", "COMPLEX": "large", "REASONING": "large"},
        "keyword_tier_rules": [{"keywords": ["answer", "tail result"], "tier": "SIMPLE"}],
        "enable_context_window_escalation": False, "max_tokens_from_tier_model": False,
        **(settings or {}),
    }
    return litellm.Router(model_list=[
        {"model_name": "auto", "litellm_params": {
            "model": "auto_router/complexity_router", "complexity_router_config": config,
        }},
        {"model_name": "small", "litellm_params": {
            "model": "openai/arbitrary-answer", "api_base": "https://answer.test/v1", "api_key": "answer-test", "max_retries": 0,
            **(answer_defaults or {}),
        }, "model_info": {"id": "pinned-answer", "max_input_tokens": window, "max_output_tokens": output}},
        {"model_name": "large", "litellm_params": {
            "model": "anthropic/summary-fixture", "api_base": "https://compact.test", "api_key": "compact-test",
            **({"stop": ["deployment policy"]} if conflict else {}),
        }, "model_info": {"id": "native-compactor", "max_input_tokens": compactor_window, "max_output_tokens": 4096}},
        {"model_name": "backup", "litellm_params": {
            "model": "anthropic/summary-fixture", "api_base": "https://compact.test", "api_key": "backup-test",
        }, "model_info": {"id": "backup-compactor", "max_input_tokens": 32000, "max_output_tokens": 4096}},
    ], enable_pre_call_checks=True, num_retries=0, disable_cooldowns=True,
        retry_policy={"InternalServerErrorRetries": 1},
        context_window_fallbacks=[{"auto": ["large"]}] if context_fallback else [])


def exchange(surface: Surface, phase: str) -> list[dict[str, object]]:
    identifier: Final = f"{phase}-call"
    result: Final = f"{phase} result"
    if surface == "responses":
        return [
            {"type": "function_call", "call_id": identifier, "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": identifier, "output": result},
        ]
    if surface == "messages":
        return [
            {"role": "assistant", "content": [{"type": "tool_use", "id": identifier, "name": "lookup", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": identifier, "content": result}]},
        ]
    return [
        {"role": "assistant", "tool_calls": [
            {"id": identifier, "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": identifier, "content": result},
    ]


def history(surface: Surface) -> dict[str, object]:
    conversation: Final = [
        {"role": "user", "content": "Project code MAPLE-47. Background detail. " * 150},
        {"role": "assistant", "content": "Recorded"},
        *exchange(surface, "prefix"),
        {"role": "user", "content": "Answer with the project code"},
        *exchange(surface, "tail"),
    ]
    function: Final = {"name": "lookup", "parameters": SCHEMA}
    if surface == "responses":
        return {"instructions": "Keep the code exact", "tools": [{"type": "function", **function}], "input": [
            {"role": "developer", "content": "Retain the original spelling"}, *conversation,
        ]}
    if surface == "messages":
        return {"system": "Keep the code exact", "tools": [{"name": "lookup", "input_schema": SCHEMA}], "messages": [
            *conversation,
        ]}
    return {"tools": [{"type": "function", "function": function}], "messages": [
        {"role": "system", "content": "Keep the code exact"}, *conversation,
    ]}


def native_reply(summary: str = "Project code MAPLE-47", signed: bool = True, truncated: bool = False) -> httpx.Response:
    return httpx.Response(200, json={
        "id": "msg_compact", "type": "message", "role": "assistant", "model": "summary-fixture",
        "content": [{"type": "compaction", "content": summary, **({"signature": "native-signature"} if signed else {})}],
        "stop_reason": "max_tokens" if truncated else "compaction", "usage": {"input_tokens": 0, "output_tokens": 0, "iterations": [
            {"type": "compaction", "input_tokens": 1200, "output_tokens": 20},
        ]},
    })


def answer_reply(request: httpx.Request, expected_model: str = "arbitrary-answer") -> httpx.Response:
    payload: Final = json.loads(request.content)
    assert payload["model"] == expected_model
    assert request.headers["authorization"] == "Bearer answer-test"
    if request.url.path.endswith("responses"):
        return httpx.Response(200, json={
            "id": "resp_answer", "object": "response", "created_at": 0, "status": "completed",
            "model": payload["model"], "output": [{"id": "msg_answer", "type": "message", "role": "assistant",
                "status": "completed", "content": [{"type": "output_text", "text": "MAPLE-47", "annotations": []}]}],
            "usage": {"input_tokens": 60, "output_tokens": 8, "total_tokens": 68},
        })
    return httpx.Response(200, json={
        "id": "answer", "object": "chat.completion", "created": 0, "model": payload["model"],
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "MAPLE-47"}}],
        "usage": {"prompt_tokens": 60, "completion_tokens": 8, "total_tokens": 68},
    })


@pytest.fixture
def wire() -> Iterator[tuple[respx.Route, respx.Route]]:
    with respx.mock(assert_all_called=False) as transport:
        compactor: Final = transport.post("https://compact.test/v1/messages").mock(return_value=native_reply())
        answer: Final = transport.route(method="POST", host="answer.test").mock(side_effect=answer_reply)
        yield compactor, answer


async def invoke(router: litellm.Router, surface: Surface, payload: Mapping[str, object], retries: int = 0) -> object:
    if surface == "responses":
        return await router.aresponses(model="auto", max_output_tokens=64, num_retries=retries, **payload)
    if surface == "messages":
        return await router.aanthropic_messages(model="auto", max_tokens=64, num_retries=retries, **payload)
    return await router.acompletion(model="auto", max_tokens=64, num_retries=retries, **payload)


@pytest.mark.parametrize("surface", ["chat", "messages", "responses"])
@pytest.mark.parametrize("near", [False, True])
@pytest.mark.parametrize("configured", [False, True])
async def test_all_surfaces_compact_and_keep_selected_answerer(
    wire: tuple[respx.Route, respx.Route], surface: Surface, near: bool, configured: bool,
) -> None:
    payload: Final = history(surface)
    original: Final = deepcopy(payload)
    counted: Final = make_router()._count_pre_call_check_tokens(payload.get("messages"), payload.get("input"), payload)
    window: Final = int((counted + 32) / ContextCompactionConfig().trigger_ratio) + 1 if near else 512
    assert (counted < window) is near
    settings: Final = {"enable_context_window_escalation": True,
        **({"context_compaction": {"model": "large", "max_tokens": 512}} if configured else {})}
    router: Final = make_router(window, settings)
    captured: Final = asyncio.Queue[Mapping[str, object]]()
    compactor, answer = wire
    retry: Final = near and configured

    def answer_after_retry(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "retry answer"}}) if answer.call_count == 0 else answer_reply(request)

    if retry:
        answer.mock(side_effect=answer_after_retry)

    async def execute(
        protocol: native.CompactionProtocol, request: Mapping[str, object], parent_model: str | None = None
    ) -> Mapping[str, object]:
        assert parent_model == "auto"
        result: Final = await native.dispatch(router, protocol, request)
        captured.put_nowait(result)
        return result

    token: Final = compaction_executor.set(execute)
    try:
        response: Final = await invoke(router, surface, payload, retries=int(retry))
    finally:
        compaction_executor.reset(token)
    assert compactor.call_count == captured.qsize() == 1
    assert answer.call_count == router.total_calls["openai/arbitrary-answer"] == 1 + int(retry)
    compact_request: Final = compactor.calls[0].request
    compact_body: Final = json.loads(compact_request.content)
    answer_body: Final = json.loads(answer.calls[0].request.content)
    assert answer_body == json.loads(answer.calls[-1].request.content)
    assert compact_body["model"] == "summary-fixture" and compact_body["compaction"] == {"type": "summarize"}
    assert ANTHROPIC_BETA_HEADER_VALUES.COMPACT_2026_09_04.value in compact_request.headers["anthropic-beta"].split(",")
    assert compact_body["max_tokens"] == (512 if configured else ContextCompactionConfig().max_tokens)
    assert "Background detail" in str(compact_body) and "tail-call" not in str(compact_body)
    assert "prefix-call" in str(compact_body) and "prefix result" in str(compact_body)
    assert "Keep the code exact" in str(compact_body["system"])
    assert compact_body["tools"][0]["input_schema"] == SCHEMA
    assert "MAPLE-47" in str(answer_body) and "Background detail" not in str(answer_body)
    assert "prefix-call" not in str(answer_body)
    assert "native-signature" not in str(answer_body) and "compaction" not in answer_body
    assert "tail-call" in str(answer_body) and "tail result" in str(answer_body)
    assert "MAPLE-47" in str(response)
    usage: Final = captured.get_nowait()["usage"]
    if surface == "messages":
        assert usage["iterations"][0]["input_tokens"] == 1200 and usage["iterations"][0]["output_tokens"] == 20
    else:
        assert usage["prompt_tokens"] == 1200 and usage["completion_tokens"] == 20
    if surface == "responses":
        assert answer_body["input"][-3:] == original["input"][-3:]
        assert answer_body["input"][0] == original["input"][0]
        assert answer_body["instructions"] == original["instructions"] and answer_body["tools"] == original["tools"]
    assert payload == original


@pytest.mark.parametrize("surface", ["chat", "messages", "responses"])
@pytest.mark.parametrize("mode", ["fitting", "false", "null"])
async def test_fitting_and_disabled_requests_do_not_compact(
    wire: tuple[respx.Route, respx.Route], surface: Surface, mode: str,
) -> None:
    settings: Final = {} if mode == "fitting" else {"context_compaction": False if mode == "false" else None}
    payload: Final = history(surface)
    original: Final = deepcopy(payload)
    counted: Final = make_router()._count_pre_call_check_tokens(payload.get("messages"), payload.get("input"), payload)
    await invoke(make_router(20000 if mode == "fitting" else counted + 64, settings), surface, payload)
    compactor, answer = wire
    assert compactor.call_count == 0 and answer.call_count == 1
    assert "Background detail" in answer.calls[0].request.content.decode()
    assert payload == original


@pytest.mark.parametrize("surface", ["chat", "messages", "responses"])
@pytest.mark.parametrize("reason", ["single", "unclosed", "no_compactor"])
async def test_fitting_request_survives_unavailable_compaction(
    wire: tuple[respx.Route, respx.Route], surface: Surface, reason: str,
) -> None:
    key: Final = "input" if surface == "responses" else "messages"
    items: Final = [{"role": "user", "content": "Answer with MAPLE-47. Detail. " * 80}]
    payload: Final = history(surface) if reason == "no_compactor" else {key: (
        items if reason == "single" else [*items, *exchange(surface, "open")[:1], {"role": "user", "content": "Answer"}]
    )}
    counted: Final = make_router()._count_pre_call_check_tokens(payload.get("messages"), payload.get("input"), payload)
    settings: Final = {"tiers": {"SIMPLE": "small", "MEDIUM": "small", "COMPLEX": "small", "REASONING": "small"}} if reason == "no_compactor" else {}
    await invoke(make_router(counted + 1, settings), surface, payload)
    compactor, answer = wire
    assert compactor.call_count == 0 and answer.call_count == 1
    assert "Detail" in str(answer.calls[0].request.content) or "Background detail" in str(answer.calls[0].request.content)


@pytest.mark.parametrize("surface", ["chat", "messages", "responses"])
async def test_uncompactable_overflow_uses_explicit_context_fallback(
    wire: tuple[respx.Route, respx.Route], surface: Surface,
) -> None:
    compactor, answer = wire
    compactor.mock(return_value=httpx.Response(200, json={**native_reply().json(),
        "content": [{"type": "text", "text": "MAPLE-47"}], "stop_reason": "end_turn"}))
    payload: Final = {"input" if surface == "responses" else "messages": [
        {"role": "user", "content": "Answer with MAPLE-47. Detail. " * 150},
    ]}
    await invoke(make_router(context_fallback=True), surface, payload)
    assert answer.call_count == 0 and compactor.call_count == 1
    assert "compaction" not in json.loads(compactor.calls[0].request.content)


@pytest.mark.parametrize("escalate", [False, True])
async def test_no_native_compactor_respects_explicit_escalation(
    wire: tuple[respx.Route, respx.Route], monkeypatch: pytest.MonkeyPatch, escalate: bool,
) -> None:
    monkeypatch.setitem(litellm.model_cost["summary-fixture"], "supports_anthropic_compaction", False)
    router: Final = make_router(settings={"enable_context_window_escalation": escalate})
    compactor, answer = wire
    compactor.mock(return_value=httpx.Response(200, json={**native_reply().json(),
        "content": [{"type": "text", "text": "MAPLE-47"}], "stop_reason": "end_turn"}))
    if not escalate:
        with pytest.raises(litellm.ContextWindowExceededError, match="No configured compactor"):
            await invoke(router, "chat", history("chat"))
        assert compactor.call_count == 0
    else:
        await invoke(router, "chat", history("chat"))
        assert compactor.call_count == 1
        assert "compaction" not in json.loads(compactor.calls[0].request.content)
    assert answer.call_count == 0


@pytest.mark.parametrize("surface", ["chat", "messages", "responses"])
async def test_undersized_native_compactor_does_not_block_explicit_escalation(
    wire: tuple[respx.Route, respx.Route], surface: Surface,
) -> None:
    router: Final = make_router(compactor_window=512, settings={
        "enable_context_window_escalation": True,
        "tiers": {"SIMPLE": "small", "MEDIUM": "large", "COMPLEX": "wide", "REASONING": "wide"},
    })
    router.add_deployment(Deployment(
        model_name="wide", litellm_params={
            "model": "openai/wide-answer", "api_base": "https://answer.test/v1", "api_key": "answer-test",
        }, model_info={"id": "wide-answer", "max_input_tokens": 32000, "max_output_tokens": 64},
    ))
    compactor, answer = wire
    answer.mock(side_effect=partial(answer_reply, expected_model="wide-answer"))
    await invoke(router, surface, history(surface))
    assert compactor.call_count == 0 and answer.call_count == 1
    assert json.loads(answer.calls[0].request.content)["model"] == "wide-answer"


@pytest.mark.parametrize(
    ("surface", "failure"),
    [(surface, failure) for surface in ("chat", "messages", "responses") for failure in ("unsigned", "oversized", "provider")]
    + [("messages", "truncated")],
)
async def test_bad_native_result_never_reaches_answerer(
    wire: tuple[respx.Route, respx.Route], surface: Surface, failure: str,
) -> None:
    compactor, answer = wire
    reply: Final = httpx.Response(500, json={"error": {"type": "api_error", "message": "failed"}}) if failure == "provider" else native_reply(
        "too large " * 2000 if failure == "oversized" else "MAPLE-47", signed=failure != "unsigned",
        truncated=failure == "truncated",
    )
    compactor.mock(return_value=reply)
    with pytest.raises((litellm.BadRequestError, litellm.InternalServerError)):
        await invoke(make_router(), surface, history(surface))
    assert compactor.call_count == 1 and answer.call_count == 0


@pytest.mark.parametrize("failure", ["item", "content", "tools", "unclosed", "missing", "duplicate", "instructions", "retained"])
@pytest.mark.parametrize("needed", [False, True])
async def test_unsafe_responses_reject_only_when_compaction_needed(
    wire: tuple[respx.Route, respx.Route], failure: str, needed: bool,
) -> None:
    payload: Final = history("responses")
    extra: Final = {
        "item": [{"type": "computer_call", "call_id": "opaque-tool"}],
        "content": [{"role": "assistant", "content": [{"type": "refusal", "refusal": "cannot"}]}],
        "unclosed": [{"type": "function_call", "call_id": "unclosed", "name": "lookup", "arguments": "{}"}],
        "missing": [{"type": "function_call", "name": "lookup", "arguments": "{}"}],
        "duplicate": exchange("responses", "prefix"),
        "instructions": [{"role": "developer", "content": "Changed instructions"}],
    }
    request: Final = {
        **payload, "input": [*payload["input"][:3], *extra.get(failure, []), *payload["input"][3:]],
        **({"tools": [{"type": "computer_use_preview", "display_width": 800, "display_height": 600}]} if failure == "tools" else {}),
        **({"instructions": "Keep every instruction " * 600} if failure == "retained" else {}),
    }
    if needed:
        with pytest.raises(litellm.BadRequestError, match="Context compaction"):
            await invoke(make_router(), "responses", request)
        assert all(route.call_count == 0 for route in wire)
    else:
        await invoke(make_router(20000), "responses", request)
        compactor, answer = wire
        assert compactor.call_count == 0 and answer.call_count == 1


@pytest.mark.parametrize("owned", [
    {"previous_response_id": "resp_parent"}, {"conversation": "conv_parent"},
    {"context_management": [{"type": "compaction", "compact_threshold": 1000}]}, {"compaction": {"type": "summarize"}},
    {"input": [{"type": "reasoning", "encrypted_content": "opaque"}]},
    {"input": [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "prior reasoning"}]}]},
    {"input": [{"type": "compaction", "encrypted_content": "opaque"}]},
    {"input": [{"type": "item_reference", "id": "item_parent"}]},
    {"input": [{"role": "assistant", "content": "visible", "encrypted_content": "opaque"}]},
    {"input": [{"role": "user", "content": [{"type": "encrypted_content", "encrypted_content": "opaque"}]}]},
])
@pytest.mark.parametrize("arm_first", [False, True])
async def test_client_owned_history_bypasses_compaction(
    wire: tuple[respx.Route, respx.Route], owned: Mapping[str, object], arm_first: bool,
) -> None:
    router: Final = make_router()
    deployment: Final = router.get_deployment(model_id="pinned-answer")
    assert deployment is not None
    state: Final = CompactionState()
    request: Final = {**history("responses"), **owned, "model": "small", "max_tokens": 64, "_context_compaction_state": state}
    original: Final = deepcopy({key: value for key, value in request.items() if key != "_context_compaction_state"})
    before_defaults: Final = {"_context_compaction_state": state} if arm_first else request
    await arm_compaction(before_defaults, ContextCompactionConfig(), ("large",))
    counted: Final = router._count_pre_call_check_tokens(None, request["input"], request)
    if arm_first and counted > 512:
        with pytest.raises(litellm.ContextWindowExceededError):
            await compact_to_fit(router, deployment.model_dump(), request, "responses")
    else:
        result: Final = await compact_to_fit(router, deployment.model_dump(), request, "responses")
        assert result is request
    assert {key: value for key, value in request.items() if key != "_context_compaction_state"} == original
    assert all(route.call_count == 0 for route in wire)


@pytest.mark.parametrize("surface", ["chat", "messages", "responses"])
@pytest.mark.parametrize("source", ["request", "deployment"])
async def test_client_managed_overflow_keeps_context_window_admission(
    wire: tuple[respx.Route, respx.Route], surface: Surface, source: str,
) -> None:
    managed: Final = {"context_management": {"edits": []}}
    router: Final = make_router(answer_defaults=managed if source == "deployment" else None)
    payload: Final = {**history(surface), **(managed if source == "request" else {})}
    compactor, answer = wire
    if source == "deployment":
        with pytest.raises(litellm.ContextWindowExceededError):
            await invoke(router, surface, payload)
        assert compactor.call_count == 0
    else:
        await invoke(router, surface, payload)
        assert compactor.call_count == 1
        assert "compaction" not in json.loads(compactor.calls[0].request.content)
    assert answer.call_count == 0


@pytest.mark.parametrize("case", ["conflicting_defaults", "small_window", "capability_false", "capability_missing"])
async def test_automatic_compactor_skips_conflicts_and_requires_capacity_and_capability(
    wire: tuple[respx.Route, respx.Route], monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    if case.startswith("capability"):
        metadata: Final = {key: value for key, value in litellm.model_cost["summary-fixture"].items()
                           if key != "supports_anthropic_compaction"}
        monkeypatch.setitem(litellm.model_cost, "summary-fixture", {
            **metadata, **({"supports_anthropic_compaction": False} if case == "capability_false" else {}),
        })
    conflict: Final = case == "conflicting_defaults"
    settings: Final = {"tiers": {"SIMPLE": "small", "MEDIUM": "large", "COMPLEX": "backup", "REASONING": "backup"}} if conflict else {}
    router: Final = make_router(settings=settings, conflict=conflict, compactor_window=512 if case == "small_window" else 32000)
    if conflict:
        await invoke(router, "chat", history("chat"))
        compactor, answer = wire
        assert compactor.call_count == answer.call_count == 1
        assert compactor.calls[0].request.headers["x-api-key"] == "backup-test"
        assert "stop_sequences" not in json.loads(compactor.calls[0].request.content)
    else:
        with pytest.raises(litellm.BadRequestError, match="No configured compactor"):
            await invoke(router, "chat", history("chat"))
        assert all(route.call_count == 0 for route in wire)


@pytest.mark.parametrize("unknown_output", [False, True])
@pytest.mark.parametrize("overflow", [False, True])
async def test_unusable_output_budget_still_enforces_known_input_window(
    wire: tuple[respx.Route, respx.Route], unknown_output: bool, overflow: bool,
) -> None:
    payload: Final = history("chat")
    counted: Final = make_router(output=None)._count_pre_call_check_tokens(payload["messages"], None, payload)
    window: Final = 512 if overflow else counted + 64
    output: Final = None if unknown_output else window
    router: Final = make_router(window, output=output)
    deployment: Final = router.get_deployment(model_id="pinned-answer")
    assert deployment is not None
    state: Final = CompactionState(config=ContextCompactionConfig(), candidates=("large",))
    request: Final = {**payload, "model": "small", "max_tokens": output, "_context_compaction_state": state}
    if overflow:
        with pytest.raises(litellm.BadRequestError, match="known input window"):
            await compact_to_fit(router, deployment.model_dump(), request, "chat")
    else:
        assert await compact_to_fit(router, deployment.model_dump(), request, "chat") is request
    assert all(route.call_count == 0 for route in wire)


@pytest.mark.parametrize("outcome", ["success", "timeout", "cancel"])
async def test_retry_reuses_summary_or_terminal_cancellation(outcome: Literal["success", "timeout", "cancel"]) -> None:
    router: Final = make_router()
    deployment: Final = router.get_deployment(model_id="pinned-answer")
    assert deployment is not None
    state: Final = CompactionState(config=ContextCompactionConfig(model="large", max_tokens=512, timeout_seconds=0.02))
    request: Final = {**history("messages"), "model": "small", "max_tokens": 64, "_context_compaction_state": state}
    calls: Final = asyncio.Queue[None]()
    started: Final = asyncio.Event()
    stopped: Final = asyncio.Event()

    async def execute(
        protocol: native.CompactionProtocol, payload: Mapping[str, object], parent_model: str | None = None
    ) -> Mapping[str, object]:
        calls.put_nowait(None)
        started.set()
        try:
            return native_reply().json() if outcome == "success" else await asyncio.Future[Mapping[str, object]]()
        finally:
            stopped.set()

    token: Final = compaction_executor.set(execute)
    try:
        first: Final = asyncio.create_task(compact_to_fit(router, deployment.model_dump(), request, "messages"))
        await asyncio.wait_for(started.wait(), timeout=2)
        if outcome == "cancel":
            first.cancel()
        if outcome == "success":
            assert await first == await compact_to_fit(router, deployment.model_dump(), request, "messages")
            changed: Final = {**request, "messages": [{"role": "user", "content": "new history"}, *request["messages"]]}
            with pytest.raises(litellm.BadRequestError, match="History changed"):
                await compact_to_fit(router, deployment.model_dump(), changed, "messages")
        else:
            error: Final = asyncio.CancelledError if outcome == "cancel" else asyncio.TimeoutError
            with pytest.raises(error):
                await first
            with pytest.raises(error):
                await compact_to_fit(router, deployment.model_dump(), request, "messages")
        assert calls.qsize() == 1 and stopped.is_set()
    finally:
        compaction_executor.reset(token)
