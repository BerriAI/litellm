import asyncio
from collections.abc import Mapping
from contextvars import Context, copy_context
from copy import deepcopy
from dataclasses import dataclass, field, replace
from queue import SimpleQueue
from typing import Final, Literal

import pytest

from litellm.litellm_core_utils.prompt_templates.compaction import NativeProtocol
from litellm.router_strategy.complexity_router import context_compaction as core

pytestmark: Final = pytest.mark.asyncio


def _budget(protocol: NativeProtocol, identifier: str = "target") -> core.ModelBudget:
    provider: Final = "openai" if protocol == "responses" else "anthropic"
    return core.ModelBudget(f"{provider}/native-test", 2_000, 256, deployment_id=identifier)


_HISTORY: Final = (
    {"role": "user", "content": "old"},
    {"role": "assistant", "content": "answer"},
    {"role": "user", "content": "latest"},
)


def _payload(protocol: NativeProtocol) -> Mapping[str, object]:
    return {
        "input" if protocol == "responses" else "messages": list(_HISTORY),
        "instructions" if protocol == "responses" else "system": "rules",
        "tools": [],
    }


@dataclass(frozen=True)
class _Rig:
    counts: tuple[int | Exception, ...] = (2_000, 100)
    local_count: int = 2_000
    error: Exception | None = None
    block: bool = False
    calls: SimpleQueue[tuple[str, core.NativeRequest, Context]] = field(default_factory=SimpleQueue)
    native: SimpleQueue[tuple[core.ModelBudget, Mapping[str, object], NativeProtocol]] = field(
        default_factory=SimpleQueue
    )
    local: SimpleQueue[str] = field(default_factory=SimpleQueue)
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def counter(self, model: str, payload: Mapping[str, object]) -> int:
        self.local.put(model)
        return self.local_count

    async def native_counter(
        self, target: core.ModelBudget, payload: Mapping[str, object], protocol: NativeProtocol, timeout: float
    ) -> int:
        assert timeout > 0
        self.native.put((target, payload, protocol))
        result: Final = self.counts[(self.native.qsize() - 1) % len(self.counts)]
        if isinstance(result, Exception):
            raise result
        return result

    async def execute(self, model: str, request: core.NativeRequest, timeout: float) -> Mapping[str, object]:
        assert core.current_native_request() is request and timeout > 0
        self.calls.put((model, request, copy_context()))
        self.started.set()
        if self.error is not None:
            raise self.error
        if self.block:
            await asyncio.Event().wait()
        if request.protocol == "responses":
            return {"output": [{"type": "compaction", "encrypted_content": "opaque"}]}
        return {
            "stop_reason": "compaction",
            "content": [{"type": "compaction", "content": "native context", "signature": "signed"}],
        }


async def _run(
    rig: _Rig,
    protocol: NativeProtocol = "responses",
    state: core.CompactionState | None = None,
    payload: Mapping[str, object] | None = None,
    budgets: tuple[core.ModelBudget, ...] | None = None,
    target: core.ModelBudget | None = None,
    route: Literal["chat", "responses", "messages"] | None = None,
) -> core.CompactedRequest | core.CompactionFailure | None:
    active: Final = core.CompactionState() if state is None else state
    active.arm(active.model or "compactors")
    recipient: Final = _budget(protocol) if target is None else target
    candidates: Final = (
        (replace(recipient, input_limit=10_000, output_limit=core.SUMMARY_OUTPUT_TOKENS, deployment_id="allowed"),)
        if budgets is None
        else budgets
    )
    return await core.prepare_compaction(
        _payload(protocol) if payload is None else payload,
        recipient,
        active,
        candidates,
        rig.execute,
        rig.counter,
        route or protocol,
        rig.native_counter,
    )


@pytest.mark.parametrize("protocol", ("responses", "messages"))
@pytest.mark.parametrize("mode", ("success", "error", "timeout", "cancel", "expired"))
async def test_candidates_recipient_and_context_cleanup(protocol: NativeProtocol, mode: str) -> None:
    target: Final = _budget(protocol)
    good: Final = replace(target, input_limit=10_000, output_limit=core.SUMMARY_OUTPUT_TOKENS, deployment_id="good")
    bad: Final = (
        replace(good, deployment_id="small", input_limit=1_000),
        replace(good, deployment_id="output", output_limit=1),
        replace(good, deployment_id=""),
        replace(good, deployment_id="other", model="anthropic/test" if protocol == "responses" else "openai/test"),
    )
    rig: Final = _Rig(
        error=RuntimeError("private native response") if mode == "error" else None, block=mode in ("timeout", "cancel")
    )
    state: Final = core.CompactionState(timeout=0 if mode == "expired" else 0.02 if mode == "timeout" else 10)
    payload: Final = _payload(protocol)
    original: Final = deepcopy(payload)
    with core.use_summary_executor(rig.execute):
        assert core.current_summary_executor() == rig.execute
        task: Final = asyncio.create_task(_run(rig, protocol, state, payload, (*bad, good), target))
        if mode == "cancel":
            await rig.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert isinstance(await task, core.CompactedRequest if mode == "success" else core.CompactionFailure)
    assert core.current_summary_executor() is None and core.current_native_request() is None and payload == original
    assert rig.calls.qsize() == (0 if mode == "expired" else 1)
    if mode != "expired":
        model, request, inherited = rig.calls.get_nowait()
        assert model == "compactors" and request.allowed_deployment_ids == frozenset({"good"})
        assert request.closed.is_set() and inherited.run(core.current_native_request) is None
    if mode == "success":
        before, after = (rig.native.get_nowait() for _ in range(2))
        result: Final = task.result()
        assert isinstance(result, core.CompactedRequest) and after[1][result.field] == result.value
        assert before[0] is target and after[0] is target and after[2] == protocol
        denied: Final = _Rig()
        assert isinstance(await _run(denied, protocol, budgets=bad), core.CompactionFailure) and denied.calls.empty()


@pytest.mark.parametrize("scenario", ("local_fit", "native_fit", "chat", "provider", "no_prefix"))
async def test_fit_and_unsupported_admission(scenario: str) -> None:
    rig: Final = _Rig(
        local_count=10 if scenario == "local_fit" else 2_000, counts=(10,) if scenario == "native_fit" else (2_000,)
    )
    target: Final = (
        replace(_budget("responses"), model="anthropic/test") if scenario == "provider" else _budget("responses")
    )
    result: Final = await _run(
        rig,
        target=target,
        payload={"input": [{"role": "user", "content": "latest"}]},
        route="chat" if scenario == "chat" else "responses",
        budgets=(),
    )
    assert (result is None) if scenario.endswith("fit") else isinstance(result, core.CompactionFailure)
    assert rig.calls.empty()
    assert rig.native.qsize() == (0 if scenario in ("local_fit", "chat", "provider") else 1)


@pytest.mark.parametrize(
    ("counts", "paid"),
    (
        *((((value,), 0)) for value in (-1, True, False, RuntimeError("private native response"))),
        *((((2_000, value), 1)) for value in (-1, True, False, 2_000, RuntimeError("private native response"))),
    ),
)
async def test_native_count_failures_do_not_retry_paid_work(counts: tuple[int | Exception, ...], paid: int) -> None:
    rig: Final = _Rig(counts=counts)
    state: Final = core.CompactionState()
    result: Final = await _run(rig, state=state)
    if not isinstance(result, core.CompactionFailure):
        pytest.fail("Invalid native count was accepted")
    assert "private native response" not in result.message
    assert isinstance(await _run(rig, state=state), core.CompactionFailure)
    assert rig.calls.qsize() == paid


@pytest.mark.parametrize(
    "change", ("same", "tail", "prefix", "instructions", "tools", "model", "protocol", "recipient")
)
async def test_memo_identity_and_one_call_per_request(change: str) -> None:
    rig: Final = _Rig()
    state: Final = core.CompactionState()
    assert isinstance(await _run(rig, state=state), core.CompactedRequest)
    changed: Final = {"role": "user", "content": "changed"}
    patches: Final = {
        "tail": {"input": [*_HISTORY[:-1], changed]},
        "prefix": {"input": [changed, _HISTORY[-1]]},
        "instructions": {"instructions": "changed"},
        "tools": {"tools": [{"type": "function", "name": "lookup"}]},
    }
    if change == "model":
        state.arm("different-compactors")
    protocol: Final = "messages" if change == "protocol" else "responses"
    target: Final = _budget(protocol, "retry" if change == "recipient" else "target")
    result: Final = await _run(rig, protocol, state, {**_payload(protocol), **patches.get(change, {})}, target=target)
    accepted: Final = change in ("same", "tail", "recipient")
    assert isinstance(result, core.CompactedRequest if accepted else core.CompactionFailure)
    assert rig.calls.qsize() == 1 and rig.native.qsize() == (4 if accepted else 3)
    if isinstance(result, core.CompactedRequest):
        assert result.value[-1] == (changed if change == "tail" else _HISTORY[-1])
    assert isinstance(await _run(_Rig()), core.CompactedRequest)


@pytest.mark.parametrize("protocol", ("responses", "messages"))
@pytest.mark.parametrize(
    "change", ("none", "id", "provider", "defaults", "oversized", "negative", "boolean", "allowance")
)
async def test_exact_selected_compactor_validation(protocol: NativeProtocol, change: str) -> None:
    rig: Final = _Rig(counts=({"oversized": 100_000, "negative": -1, "boolean": True}.get(change, 100),))
    target: Final = replace(_budget(protocol), input_limit=10_000, output_limit=core.SUMMARY_OUTPUT_TOKENS)
    body: Final = {
        **_payload(protocol),
        **({"max_tokens": core.SUMMARY_OUTPUT_TOKENS} if protocol == "messages" else {}),
    }
    request: Final = core.NativeRequest(protocol, body, frozenset({target.deployment_id}))
    selected: Final = replace(
        target,
        deployment_id="outsider" if change == "id" else target.deployment_id,
        model=("anthropic/test" if protocol == "responses" else "openai/test")
        if change == "provider"
        else target.model,
    )
    overrides: Final = {"defaults": {"instructions": "overridden"}, "allowance": {"max_tokens": 1}}
    payload: Final = {**body, **overrides.get(change, {})}
    result: Final = await core.validate_native_recipient(request, selected, payload, 10, rig.native_counter)
    accepted: Final = change == "none" or (change == "allowance" and protocol == "responses")
    assert (result is None) if accepted else isinstance(result, core.CompactionFailure)
    assert rig.native.qsize() == (1 if accepted or change in ("oversized", "negative", "boolean") else 0)


@pytest.mark.parametrize(
    ("patch", "reserve"),
    (
        *(
            ({key: value}, reserve)
            for key in ("max_tokens", "max_completion_tokens", "max_output_tokens")
            for value, reserve in ((None, 256), (64, 64), (0, None), (True, None), ("64", None), (257, None))
        ),
        (dict[str, object](), 256),
        ({"max_tokens": 32, "max_output_tokens": 64}, 64),
        ({"thinking": {"budget_tokens": 64}}, 256),
        ({"max_tokens": 64, "thinking": {"budget_tokens": 64}}, None),
        ({"thinking": {"budget_tokens": True}}, None),
        *(({"extra_body": body}, None) for body in (False, list[object](), "invalid")),
        *(
            ({"extra_body": {key: "override"}}, None)
            for key in ("input", "messages", "tools", "system", "instructions", "max_tokens", "thinking")
        ),
        ({"extra_body": {"provider_feature_enabled": True}}, 256),
    ),
)
async def test_output_reservation_and_extra_body_admission(patch: Mapping[str, object], reserve: int | None) -> None:
    rig: Final = _Rig(local_count=1_750, counts=(1_750, 100))
    result: Final = await _run(rig, payload={**_payload("responses"), **patch})
    if reserve is None:
        assert isinstance(result, core.CompactionFailure)
        assert rig.local.empty() and rig.native.empty() and rig.calls.empty()
    elif reserve == 64:
        assert result is None and rig.calls.empty()
    else:
        assert isinstance(result, core.CompactedRequest) and rig.calls.qsize() == 1
