import asyncio
import json
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone, tzinfo
from types import SimpleNamespace
from typing import Final, Literal

import pytest
from fastapi import HTTPException
from pydantic import JsonValue

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.models.workflow_definition import ChatCompletionNode
from litellm.models.workflow_execution import ChildCallResult, ExecutionError, workflow_call_id
from litellm.proxy import proxy_server
from litellm.proxy._types import (
    LiteLLM_EndUserTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import cache_key_object
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    end_user_cache_key,
    team_membership_auth_cache_key,
    team_membership_reservation_cache_key,
)
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.proxy.utils import ProxyLogging
from litellm.proxy.workflows import gateway
from litellm.types.utils import ModelResponse
from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters

_NOW: Final = datetime(2026, 11, 6, tzinfo=timezone.utc)
_PARENT_STATE: Final = ContextVar("workflow_gateway_test_parent", default="isolated")
_KEY: Final = "a" * 64


class _Clock(datetime):
    @classmethod
    def now(cls, tz: tzinfo | None = None) -> datetime:
        return _NOW


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(gateway, "datetime", _Clock)
    token: Final = _PARENT_STATE.set("parent")
    yield
    _PARENT_STATE.reset(token)


def _call(*, structured: bool = False) -> gateway._NodeCall:
    node: Final = ChatCompletionNode.model_validate(
        {
            "id": "answer",
            "type": "chat_completion",
            "model": "workflow-model",
            "messages": {"from": "$input", "output": "messages"},
            "timeout_seconds": 60,
            "retry": {"max_attempts": 2, "backoff_seconds": 0},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": True,
                    "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
                },
            }
            if structured
            else None,
        }
    )
    return gateway._NodeCall(
        key_hash=_KEY,
        team_id="team",
        node=node,
        messages=[{"role": "user", "content": "question"}],
        workflow_id="workflow",
        run_id="run",
        session_id="session",
        attempt=1,
        deadline=_NOW + timedelta(minutes=2),
        end_user_id="end-user",
    )


@dataclass(frozen=True, slots=True)
class _Gateway:
    content: str | None = "answer"
    failure: Exception | None = None
    blocked_at: Literal["authorize", "process", "validate"] | None = None
    response_body: Mapping[str, object] | None = None
    receipt_ids: tuple[str, ...] = ("provider-id_cache_hit123",)
    schema_error: ExecutionError | None = None
    contexts: asyncio.Queue[gateway._RequestContext] = field(default_factory=asyncio.Queue)
    schema_inputs: asyncio.Queue[JsonValue] = field(default_factory=asyncio.Queue)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    stopped: asyncio.Event = field(default_factory=asyncio.Event)
    processed: asyncio.Event = field(default_factory=asyncio.Event)
    released: asyncio.Event = field(default_factory=asyncio.Event)

    async def _block(self, stage: str) -> None:
        if self.blocked_at == stage:
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.stopped.set()

    async def authorize(self, call: gateway._NodeCall, context: gateway._RequestContext) -> UserAPIKeyAuth:
        assert _PARENT_STATE.get() == "isolated"
        context.request.state.budget_reservation = {"reservation_id": context.request.headers["x-litellm-call-id"]}
        self.contexts.put_nowait(context)
        await self._block("authorize")
        if self.failure is not None:
            raise self.failure
        return UserAPIKeyAuth(token=call.key_hash, team_id=call.team_id, end_user_id=call.end_user_id)

    async def process(self, context: gateway._RequestContext, auth: UserAPIKeyAuth) -> object:
        self.processed.set()
        await self._block("process")
        context.response.headers["x-litellm-response-cost"] = "0.125"
        return (
            self.response_body
            if self.response_body is not None
            else {
                "id": "provider-id",
                "choices": [{"message": {"role": "assistant", "content": self.content}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            }
        )

    async def receipts(self, call: gateway._NodeCall, call_id: str) -> tuple[str, ...]:
        return self.receipt_ids

    async def validate_output(
        self, output: JsonValue, schema: Mapping[str, JsonValue], deadline: float
    ) -> ExecutionError | None:
        self.schema_inputs.put_nowait(output)
        await self._block("validate")
        return self.schema_error

    async def release(self, context: gateway._RequestContext) -> None:
        self.released.set()


@pytest.mark.asyncio
async def test_attempts_keep_identity_and_unique_call_contexts_without_inheriting_parent() -> None:
    port: Final = _Gateway()
    call: Final = _call()
    first: Final = await gateway._execute_isolated(call, port)
    second: Final = await gateway._execute_isolated(replace(call, attempt=2), port)
    assert first == second == ChildCallResult("answer", 7, 3, "provider-id_cache_hit123", 0.125)
    assert _PARENT_STATE.get() == "parent"
    first_context: Final = port.contexts.get_nowait()
    second_context: Final = port.contexts.get_nowait()
    assert first_context.request.state is not second_context.request.state
    assert first_context.request.headers["x-litellm-call-id"] == workflow_call_id("run", "answer", 1)
    assert second_context.request.headers["x-litellm-call-id"] == workflow_call_id("run", "answer", 2)
    assert first_context.request.headers["x-litellm-call-id"] != second_context.request.headers["x-litellm-call-id"]
    assert first_context.data["metadata"] == {
        "workflow_id": "workflow",
        "run_id": "run",
        "node_id": "answer",
        "attempt": 1,
        "session_id": "session",
        "spend_logs_metadata": {
            "workflow_id": "workflow",
            "run_id": "run",
            "node_id": "answer",
            "attempt": 1,
            "session_id": "session",
        },
    }
    assert second_context.data["metadata"] == {
        "workflow_id": "workflow",
        "run_id": "run",
        "node_id": "answer",
        "attempt": 2,
        "session_id": "session",
        "spend_logs_metadata": {
            "workflow_id": "workflow",
            "run_id": "run",
            "node_id": "answer",
            "attempt": 2,
            "session_id": "session",
        },
    }
    assert first_context.data["user"] == "end-user"
    assert first_context.data["stream"] is False
    assert first_context.data["messages"] == call.messages
    assert call.messages == [{"role": "user", "content": "question"}]
    assert port.released.is_set()


@pytest.mark.asyncio
async def test_recovered_attempt_uses_persisted_call_id_with_fresh_request_state() -> None:
    port: Final = _Gateway(receipt_ids=())
    call: Final = _call()
    first: Final = await gateway._execute_isolated(call, port)
    recovered: Final = await gateway._execute_isolated(call, port)
    assert first == recovered == ChildCallResult("answer", 7, 3, None, 0.125)
    first_context: Final = port.contexts.get_nowait()
    recovered_context: Final = port.contexts.get_nowait()
    assert first_context.request.headers["x-litellm-call-id"] == recovered_context.request.headers["x-litellm-call-id"]
    assert first_context.request.state is not recovered_context.request.state


@pytest.mark.asyncio
@pytest.mark.parametrize("status,code", [(403, "authorization_failed"), (429, "rate_limit"), (503, "provider_error")])
async def test_policy_denial_prevents_dispatch_and_redacts_failure(status: int, code: str) -> None:
    port: Final = _Gateway(failure=HTTPException(status_code=status, detail="private provider configuration"))
    result: Final = await gateway._execute_isolated(_call(), port)
    assert isinstance(result, ExecutionError)
    assert result.code == code
    assert "private" not in result.message
    assert not port.processed.is_set()
    assert port.released.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["authorize", "process"])
@pytest.mark.parametrize("expire", [False, True])
async def test_cancellation_and_deadline_drain_auth_or_processing(
    monkeypatch: pytest.MonkeyPatch, stage: Literal["authorize", "process"], expire: bool
) -> None:
    port: Final = _Gateway(blocked_at=stage)
    task: Final = asyncio.create_task(gateway._execute_isolated(_call(), port))
    await port.started.wait()
    if expire:
        loop: Final = asyncio.get_running_loop()
        original_clock: Final = loop.time
        monkeypatch.setattr(loop, "time", lambda: original_clock() + 3600)
        result: Final = await task
        assert result == ExecutionError("timeout", "The node deadline expired")
    else:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert port.stopped.is_set()
    assert port.released.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [(), ("first", "second")])
async def test_missing_or_ambiguous_spend_receipt_never_substitutes_provider_id(ids: tuple[str, ...]) -> None:
    result: Final = await gateway._execute_isolated(_call(), _Gateway(receipt_ids=ids))
    assert result == ChildCallResult("answer", 7, 3, None, 0.125)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [False, True])
async def test_structured_result_is_parsed_and_schema_validation_is_required(invalid: bool) -> None:
    error: Final = ExecutionError("output_invalid", "Schema rejected") if invalid else None
    port: Final = _Gateway(content='{"answer":"structured"}', schema_error=error)
    result: Final = await gateway._execute_isolated(_call(structured=True), port)
    assert port.schema_inputs.get_nowait() == {"answer": "structured"}
    assert result == (
        ExecutionError(
            "output_invalid",
            "Schema rejected",
            receipt=ChildCallResult(None, 7, 3, "provider-id_cache_hit123", 0.125),
        )
        if invalid
        else ChildCallResult({"answer": "structured"}, 7, 3, "provider-id_cache_hit123", 0.125)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ['{"value":NaN}', '{"value":Infinity}', "not json"])
@pytest.mark.parametrize("ids", [(), ("provider-id_cache_hit123",), ("first", "second")])
async def test_invalid_json_retains_billing_without_reaching_schema_validator(
    content: str, ids: tuple[str, ...]
) -> None:
    port: Final = _Gateway(content=content, receipt_ids=ids)
    result: Final = await gateway._execute_isolated(_call(structured=True), port)
    assert result == ExecutionError(
        "output_invalid",
        "The model did not return valid JSON",
        receipt=ChildCallResult(None, 7, 3, ids[0] if len(ids) == 1 else None, 0.125),
    )
    assert port.schema_inputs.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content,code,message",
    [
        (None, "invalid_response", "The model returned no assistant content"),
        ("x" * (gateway._MAX_OUTPUT_BYTES + 1), "output_too_large", "The model output exceeded the workflow limit"),
    ],
    ids=["missing-content", "oversized-content"],
)
async def test_unusable_content_retains_billing_without_retaining_content(
    content: str | None, code: str, message: str
) -> None:
    result: Final = await gateway._execute_isolated(_call(), _Gateway(content=content))
    assert result == ExecutionError(
        code, message, receipt=ChildCallResult(None, 7, 3, "provider-id_cache_hit123", 0.125)
    )


@pytest.mark.asyncio
async def test_malformed_chat_choices_retain_valid_reported_usage() -> None:
    port: Final = _Gateway(response_body={"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
    result: Final = await gateway._execute_isolated(_call(), port)
    assert result == ExecutionError(
        "invalid_response",
        "The gateway returned an invalid chat response",
        receipt=ChildCallResult(None, 7, 3, "provider-id_cache_hit123", 0.125),
    )


@pytest.mark.asyncio
async def test_schema_validation_deadline_retains_billing_for_delayed_spend_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port: Final = _Gateway(content='{"answer":"structured"}', blocked_at="validate", receipt_ids=())
    task: Final = asyncio.create_task(gateway._execute_isolated(_call(structured=True), port))
    await port.started.wait()
    loop: Final = asyncio.get_running_loop()
    original_clock: Final = loop.time
    monkeypatch.setattr(loop, "time", lambda: original_clock() + 3600)
    result: Final = await task
    assert result == ExecutionError(
        "timeout", "The node deadline expired", receipt=ChildCallResult(None, 7, 3, None, 0.125)
    )
    assert port.stopped.is_set()
    assert port.released.is_set()


@pytest.mark.asyncio
async def test_expired_deadline_and_unpersisted_key_are_rejected_before_auth() -> None:
    port: Final = _Gateway()
    call: Final = _call()
    assert await gateway._execute_isolated(replace(call, deadline=_NOW), port) == ExecutionError(
        "timeout", "The node deadline expired"
    )
    assert await gateway._execute_isolated(replace(call, key_hash="sk-unpersisted"), port) == ExecutionError(
        "authorization_failed", "A persisted team virtual key is required"
    )
    assert port.contexts.empty()


@pytest.mark.asyncio
async def test_internal_receive_yields_body_once_then_waits_without_signalling_disconnect() -> None:
    context: Final = gateway._internal_request({"model": "workflow-model"}, "call", "session")
    assert await context.request.body() == b'{"model": "workflow-model"}'
    pending: Final = asyncio.create_task(context.request.receive())
    await asyncio.sleep(0)
    assert not pending.done()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending


class _SpendSink(CustomLogger):
    def __init__(self, outcome: str) -> None:
        super().__init__()
        self.outcome: Final = outcome
        self.output_received: Final = asyncio.Event()
        self.failure_received: Final = asyncio.Event()
        self.admitted: Final[asyncio.Queue[UserAPIKeyAuth]] = asyncio.Queue()
        self.rows: Final[asyncio.Queue[tuple[str, Mapping[str, object]]]] = asyncio.Queue()

    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: object, data: Mapping[str, object], call_type: str
    ) -> None:
        self.admitted.put_nowait(user_api_key_dict)

    async def async_post_call_success_hook(
        self, data: Mapping[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> object:
        self.output_received.set()
        if self.outcome in {"cancelled", "timeout"}:
            await asyncio.Event().wait()
        if self.outcome in {"rejected", "failure_timeout"}:
            raise HTTPException(status_code=422, detail="Private guardrail detail")
        assert isinstance(response, ModelResponse)
        return response.model_copy(
            update={
                "choices": [litellm.Choices(message=litellm.Message(content="guarded answer", role="assistant"))],
            }
        )

    async def async_post_call_failure_hook(
        self,
        request_data: Mapping[str, object],
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        self.failure_received.set()
        if self.outcome == "failure_timeout":
            await asyncio.Event().wait()

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.rows.put_nowait(
            (
                "success",
                get_logging_payload(
                    kwargs=dict(kwargs),
                    response_obj=response_obj,
                    start_time=start_time,
                    end_time=end_time,
                ),
            )
        )

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.rows.put_nowait(
            (
                "failure",
                get_logging_payload(
                    kwargs=dict(kwargs),
                    response_obj=response_obj,
                    start_time=start_time,
                    end_time=end_time,
                ),
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "rejected", "cancelled", "timeout", "failure_timeout"])
async def test_full_processor_reserves_child_budget_runs_hooks_and_emits_one_correlated_spend_payload(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    cache: Final = UserApiKeyCache()
    key: Final = UserAPIKeyAuth(
        token=_KEY, team_id="team", user_id="owner", max_budget=100.0, models=["workflow-model"]
    )
    await cache_key_object(hashed_token=_KEY, user_api_key_obj=key, user_api_key_cache=cache, proxy_logging_obj=None)
    await cache.async_set_cache(key="team_id:team", value=LiteLLM_TeamTableCachedObj(team_id="team"))
    await cache.async_set_cache(key="owner", value=LiteLLM_UserTable(user_id="owner"))
    await cache.async_set_cache(
        key=team_membership_auth_cache_key(team_id="team", user_id="owner"),
        value=LiteLLM_TeamMembership(user_id="owner", team_id="team", spend=0),
    )
    await cache.async_set_cache(
        key=team_membership_reservation_cache_key(team_id="team", user_id="owner"),
        value=LiteLLM_TeamMembership(user_id="owner", team_id="team", spend=0),
    )
    await cache.async_set_cache(
        key=end_user_cache_key("end-user"), value=LiteLLM_EndUserTable(user_id="end-user", blocked=False)
    )
    sink: Final = _SpendSink(outcome)
    counters: Final = FakeSpendCounters(redis_default=0.0)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "workflow-model",
                "litellm_params": {
                    "model": "openai/workflow-fixture",
                    "api_key": "unit-only",
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                    "mock_response": ModelResponse(
                        id="provider-fixture",
                        choices=[{"message": {"role": "assistant", "content": "unfiltered answer"}}],
                        usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
                    ),
                },
            }
        ],
        num_retries=0,
    )
    monkeypatch.setattr(litellm, "callbacks", [sink])
    monkeypatch.setattr(litellm, "max_budget", 0.0)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "spend_counter_store", counters.store())
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", ProxyLogging(user_api_key_cache=cache))
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace())
    monkeypatch.setattr(proxy_server, "general_settings", {"cancel_on_disconnect": True})
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "proxy_config", proxy_server.ProxyConfig())
    call: Final = _call()
    task: Final = asyncio.create_task(
        gateway._execute_isolated(
            replace(call, node=call.node.model_copy(update={"max_tokens": 3})), gateway._ProxyGateway()
        )
    )
    result: Final[ChildCallResult | ExecutionError | None]
    if outcome == "cancelled":
        await asyncio.wait_for(sink.output_received.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        result = None
    elif outcome in {"timeout", "failure_timeout"}:
        await asyncio.wait_for(
            (sink.failure_received if outcome == "failure_timeout" else sink.output_received).wait(), timeout=5
        )
        loop: Final = asyncio.get_running_loop()
        original_clock: Final = loop.time
        monkeypatch.setattr(loop, "time", lambda: original_clock() + 3600)
        result = await task
    else:
        result = await task
    assert sink.admitted.qsize() == 1, result
    admitted: Final = await asyncio.wait_for(sink.admitted.get(), timeout=5)
    logged_status, row = await asyncio.wait_for(sink.rows.get(), timeout=5)
    assert admitted.budget_reservation is not None
    assert (admitted.user_id, admitted.team_id, admitted.end_user_id) == ("owner", "team", "end-user")
    assert row["session_id"] == "session" and row["api_key"] == _KEY and row["team_id"] == "team"
    assert row["end_user"] == "end-user" and row["user"] == "owner"
    assert row["litellm_call_id"] == workflow_call_id("run", "answer", 1)
    assert row["spend"] == pytest.approx(7 * 0.001 + 3 * 0.002)
    assert isinstance(row["metadata"], str)
    assert json.loads(row["metadata"])["spend_logs_metadata"] == {
        "workflow_id": "workflow",
        "run_id": "run",
        "node_id": "answer",
        "attempt": 1,
        "session_id": "session",
    }
    assert sink.rows.empty()
    if outcome != "success":
        assert logged_status == "failure"
        if outcome == "cancelled":
            assert result is None
        else:
            assert isinstance(result, ExecutionError)
            assert result.code == ("request_rejected" if outcome == "rejected" else "timeout")
            assert result.receipt == ChildCallResult(None, 7, 3, None, row["spend"])
    else:
        assert logged_status == "success"
        assert isinstance(result, ChildCallResult)
        assert (result.output, result.prompt_tokens, result.completion_tokens) == ("guarded answer", 7, 3)
        assert result.response_cost == pytest.approx(row["spend"])
