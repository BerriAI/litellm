import asyncio
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timedelta, timezone, tzinfo
from types import SimpleNamespace
from typing import Final

import pytest
from fastapi import HTTPException, Request
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.models.workflow_definition import ChatCompletionNode
from litellm.models.workflow_execution import (
    ChildCallResult,
    EngineRun,
    ExecutionError,
    RunInput,
    RunState,
    StepTrace,
    workflow_call_id,
)
from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_TeamTableCachedObj, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import cache_key_object
from litellm.proxy.auth.user_api_key_auth import authorize_internal_virtual_key
from litellm.proxy.common_request_processing import process_workflow_request
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.deps.database import get_database
from litellm.proxy.hooks.parallel_request_limiter_v3 import PROXY_MaxParallelRequestsHandler_v3, get_request_stash
from litellm.proxy.policy_engine.attachment_registry import get_attachment_registry
from litellm.proxy.policy_engine.policy_registry import get_policy_registry
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.proxy.utils import ProxyLogging
from litellm.proxy.workflows import gateway
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.policy_engine import Policy, PolicyAttachment, PolicyGuardrails
from litellm.types.utils import ModelResponse

_NOW: Final = datetime(2026, 11, 6, tzinfo=timezone.utc)
_KEY: Final = "a" * 64
_FIELDS: Final = TypeAdapter(dict[str, object])
_MESSAGES: Final = [{"role": "user", "content": "private input"}]


class _Clock(datetime):
    @classmethod
    def now(cls, tz: tzinfo | None = None) -> datetime:
        return _NOW


def _response(content: str) -> ModelResponse:
    return ModelResponse(
        model="workflow-model",
        choices=[{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    )


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "app": proxy_server.app,
            "http_version": "1.1",
            "method": "POST",
            "scheme": "https",
            "path": "/v1/chat/completions",
            "query_string": b"",
            "server": ("gateway.example", 443),
            "client": ("192.0.2.10", 1234),
            "headers": [
                (b"x-litellm-trace-id", b"trace-root"),
                (b"x-litellm-session-id", b"caller-session"),
                (b"x-forwarded-for", b"198.51.100.8"),
                (b"content-type", b"application/json"),
                (b"authorization", b"Bearer caller-secret"),
            ],
            "state": {"parent": "untouched", "database": None},
        }
    )


def _run() -> EngineRun:
    return EngineRun(
        run_id="run",
        session_id="trace-root",
        workflow_id="review",
        snapshot_id="snapshot",
        team_id="team",
        created_by=_KEY,
        status=RunState.running,
        created_at=_NOW,
        updated_at=_NOW,
        request_digest="digest",
        input=RunInput(messages=tuple(_MESSAGES), deadline=_NOW + timedelta(minutes=2), draft=False),
    )


def _node() -> ChatCompletionNode:
    return ChatCompletionNode.model_validate(
        {
            "id": "answer",
            "type": "chat_completion",
            "model": "workflow-model",
            "messages": {"from": "$input", "output": "messages"},
            "timeout_seconds": 60,
            "max_tokens": 3,
            "retry": {"max_attempts": 1, "backoff_seconds": 0},
        }
    )


def _step(attempt: int = 1) -> StepTrace:
    return StepTrace(
        step_id=f"step-{attempt}",
        run_id="run",
        node_id="answer",
        state=RunState.running,
        input=_MESSAGES,
        attempts=attempt,
        deadline=_NOW + timedelta(minutes=1),
        started_at=_NOW,
        call_id=workflow_call_id("run", "answer", attempt),
    )


class _Observer(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.outcome = "success"
        self.failure_status = 500
        self.entered: Final = asyncio.Event()
        self.failure_entered: Final = asyncio.Event()
        self.completed: Final[asyncio.Queue[str]] = asyncio.Queue()
        self.admissions: Final[asyncio.Queue[UserAPIKeyAuth]] = asyncio.Queue()
        self.rows: Final[asyncio.Queue[Mapping[str, object]]] = asyncio.Queue()
        self.requests: Final[asyncio.Queue[Mapping[str, object]]] = asyncio.Queue()

    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: object, data: Mapping[str, object], call_type: str
    ) -> Mapping[str, object] | None:
        self.admissions.put_nowait(user_api_key_dict)
        self.requests.put_nowait(_FIELDS.validate_python(data["proxy_server_request"]))
        if self.outcome == "pre":
            raise HTTPException(self.failure_status, "private processor failure")

    async def async_post_call_success_hook(
        self, data: Mapping[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> object:
        self.completed.put_nowait(str(data["model"]))
        if data["model"] == "workflow-model":
            self.entered.set()
            if self.outcome == "post":
                raise HTTPException(self.failure_status, "private post-call failure")
            if self.outcome in {"cancel", "timeout"}:
                await asyncio.Event().wait()
            if self.outcome in {"reject", "failure_timeout"}:
                raise HTTPException(422, "private guardrail detail")
        return response

    async def async_post_call_failure_hook(
        self,
        request_data: Mapping[str, object],
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        self.failure_entered.set()
        if self.outcome == "failure_timeout":
            await asyncio.Event().wait()

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.rows.put_nowait(
            get_logging_payload(
                kwargs=dict(kwargs),
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
            )
        )

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await self.async_log_success_event(kwargs, response_obj, start_time, end_time)


class _WorkflowPolicy(CustomGuardrail):
    def __init__(self, reject: bool = False) -> None:
        super().__init__(
            guardrail_name="workflow-policy",
            event_hook=[GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call],
        )
        self.reject: Final = reject

    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: object, data: Mapping[str, object], call_type: str
    ) -> Mapping[str, object] | None:
        return {**data, "messages": [{"role": "user", "content": "guarded input"}]}

    async def async_post_call_success_hook(
        self, data: Mapping[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> object:
        if self.reject:
            raise HTTPException(422, "private Python output")
        assert isinstance(response, ModelResponse)
        return response.model_copy(update={"choices": _response("guarded Python output").choices})


@pytest.fixture
async def proxy(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[_Observer, UserAPIKeyAuth]]:
    cache: Final = UserApiKeyCache()
    key: Final = UserAPIKeyAuth(
        token=_KEY,
        team_id="team",
        max_budget=100.0,
        max_parallel_requests=1,
        models=["workflow-model", "workflow/review"],
    )
    await cache_key_object(hashed_token=_KEY, user_api_key_obj=key, user_api_key_cache=cache, proxy_logging_obj=None)
    await cache.async_set_cache(
        key="team_id:team",
        value=LiteLLM_TeamTableCachedObj(
            team_id="team",
            metadata={"require_trace_id": True},
        ),
    )
    observer: Final = _Observer()
    logging: Final = ProxyLogging(user_api_key_cache=cache)
    limiter: Final = PROXY_MaxParallelRequestsHandler_v3(logging.internal_usage_cache)
    counters: Final = DualCache()
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "workflow-model",
                "litellm_params": {
                    "model": "openai/workflow-fixture",
                    "api_key": "unit-only",
                    "mock_response": _response("child answer"),
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                },
            }
        ],
        num_retries=0,
    )
    monkeypatch.setattr(gateway, "datetime", _Clock)
    monkeypatch.setattr(litellm, "callbacks", [observer, limiter])
    monkeypatch.setattr(litellm, "max_budget", 0.0)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "spend_counter_cache", counters)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", logging)
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace())
    monkeypatch.setattr(proxy_server, "general_settings", {"allowed_ips": ["192.0.2.10"], "cancel_on_disconnect": True})
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "proxy_config", proxy_server.ProxyConfig())
    yield observer, key
    get_policy_registry().clear()
    get_attachment_registry().clear()
    router.reset()


@pytest.mark.asyncio
@pytest.mark.parametrize("forwarded", [False, True])
async def test_child_preserves_trusted_context_and_rechecks_model_access(
    proxy: tuple[_Observer, UserAPIKeyAuth], forwarded: bool
) -> None:
    observer, key = proxy
    proxy_server.general_settings.update(
        {
            "use_x_forwarded_for": forwarded,
            "allowed_ips": ["198.51.100.8" if forwarded else "192.0.2.10"],
        }
    )
    request: Final = _request()
    child: Final = gateway._child_request(request, {}, workflow_call_id("run", "answer", 1), "trace-root")
    assert await get_database(child) is None
    assert "authorization" not in child.headers
    result: Final = await gateway.execute_chat_node(request, _run(), _node(), _step())
    assert isinstance(result, ChildCallResult), result
    row: Final = await asyncio.wait_for(observer.rows.get(), 5)
    seen: Final = observer.requests.get_nowait()
    assert _FIELDS.validate_python(seen["headers"])["x-litellm-trace-id"] == "trace-root"
    assert (row["session_id"], row["litellm_call_id"], row["api_key"]) == ("trace-root", _step().call_id, _KEY)
    assert result.request_id is None
    assert result.response_cost == pytest.approx(7 * 0.001 + 3 * 0.002)
    assert request.scope["state"] == {"parent": "untouched", "database": None}
    assert "caller-secret" not in str(seen)
    await cache_key_object(
        hashed_token=_KEY,
        user_api_key_obj=key.model_copy(update={"models": ["another-model"]}),
        user_api_key_cache=proxy_server.user_api_key_cache,
        proxy_logging_obj=None,
    )
    denied: Final = await gateway.execute_chat_node(request, _run(), _node(), _step(2))
    assert isinstance(denied, ExecutionError), denied
    assert denied.receipt is None and observer.requests.empty()
    assert "private" not in denied.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "status", "code"),
    [
        ("auth", 429, "admission_rate_limit"),
        ("auth", 500, "provider_error"),
        ("pre", 429, "rate_limit"),
        ("pre", 500, "provider_error"),
        ("post", 429, "rate_limit"),
        ("post", 500, "provider_error"),
    ],
)
async def test_only_auth_rate_limits_are_safe_to_retry(
    proxy: tuple[_Observer, UserAPIKeyAuth], phase: str, status: int, code: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    observer, _ = proxy
    observer.outcome = phase
    observer.failure_status = status

    async def reject_auth(
        key_hash: str,
        request: Request,
        request_data: Mapping[str, object],
        *,
        route: str | None = None,
        expected_team_id: str | None = None,
    ) -> UserAPIKeyAuth:
        raise HTTPException(status, "private authentication failure")

    if phase == "auth":
        monkeypatch.setattr(gateway, "authorize_internal_virtual_key", reject_auth)
    result: Final = await gateway.execute_chat_node(_request(), _run(), _node(), _step())
    assert isinstance(result, ExecutionError), result
    assert result.code == code and "private" not in result.message
    assert observer.admissions.qsize() == (0 if phase == "auth" else 1)
    if phase == "post":
        assert result.receipt is not None
        assert result.receipt.response_cost == pytest.approx(7 * 0.001 + 3 * 0.002)
        await asyncio.wait_for(observer.rows.get(), 5)
    else:
        assert result.receipt is None
    if phase == "auth":
        assert observer.requests.empty() and observer.rows.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["reject", "cancel", "timeout", "failure_timeout"])
async def test_billed_failure_and_cancellation_release_child_slot(
    proxy: tuple[_Observer, UserAPIKeyAuth], outcome: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    observer, _ = proxy
    observer.outcome = outcome
    task: Final = asyncio.create_task(gateway.execute_chat_node(_request(), _run(), _node(), _step()))
    await asyncio.wait_for((observer.failure_entered if outcome == "failure_timeout" else observer.entered).wait(), 5)
    if outcome == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        if outcome in {"timeout", "failure_timeout"}:
            loop: Final = asyncio.get_running_loop()
            original_time: Final = loop.time
            monkeypatch.setattr(loop, "time", lambda: original_time() + 120)
        failure: Final = await task
        assert isinstance(failure, ExecutionError) and failure.receipt is not None, failure
        assert failure.receipt.response_cost == pytest.approx(7 * 0.001 + 3 * 0.002)
        assert failure.message == (
            "The node deadline expired"
            if outcome in {"timeout", "failure_timeout"}
            else "The gateway rejected this node"
        )
    await asyncio.wait_for(observer.rows.get(), 5)
    admitted: Final = observer.admissions.get_nowait()
    if outcome in {"cancel", "timeout", "failure_timeout"}:
        assert admitted.budget_reservation is not None and admitted.budget_reservation["finalized"] is True
    observer.outcome = "success"
    retried: Final = await gateway.execute_chat_node(_request(), _run(), _node(), _step(2))
    assert isinstance(retried, ChildCallResult), retried


@pytest.mark.asyncio
@pytest.mark.parametrize("detached", [False, True])
@pytest.mark.parametrize("reject", [False, True])
async def test_workflow_policy_guards_modified_input_and_final_python_output_without_wrapper_charge(
    proxy: tuple[_Observer, UserAPIKeyAuth],
    detached: bool,
    reject: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer, key = proxy
    policy: Final = _WorkflowPolicy(reject)
    monkeypatch.setattr(litellm, "callbacks", [*litellm.callbacks, policy])
    get_policy_registry().add_policy("workflow-dlp", Policy(guardrails=PolicyGuardrails(add=["workflow-policy"])))
    get_attachment_registry().add_attachment(PolicyAttachment(policy="workflow-dlp", models=["workflow/review"]))
    parent_request: Final = _request()
    parent_auth: Final = await authorize_internal_virtual_key(
        key_hash=_KEY,
        request=parent_request,
        request_data={"model": "workflow-model", "messages": _MESSAGES, "max_tokens": 3},
        expected_team_id="team",
    )
    assert parent_auth.budget_reservation is not None
    inputs: Final[asyncio.Queue[JsonValue]] = asyncio.Queue()

    async def execute(messages: JsonValue) -> ModelResponse | ExecutionError:
        inputs.put_nowait(messages)
        assert get_request_stash() is None
        assert parent_auth.budget_reservation["finalized"] is True
        assert proxy_server.spend_counter_cache.in_memory_cache.get_cache(key=f"spend:key:{_KEY}") == 0
        child: Final = await gateway.execute_chat_node(
            _request(),
            _run(),
            _node(),
            _step().model_copy(update={"input": messages}),
        )
        assert isinstance(child, ChildCallResult), child
        return _response("private Python output")

    invocation: Final = process_workflow_request(
        request=parent_request,
        workflow_model="workflow/review",
        caller=parent_auth,
        messages=_MESSAGES,
        execute=execute,
    )
    result: Final = await (asyncio.create_task(invocation) if detached else invocation)
    assert inputs.get_nowait() == [{"role": "user", "content": "guarded input"}]
    if reject:
        assert isinstance(result, ExecutionError), result
        assert "private" not in result.message
    else:
        assert isinstance(result, ModelResponse), result
        assert result.choices[0].message.content == "guarded Python output"
    row: Final = await asyncio.wait_for(observer.rows.get(), 5)
    assert row["litellm_call_id"] == _step().call_id
    assert observer.rows.empty()
    assert observer.admissions.qsize() == 1
    assert observer.completed.get_nowait() == "workflow-model" and observer.completed.empty()
