import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from typing import Final, cast
from unittest.mock import patch

import httpx
import pytest
import respx

import litellm
from litellm.constants import (
    CALLER_SESSION_ID_METADATA_KEY,
    SESSION_ID_GENERATED_METADATA_KEY,
    SESSION_ID_INHERITED_METADATA_KEY,
    SESSION_ID_OMITTED_METADATA_KEY,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.langfuse.langfuse import LangFuseLogger, langfuse_trace_id
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.anthropic.messages.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
    request_safeguards_evaluator,
)
from litellm.llms.anthropic.messages.caller_model_checks import CallerModelVerdict
from litellm.llms.anthropic.messages.safeguards import ToolUseUnderReview, build_safeguards_evaluator
from litellm.llms.anthropic.messages.utils import proxy_sub_call_forwarded_headers
from litellm.proxy._types import LiteLLM_TeamTableCachedObj, SpendLogsPayload, UserAPIKeyAuth
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.router import Router
from litellm.types.router import ForwardedClientHeaders, ModelGroupSettings
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    ChatCompletionMessageToolCall,
    Choices,
    Delta,
    Function,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)
from tests.unit.llms.anthropic.messages.conftest import ProxyAuthState

SAFEGUARDS: Final = [{"type": "dangerous_tool_use", "classifier_context": {"permission_mode": "auto"}}]
BASH_TOOL: Final = {"name": "Bash", "description": "run", "input_schema": {"type": "object", "properties": {}}}
MESSAGES: Final = [{"role": "user", "content": "install the deps"}]
LITELLM_METADATA: Final = {"user_api_key": "hashed-key", "user_api_key_team_id": "team-1", "headers": {}}


def _tool_call_response() -> ModelResponse:
    tool_call: Final = ChatCompletionMessageToolCall(
        id="call_main", type="function", function=Function(name="Bash", arguments=json.dumps({"command": "npm ci"}))
    )
    return ModelResponse(
        id="chatcmpl-main",
        model="kimi",
        choices=[Choices(index=0, finish_reason="tool_calls", message=Message(content=None, tool_calls=[tool_call]))],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _tool_call_stream() -> Sequence[ModelResponseStream]:
    def chunk(delta: Delta, finish_reason: str | None = None) -> ModelResponseStream:
        return ModelResponseStream(
            id="chatcmpl-s", model="kimi", choices=[StreamingChoices(index=0, finish_reason=finish_reason, delta=delta)]
        )

    return (
        chunk(
            Delta(
                content=None,
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id="call_stream",
                        type="function",
                        index=0,
                        function=Function(name="Bash", arguments='{"command": "npm'),
                    )
                ],
            )
        ),
        chunk(
            Delta(
                content=None,
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id=None, type="function", index=0, function=Function(name=None, arguments=' ci"}')
                    )
                ],
            )
        ),
        chunk(Delta(content=None), finish_reason="tool_calls"),
        ModelResponseStream(
            id="chatcmpl-s",
            model="kimi",
            choices=[],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        ),
    )


class _AsyncStream:
    def __init__(self, items: Sequence[ModelResponseStream]) -> None:
        self._it = iter(items)

    def __aiter__(self) -> "_AsyncStream":
        return self

    async def __anext__(self) -> ModelResponseStream:
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class _Backend:
    def __init__(self, stream: bool) -> None:
        self._stream = stream
        self.calls: list[dict[str, object]] = []

    async def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return _AsyncStream(_tool_call_stream()) if self._stream else _tool_call_response()


class _ClassifierRouter:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[dict[str, object]] = []

    async def acompletion(self, **kwargs: object) -> ModelResponse:
        self.calls.append(kwargs)
        return ModelResponse(choices=[Choices(index=0, finish_reason="stop", message=Message(content=self._reply))])


class _ClassifierCall:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def __call__(self, *, messages: Sequence[Mapping[str, object]], **kwargs: object) -> ModelResponse:
        self.calls.append({"messages": messages, **kwargs})
        reply: Final = '{"verdicts": {"call_1": {"flagged": false}}}'
        return ModelResponse(choices=[Choices(index=0, finish_reason="stop", message=Message(content=reply))])


async def _allowed(model: str) -> CallerModelVerdict:
    return "allowed"


async def _call(
    backend: _Backend, router: _ClassifierRouter, stream: bool, settings: Mapping[str, object], **extra: object
) -> object:
    with (
        patch("litellm.acompletion", new=backend),
        patch("litellm.proxy.proxy_server.general_settings", dict(settings)),
    ):
        return await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
            max_tokens=256,
            messages=list(MESSAGES),
            model="fireworks_ai/kimi",
            tools=[dict(BASH_TOOL)],
            stream=stream,
            litellm_router=router,
            litellm_metadata=dict(LITELLM_METADATA),
            **extra,
        )


def _events(sse: bytes) -> Sequence[Mapping[str, object]]:
    return tuple(json.loads(line[len("data: ") :]) for line in sse.decode().splitlines() if line.startswith("data: "))


async def _drain_stream(stream: object) -> None:
    assert hasattr(stream, "__aiter__")
    _ = [chunk async for chunk in stream]  # pyright: ignore[reportGeneralTypeIssues]  # the handler's union narrows to an async iterator once stream is on


def _available_tool_uses(results: object) -> Mapping[str, object]:
    assert isinstance(results, (list, tuple)) and len(results) == 1
    assert results[0]["type"] == "dangerous_tool_use"
    assert results[0]["status"]["type"] == "available"
    return results[0]["status"]["tool_uses"]


@pytest.mark.asyncio
async def test_non_stream_response_carries_a_verdict_for_the_tool_use_and_bills_the_classifier_to_the_caller():
    backend: Final = _Backend(stream=False)
    router: Final = _ClassifierRouter('{"verdicts": {"call_main": {"flagged": false, "explanation": "install"}}}')
    response: Final = await _call(
        backend, router, False, {"safeguards_classifier_model": "classifier"}, safeguards=SAFEGUARDS
    )
    assert isinstance(response, dict)
    assert _available_tool_uses(response["safeguard_results"]) == {
        "call_main": {"type": "evaluated", "outcome": "not_flagged"}
    }
    assert "safeguards" not in backend.calls[0]
    classifier_call: Final = router.calls[0]
    assert classifier_call["model"] == "classifier"
    assert classifier_call["litellm_metadata"] == {
        "user_api_key": "hashed-key",
        "user_api_key_team_id": "team-1",
        "headers": {},
        SESSION_ID_INHERITED_METADATA_KEY: True,
    }
    assert json.dumps({"command": "npm ci"}) in str(classifier_call["messages"][1]["content"])


def _forwarded_stamp(headers: Mapping[str, str], *, model_group_scoped: bool = False) -> Mapping[str, object]:
    stamp: Final = ForwardedClientHeaders.of(headers, model_group_scoped=model_group_scoped).model_dump()
    return {"proxy_server_request": {"forwarded_client_headers": stamp}}


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
async def test_classifier_call_carries_the_caller_headers_the_proxy_forwards_to_the_parent_model(stream: bool):
    backend: Final = _Backend(stream=stream)
    router: Final = _ClassifierRouter('{"verdicts": {"call_main": {"flagged": false}}}')
    caller_headers: Final = {"X-User-Id": "user-42", "X-Tenant": "acme"}
    response: Final = await _call(
        backend,
        router,
        stream,
        {"safeguards_classifier_model": "classifier"},
        safeguards=SAFEGUARDS,
        headers=caller_headers,
        **_forwarded_stamp(caller_headers),
    )
    if stream:
        await _drain_stream(response)
    assert router.calls[0]["headers"] == caller_headers


@pytest.mark.asyncio
async def test_classifier_call_drops_a_header_the_proxy_did_not_forward_to_the_parent_model():
    backend: Final = _Backend(stream=False)
    router: Final = _ClassifierRouter('{"verdicts": {"call_main": {"flagged": false}}}')
    caller_headers: Final = {"X-User-Id": "user-42"}
    response: Final = await _call(
        backend,
        router,
        False,
        {"safeguards_classifier_model": "classifier"},
        safeguards=SAFEGUARDS,
        headers={**caller_headers, "X-Gateway-Token": "secret"},
        **_forwarded_stamp(caller_headers),
    )
    assert response
    assert router.calls[0]["headers"] == caller_headers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope, expected",
    [(["classifier"], {"X-User-Id": "user-42"}), (["parent-group"], None)],
    ids=["scope-covers-the-classifier", "scope-covers-only-the-parent"],
)
async def test_a_model_group_scoped_forwarding_reaches_the_classifier_only_when_its_scope_covers_it(
    monkeypatch: pytest.MonkeyPatch, scope: Sequence[str], expected: Mapping[str, str] | None
) -> None:
    router: Final = Router(model_list=[_mock_deployment("classifier", '{"verdicts": {"call_1": {"flagged": false}}}')])
    recorder: Final = _ClassifierCall()
    monkeypatch.setattr(
        litellm, "model_group_settings", ModelGroupSettings(forward_client_headers_to_llm_api=list(scope))
    )
    with patch("litellm.proxy.proxy_server.general_settings", {"safeguards_classifier_model": "classifier"}):
        evaluator: Final = build_safeguards_evaluator(
            safeguards=SAFEGUARDS,
            messages=MESSAGES,
            litellm_metadata={"user_api_key_auth": UserAPIKeyAuth(api_key="sk-test", team_id="team-1")},
            user_api_key_auth=UserAPIKeyAuth(api_key="sk-test", team_id="team-1"),
            llm_router=router,
            logging_kwargs={},
            forwarded_headers=proxy_sub_call_forwarded_headers(
                {
                    **_forwarded_stamp({"X-User-Id": "user-42"}, model_group_scoped=True),
                    "headers": {"X-User-Id": "user-42"},
                }
            ),
        )
    assert evaluator is not None
    recorded: Final = replace(evaluator, acompletion=recorder, caller_model_verdict=_allowed)
    await recorded.evaluate((ToolUseUnderReview(id="call_1", name="Bash", input={"command": "ls"}),))
    assert recorder.calls[0].get("headers") == expected
    assert (recorder.calls[0].get("proxy_server_request") is not None) is (expected is not None)


@pytest.mark.asyncio
async def test_stream_carries_the_verdict_inside_the_final_message_delta():
    backend: Final = _Backend(stream=True)
    router: Final = _ClassifierRouter('{"verdicts": {"call_stream": {"flagged": true, "explanation": "unrequested"}}}')
    stream: Final = await _call(
        backend, router, True, {"safeguards_classifier_model": "classifier"}, safeguards=SAFEGUARDS
    )
    events: Final = _events(b"".join([chunk async for chunk in stream]))
    message_deltas: Final = [event for event in events if event["type"] == "message_delta"]
    assert len(message_deltas) == 1
    assert message_deltas[0]["delta"]["stop_reason"] == "tool_use"
    assert _available_tool_uses(message_deltas[0]["delta"]["safeguard_results"]) == {
        "call_stream": {"type": "evaluated", "outcome": "flagged", "explanation": "unrequested"}
    }
    assert message_deltas[0]["usage"]["output_tokens"] == 5
    assert all("safeguard_results" not in event for event in events if event["type"] != "message_delta")
    reviewed: Final = str(router.calls[0]["messages"][1]["content"])
    assert json.dumps([{"id": "call_stream", "name": "Bash", "input": {"command": "npm ci"}}]) in reviewed


@pytest.mark.asyncio
async def test_classifier_reads_the_turns_before_a_compaction_block_the_agent_model_no_longer_gets():
    backend: Final = _Backend(stream=False)
    router: Final = _ClassifierRouter('{"verdicts": {"call_main": {"flagged": false}}}')
    with (
        patch("litellm.acompletion", new=backend),
        patch("litellm.proxy.proxy_server.general_settings", {"safeguards_classifier_model": "classifier"}),
    ):
        await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
            max_tokens=256,
            messages=[
                {"role": "user", "content": "install the deps"},
                {"role": "assistant", "content": [{"type": "compaction", "content": "Deps were requested."}]},
                {"role": "user", "content": "go on"},
            ],
            model="fireworks_ai/kimi",
            tools=[dict(BASH_TOOL)],
            stream=False,
            litellm_router=router,
            litellm_metadata=dict(LITELLM_METADATA),
            safeguards=SAFEGUARDS,
        )
    assert "install the deps" not in json.dumps(backend.calls[0]["messages"])
    classifier_prompt: Final = str(router.calls[0]["messages"][1]["content"])
    assert '{"user": "install the deps"}' in classifier_prompt
    assert '{"conversation_summary": "Deps were requested."}' in classifier_prompt
    assert '{"user": "go on"}' in classifier_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_no_classifier_setting_means_no_results_and_no_classifier_call(stream: bool):
    backend: Final = _Backend(stream=stream)
    router: Final = _ClassifierRouter("{}")
    response: Final = await _call(backend, router, stream, {}, safeguards=SAFEGUARDS)
    if stream:
        events: Final = _events(b"".join([chunk async for chunk in response]))
        assert all("safeguard_results" not in event.get("delta", {}) for event in events)
    else:
        assert isinstance(response, dict) and "safeguard_results" not in response
    assert router.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_request_without_safeguards_gets_no_results_even_with_a_classifier_configured(stream: bool):
    backend: Final = _Backend(stream=stream)
    router: Final = _ClassifierRouter("{}")
    response: Final = await _call(backend, router, stream, {"safeguards_classifier_model": "classifier"})
    if stream:
        events: Final = _events(b"".join([chunk async for chunk in response]))
        assert all("safeguard_results" not in event.get("delta", {}) for event in events)
    else:
        assert isinstance(response, dict) and "safeguard_results" not in response
    assert router.calls == []


class _ClassifierLogs(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.logged: Final[list[object]] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.logged.append(kwargs["standard_logging_object"])


async def _drain_logging_worker() -> None:
    for _ in range(10):
        await asyncio.sleep(0)
        await GLOBAL_LOGGING_WORKER.flush()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parent_kwargs, headers, redacted",
    [
        ({}, {}, False),
        ({"turn_off_message_logging": True}, {}, True),
        ({}, {"x-litellm-enable-message-redaction": "true"}, True),
    ],
    ids=["parent-logs-messages", "parent-turns-message-logging-off", "parent-enables-redaction-by-header"],
)
async def test_the_classifier_call_logs_under_the_parent_requests_message_redaction(
    monkeypatch: pytest.MonkeyPatch, parent_kwargs: dict[str, object], headers: dict[str, str], redacted: bool
) -> None:
    await _drain_logging_worker()
    logs: Final = _ClassifierLogs()
    monkeypatch.setattr(litellm, "callbacks", [logs])
    monkeypatch.setattr(litellm, "_async_success_callback", [logs])
    router: Final = Router(
        model_list=[
            {
                "model_name": "classifier",
                "litellm_params": {
                    "model": "openai/gpt-5.4-mini",
                    "api_key": "sk-fake",
                    "mock_response": '{"verdicts": {"call_ls": {"flagged": false}}}',
                },
            }
        ]
    )
    parent_logging: Final = Logging(
        model="fireworks_ai/kimi",
        messages=list(MESSAGES),
        stream=False,
        call_type="anthropic_messages",
        start_time=datetime.now(),
        litellm_call_id="parent-call",
        function_id="parent",
        kwargs=parent_kwargs,
    )
    with patch("litellm.proxy.proxy_server.general_settings", {"safeguards_classifier_model": "classifier"}):
        evaluator: Final = request_safeguards_evaluator(
            {
                "safeguards": SAFEGUARDS,
                "litellm_metadata": {**LITELLM_METADATA, "headers": headers},
                "litellm_router": router,
                "litellm_logging_obj": parent_logging,
            },
            MESSAGES,
        )
    assert evaluator is not None

    await evaluator.evaluate((ToolUseUnderReview(id="call_ls", name="Bash", input={"command": "ls"}),))
    await _drain_logging_worker()

    assert len(logs.logged) == 1
    assert ("install the deps" not in json.dumps(logs.logged[0]["messages"])) is redacted


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "disable_header, logged",
    [("classifier_logs", False), ("Langfuse, CLASSIFIER_LOGS", False), ("langfuse", True)],
    ids=["header-disables-the-callback", "header-lists-it-among-others", "header-disables-another-callback"],
)
async def test_the_classifier_call_skips_a_callback_the_parents_disable_callbacks_header_turned_off(
    monkeypatch: pytest.MonkeyPatch, disable_header: str, logged: bool
) -> None:
    from litellm.litellm_core_utils.custom_logger_registry import CustomLoggerRegistry

    await _drain_logging_worker()
    logs: Final = _ClassifierLogs()
    monkeypatch.setattr(litellm, "callbacks", [logs])
    monkeypatch.setattr(litellm, "_async_success_callback", [logs])
    router: Final = Router(
        model_list=[_mock_deployment("classifier", '{"verdicts": {"call_ls": {"flagged": false}}}')], num_retries=0
    )
    parent_logging: Final = Logging(
        model="fireworks_ai/kimi",
        messages=list(MESSAGES),
        stream=False,
        call_type="anthropic_messages",
        start_time=datetime.now(),
        litellm_call_id="parent-call",
        function_id="parent",
    )
    with (
        patch.dict(CustomLoggerRegistry.CALLBACK_CLASS_STR_TO_CLASS_TYPE, {"classifier_logs": _ClassifierLogs}),
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch("litellm.proxy.proxy_server.general_settings", {"safeguards_classifier_model": "classifier"}),
    ):
        evaluator: Final = request_safeguards_evaluator(
            {
                "safeguards": SAFEGUARDS,
                "litellm_metadata": dict(LITELLM_METADATA),
                "litellm_router": router,
                "litellm_logging_obj": parent_logging,
                "proxy_server_request": {"headers": {"x-litellm-disable-callbacks": disable_header}},
            },
            MESSAGES,
        )
        assert evaluator is not None
        await evaluator.evaluate((ToolUseUnderReview(id="call_ls", name="Bash", input={"command": "ls"}),))
        await _drain_logging_worker()

    assert (len(logs.logged) == 1) is logged


def _mock_deployment(model_name: str, reply: str) -> dict[str, object]:
    return {
        "model_name": model_name,
        "litellm_params": {"model": "openai/gpt-5.4-mini", "api_key": "sk-fake", "mock_response": reply},
    }


@pytest.mark.asyncio
async def test_a_subagent_turns_classifier_call_stays_on_the_classifier_model_when_its_session_is_bound_to_a_router():
    router: Final = Router(
        model_list=[
            _mock_deployment("classifier", '{"verdicts": {"call_ls": {"flagged": false}}}'),
            _mock_deployment("cheap-model", "a reply from the session's routed model"),
            {
                "model_name": "smart-router",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_default_model": "cheap-model",
                    "complexity_router_config": {
                        "tiers": {tier: "cheap-model" for tier in ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING")}
                    },
                },
            },
        ],
        num_retries=0,
    )
    session_headers: Final = {"x-claude-code-session-id": "session-1234"}
    proxy_session_ids: Final = {"litellm_session_id": "session-1234", "litellm_trace_id": "session-1234"}
    await router.acompletion(
        model="smart-router",
        messages=list(MESSAGES),
        metadata={"user_api_key_hash": "key-hash-a"},
        proxy_server_request={"headers": session_headers},
        **proxy_session_ids,
    )
    subagent_headers: Final = {**session_headers, "x-claude-code-agent-id": "agent-1234"}
    with patch("litellm.proxy.proxy_server.general_settings", {"safeguards_classifier_model": "classifier"}):
        evaluator: Final = request_safeguards_evaluator(
            {
                "safeguards": SAFEGUARDS,
                "litellm_metadata": {
                    **LITELLM_METADATA,
                    "user_api_key_hash": "key-hash-a",
                    "headers": subagent_headers,
                },
                "litellm_router": router,
                "proxy_server_request": {"headers": subagent_headers},
                **proxy_session_ids,
            },
            MESSAGES,
        )
    assert evaluator is not None

    results: Final = await evaluator.evaluate((ToolUseUnderReview(id="call_ls", name="Bash", input={"command": "ls"}),))

    assert _available_tool_uses(results) == {"call_ls": {"type": "evaluated", "outcome": "not_flagged"}}


@pytest.mark.asyncio
async def test_a_trace_id_sent_in_the_body_carries_the_classifier_and_summary_calls_past_the_teams_trace_requirement(
    proxy_auth_state: ProxyAuthState,
) -> None:
    proxy_auth_state.seed(
        LiteLLM_TeamTableCachedObj(
            team_id="team-1", models=["classifier", "summarizer"], metadata={"require_trace_id": True}
        )
    )
    router: Final = Router(
        model_list=[
            _mock_deployment("classifier", '{"verdicts": {"call_main": {"flagged": false}}}'),
            _mock_deployment("summarizer", "<summary>The user asked to install the deps.</summary>"),
        ],
        num_retries=0,
    )
    backend: Final = _Backend(stream=False)
    router_deployment_call: Final = litellm.acompletion

    async def agent_model_or_router_deployment(**kwargs: object) -> object:
        if kwargs.get("model") == "fireworks_ai/kimi":
            return await backend(**kwargs)
        return await router_deployment_call(**kwargs)

    settings: Final = {"safeguards_classifier_model": "classifier", "context_management_summary_model": "summarizer"}
    with (
        patch("litellm.acompletion", new=agent_model_or_router_deployment),
        patch("litellm.proxy.proxy_server.general_settings", settings),
        patch("litellm.token_counter", return_value=200_000),
    ):
        response: Final = await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
            max_tokens=256,
            messages=list(MESSAGES),
            model="fireworks_ai/kimi",
            tools=[dict(BASH_TOOL)],
            stream=False,
            litellm_router=router,
            litellm_metadata={
                **LITELLM_METADATA,
                "user_api_key_auth": UserAPIKeyAuth(api_key="sk-team", team_id="team-1"),
            },
            safeguards=SAFEGUARDS,
            context_management={"edits": [{"type": "compact_20260112"}]},
            litellm_trace_id="trace-from-the-body",
        )

    assert isinstance(response, dict)
    assert _available_tool_uses(response["safeguard_results"]) == {
        "call_main": {"type": "evaluated", "outcome": "not_flagged"}
    }
    assert "The user asked to install the deps." in json.dumps(backend.calls[0]["messages"])


class _SpendRows(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.rows: Final[list[SpendLogsPayload]] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.rows.append(get_logging_payload(kwargs, response_obj, start_time, end_time))


def _classifier_and_summarizer_router(classifier_params: Mapping[str, object] | None = None) -> Router:
    classifier: Final = _mock_deployment("classifier", '{"verdicts": {"call_main": {"flagged": false}}}')
    return Router(
        model_list=[
            {**classifier, "litellm_params": classifier_params} if classifier_params else classifier,
            _mock_deployment("summarizer", "<summary>The user asked to install the deps.</summary>"),
        ],
        num_retries=0,
    )


async def _send_parent_request(
    router: Router,
    general_settings: Mapping[str, object],
    litellm_metadata: Mapping[str, object],
    parent_ids: Mapping[str, str],
    request_headers: Mapping[str, str] | None = None,
    anthropic_metadata: Mapping[str, str] | None = None,
) -> None:
    parent_logging, _ = litellm.utils.function_setup(
        "anthropic_messages",
        litellm.utils.Rules(),
        datetime.now(),
        model="fireworks_ai/kimi",
        messages=list(MESSAGES),
        stream=False,
        litellm_call_id="parent-call",
        litellm_trace_id=parent_ids.get("litellm_trace_id", "trace-the-parent-minted"),
        litellm_metadata=dict(litellm_metadata),
        **({} if anthropic_metadata is None else {"metadata": dict(anthropic_metadata)}),
    )
    backend: Final = _Backend(stream=False)
    router_deployment_call: Final = litellm.acompletion

    async def agent_model_or_router_deployment(**kwargs: object) -> object:
        if kwargs.get("model") == "fireworks_ai/kimi":
            return await backend(**kwargs)
        return await router_deployment_call(**kwargs)

    with (
        patch("litellm.acompletion", new=agent_model_or_router_deployment),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {
                "safeguards_classifier_model": "classifier",
                "context_management_summary_model": "summarizer",
                **general_settings,
            },
        ),
        patch("litellm.token_counter", return_value=200_000),
    ):
        await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
            max_tokens=256,
            messages=list(MESSAGES),
            model="fireworks_ai/kimi",
            tools=[dict(BASH_TOOL)],
            stream=False,
            litellm_router=router,
            litellm_metadata=dict(litellm_metadata),
            litellm_logging_obj=parent_logging,
            **({} if anthropic_metadata is None else {"metadata": dict(anthropic_metadata)}),
            safeguards=SAFEGUARDS,
            context_management={"edits": [{"type": "compact_20260112"}]},
            **({} if request_headers is None else {"proxy_server_request": {"headers": dict(request_headers)}}),
            **parent_ids,
        )
        await _drain_logging_worker()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_correlation_in_logs, general_settings, parent_metadata, parent_ids, parent_session",
    [
        (
            False,
            {},
            {},
            {"litellm_session_id": "claude-code-session", "litellm_trace_id": "claude-code-session"},
            "claude-code-session",
        ),
        (False, {}, {}, {"litellm_session_id": "conversation-1", "litellm_trace_id": "trace-1"}, "conversation-1"),
        (False, {}, {}, {}, "trace-the-parent-minted"),
        (
            True,
            {},
            {"session_id": "client-trace", "trace_id": "client-trace", CALLER_SESSION_ID_METADATA_KEY: "claude-code-session"},
            {"litellm_session_id": "client-trace", "litellm_trace_id": "client-trace"},
            "claude-code-session",
        ),
        (False, {}, {"session_id": "conversation-in-metadata"}, {}, "conversation-in-metadata"),
        (
            False,
            {"missing_session_id": "omit"},
            {"session_id": "claude-code-session", "trace_id": "claude-code-session", SESSION_ID_OMITTED_METADATA_KEY: True},
            {"litellm_session_id": "claude-code-session", "litellm_trace_id": "claude-code-session"},
            "claude-code-session",
        ),
        (False, {"missing_session_id": "omit"}, {SESSION_ID_OMITTED_METADATA_KEY: True}, {}, None),
    ],
    ids=[
        "claude-code-session-header",
        "session-and-trace-sent-apart",
        "no-session-sent",
        "trace-header-beside-the-claude-code-session-under-request-correlation",
        "session-sent-only-in-litellm-metadata",
        "claude-code-session-header-under-omit",
        "no-session-sent-under-omit",
    ],
)
async def test_the_classifier_and_summary_spend_rows_land_in_the_parent_requests_session(
    monkeypatch: pytest.MonkeyPatch,
    request_correlation_in_logs: bool,
    general_settings: Mapping[str, object],
    parent_metadata: Mapping[str, object],
    parent_ids: Mapping[str, str],
    parent_session: str | None,
) -> None:
    await _drain_logging_worker()
    monkeypatch.setattr(litellm, "request_correlation_in_logs", request_correlation_in_logs)
    spend: Final = _SpendRows()
    monkeypatch.setattr(litellm, "callbacks", [spend])
    monkeypatch.setattr(litellm, "_async_success_callback", [spend])

    await _send_parent_request(
        _classifier_and_summarizer_router(), general_settings, {**LITELLM_METADATA, **parent_metadata}, parent_ids
    )

    assert {row["model_group"]: row["session_id"] for row in spend.rows} == {
        "classifier": parent_session,
        "summarizer": parent_session,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parent_metadata, affinity_header",
    [
        ({"session_id": "conversation-1"}, "conversation-1"),
        ({"session_id": "minted-1", SESSION_ID_GENERATED_METADATA_KEY: True}, None),
    ],
    ids=["session-the-caller-sent", "session-the-proxy-minted"],
)
async def test_the_classifier_call_sends_the_parents_session_as_its_affinity_header_unless_the_proxy_minted_it(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    parent_metadata: Mapping[str, object],
    affinity_header: str | None,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    session: Final = str(parent_metadata["session_id"])
    classifier_route: Final = respx_mock.post("https://classifier.example/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-classifier",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-5.4-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"verdicts": {"call_main": {"flagged": false}}}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )
    )
    router: Final = _classifier_and_summarizer_router(
        {
            "model": "openai/gpt-5.4-mini",
            "api_key": "sk-fake",
            "api_base": "https://classifier.example/v1",
            "provider_affinity_header": "x-session-affinity",
        }
    )

    await _send_parent_request(
        router,
        {},
        {**LITELLM_METADATA, **parent_metadata},
        {"litellm_session_id": session, "litellm_trace_id": session},
    )

    assert classifier_route.call_count == 1
    assert classifier_route.calls.last.request.headers.get("x-session-affinity") == affinity_header


class _LoggedCalls(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.calls: Final[list[Mapping[str, object]]] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.calls.append(kwargs)


def _langfuse_trace(logged_call: Mapping[str, object]) -> tuple[object, bool]:
    litellm_params: Final = cast(dict[str, object], logged_call["litellm_params"])
    trace_metadata: Final = LangFuseLogger.add_metadata_from_header(
        litellm_params, dict(cast(Mapping[str, object], litellm_params.get("metadata") or {}))
    )
    standard_logging_object: Final = cast(Mapping[str, object], logged_call["standard_logging_object"])
    trace_id: Final = langfuse_trace_id(
        metadata=trace_metadata,
        standard_trace_id=standard_logging_object["trace_id"],
        litellm_trace_id=logged_call.get("litellm_trace_id"),
        litellm_call_id=logged_call.get("litellm_call_id"),
        proxy_server_request=litellm_params.get("proxy_server_request"),
    )
    return trace_id, "existing_trace_id" in trace_metadata


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parent_metadata, parent_ids, request_headers, anthropic_metadata, parent_trace",
    [
        (
            {"session_id": "claude-code-session"},
            {"litellm_session_id": "claude-code-session", "litellm_trace_id": "claude-code-session"},
            {"x-claude-code-session-id": "claude-code-session"},
            None,
            "parent-call",
        ),
        (
            {"session_id": "claude-code-session"},
            {"litellm_session_id": "claude-code-session", "litellm_trace_id": "claude-code-session"},
            {"x-claude-code-session-id": "claude-code-session"},
            {"user_id": '{"device_id":"laptop","session_id":"claude-code-session"}'},
            "parent-call",
        ),
        ({}, {}, None, None, "trace-the-parent-minted"),
        (
            {},
            {},
            {"langfuse_existing_trace_id": "trace-the-client-continues"},
            None,
            "trace-the-client-continues",
        ),
    ],
    ids=[
        "claude-code-session-header",
        "claude-code-sends-its-anthropic-metadata-user-id",
        "no-session-sent",
        "client-continues-its-own-langfuse-trace",
    ],
)
async def test_the_classifier_and_summary_join_the_parent_requests_langfuse_trace_without_renaming_it(
    monkeypatch: pytest.MonkeyPatch,
    parent_metadata: Mapping[str, object],
    parent_ids: Mapping[str, str],
    request_headers: Mapping[str, str] | None,
    anthropic_metadata: Mapping[str, str] | None,
    parent_trace: str,
) -> None:
    await _drain_logging_worker()
    logged: Final = _LoggedCalls()
    monkeypatch.setattr(litellm, "callbacks", [logged])
    monkeypatch.setattr(litellm, "_async_success_callback", [logged])

    await _send_parent_request(
        _classifier_and_summarizer_router(),
        {},
        {**LITELLM_METADATA, **parent_metadata},
        parent_ids,
        request_headers,
        anthropic_metadata,
    )

    assert {
        cast(Mapping[str, object], call["standard_logging_object"])["model_group"]: _langfuse_trace(call)
        for call in logged.calls
    } == {"classifier": (parent_trace, True), "summarizer": (parent_trace, True)}
