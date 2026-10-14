import asyncio
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Final
from unittest.mock import patch

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.anthropic.pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
    request_safeguards_evaluator,
)
from litellm.llms.anthropic.pass_through.safeguards import ToolUseUnderReview
from litellm.proxy._types import LiteLLM_TeamTableCachedObj, UserAPIKeyAuth
from litellm.router import Router
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
from tests.unit.llms.anthropic.pass_through.conftest import ProxyAuthState

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
    }
    assert json.dumps({"command": "npm ci"}) in str(classifier_call["messages"][1]["content"])


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
    await router.acompletion(
        model="smart-router",
        messages=list(MESSAGES),
        metadata={"user_api_key_hash": "key-hash-a"},
        proxy_server_request={"headers": session_headers},
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
