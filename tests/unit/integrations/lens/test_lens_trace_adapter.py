from datetime import datetime, timezone
from typing import Final

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from litellm.integrations.lens.lens_trace_adapter import (
    AgentCall,
    ToolExchange,
    _LensTraceLogger,
    agent_call,
    agent_name,
    tool_exchanges,
)
from litellm.integrations.otel.logger import OpenTelemetryV2
from litellm.integrations.otel.model.config import OpenTelemetryV2Config
from litellm.integrations.otel.plumbing.providers import build_tracer_provider
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

_LS = ToolExchange(call_id="call-ls", name="Bash", arguments='{"command": "ls"}', result="a.txt")
_PWD = ToolExchange(call_id="call-pwd", name="Bash", arguments='{"command": "pwd"}', result="/work")


@pytest.mark.parametrize(
    ("user_agent", "expected"),
    [
        ("claude-cli/2.1.329 (external, sdk-cli)", "claude-code"),
        ("codex_exec/0.171.0 (Ubuntu 22.04; x86_64)", "codex"),
        ("Codex Desktop/1.0", "codex"),
        ("OpenAI/Python 1.99.0", "openai"),
        ("my-agent", "my-agent"),
        ("", None),
        (None, None),
    ],
)
def test_agent_is_named_from_the_caller_user_agent(user_agent: str | None, expected: str | None) -> None:
    assert agent_name(user_agent) == expected


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param(
            [
                {"role": "user", "content": "list then print"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "running"},
                        {"type": "tool_use", "id": "call-ls", "name": "Bash", "input": {"command": "ls"}},
                        {"type": "tool_use", "id": "call-pwd", "name": "Bash", "input": {"command": "pwd"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call-ls", "content": "a.txt"},
                        {"type": "tool_result", "tool_use_id": "call-pwd", "content": "/work"},
                    ],
                },
            ],
            id="anthropic-blocks",
        ),
        pytest.param(
            [
                {"role": "user", "content": "list then print"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-ls",
                            "type": "function",
                            "function": {"name": "Bash", "arguments": '{"command": "ls"}'},
                        },
                        {
                            "id": "call-pwd",
                            "type": "function",
                            "function": {"name": "Bash", "arguments": '{"command": "pwd"}'},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call-ls", "content": "a.txt"},
                {"role": "tool", "tool_call_id": "call-pwd", "content": "/work"},
            ],
            id="openai-chat",
        ),
        pytest.param(
            [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "list then print"}]},
                {"type": "function_call", "name": "Bash", "arguments": '{"command": "ls"}', "call_id": "call-ls"},
                {"type": "function_call", "name": "Bash", "arguments": '{"command": "pwd"}', "call_id": "call-pwd"},
                {"type": "function_call_output", "call_id": "call-ls", "output": "a.txt"},
                {"type": "function_call_output", "call_id": "call-pwd", "output": "/work"},
            ],
            id="openai-responses",
        ),
    ],
)
def test_parallel_tool_results_become_one_exchange_each_in_every_request_shape(messages: list[object]) -> None:
    assert tool_exchanges(messages) == (_LS, _PWD)


def test_a_result_already_answered_by_the_model_is_not_reported_again() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "call-ls", "function": {"name": "Bash", "arguments": '{"command": "ls"}'}}],
        },
        {"role": "tool", "tool_call_id": "call-ls", "content": "a.txt"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "call-pwd", "function": {"name": "Bash", "arguments": '{"command": "pwd"}'}}],
        },
        {"role": "tool", "tool_call_id": "call-pwd", "content": "/work"},
    ]

    assert tool_exchanges(messages) == (_PWD,)


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param([{"role": "user", "content": "hello"}], id="no-tools"),
        pytest.param(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "call-ls", "function": {"name": "Bash", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "call-ls", "content": "a.txt"},
                {"role": "assistant", "content": "done"},
                {"role": "user", "content": "thanks"},
            ],
            id="result-before-the-last-model-turn",
        ),
        pytest.param([{"role": "tool", "tool_call_id": "unknown", "content": "x"}], id="result-without-its-call"),
        pytest.param(
            [{"role": "assistant", "tool_calls": [{"id": "call-ls", "function": {"name": "Bash"}}]}],
            id="call-without-result",
        ),
        pytest.param("a plain prompt", id="string-prompt"),
        pytest.param([7, None, {"role": 3}], id="malformed-items"),
        pytest.param(None, id="no-prompt"),
    ],
)
def test_a_prompt_with_no_new_paired_tool_result_yields_no_exchange(messages: object) -> None:
    assert tool_exchanges(messages) == ()


def test_structured_tool_output_is_kept_as_json() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call-ls", "name": "Bash", "input": {"command": "ls"}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call-ls", "content": [{"type": "text", "text": "a.txt"}]}
            ],
        },
    ]

    assert tool_exchanges(messages) == (
        ToolExchange(
            call_id="call-ls", name="Bash", arguments='{"command": "ls"}', result='[{"type": "text", "text": "a.txt"}]'
        ),
    )


def test_a_model_call_payload_gives_the_agent_and_its_new_tool_results() -> None:
    payload = {
        "metadata": {"user_agent": "claude-cli/2.1.329"},
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call-ls", "function": {"name": "Bash", "arguments": '{"command": "ls"}'}}],
            },
            {"role": "tool", "tool_call_id": "call-ls", "content": "a.txt"},
        ],
    }

    assert agent_call(payload) == AgentCall(agent="claude-code", tool_exchanges=(_LS,))


@pytest.mark.parametrize("payload", [None, "text", {"metadata": "broken"}])
def test_an_unreadable_payload_names_no_agent_and_no_tool_result(payload: object) -> None:
    assert agent_call(payload) == AgentCall()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", (False, True), ids=("success", "failure"))
@pytest.mark.parametrize("enabled", (False, True), ids=("disabled", "enabled"))
async def test_lens_control_applies_to_model_and_tool_spans_without_disabling_other_telemetry(
    failed: bool, enabled: bool
) -> None:
    config: Final = OpenTelemetryV2Config(exporter="in_memory")
    lens_spans: Final = InMemorySpanExporter()
    other_spans: Final = InMemorySpanExporter()
    lens: Final = _LensTraceLogger(config=config, tracer_provider=build_tracer_provider(config, exporter=lens_spans))
    other: Final = OpenTelemetryV2(config=config, tracer_provider=build_tracer_provider(config, exporter=other_spans))
    at: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    kwargs: Final = {
        "api_call_start_time": at,
        "litellm_params": {
            "litellm_session_id": "session-policy",
            "metadata": {
                "user_api_key_auth": UserAPIKeyAuth(metadata={"lens_enabled": enabled}),
                "lens_enabled": not enabled,
            },
        },
        "standard_logging_object": {
            "id": "response-policy",
            "litellm_call_id": "call-policy",
            "call_type": "acompletion",
            "model": "test-model",
            "custom_llm_provider": "openai",
            "status": "failure" if failed else "success",
            "metadata": {"user_agent": "codex"},
            "messages": [
                {"role": "assistant", "tool_calls": [{"id": "call-ls", "function": {"name": "Bash"}}]},
                {"role": "tool", "tool_call_id": "call-ls", "content": "a.txt"},
            ],
            "response": {"id": "response-policy", "choices": [{"finish_reason": "stop"}]},
        },
    }
    lens.log_pre_api_call("test-model", [], kwargs)
    other.log_pre_api_call("test-model", [], kwargs)
    lens_callback: Final = lens.async_log_failure_event if failed else lens.async_log_success_event
    other_callback: Final = other.async_log_failure_event if failed else other.async_log_success_event
    await lens_callback(kwargs, None, at, at)
    await other_callback(kwargs, None, at, at)
    assert len(lens_spans.get_finished_spans()) == (2 if enabled else 0)
    assert len(other_spans.get_finished_spans()) == 1
    lens.tracer_provider.shutdown()
    other.tracer_provider.shutdown()
