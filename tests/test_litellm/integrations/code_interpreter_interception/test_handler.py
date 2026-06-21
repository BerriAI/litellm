"""
Unit tests for CodeInterpreterInterceptionLogger.

All sandbox dependencies are injected (dependency injection, no monkeypatch):
a FakeSandbox stands in for the real e2b config and records how it is called.
"""

import pytest

from litellm.integrations.code_interpreter_interception.handler import (
    CodeInterpreterInterceptionLogger,
    LITELLM_CODE_EXECUTION_TOOL_NAME,
)
from litellm.llms.base_llm.sandbox.transformation import CodeExecutionResult
from litellm.types.utils import CallTypes


class FakeHandle:
    def __init__(self, sandbox_id="sbx_fake"):
        self.id = sandbox_id


class FakeSandbox:
    """Records acreate_sandbox / arun_code calls and returns canned results."""

    def __init__(self, stdout="42"):
        self.stdout = stdout
        self.create_calls = []
        self.run_calls = []

    async def acreate_sandbox(self, **kwargs):
        self.create_calls.append(kwargs)
        return FakeHandle()

    async def arun_code(self, *, container, code, **kwargs):
        self.run_calls.append({"container": container, "code": code})
        return CodeExecutionResult(stdout=self.stdout)


class FakeLogging:
    def __init__(self, litellm_call_id="k1"):
        self.litellm_call_id = litellm_call_id
        self.model_call_details = {}


def _function_call_item(call_id="c1", name=LITELLM_CODE_EXECUTION_TOOL_NAME):
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": '{"code":"print(40 + 2)"}',
    }


class FakeResponse:
    def __init__(self, output):
        self.output = output


def _iter_messages(plan):
    patch = plan.request_patch
    assert patch is not None, "plan.request_patch must be set"
    assert patch.messages is not None, "plan.request_patch.messages must be set"
    return patch.messages


@pytest.mark.asyncio
async def test_build_plan_runs_code_and_feeds_output_back():
    sandbox = FakeSandbox(stdout="42")
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)
    response = FakeResponse(output=[_function_call_item()])

    plan = await logger.async_build_agentic_loop_plan(
        tools={
            "tool_calls": [
                {
                    "call_id": "c1",
                    "name": LITELLM_CODE_EXECUTION_TOOL_NAME,
                    "arguments": '{"code":"print(40 + 2)"}',
                }
            ]
        },
        model="gpt-5",
        messages=[{"role": "user", "content": "x"}],
        response=response,
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={"tools": []},
        logging_obj=FakeLogging(litellm_call_id="k1"),
        stream=False,
        kwargs={"litellm_call_id": "k1"},
    )

    assert sandbox.run_calls, "sandbox.arun_code must be invoked"
    assert sandbox.run_calls[0]["code"] == "print(40 + 2)"

    messages = _iter_messages(plan)
    outputs = [
        m
        for m in messages
        if isinstance(m, dict) and m.get("type") == "function_call_output"
    ]
    assert outputs, "expected a function_call_output item appended"
    output_item = next(m for m in outputs if m.get("call_id") == "c1")
    assert "42" in str(output_item["output"])


@pytest.mark.asyncio
async def test_pre_call_converts_code_interpreter_tool():
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    kwargs = {
        "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
        "custom_llm_provider": "openai",
    }

    result = await logger.async_pre_call_deployment_hook(kwargs, CallTypes.aresponses)

    assert result is not None
    tools = result["tools"]
    assert not any(
        t.get("type") == "code_interpreter" for t in tools
    ), "code_interpreter tool must be removed"
    names = [t.get("name") or (t.get("function") or {}).get("name") for t in tools]
    assert LITELLM_CODE_EXECUTION_TOOL_NAME in names


@pytest.mark.asyncio
async def test_pre_call_noop_on_non_responses():
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    kwargs = {
        "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
        "custom_llm_provider": "openai",
    }

    result = await logger.async_pre_call_deployment_hook(kwargs, CallTypes.acompletion)

    assert result is None


@pytest.mark.asyncio
async def test_should_run_detects_only_matching_function_call():
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())

    match = FakeResponse(
        output=[_function_call_item(name=LITELLM_CODE_EXECUTION_TOOL_NAME)]
    )
    should_run, payload = await logger.async_should_run_agentic_loop(
        response=match,
        model="gpt-5",
        messages=[{"role": "user", "content": "x"}],
        tools=[],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )
    assert should_run is True
    assert payload.get("tool_calls")

    no_match = FakeResponse(output=[_function_call_item(name="something_else")])
    should_run2, payload2 = await logger.async_should_run_agentic_loop(
        response=no_match,
        model="gpt-5",
        messages=[{"role": "user", "content": "x"}],
        tools=[],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )
    assert should_run2 is False
    assert payload2 == {}


@pytest.mark.asyncio
async def test_container_reused_across_calls_same_call_id():
    sandbox = FakeSandbox(stdout="42")
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)
    response = FakeResponse(output=[_function_call_item()])

    common = dict(
        tools={
            "tool_calls": [
                {
                    "call_id": "c1",
                    "name": LITELLM_CODE_EXECUTION_TOOL_NAME,
                    "arguments": '{"code":"print(40 + 2)"}',
                }
            ]
        },
        model="gpt-5",
        messages=[{"role": "user", "content": "x"}],
        response=response,
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={"tools": []},
        stream=False,
    )

    await logger.async_build_agentic_loop_plan(
        logging_obj=FakeLogging(litellm_call_id="k1"),
        kwargs={"litellm_call_id": "k1"},
        **common,
    )
    await logger.async_build_agentic_loop_plan(
        logging_obj=FakeLogging(litellm_call_id="k1"),
        kwargs={"litellm_call_id": "k1"},
        **common,
    )

    assert (
        len(sandbox.create_calls) == 1
    ), "sandbox must be created once and reused across calls with the same litellm_call_id"
