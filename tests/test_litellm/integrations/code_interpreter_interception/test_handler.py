"""
Unit tests for CodeInterpreterInterceptionLogger.

All sandbox dependencies are injected: a FakeSandbox stands in for the real e2b
config and records how it is called.
"""

import time

import pytest

from litellm.integrations.code_interpreter_interception.handler import (
    CodeInterpreterInterceptionLogger,
    LITELLM_CODE_EXECUTION_TOOL_NAME,
    _INTERCEPTION_ACTIVE_KEY as _ACTIVE_KEY,
    _SANDBOX_KEY,
    _SESSION_SCOPED_KEY,
)
from litellm.types.integrations.custom_logger import (
    CHAT_COMPLETION_AGENTIC_SURFACE,
    NON_CODE_INTERPRETER_INTERCEPTION_INTERNAL_PREFIXES,
    is_interception_internal_key,
)
from litellm.llms.base_llm.sandbox.transformation import CodeExecutionResult
from litellm.types.utils import CallTypes


class FakeHandle:
    def __init__(self, sandbox_id="sbx_fake"):
        self.id = sandbox_id


class FakeSandbox:
    """Records acreate_sandbox / arun_code / adelete_sandbox calls."""

    def __init__(self, stdout="42"):
        self.stdout = stdout
        self.create_calls = []
        self.run_calls = []
        self.delete_calls = []

    async def acreate_sandbox(self, **kwargs):
        self.create_calls.append(kwargs)
        return FakeHandle()

    async def arun_code(self, *, container, code, **kwargs):
        self.run_calls.append({"container": container, "code": code})
        return CodeExecutionResult(stdout=self.stdout)

    async def adelete_sandbox(self, *, container, **kwargs):
        self.delete_calls.append({"container": container})
        return True


class FakeLogging:
    def __init__(self, litellm_call_id="k1"):
        self.litellm_call_id = litellm_call_id
        self.model_call_details = {}
        self.dynamic_success_callbacks = []

    def pre_call(self, *args, **kwargs):
        return None

    def post_call(self, *args, **kwargs):
        return None


def _function_call_item(call_id="c1", name=LITELLM_CODE_EXECUTION_TOOL_NAME):
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": '{"code":"print(40 + 2)"}',
    }


def _chat_function_call_item(call_id="call_1", name=LITELLM_CODE_EXECUTION_TOOL_NAME):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": '{"code":"print(40 + 2)"}',
        },
    }


class FakeResponse:
    def __init__(self, output):
        self.output = output


def _iter_messages(plan):
    patch = plan.request_patch
    assert patch is not None, "plan.request_patch must be set"
    assert patch.messages is not None, "plan.request_patch.messages must be set"
    return patch.messages


def test_interception_internal_key_prefix_sets_preserve_code_interpreter_state():
    assert is_interception_internal_key("_code_interpreter_interception_active")
    assert not is_interception_internal_key(
        "_code_interpreter_interception_active",
        prefixes=NON_CODE_INTERPRETER_INTERCEPTION_INTERNAL_PREFIXES,
    )
    assert is_interception_internal_key(
        "_websearch_interception_converted_stream",
        prefixes=NON_CODE_INTERPRETER_INTERCEPTION_INTERNAL_PREFIXES,
    )


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
        kwargs={"litellm_call_id": "k1", _SANDBOX_KEY: "sbxkey1"},
    )

    assert sandbox.run_calls, "sandbox.arun_code must be invoked"
    assert sandbox.run_calls[0]["code"] == "print(40 + 2)"

    messages = _iter_messages(plan)
    outputs = [m for m in messages if isinstance(m, dict) and m.get("type") == "function_call_output"]
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
    assert not any(t.get("type") == "code_interpreter" for t in tools), "code_interpreter tool must be removed"
    names = [t.get("name") or (t.get("function") or {}).get("name") for t in tools]
    assert LITELLM_CODE_EXECUTION_TOOL_NAME in names


@pytest.mark.asyncio
async def test_pre_call_converts_code_interpreter_tool_for_chat_completions():
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    kwargs = {
        "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
        "tool_choice": {"type": "code_interpreter"},
        "custom_llm_provider": "openai",
    }

    result = await logger.async_pre_call_deployment_hook(kwargs, CallTypes.acompletion)

    assert result is not None
    tool = result["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == LITELLM_CODE_EXECUTION_TOOL_NAME
    assert tool["function"]["parameters"]["required"] == ["code"]
    assert result["tool_choice"] == {
        "type": "function",
        "function": {"name": LITELLM_CODE_EXECUTION_TOOL_NAME},
    }
    assert result["litellm_metadata"][_ACTIVE_KEY] is True
    assert result["litellm_metadata"][_SANDBOX_KEY] == result[_SANDBOX_KEY]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_choice",
    [
        {"type": "code_interpreter"},
        {"type": "hosted_tool", "name": "code_interpreter"},
    ],
)
async def test_pre_call_rewrites_forced_code_interpreter_tool_choice(tool_choice):
    """A forced tool_choice targeting the native code_interpreter tool must be
    rewritten to the generated function tool; otherwise the outbound request
    references a tool that no longer exists and the provider rejects it."""
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    kwargs = {
        "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
        "tool_choice": tool_choice,
        "custom_llm_provider": "openai",
    }

    result = await logger.async_pre_call_deployment_hook(kwargs, CallTypes.aresponses)

    assert result is not None
    assert result["tool_choice"] == {
        "type": "function",
        "name": LITELLM_CODE_EXECUTION_TOOL_NAME,
    }


@pytest.mark.asyncio
async def test_pre_call_leaves_unrelated_tool_choice_untouched():
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    kwargs = {
        "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
        "tool_choice": "auto",
        "custom_llm_provider": "openai",
    }

    result = await logger.async_pre_call_deployment_hook(kwargs, CallTypes.aresponses)

    assert result is not None
    assert result["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_pre_call_noop_on_non_responses():
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    kwargs = {
        "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
        "custom_llm_provider": "openai",
    }

    result = await logger.async_pre_call_deployment_hook(kwargs, CallTypes.aembedding)

    assert result is None


@pytest.mark.asyncio
async def test_pre_call_noop_on_chat_completion_without_code_interpreter_tool():
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    kwargs = {
        "tools": [{"type": "web_search"}],
        "custom_llm_provider": "openai",
    }

    result = await logger.async_pre_call_deployment_hook(kwargs, CallTypes.acompletion)

    assert result is None
    assert _ACTIVE_KEY not in kwargs
    assert _SANDBOX_KEY not in kwargs


@pytest.mark.asyncio
async def test_should_run_detects_only_matching_function_call():
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())

    active_kwargs = {"_code_interpreter_interception_active": True}
    match = FakeResponse(output=[_function_call_item(name=LITELLM_CODE_EXECUTION_TOOL_NAME)])
    should_run, payload = await logger.async_should_run_agentic_loop(
        response=match,
        model="gpt-5",
        messages=[{"role": "user", "content": "x"}],
        tools=[],
        stream=False,
        custom_llm_provider="openai",
        kwargs=active_kwargs,
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
        kwargs=active_kwargs,
    )
    assert should_run2 is False
    assert payload2 == {}


@pytest.mark.asyncio
async def test_container_reused_within_request_via_server_sandbox_key():
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
        kwargs={"litellm_call_id": "k1", _SANDBOX_KEY: "server-nonce-1"},
        **common,
    )
    await logger.async_build_agentic_loop_plan(
        logging_obj=FakeLogging(litellm_call_id="k1"),
        kwargs={"litellm_call_id": "k1", _SANDBOX_KEY: "server-nonce-1"},
        **common,
    )

    assert len(sandbox.create_calls) == 1, "the sandbox is reused across loop iterations sharing one server sandbox key"


@pytest.mark.asyncio
async def test_colliding_caller_call_id_does_not_share_sandbox():
    """Two requests with the same caller-controlled litellm_call_id but distinct
    server-minted sandbox keys must NOT share a container; otherwise one user's
    code could read another in-flight request's sandbox state."""
    sandbox = FakeSandbox(stdout="42")
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)
    common = dict(
        tools={
            "tool_calls": [
                {
                    "call_id": "c1",
                    "name": LITELLM_CODE_EXECUTION_TOOL_NAME,
                    "arguments": '{"code":"print(1)"}',
                }
            ]
        },
        model="gpt-5",
        messages=[{"role": "user", "content": "x"}],
        response=FakeResponse(output=[_function_call_item()]),
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={"tools": []},
        stream=False,
    )

    await logger.async_build_agentic_loop_plan(
        logging_obj=FakeLogging(litellm_call_id="shared"),
        kwargs={"litellm_call_id": "shared", _SANDBOX_KEY: "nonce-A"},
        **common,
    )
    await logger.async_build_agentic_loop_plan(
        logging_obj=FakeLogging(litellm_call_id="shared"),
        kwargs={"litellm_call_id": "shared", _SANDBOX_KEY: "nonce-B"},
        **common,
    )

    assert len(sandbox.create_calls) == 2, (
        "distinct server sandbox keys must isolate sandboxes despite a colliding call id"
    )


@pytest.mark.asyncio
async def test_pre_call_mints_server_sandbox_key():
    """The interceptor mints a server-side sandbox key (not derived from the
    caller-controlled call id) when it activates."""
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    kwargs = {
        "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
        "custom_llm_provider": "openai",
        "litellm_call_id": "caller-supplied",
        _SANDBOX_KEY: "caller-forged",
    }

    result = await logger.async_pre_call_deployment_hook(kwargs, CallTypes.aresponses)

    assert result is not None
    assert result[_SANDBOX_KEY] not in ("caller-forged", "caller-supplied")
    assert len(result[_SANDBOX_KEY]) >= 16


@pytest.mark.asyncio
async def test_build_plan_records_code_interpreter_call_metadata():
    sandbox = FakeSandbox(stdout="42")
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)
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
        response=FakeResponse(output=[_function_call_item()]),
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={"tools": []},
        logging_obj=FakeLogging(litellm_call_id="k1"),
        stream=False,
        kwargs={"litellm_call_id": "k1", _SANDBOX_KEY: "sbxkey1"},
    )

    calls = plan.metadata["code_interpreter_calls"]
    assert calls, "build_plan must record a code_interpreter_call for re-injection"
    assert calls[0]["code"] == "print(40 + 2)"
    assert calls[0]["container_id"] == "sbx_fake"
    assert calls[0]["type"] == "code_interpreter_call"
    assert calls[0]["status"] == "completed"
    assert calls[0]["outputs"] == [{"type": "logs", "logs": "42"}], (
        "outputs must be an OpenAI-shaped logs array (not None) so clients that "
        "iterate over code_interpreter_call.outputs do not break"
    )


@pytest.mark.asyncio
async def test_build_plan_outputs_empty_array_when_no_stdout():
    """No stdout must still yield an iteration-safe empty array, never None."""
    sandbox = FakeSandbox(stdout="")
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)
    plan = await logger.async_build_agentic_loop_plan(
        tools={
            "tool_calls": [
                {
                    "call_id": "c1",
                    "name": LITELLM_CODE_EXECUTION_TOOL_NAME,
                    "arguments": '{"code":"pass"}',
                }
            ]
        },
        model="gpt-5",
        messages=[{"role": "user", "content": "x"}],
        response=FakeResponse(output=[_function_call_item()]),
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={"tools": []},
        logging_obj=FakeLogging(litellm_call_id="k1"),
        stream=False,
        kwargs={"litellm_call_id": "k1", _SANDBOX_KEY: "sbxkey1"},
    )

    assert plan.metadata["code_interpreter_calls"][0]["outputs"] == []


@pytest.mark.asyncio
async def test_post_hook_injects_code_interpreter_call_matching_openai_shape():
    from litellm.types.integrations.custom_logger import AgenticLoopPlan

    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    ci_item = {
        "id": "ci_x",
        "type": "code_interpreter_call",
        "status": "completed",
        "code": "print(1)",
        "container_id": "sbx_fake",
        "outputs": [{"type": "logs", "logs": "1"}],
    }
    plan = AgenticLoopPlan(
        run_agentic_loop=True,
        metadata={"code_interpreter_calls": [ci_item]},
    )
    response = FakeResponse(output=[{"type": "message", "content": []}])

    out = await logger.async_post_agentic_loop_response_hook(response=response, plan=plan, kwargs={})

    types = [item.get("type") for item in out.output]
    assert types == ["code_interpreter_call", "message"], (
        "code_interpreter_call must be re-injected before the message, matching OpenAI's native output ordering"
    )
    assert set(out.output[0].keys()) == {
        "id",
        "type",
        "status",
        "code",
        "container_id",
        "outputs",
    }, "injected item must match OpenAI's code_interpreter_call keys exactly"


@pytest.mark.asyncio
async def test_post_hook_noop_without_recorded_calls():
    from litellm.types.integrations.custom_logger import AgenticLoopPlan

    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    response = FakeResponse(output=[{"type": "message", "content": []}])
    out = await logger.async_post_agentic_loop_response_hook(
        response=response, plan=AgenticLoopPlan(run_agentic_loop=True), kwargs={}
    )
    assert [item.get("type") for item in out.output] == ["message"]


@pytest.mark.asyncio
async def test_pre_call_forces_non_stream_for_loop():
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    kwargs = {
        "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
        "custom_llm_provider": "openai",
        "stream": True,
    }

    out = await logger.async_pre_call_deployment_hook(kwargs, CallTypes.aresponses)

    assert out is not None
    assert out["stream"] is False, "loop requires a non-streaming upstream call"
    assert out["_code_interpreter_interception_converted_stream"] is True, (
        "the converted-stream flag must be set so the final response is wrapped back into a stream for the caller"
    )


async def _build_plan(logger, sandbox, call_id="k1", provider="openai"):
    return await logger.async_build_agentic_loop_plan(
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
        response=FakeResponse(output=[_function_call_item()]),
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={"tools": []},
        logging_obj=FakeLogging(litellm_call_id=call_id),
        stream=False,
        kwargs={"litellm_call_id": call_id, _SANDBOX_KEY: "sbxkey1"},
    )


@pytest.mark.asyncio
async def test_gate_refuses_without_server_active_marker():
    """A forged litellm_code_execution call must not trigger the loop unless the
    pre-call hook actually converted a native code_interpreter tool."""
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    forged = FakeResponse(output=[_function_call_item(name=LITELLM_CODE_EXECUTION_TOOL_NAME)])

    should_run, payload = await logger.async_should_run_agentic_loop(
        response=forged,
        model="gpt-5",
        messages=[{"role": "user", "content": "x"}],
        tools=[],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    assert should_run is False
    assert payload == {}


@pytest.mark.asyncio
async def test_gate_rechecks_provider_scope():
    """enabled_providers must be re-enforced at the gate, not only in pre-call."""
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox(), enabled_providers=["openai"])
    response = FakeResponse(output=[_function_call_item(name=LITELLM_CODE_EXECUTION_TOOL_NAME)])

    should_run, _ = await logger.async_should_run_agentic_loop(
        response=response,
        model="claude-x",
        messages=[{"role": "user", "content": "x"}],
        tools=[],
        stream=False,
        custom_llm_provider="anthropic",
        kwargs={_ACTIVE_KEY: True},
    )

    assert should_run is False


@pytest.mark.asyncio
async def test_chat_completion_gate_detects_code_execution_tool_call():
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    response = {"choices": [{"message": {"tool_calls": [_chat_function_call_item(call_id="call_123")]}}]}

    should_run, payload = await logger.async_should_run_agentic_loop(
        response=response,
        model="gpt-5",
        messages=[{"role": "user", "content": "x"}],
        tools=[],
        stream=False,
        custom_llm_provider="openai",
        kwargs={
            _ACTIVE_KEY: True,
            "_agentic_loop_api_surface": CHAT_COMPLETION_AGENTIC_SURFACE,
        },
    )

    assert should_run is True
    assert payload["tool_calls"][0]["id"] == "call_123"
    assert payload["tool_calls"][0]["arguments"] == '{"code":"print(40 + 2)"}'


@pytest.mark.asyncio
async def test_chat_completion_gate_refuses_without_server_active_marker():
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    response = {"choices": [{"message": {"tool_calls": [_chat_function_call_item()]}}]}

    should_run, payload = await logger.async_should_run_agentic_loop(
        response=response,
        model="gpt-5",
        messages=[{"role": "user", "content": "x"}],
        tools=[],
        stream=False,
        custom_llm_provider="openai",
        kwargs={"_agentic_loop_api_surface": CHAT_COMPLETION_AGENTIC_SURFACE},
    )

    assert should_run is False
    assert payload == {}


@pytest.mark.asyncio
async def test_chat_completion_build_plan_runs_code_and_appends_tool_message():
    sandbox = FakeSandbox(stdout="42")
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)
    native_chat_tool = {"type": "code_interpreter", "container": {"type": "auto"}}

    plan = await logger.async_build_agentic_loop_plan(
        tools={
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": LITELLM_CODE_EXECUTION_TOOL_NAME,
                    "arguments": '{"code":"print(40 + 2)"}',
                }
            ]
        },
        model="gpt-5",
        messages=[{"role": "user", "content": "x"}],
        response={"choices": [{"message": {"tool_calls": [_chat_function_call_item()]}}]},
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={
            "tools": [native_chat_tool],
            "tool_choice": {"type": "code_interpreter", "container": {"type": "auto"}},
            "temperature": 0,
        },
        logging_obj=FakeLogging(litellm_call_id="k1"),
        stream=False,
        kwargs={
            "acompletion": True,
            "litellm_call_id": "k1",
            _ACTIVE_KEY: True,
            _SANDBOX_KEY: "sbxkey1",
            "_code_interpreter_interception_converted_stream": True,
            "_agentic_loop_api_surface": CHAT_COMPLETION_AGENTIC_SURFACE,
        },
    )

    assert sandbox.run_calls[0]["code"] == "print(40 + 2)"
    patch = plan.request_patch
    assert patch is not None
    assert patch.tools == [
        {
            "type": "function",
            "function": {
                "name": LITELLM_CODE_EXECUTION_TOOL_NAME,
                "description": "Execute python code in a sandbox and return stdout.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }
    ]
    assert patch.optional_params == {"temperature": 0}
    assert patch.kwargs == {
        "litellm_call_id": "k1",
        _ACTIVE_KEY: True,
        _SANDBOX_KEY: "sbxkey1",
        "_code_interpreter_interception_converted_stream": True,
        "_agentic_loop_api_surface": CHAT_COMPLETION_AGENTIC_SURFACE,
    }
    assert patch.messages is not None
    assert patch.messages[-2]["role"] == "assistant"
    assert patch.messages[-2]["tool_calls"][0]["id"] == "call_1"
    assert patch.messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "42",
    }
    assert plan.metadata["code_interpreter_calls"][0]["code"] == "print(40 + 2)"


@pytest.mark.asyncio
async def test_pre_call_strips_client_forged_marker_on_initial_request():
    """A client cannot pre-set the active marker on the original request: with no
    native code_interpreter tool, any client-supplied interception markers in
    litellm_metadata are scrubbed and the active flag in kwargs is cleared."""
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    kwargs = {
        "tools": [{"type": "web_search"}],
        "custom_llm_provider": "openai",
        _ACTIVE_KEY: True,
        "litellm_metadata": {
            _ACTIVE_KEY: True,
            _SANDBOX_KEY: "client-forged",
            "safe_user_value": "kept",
        },
    }

    await logger.async_pre_call_deployment_hook(kwargs, CallTypes.aresponses)

    assert _ACTIVE_KEY not in kwargs, (
        "no native code_interpreter tool was present, so a client-supplied active marker must be cleared"
    )
    assert kwargs["litellm_metadata"] == {"safe_user_value": "kept"}


@pytest.mark.asyncio
async def test_pre_call_strips_forged_loop_controls_then_mints_own_markers():
    """On an INITIAL request (no server-set _agentic_loop_depth) a client cannot
    smuggle loop-control state: forged _agentic_loop_depth / max_agentic_loops and
    interception markers in litellm_metadata are stripped before the interceptor
    activates, so the only interception markers that survive are the ones the
    server mints for the converted code_interpreter tool."""
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    kwargs = {
        "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
        "custom_llm_provider": "openai",
        "litellm_metadata": {
            _ACTIVE_KEY: True,
            _SANDBOX_KEY: "client-forged",
            "_agentic_loop_depth": 99,
            "max_agentic_loops": 999,
            "safe_user_value": "kept",
        },
    }

    result = await logger.async_pre_call_deployment_hook(kwargs, CallTypes.acompletion)

    assert result is not None
    metadata = result["litellm_metadata"]
    assert metadata["safe_user_value"] == "kept"
    assert "_agentic_loop_depth" not in metadata, "forged loop depth must be stripped"
    assert "max_agentic_loops" not in metadata, "forged loop cap must be stripped"
    assert metadata[_ACTIVE_KEY] is True
    assert metadata[_SANDBOX_KEY] == result[_SANDBOX_KEY]
    assert metadata[_SANDBOX_KEY] != "client-forged", (
        "the surviving sandbox key must be the server-minted one, not the forged value the client supplied"
    )


@pytest.mark.asyncio
async def test_pre_call_preserves_marker_on_server_followup():
    """On a server-driven followup (depth>0) the marker is trusted and kept."""
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    kwargs = {
        "tools": [{"type": "function", "name": LITELLM_CODE_EXECUTION_TOOL_NAME}],
        "custom_llm_provider": "openai",
        "_agentic_loop_depth": 1,
        _ACTIVE_KEY: True,
    }

    await logger.async_pre_call_deployment_hook(kwargs, CallTypes.aresponses)

    assert kwargs.get(_ACTIVE_KEY) is True, (
        "the server-set marker must survive followup requests so multi-round code execution keeps working"
    )


@pytest.mark.asyncio
async def test_sandbox_deleted_after_loop_completes():
    sandbox = FakeSandbox(stdout="42")
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)

    plan = await _build_plan(logger, sandbox, call_id="k1")
    assert sandbox.create_calls, "sandbox must be created during the loop"
    assert not sandbox.delete_calls, "sandbox must outlive the loop until the final hook"

    await logger.async_post_agentic_loop_response_hook(
        response=FakeResponse(output=[{"type": "message", "content": []}]),
        plan=plan,
        kwargs={},
    )

    assert len(sandbox.delete_calls) == 1, (
        "the sandbox must be deleted once the final response is assembled, otherwise it keeps running and billing"
    )
    assert "sbxkey1" not in logger._container_cache


@pytest.mark.asyncio
async def test_post_hook_delete_is_idempotent_across_loop_levels():
    sandbox = FakeSandbox(stdout="42")
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)
    plan = await _build_plan(logger, sandbox, call_id="k1")
    response = FakeResponse(output=[{"type": "message", "content": []}])

    await logger.async_post_agentic_loop_response_hook(response=response, plan=plan, kwargs={})
    await logger.async_post_agentic_loop_response_hook(response=response, plan=plan, kwargs={})

    assert len(sandbox.delete_calls) == 1, (
        "deleting an already-removed container must be a no-op so unwinding loop levels do not double-delete"
    )


@pytest.mark.asyncio
async def test_build_plan_deletes_sandbox_when_execution_raises():
    """If sandbox execution raises before a plan is built (e.g. E2B aborts
    output over its cap), the cached sandbox must be deleted before re-raising,
    otherwise a caller can leak paid containers until the prune TTL."""

    class RaisingSandbox(FakeSandbox):
        async def arun_code(self, *, container, code, **kwargs):
            raise ValueError("output exceeded cap")

    sandbox = RaisingSandbox()
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)

    with pytest.raises(ValueError, match="exceeded cap"):
        await _build_plan(logger, sandbox, call_id="k1")

    assert len(sandbox.create_calls) == 1, "the sandbox must have been created"
    assert len(sandbox.delete_calls) == 1, (
        "a build failure must delete the cached sandbox so it does not keep running and billing"
    )
    assert "sbxkey1" not in logger._container_cache


@pytest.mark.asyncio
async def test_cleanup_hook_deletes_sandbox():
    sandbox = FakeSandbox(stdout="42")
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)
    plan = await _build_plan(logger, sandbox, call_id="k1")

    await logger.async_agentic_loop_cleanup_hook(plan=plan, kwargs={})

    assert len(sandbox.delete_calls) == 1, (
        "the cleanup hook must delete the sandbox so a rerun failure cannot leak a running container"
    )
    assert "sbxkey1" not in logger._container_cache


@pytest.mark.asyncio
async def test_cleanup_hook_is_idempotent_with_post_hook():
    sandbox = FakeSandbox(stdout="42")
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)
    plan = await _build_plan(logger, sandbox, call_id="k1")

    await logger.async_post_agentic_loop_response_hook(
        response=FakeResponse(output=[{"type": "message", "content": []}]),
        plan=plan,
        kwargs={},
    )
    await logger.async_agentic_loop_cleanup_hook(plan=plan, kwargs={})

    assert len(sandbox.delete_calls) == 1, (
        "cleanup running in finally after the success-path post hook already deleted the sandbox must not double-delete"
    )


@pytest.mark.asyncio
async def test_responses_plan_cleans_up_sandbox_when_followup_raises():
    """If the agentic rerun fails, _execute_responses_agentic_plan must still
    invoke the cleanup hook so the sandbox is not left running."""
    import litellm
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    from litellm.types.integrations.custom_logger import (
        AgenticLoopPlan,
        AgenticLoopRequestPatch,
    )

    cleanup_calls = []

    class CleanupCallback(CustomLogger):
        async def async_post_agentic_loop_response_hook(self, response, plan, kwargs):
            return response

        async def async_agentic_loop_cleanup_hook(self, plan, kwargs):
            cleanup_calls.append(plan)

    plan = AgenticLoopPlan(
        run_agentic_loop=True,
        request_patch=AgenticLoopRequestPatch(model="gpt-5", messages=[{"role": "user", "content": "x"}]),
        metadata={"sandbox_key": "sbxkey1"},
    )

    original = litellm.aresponses

    async def _boom(*args, **kwargs):
        raise RuntimeError("upstream blew up")

    litellm.aresponses = _boom
    try:
        with pytest.raises(RuntimeError, match="upstream blew up"):
            await BaseLLMHTTPHandler()._execute_responses_agentic_plan(
                plan=plan,
                model="gpt-5",
                response_api_optional_request_params={},
                logging_obj=FakeLogging(litellm_call_id="k1"),
                kwargs={},
                depth=0,
                max_loops=3,
                fingerprints=[],
                fingerprint="fp",
                callback=CleanupCallback(),
            )
    finally:
        litellm.aresponses = original

    assert cleanup_calls == [plan], (
        "cleanup hook must run in finally even when the rerun raises, otherwise "
        "the sandbox keeps running until the prune TTL"
    )


@pytest.mark.asyncio
async def test_run_code_does_not_re_resolve_registry(monkeypatch):
    """Params resolved once at create time must be reused for running code, so a
    registry clear between create and run cannot turn into a create-then-fail."""
    import litellm
    from litellm.sandbox import sandbox_tools

    sandbox_tools.register_sandbox_tools(
        [
            {
                "sandbox_tool_name": "e2b_default",
                "litellm_params": {"sandbox_provider": "e2b", "api_key": "sk-x"},
            }
        ]
    )

    create_kwargs = {}
    run_kwargs = {}

    async def fake_acreate_sandbox(**kwargs):
        create_kwargs.update(kwargs)
        return FakeHandle()

    async def fake_arun_code(**kwargs):
        run_kwargs.update(kwargs)
        return CodeExecutionResult(stdout="ok")

    monkeypatch.setattr(litellm, "acreate_sandbox", fake_acreate_sandbox)
    monkeypatch.setattr(litellm, "arun_code", fake_arun_code)

    logger = CodeInterpreterInterceptionLogger(sandbox_tool_name="e2b_default")
    try:
        container, params = await logger._get_or_create_container(cache_key="k1")
        assert params is not None and params["sandbox_provider"] == "e2b"

        sandbox_tools.clear_sandbox_tools()

        stdout = await logger._run_tool_call(container=container, params=params, arguments='{"code":"print(1)"}')
    finally:
        sandbox_tools.clear_sandbox_tools()

    assert stdout == "ok", "run must succeed using the params captured at create time"
    assert run_kwargs["provider"] == "e2b"


@pytest.mark.asyncio
async def test_run_tool_call_surfaces_execution_error():
    """A sandbox execution error must be fed back to the model as a labelled
    string, not raised, so the agentic loop can react to it."""

    class ErroringSandbox(FakeSandbox):
        async def arun_code(self, *, container, code, **kwargs):
            self.run_calls.append({"container": container, "code": code})
            return CodeExecutionResult(stdout="", error={"name": "ValueError", "value": "boom"})

    sandbox = ErroringSandbox()
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)
    container = await logger._create_container()

    stdout = await logger._run_tool_call(
        container=container[0], params=None, arguments='{"code":"raise ValueError(1)"}'
    )

    assert stdout == "[execution error] boom"


@pytest.mark.asyncio
async def test_run_tool_call_reports_unparseable_arguments():
    """Malformed tool arguments must produce a parse error string the model can
    see rather than crashing the interceptor."""
    sandbox = FakeSandbox()
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)
    container = await logger._create_container()

    stdout = await logger._run_tool_call(container=container[0], params=None, arguments="not-json")

    assert stdout == "[invalid tool arguments: could not parse code]"
    assert not sandbox.run_calls, "code must not run when arguments cannot be parsed"


@pytest.mark.asyncio
async def test_pre_call_skips_provider_outside_scope():
    """enabled_providers must filter the pre-call conversion so a request to an
    out-of-scope provider is left untouched."""
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox(), enabled_providers=["openai"])
    kwargs = {
        "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
        "custom_llm_provider": "anthropic",
    }

    result = await logger.async_pre_call_deployment_hook(kwargs, CallTypes.aresponses)

    assert result is None
    assert kwargs["tools"][0]["type"] == "code_interpreter", "tool must be untouched"
    assert _ACTIVE_KEY not in kwargs


@pytest.mark.asyncio
async def test_resolve_provider_falls_back_to_model_lookup():
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())

    assert logger._resolve_provider({"custom_llm_provider": "openai"}) == "openai"
    assert logger._resolve_provider({"model": "gpt-5"}) == "openai"
    assert logger._resolve_provider({"model": 123}) is None
    assert logger._resolve_provider({"model": "no-such-provider-xyz"}) is None


@pytest.mark.asyncio
async def test_create_container_without_sandbox_raises():
    """The registry path must raise a clear error when no sandbox is resolvable
    instead of silently creating nothing."""
    logger = CodeInterpreterInterceptionLogger(sandbox_tool_name="missing")

    with pytest.raises(ValueError, match="no sandbox available"):
        await logger._create_container()


@pytest.mark.asyncio
async def test_run_code_without_params_raises():
    logger = CodeInterpreterInterceptionLogger(sandbox_tool_name="missing")

    with pytest.raises(ValueError, match="no sandbox available to run code"):
        await logger._run_code(container=FakeHandle(), params=None, code="print(1)")


@pytest.mark.asyncio
async def test_delete_container_swallows_errors():
    """A delete failure must not propagate; the request already succeeded."""

    class FailingDeleteSandbox(FakeSandbox):
        async def adelete_sandbox(self, *, container, **kwargs):
            raise RuntimeError("e2b unreachable")

    sandbox = FailingDeleteSandbox()
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)
    container, params = await logger._create_container()

    await logger._delete_container(container=container, params=params)


@pytest.mark.asyncio
async def test_prune_expired_cache_deletes_underlying_container():
    """Expired cache entries must have their sandbox deleted, not just dropped,
    otherwise an orphaned sandbox keeps running."""
    import litellm.integrations.code_interpreter_interception.handler as handler_mod

    sandbox = FakeSandbox()
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)
    container, params = await logger._create_container()
    logger._container_cache["old"] = (
        container,
        params,
        time.time() - handler_mod._CACHE_TTL_SECONDS - 1,
        None,
    )

    await logger._prune_expired_cache()

    assert "old" not in logger._container_cache
    assert len(sandbox.delete_calls) == 1, "expired sandbox must be deleted"


@pytest.mark.asyncio
async def test_normalize_messages_handles_str_and_unknown():
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())

    assert logger._normalize_messages("hi") == [{"role": "user", "content": "hi"}]
    assert logger._normalize_messages([{"role": "user"}]) == [{"role": "user"}]
    assert logger._normalize_messages(42) == []


def test_from_config_yaml_reads_fields():
    cfg = {
        "enabled": False,
        "enabled_providers": ["openai"],
        "sandbox_tool_name": "e2b_default",
    }
    logger = CodeInterpreterInterceptionLogger.from_config_yaml(cfg)

    assert logger.enabled is False
    assert logger.enabled_providers == ["openai"]
    assert logger.sandbox_tool_name == "e2b_default"


def test_initialize_from_proxy_config_prefers_litellm_settings():
    logger = CodeInterpreterInterceptionLogger.initialize_from_proxy_config(
        litellm_settings={
            "code_interpreter_interception_params": {
                "enabled_providers": ["openai"],
                "sandbox_tool_name": "e2b_default",
            }
        },
        callback_specific_params={},
    )

    assert logger.enabled_providers == ["openai"]
    assert logger.sandbox_tool_name == "e2b_default"


@pytest.mark.asyncio
async def test_build_plan_handles_dict_shaped_response():
    """A responses payload delivered as a plain dict (not an object) must flow
    through detection, execution, and re-injection the same as the typed form."""
    sandbox = FakeSandbox(stdout="42")
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)
    dict_response = {"output": [_function_call_item()]}

    plan = await logger.async_build_agentic_loop_plan(
        tools={"tool_calls": logger._extract_code_execution_tool_calls(dict_response)},
        model="gpt-5",
        messages=[{"role": "user", "content": "x"}],
        response=dict_response,
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={"tools": []},
        logging_obj=FakeLogging(litellm_call_id="k1"),
        stream=False,
        kwargs={"litellm_call_id": "k1", _SANDBOX_KEY: "sbxkey1"},
    )

    assert sandbox.run_calls, "code must run for a dict-shaped response"
    assert plan.metadata["code_interpreter_calls"][0]["code"] == "print(40 + 2)"

    out = await logger.async_post_agentic_loop_response_hook(
        response={"output": [{"type": "message", "content": []}]},
        plan=plan,
        kwargs={},
    )

    assert [item.get("type") for item in out["output"]] == [
        "code_interpreter_call",
        "message",
    ], "the dict-shaped response must get the code_interpreter_call re-injected"


@pytest.mark.asyncio
async def test_extract_tool_calls_reads_object_attributes():
    """Detection must work when output items are objects with attributes, not
    only dicts."""

    class Item:
        def __init__(self):
            self.type = "function_call"
            self.name = LITELLM_CODE_EXECUTION_TOOL_NAME
            self.call_id = "c9"
            self.arguments = '{"code":"print(1)"}'

    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    calls = logger._extract_code_execution_tool_calls(FakeResponse(output=[Item()]))

    assert len(calls) == 1
    assert calls[0]["call_id"] == "c9"
    assert calls[0]["arguments"] == '{"code":"print(1)"}'


# ---------------------------------------------------------------------------
# Sticky session tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_call_uses_session_id_from_metadata_as_sandbox_key():
    """When session_id is in request metadata, it becomes the sandbox key so the
    container is shared across requests in the same session."""
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    session_id = "conv-abc-123"
    kwargs = {
        "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
        "custom_llm_provider": "openai",
        "metadata": {"session_id": session_id},
    }

    result = await logger.async_pre_call_deployment_hook(kwargs, CallTypes.acompletion)

    assert result is not None
    assert result[_SANDBOX_KEY] == session_id
    assert result[_SESSION_SCOPED_KEY] is True
    assert result["litellm_metadata"][_SANDBOX_KEY] == session_id
    assert result["litellm_metadata"][_SESSION_SCOPED_KEY] is True


@pytest.mark.asyncio
async def test_pre_call_uses_session_id_from_litellm_metadata():
    """session_id in litellm_metadata also works as the sticky key."""
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    session_id = "sess-xyz-789"
    kwargs = {
        "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
        "custom_llm_provider": "openai",
        "litellm_metadata": {"session_id": session_id},
    }

    result = await logger.async_pre_call_deployment_hook(kwargs, CallTypes.acompletion)

    assert result is not None
    assert result[_SANDBOX_KEY] == session_id
    assert result[_SESSION_SCOPED_KEY] is True


@pytest.mark.asyncio
async def test_pre_call_without_session_id_still_mints_random_key():
    """Requests without a session_id still get a server-minted random sandbox key."""
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    kwargs = {
        "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
        "custom_llm_provider": "openai",
    }

    result = await logger.async_pre_call_deployment_hook(kwargs, CallTypes.acompletion)

    assert result is not None
    assert _SESSION_SCOPED_KEY not in result or result[_SESSION_SCOPED_KEY] is False
    assert len(result[_SANDBOX_KEY]) >= 16


@pytest.mark.asyncio
async def test_session_scoped_sandbox_survives_agentic_loop_cleanup():
    """A session-scoped sandbox must NOT be deleted by the cleanup or post hooks;
    it needs to persist across requests within the same session."""
    sandbox = FakeSandbox(stdout="42")
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)
    session_id = "conv-persist-me"

    plan = await logger.async_build_agentic_loop_plan(
        tools={
            "tool_calls": [
                {
                    "call_id": "c1",
                    "name": LITELLM_CODE_EXECUTION_TOOL_NAME,
                    "arguments": '{"code":"x = 10"}',
                }
            ]
        },
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "set x"}],
        response=FakeResponse(output=[_function_call_item()]),
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={"tools": []},
        logging_obj=FakeLogging(litellm_call_id="k1"),
        stream=False,
        kwargs={
            "litellm_call_id": "k1",
            _SANDBOX_KEY: session_id,
            _SESSION_SCOPED_KEY: True,
        },
    )

    assert plan.metadata["is_session_scoped"] is True

    await logger.async_post_agentic_loop_response_hook(
        response=FakeResponse(output=[{"type": "message", "content": []}]),
        plan=plan,
        kwargs={},
    )
    await logger.async_agentic_loop_cleanup_hook(plan=plan, kwargs={})

    assert not sandbox.delete_calls, (
        "session-scoped sandbox must not be deleted after a single agentic loop; "
        "it must persist for the next request in the session"
    )
    assert session_id in logger._container_cache, "session-scoped container must remain in cache after loop ends"


@pytest.mark.asyncio
async def test_session_scoped_sandbox_reused_across_sequential_requests():
    """Two sequential requests with the same session_id must share one container,
    confirming state (e.g. assigned variables) can persist across HTTP requests."""
    sandbox = FakeSandbox(stdout="42")
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)
    session_id = "conv-reuse-me"

    common_plan_args = dict(
        tools={
            "tool_calls": [
                {
                    "call_id": "c1",
                    "name": LITELLM_CODE_EXECUTION_TOOL_NAME,
                    "arguments": '{"code":"print(1)"}',
                }
            ]
        },
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "x"}],
        response=FakeResponse(output=[_function_call_item()]),
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={"tools": []},
        stream=False,
    )
    session_kwargs = {_SANDBOX_KEY: session_id, _SESSION_SCOPED_KEY: True}

    plan1 = await logger.async_build_agentic_loop_plan(
        logging_obj=FakeLogging(litellm_call_id="req1"),
        kwargs={"litellm_call_id": "req1", **session_kwargs},
        **common_plan_args,
    )
    await logger.async_post_agentic_loop_response_hook(
        response=FakeResponse(output=[{"type": "message", "content": []}]),
        plan=plan1,
        kwargs={},
    )

    plan2 = await logger.async_build_agentic_loop_plan(
        logging_obj=FakeLogging(litellm_call_id="req2"),
        kwargs={"litellm_call_id": "req2", **session_kwargs},
        **common_plan_args,
    )
    await logger.async_post_agentic_loop_response_hook(
        response=FakeResponse(output=[{"type": "message", "content": []}]),
        plan=plan2,
        kwargs={},
    )

    assert len(sandbox.create_calls) == 1, (
        "a single container must serve both requests in the same session; "
        "two creates means state cannot persist between requests"
    )
    assert len(sandbox.delete_calls) == 0, "the session container must still be alive after both requests complete"


@pytest.mark.asyncio
async def test_non_session_sandbox_still_deleted_after_loop():
    """Without a session_id, the existing per-request ephemeral behavior is unchanged."""
    sandbox = FakeSandbox(stdout="42")
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)

    plan = await _build_plan(logger, sandbox, call_id="k1")
    await logger.async_post_agentic_loop_response_hook(
        response=FakeResponse(output=[{"type": "message", "content": []}]),
        plan=plan,
        kwargs={},
    )

    assert len(sandbox.delete_calls) == 1, "non-session sandbox must still be cleaned up after each request"


@pytest.mark.asyncio
async def test_sandbox_key_scoped_to_api_key_hash_isolates_users():
    """Two callers supplying the same session_id but different API key hashes must
    each get their own sandbox; sharing across tenants would let one read or mutate
    the other's interpreter state."""
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    session_id = "same-session-id"

    result_a = await logger.async_pre_call_deployment_hook(
        {
            "tools": [{"type": "code_interpreter"}],
            "custom_llm_provider": "openai",
            "metadata": {"session_id": session_id},
            "user_api_key_hash": "hash-for-tenant-a",
        },
        CallTypes.acompletion,
    )
    result_b = await logger.async_pre_call_deployment_hook(
        {
            "tools": [{"type": "code_interpreter"}],
            "custom_llm_provider": "openai",
            "metadata": {"session_id": session_id},
            "user_api_key_hash": "hash-for-tenant-b",
        },
        CallTypes.acompletion,
    )

    assert result_a is not None and result_b is not None
    assert result_a[_SANDBOX_KEY] != result_b[_SANDBOX_KEY], (
        "same session_id from different API keys must yield different sandbox keys; "
        "otherwise tenant A can read tenant B's sandbox state"
    )
    assert "hash-for-tenant-a" in result_a[_SANDBOX_KEY]
    assert "hash-for-tenant-b" in result_b[_SANDBOX_KEY]


@pytest.mark.asyncio
async def test_per_identity_cap_evicts_lru_session():
    """When a single identity holds the cap limit of session sandboxes and opens a
    new one, the least-recently-used session is evicted so the allocation stays
    bounded.  Without this, rotating session IDs is an unbounded sandbox leak."""
    from litellm.integrations.code_interpreter_interception.handler import _SESSION_SCOPED_PER_IDENTITY_CAP

    sandbox = FakeSandbox(stdout="ok")
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)
    identity = "hash-for-identity-x"

    for i in range(_SESSION_SCOPED_PER_IDENTITY_CAP):
        await logger._get_or_create_container(
            cache_key=f"{identity}:session-{i}",
            identity=identity,
        )
        logger._container_cache[f"{identity}:session-{i}"] = (
            logger._container_cache[f"{identity}:session-{i}"][0],
            logger._container_cache[f"{identity}:session-{i}"][1],
            float(i),
            identity,
        )

    assert len(logger._container_cache) == _SESSION_SCOPED_PER_IDENTITY_CAP

    await logger._get_or_create_container(
        cache_key=f"{identity}:session-new",
        identity=identity,
    )

    assert len(logger._container_cache) == _SESSION_SCOPED_PER_IDENTITY_CAP, (
        "adding a new session beyond the cap must evict one entry so total stays bounded"
    )
    assert f"{identity}:session-0" not in logger._container_cache, (
        "the entry with the oldest last_accessed timestamp must be evicted first (LRU)"
    )
    assert len(sandbox.delete_calls) == 1, "evicted sandbox must be deleted, not just removed from cache"
