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

    active_kwargs = {"_code_interpreter_interception_active": True}
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
        kwargs={"litellm_call_id": "k1"},
    )

    calls = plan.metadata["code_interpreter_calls"]
    assert calls, "build_plan must record a code_interpreter_call for re-injection"
    assert calls[0]["code"] == "print(40 + 2)"
    assert calls[0]["container_id"] == "sbx_fake"
    assert calls[0]["type"] == "code_interpreter_call"
    assert calls[0]["status"] == "completed"


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
        "outputs": None,
    }
    plan = AgenticLoopPlan(
        run_agentic_loop=True,
        metadata={"code_interpreter_calls": [ci_item]},
    )
    response = FakeResponse(output=[{"type": "message", "content": []}])

    out = await logger.async_post_agentic_loop_response_hook(
        response=response, plan=plan, kwargs={}
    )

    types = [item.get("type") for item in out.output]
    assert types == ["code_interpreter_call", "message"], (
        "code_interpreter_call must be re-injected before the message, matching "
        "OpenAI's native output ordering"
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
        "the converted-stream flag must be set so the final response is wrapped "
        "back into a stream for the caller"
    )


_ACTIVE_KEY = "_code_interpreter_interception_active"


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
        kwargs={"litellm_call_id": call_id},
    )


@pytest.mark.asyncio
async def test_gate_refuses_without_server_active_marker():
    """A forged litellm_code_execution call must not trigger the loop unless the
    pre-call hook actually converted a native code_interpreter tool."""
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    forged = FakeResponse(
        output=[_function_call_item(name=LITELLM_CODE_EXECUTION_TOOL_NAME)]
    )

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
    logger = CodeInterpreterInterceptionLogger(
        sandbox_config=FakeSandbox(), enabled_providers=["openai"]
    )
    response = FakeResponse(
        output=[_function_call_item(name=LITELLM_CODE_EXECUTION_TOOL_NAME)]
    )

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
async def test_pre_call_strips_client_forged_marker_on_initial_request():
    """A client cannot pre-set the active marker on the original request."""
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandbox())
    kwargs = {
        "tools": [{"type": "web_search"}],
        "custom_llm_provider": "openai",
        _ACTIVE_KEY: True,
    }

    await logger.async_pre_call_deployment_hook(kwargs, CallTypes.aresponses)

    assert _ACTIVE_KEY not in kwargs, (
        "no native code_interpreter tool was present, so a client-supplied "
        "active marker must be cleared"
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
        "the server-set marker must survive followup requests so multi-round "
        "code execution keeps working"
    )


@pytest.mark.asyncio
async def test_sandbox_deleted_after_loop_completes():
    sandbox = FakeSandbox(stdout="42")
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)

    plan = await _build_plan(logger, sandbox, call_id="k1")
    assert sandbox.create_calls, "sandbox must be created during the loop"
    assert (
        not sandbox.delete_calls
    ), "sandbox must outlive the loop until the final hook"

    await logger.async_post_agentic_loop_response_hook(
        response=FakeResponse(output=[{"type": "message", "content": []}]),
        plan=plan,
        kwargs={},
    )

    assert len(sandbox.delete_calls) == 1, (
        "the sandbox must be deleted once the final response is assembled, "
        "otherwise it keeps running and billing"
    )
    assert "k1" not in logger._container_cache_by_call_id


@pytest.mark.asyncio
async def test_post_hook_delete_is_idempotent_across_loop_levels():
    sandbox = FakeSandbox(stdout="42")
    logger = CodeInterpreterInterceptionLogger(sandbox_config=sandbox)
    plan = await _build_plan(logger, sandbox, call_id="k1")
    response = FakeResponse(output=[{"type": "message", "content": []}])

    await logger.async_post_agentic_loop_response_hook(
        response=response, plan=plan, kwargs={}
    )
    await logger.async_post_agentic_loop_response_hook(
        response=response, plan=plan, kwargs={}
    )

    assert len(sandbox.delete_calls) == 1, (
        "deleting an already-removed container must be a no-op so unwinding "
        "loop levels do not double-delete"
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
        container, params = await logger._get_or_create_container(call_id="k1")
        assert params is not None and params["sandbox_provider"] == "e2b"

        sandbox_tools.clear_sandbox_tools()

        stdout = await logger._run_tool_call(
            container=container, params=params, arguments='{"code":"print(1)"}'
        )
    finally:
        sandbox_tools.clear_sandbox_tools()

    assert stdout == "ok", "run must succeed using the params captured at create time"
    assert run_kwargs["provider"] == "e2b"
