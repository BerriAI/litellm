"""
Tests for the provider-agnostic chat completion agentic loop dispatcher
(`litellm/litellm_core_utils/chat_completion_agentic_loop.py`) and the
code-interpreter interception integration that drives it.

The load-bearing regression here protects a reviewer requirement: the internal
agentic/interception control fields must NEVER reach the outbound provider HTTP
request body. The relevant fields are:

    _agentic_loop_depth
    _agentic_loop_fingerprints
    _agentic_loop_api_surface
    max_agentic_loops
    _code_interpreter_interception_active
    _code_interpreter_interception_sandbox_key
    _code_interpreter_interception_converted_stream

A scrubber in gpt_transformation.py used to strip these. That scrubber was
removed, so `test_internal_control_fields_never_leak_into_provider_body` proves
they stay out of the body even without it.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.code_interpreter_interception.handler import (
    CodeInterpreterInterceptionLogger,
)
from litellm.litellm_core_utils.chat_completion_agentic_loop import (
    maybe_run_chat_completion_agentic_loop,
)
from litellm.types.integrations.custom_logger import (
    AgenticLoopPlan,
    AgenticLoopRequestPatch,
)
from litellm.types.utils import (
    Choices,
    Function,
    ChatCompletionMessageToolCall,
    Message,
    ModelResponse,
)

# The internal control fields that must never reach a provider request body.
_INTERNAL_CONTROL_FIELDS = (
    "_agentic_loop_depth",
    "_agentic_loop_fingerprints",
    "_agentic_loop_api_surface",
    "max_agentic_loops",
    "_code_interpreter_interception_active",
    "_code_interpreter_interception_sandbox_key",
    "_code_interpreter_interception_converted_stream",
    "litellm_metadata",
)


@pytest.fixture
def restore_callbacks():
    """Save/restore litellm.callbacks so a registered fake logger never pollutes
    other tests in the suite."""
    saved = list(litellm.callbacks)
    try:
        yield
    finally:
        litellm.callbacks = saved


class _SandboxResult:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.error = None


class FakeSandboxConfig:
    """Injected sandbox so the interception loop runs no real network / E2B."""

    def __init__(self) -> None:
        self.created = 0
        self.deleted = 0
        self.run_codes: List[str] = []

    async def acreate_sandbox(self) -> Any:
        self.created += 1
        return MagicMock(id="sandbox-123")

    async def arun_code(self, container: Any, code: str) -> _SandboxResult:
        self.run_codes.append(code)
        return _SandboxResult(stdout="42\n")

    async def adelete_sandbox(self, container: Any) -> None:
        self.deleted += 1


def _tool_call_model_response() -> ModelResponse:
    return ModelResponse(
        choices=[
            Choices(
                finish_reason="tool_calls",
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_abc",
                            type="function",
                            function=Function(
                                name="litellm_code_execution",
                                arguments='{"code": "print(6*7)"}',
                            ),
                        )
                    ],
                ),
            )
        ]
    )


def _plain_model_response(content: str = "The answer is 42") -> ModelResponse:
    return ModelResponse(
        choices=[
            Choices(
                finish_reason="stop",
                message=Message(role="assistant", content=content),
            )
        ]
    )


def _raw_response_for(model_response: ModelResponse) -> MagicMock:
    """Wrap a ModelResponse as the OpenAI `with_raw_response.create` return value
    (an object exposing `.headers` and `.parse()` -> something with model_dump)."""
    parsed = MagicMock()
    parsed.model_dump.return_value = model_response.model_dump()
    raw = MagicMock()
    raw.headers = {}
    raw.parse.return_value = parsed
    return raw


# ---------------------------------------------------------------------------
# A) PROVIDER-PAYLOAD REGRESSION
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_internal_control_fields_never_leak_into_provider_body(restore_callbacks):
    """Drive a real acompletion with a native code_interpreter tool through the
    interception logger + agentic loop, capturing every outbound OpenAI request
    body. None of the internal control fields may appear at top-level or inside
    extra_body on ANY of the captured calls."""
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandboxConfig())
    litellm.callbacks = [logger]

    # First create -> model emits a code_execution tool call (triggers the loop).
    # Second create -> model returns a plain answer (loop terminates).
    create = AsyncMock(
        side_effect=[
            _raw_response_for(_tool_call_model_response()),
            _raw_response_for(_plain_model_response()),
        ]
    )
    mock_client = MagicMock()
    mock_client.chat.completions.with_raw_response.create = create

    response = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "what is 6*7?"}],
        tools=[{"type": "code_interpreter"}],
        tool_choice={"type": "code_interpreter"},
        api_key="sk-test",
        client=mock_client,
    )

    # The loop must have actually fired (sanity: two provider calls).
    assert create.await_count == 2, (
        f"expected the agentic loop to issue a follow-up provider call; got {create.await_count} call(s)"
    )

    for idx, call in enumerate(create.await_args_list):
        body = call.kwargs
        extra_body = body.get("extra_body") or {}
        for field in _INTERNAL_CONTROL_FIELDS:
            assert field not in body, (
                f"provider call #{idx}: internal field {field!r} leaked into "
                f"top-level request body: {sorted(body.keys())}"
            )
            assert field not in extra_body, (
                f"provider call #{idx}: internal field {field!r} leaked into extra_body: {sorted(extra_body.keys())}"
            )
        # The native code_interpreter tool must have been swapped for the
        # function tool, never sent raw to OpenAI as a chat-completions request.
        for tool in body.get("tools") or []:
            assert tool.get("type") != "code_interpreter"

    # The final response is the post-loop answer, not the tool-call turn.
    assert response.choices[0].message.content == "The answer is 42"


# ---------------------------------------------------------------------------
# B) DISPATCHER UNIT TESTS
# ---------------------------------------------------------------------------


class _LoggingStub:
    """Minimal logging_obj: dispatcher only reads dynamic_success_callbacks and
    litellm_call_id off it."""

    litellm_call_id = "call-test"
    dynamic_success_callbacks: List[Any] = []


class _GateOnlyLogger(CustomLogger):
    """Overrides the gate to fire, but builds a plan from request_patch."""

    def __init__(self, plan: AgenticLoopPlan, tool_calls: Dict[str, Any]) -> None:
        super().__init__()
        self._plan = plan
        self._tool_calls = tool_calls
        self.cleanup_calls = 0

    async def async_should_run_agentic_loop(
        self,
        response: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        stream: bool,
        custom_llm_provider: str,
        kwargs: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        return True, self._tool_calls

    async def async_build_agentic_loop_plan(
        self,
        tools: Dict[str, Any],
        model: str,
        messages: List[Dict[str, Any]],
        response: Any,
        anthropic_messages_provider_config: Any,
        anthropic_messages_optional_request_params: Dict[str, Any],
        logging_obj: Any,
        stream: bool,
        kwargs: Dict[str, Any],
    ) -> AgenticLoopPlan:
        return self._plan

    async def async_agentic_loop_cleanup_hook(self, plan: AgenticLoopPlan, kwargs: Dict[str, Any]) -> None:
        self.cleanup_calls += 1


def _patched_messages() -> List[Dict[str, Any]]:
    return [
        {"role": "user", "content": "what is 6*7?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "litellm_code_execution",
                        "arguments": '{"code": "print(6*7)"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_abc", "content": "42\n"},
    ]


@pytest.mark.asyncio
async def test_dispatcher_returns_none_when_no_callback_gates(restore_callbacks):
    """No callback overrides the gate -> dispatcher returns None so the caller
    keeps the original response untouched."""
    litellm.callbacks = []

    result = await maybe_run_chat_completion_agentic_loop(
        response=_plain_model_response(),
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        kwargs={},
        logging_obj=_LoggingStub(),
        custom_llm_provider="openai",
        stream=False,
    )

    assert result is None


@pytest.mark.asyncio
async def test_dispatcher_runs_followup_with_incremented_depth_and_patched_messages(
    restore_callbacks,
):
    """A gating logger with a request_patch -> the dispatcher calls
    litellm.acompletion exactly once with _agentic_loop_depth == 1 and the
    patched messages. Loop-control state rides as litellm-level kwargs and is
    mirrored into litellm_metadata; the provider-surface transient
    _agentic_loop_api_surface is never forwarded. (Provider-body stripping of
    these litellm-level kwargs is asserted separately in test A.)"""
    followup = _plain_model_response("done")
    plan = AgenticLoopPlan(
        run_agentic_loop=True,
        request_patch=AgenticLoopRequestPatch(messages=_patched_messages()),
    )
    logger = _GateOnlyLogger(plan=plan, tool_calls={"tool_calls": [{"id": "call_abc"}]})
    litellm.callbacks = [logger]

    acompletion_mock = AsyncMock(return_value=followup)
    with patch.object(litellm, "acompletion", acompletion_mock):
        result = await maybe_run_chat_completion_agentic_loop(
            response=_tool_call_model_response(),
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "what is 6*7?"}],
            optional_params={"temperature": 0.1},
            kwargs={"_code_interpreter_interception_active": True},
            logging_obj=_LoggingStub(),
            custom_llm_provider="openai",
            stream=False,
        )

    assert result is followup
    acompletion_mock.assert_awaited_once()
    call_kwargs = acompletion_mock.await_args.kwargs

    assert call_kwargs["_agentic_loop_depth"] == 1
    assert call_kwargs["messages"] == _patched_messages()
    # Preserved non-internal optional param survives the rerun.
    assert call_kwargs["temperature"] == 0.1
    # Loop-control state is carried at the litellm level for the follow-up.
    assert call_kwargs["max_agentic_loops"] >= 1
    assert "_agentic_loop_fingerprints" in call_kwargs
    # Interception markers are mirrored into litellm_metadata for the follow-up.
    assert call_kwargs["litellm_metadata"]["_code_interpreter_interception_active"] is True
    # The transient surface marker is NOT forwarded to the follow-up call.
    assert "_agentic_loop_api_surface" not in call_kwargs
    # Cleanup hook always runs.
    assert logger.cleanup_calls == 1


@pytest.mark.asyncio
async def test_dispatcher_raises_when_depth_reaches_max_agentic_loops(
    restore_callbacks,
):
    """depth >= max_agentic_loops -> ValueError mentioning max_agentic_loops,
    before any follow-up call is attempted."""
    logger = _GateOnlyLogger(
        plan=AgenticLoopPlan(run_agentic_loop=True),
        tool_calls={"tool_calls": [{"id": "call_abc"}]},
    )
    litellm.callbacks = [logger]

    acompletion_mock = AsyncMock()
    with patch.object(litellm, "acompletion", acompletion_mock):
        with pytest.raises(ValueError, match="max_agentic_loops"):
            await maybe_run_chat_completion_agentic_loop(
                response=_tool_call_model_response(),
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                optional_params={},
                kwargs={"_agentic_loop_depth": 3, "max_agentic_loops": 3},
                logging_obj=_LoggingStub(),
                custom_llm_provider="openai",
                stream=False,
            )

    acompletion_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatcher_raises_on_repeated_tool_call_fingerprint(restore_callbacks):
    """A tool_calls fingerprint already present in _agentic_loop_fingerprints ->
    ValueError about the repeated fingerprint (cycle guard), with no follow-up
    call."""
    import json

    # The dispatcher fingerprints the whole value the gate returns as its second
    # tuple element, so the seeded fingerprint must mirror that dict exactly.
    gate_tool_calls = {"tool_calls": [{"id": "call_abc", "name": "litellm_code_execution"}]}
    fingerprint = json.dumps(gate_tool_calls, sort_keys=True, default=str)

    logger = _GateOnlyLogger(
        plan=AgenticLoopPlan(run_agentic_loop=True),
        tool_calls=gate_tool_calls,
    )
    litellm.callbacks = [logger]

    acompletion_mock = AsyncMock()
    with patch.object(litellm, "acompletion", acompletion_mock):
        with pytest.raises(ValueError, match="fingerprint"):
            await maybe_run_chat_completion_agentic_loop(
                response=_tool_call_model_response(),
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                optional_params={},
                kwargs={
                    "_agentic_loop_depth": 0,
                    "max_agentic_loops": 3,
                    "_agentic_loop_fingerprints": [fingerprint],
                },
                logging_obj=_LoggingStub(),
                custom_llm_provider="openai",
                stream=False,
            )

    acompletion_mock.assert_not_awaited()
