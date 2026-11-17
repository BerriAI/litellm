from __future__ import annotations

from collections.abc import Coroutine
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Final

import pytest
from pydantic import BaseModel, ConfigDict

import litellm
from litellm import Agent
from litellm.harness import runtime, sync
from litellm.harness.errors import CapabilityUnsupported
from litellm.harness.options import ClaudeCodeOptions, ToolLoopOptions
from litellm.harness.sandbox import local
from litellm.harness.types import Approval, Harness, Result, ToolResult
from tests.unit.harness.handlers.test_tool_loop_handler import (
    ScriptedCompletion,
    function_call,
    model_response,
)


class Answer(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: int


def double(value: int) -> int:
    return value * 2


def deny(_: Approval) -> bool:
    return False


def test_should_define_an_agent_without_resolving_credentials_or_reading_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LITELLM_PROXY_API_BASE", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
    source_tools: Final = [double]  # mutable-ok: verify caller edits cannot change a reusable definition
    source_metadata: Final = {"workflow": "support"}  # mutable-ok: verify the definition snapshots caller metadata
    definition: Final = Agent(
        Harness.CLAUDE_CODE,
        "litellm_proxy/support",
        tools=source_tools,
        skills=(tmp_path / "not-created",),
        metadata=source_metadata,
        install=True,
    )
    source_tools.clear()
    source_metadata["workflow"] = "changed"

    assert definition.model == "litellm_proxy/support"
    assert definition.tools == (double,)
    assert definition.skills == (tmp_path / "not-created",)
    assert definition.metadata == {"workflow": "support"}
    assert not (tmp_path / "not-created").exists()
    with pytest.raises(FrozenInstanceError):
        setattr(definition, "instructions", "changed")


def test_should_hide_connection_credentials_and_runtime_environment_from_agent_repr() -> None:
    definition: Final = Agent(
        Harness.CLAUDE_CODE,
        api_key="example-credential",
        api_base="https://example.invalid/private",
        options=ClaudeCodeOptions(env={"EXAMPLE_CREDENTIAL": "example-environment-value"}),
    )

    assert "example-credential" not in repr(definition)
    assert "example-environment-value" not in repr(definition)
    assert "https://example.invalid/private" not in repr(definition)


def test_should_run_a_reusable_agent_with_its_instructions_tools_options_and_structured_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion: Final = ScriptedCompletion(
        (
            model_response(tool_calls=(function_call("double", '{"value": 3}'),)),
            model_response(content='{"value": 6}'),
        )
    )
    monkeypatch.setattr(litellm, "acompletion", completion)
    definition: Final = Agent(
        Harness.TOOL_LOOP,
        "openai/agent-fixture",
        instructions="Use double to calculate the answer.",
        tools=(double,),
        output=Answer,
        options=ToolLoopOptions(completion_kwargs={"temperature": 0}),
    )

    result: Final = sync.agent(definition, "Double three", sandbox=local(tmp_path))

    assert isinstance(result, Result)
    assert result.output == Answer(value=6)
    assert completion.calls[0]["messages"] == [
        {"role": "system", "content": "Use double to calculate the answer."},
        {"role": "user", "content": "Double three"},
    ]
    assert completion.calls[0]["response_format"] is Answer
    assert completion.calls[0]["temperature"] == 0
    assert tuple(event for event in result.events if isinstance(event, ToolResult)) == (
        ToolResult(id="call-1", output="6", is_error=False),
    )


async def test_should_preserve_an_agents_approval_policy_when_execution_arguments_are_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output: Final = tmp_path / "must-not-exist.txt"

    def write_file() -> str:
        output.write_text("executed")
        return "executed"

    completion: Final = ScriptedCompletion(
        (
            model_response(tool_calls=(function_call("write_file", "{}"),)),
            model_response(content="Request denied"),
        )
    )
    monkeypatch.setattr(litellm, "acompletion", completion)
    definition: Final = Agent(
        Harness.TOOL_LOOP,
        "openai/agent-fixture",
        tools=(write_file,),
        permissions="ask",
        on_approval=deny,
    )
    pending: Final = runtime.aagent(definition, "Write the file", sandbox=local(tmp_path))
    assert isinstance(pending, Coroutine)

    result: Final = await pending

    assert result.text == "Request denied"
    assert not output.exists()
    assert any(isinstance(event, ToolResult) and event.is_error for event in result.events)


async def test_should_apply_explicit_default_values_and_none_over_agent_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion: Final = ScriptedCompletion((model_response(content="plain answer"),))
    monkeypatch.setattr(litellm, "acompletion", completion)
    definition: Final = Agent(
        Harness.TOOL_LOOP,
        "openai/agent-fixture",
        instructions="Return a structured answer.",
        tools=(double,),
        permissions="ask",
        on_approval=deny,
        output=Answer,
        max_turns=1,
        timeout=5,
        metadata={"original": True},
        options=ToolLoopOptions(completion_kwargs={"temperature": 0}),
    )
    async with runtime.aagent_session(
        definition,
        sandbox=local(tmp_path),
        instructions=None,
        tools=(),
        permissions="full",
        on_approval=None,
        output=None,
        max_turns=None,
        timeout=None,
        metadata=None,
        options=None,
        install=False,
    ) as session:
        result: Final = await session.arun("Answer directly")
        assert session.config.max_turns is None
        assert session.config.timeout is None
        assert session.config.metadata == {}

    assert result.output is None
    assert result.text == "plain answer"
    assert completion.calls[0]["messages"] == [{"role": "user", "content": "Answer directly"}]
    assert "tools" not in completion.calls[0]
    assert "response_format" not in completion.calls[0]
    assert "temperature" not in completion.calls[0]
    assert definition.permissions == "ask"
    assert definition.tools == (double,)
    assert definition.metadata == {"original": True}


def test_should_allow_an_explicit_full_permission_override_to_execute_a_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output: Final = tmp_path / "approved.txt"

    def write_file() -> str:
        output.write_text("executed")
        return "executed"

    completion: Final = ScriptedCompletion(
        (
            model_response(tool_calls=(function_call("write_file", "{}"),)),
            model_response(content="File written"),
        )
    )
    monkeypatch.setattr(litellm, "acompletion", completion)
    definition: Final = Agent(
        Harness.TOOL_LOOP,
        "openai/agent-fixture",
        tools=(write_file,),
        permissions="ask",
        on_approval=deny,
    )

    result: Final = sync.agent(
        definition, "Write the file", sandbox=local(tmp_path), permissions="full", on_approval=None
    )

    assert isinstance(result, Result)
    assert result.text == "File written"
    assert output.read_text() == "executed"
    assert definition.permissions == "ask"
    assert definition.on_approval is deny


async def test_should_route_agent_model_calls_with_inherited_and_overridden_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion: Final = ScriptedCompletion((model_response(content="first"), model_response(content="second")))
    monkeypatch.setattr(litellm, "acompletion", completion)
    definition: Final = Agent(
        Harness.TOOL_LOOP,
        "litellm_proxy/support",
        api_base="https://original.example.com/",
        api_key="example-original-key",
    )
    async with runtime.aagent_session(definition, sandbox=local(tmp_path)) as inherited:
        assert (await inherited.arun("First")).text == "first"
    async with runtime.aagent_session(
        definition,
        sandbox=local(tmp_path),
        model="litellm_proxy/candidate",
        api_base="https://candidate.example.com/",
        api_key="example-candidate-key",
    ) as overridden:
        assert (await overridden.arun("Second")).text == "second"

    assert completion.calls[0]["model"] == "litellm_proxy/support"
    assert completion.calls[0]["api_base"] == "https://original.example.com"
    assert completion.calls[0]["api_key"] == "example-original-key"
    assert completion.calls[1]["model"] == "litellm_proxy/candidate"
    assert completion.calls[1]["api_base"] == "https://candidate.example.com"
    assert completion.calls[1]["api_key"] == "example-candidate-key"
    assert definition.model == "litellm_proxy/support"


def test_should_validate_inherited_skills_disabled_tools_and_install_policy(tmp_path: Path) -> None:
    skill: Final = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# Example skill")
    definitions: Final = (
        (Agent(Harness.TOOL_LOOP, "openai/agent-fixture", skills=(skill,)), "skills"),
        (Agent(Harness.TOOL_LOOP, "openai/agent-fixture", disable_tools=("shell",)), "disable_tools"),
        (Agent(Harness.TOOL_LOOP, "openai/agent-fixture", install=True), "install"),
    )

    for definition, error in definitions:
        with pytest.raises(CapabilityUnsupported, match=error):
            sync.agent_session(definition, sandbox=local(tmp_path))
