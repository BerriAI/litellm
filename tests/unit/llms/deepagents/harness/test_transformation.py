import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from litellm.harness.context import GatewayTarget, SessionContext
from litellm.harness.errors import OptionsMismatch
from litellm.harness.options import CodexOptions, DeepAgentsOptions
from litellm.harness.sandbox.local import LocalSandbox
from litellm.harness.types import Harness, Reasoning, Text, ToolCall, ToolResult
from litellm.llms.deepagents.harness import transformation as da


def make_ctx(tmp_path: Path, **kwargs: Any) -> SessionContext:
    base: dict[str, Any] = {
        "harness": Harness.DEEPAGENTS,
        "sandbox": LocalSandbox(tmp_path),
        "session_id": f"s-{os.urandom(4).hex()}",
        "model": "gpt-4o-mini",
    }
    return SessionContext(**{**base, **kwargs})


def msg(kind: str, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(type=kind, **fields)


def test_blocked_tools_modes() -> None:
    assert da.blocked_tools("full", []) == frozenset()
    assert da.blocked_tools("edit", []) == frozenset({"execute"})
    assert {"write_file", "edit_file", "delete", "execute"} <= da.blocked_tools(
        "read-only", []
    )
    assert da.blocked_tools("full", ["read", "ls"]) == frozenset({"read_file", "ls"})
    assert da.blocked_tools("full", ["bash", "grep"]) == frozenset({"execute", "grep"})


def test_interrupt_config_only_for_ask() -> None:
    assert da.interrupt_config("full", frozenset()) is None
    config = da.interrupt_config("ask", frozenset({"execute"}))
    assert set(config) == {"write_file", "edit_file", "delete"}
    assert all(
        v == {"allowed_decisions": ["approve", "reject"]} for v in config.values()
    )


def test_normalized_tool_name() -> None:
    assert da.normalized_tool_name("write_file") == "write"
    assert da.normalized_tool_name("read_file") == "read"
    assert da.normalized_tool_name("edit_file") == "edit"
    assert da.normalized_tool_name("execute") == "bash"
    assert da.normalized_tool_name("add") == "add"


def test_chat_model_kwargs_gateway_and_sdk(tmp_path: Path) -> None:
    gw = GatewayTarget(api_base="https://gw.example.com", api_key="sk-virtual")
    ctx = make_ctx(tmp_path, gateway=gw, metadata={"team": "a"})
    kwargs = da.chat_model_kwargs(ctx)
    assert kwargs["model"] == "litellm_proxy/gpt-4o-mini"
    assert kwargs["api_base"] == "https://gw.example.com"
    assert kwargs["api_key"] == "sk-virtual"
    assert kwargs["extra_headers"]["x-litellm-tags"] == "harness,deepagents"
    assert '"team": "a"' in kwargs["extra_headers"]["x-litellm-spend-logs-metadata"]
    no_meta = da.chat_model_kwargs(make_ctx(tmp_path, gateway=gw))
    assert "x-litellm-spend-logs-metadata" not in no_meta["extra_headers"]

    sdk = da.chat_model_kwargs(make_ctx(tmp_path, api_key="k", api_base="http://b"))
    assert sdk == {"model": "gpt-4o-mini", "api_key": "k", "api_base": "http://b"}
    with pytest.raises(ValueError, match="needs model="):
        da.chat_model_kwargs(make_ctx(tmp_path, model=None))


def test_recursion_limit(tmp_path: Path) -> None:
    assert (
        da.recursion_limit(make_ctx(tmp_path)) == da.DEEPAGENTS_DEFAULT_RECURSION_LIMIT
    )
    assert da.recursion_limit(make_ctx(tmp_path, max_turns=2)) == (
        da.DEEPAGENTS_BASE_RECURSION_LIMIT + 2 * da.DEEPAGENTS_STEPS_PER_TURN
    )
    opts = DeepAgentsOptions(recursion_limit=7)
    assert da.recursion_limit(make_ctx(tmp_path, max_turns=2, options=opts)) == 7


def test_stream_events_text_and_reasoning() -> None:
    assert da.stream_events(msg("human", content="hi")) == []
    events = da.stream_events(
        msg(
            "AIMessageChunk",
            content=[
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "a"},
                "b",
            ],
            additional_kwargs={},
        )
    )
    assert events == [Reasoning(delta="hmm"), Text(delta="ab")]
    extra = da.stream_events(
        msg("ai", content="x", additional_kwargs={"reasoning_content": "r"})
    )
    assert extra == [Reasoning(delta="r"), Text(delta="x")]


def test_update_events_tool_calls_results_and_skip() -> None:
    ai = msg(
        "ai",
        tool_calls=[
            {"name": "write_file", "args": {"file_path": "/a"}, "id": "c1"},
            {"name": "Answer", "args": {"city": "Paris"}, "id": "c2"},
            {"name": "add", "args": None, "id": "c3"},
        ],
    )
    tool = msg("tool", name="write_file", tool_call_id="c1", content="ok", status=None)
    err = msg("tool", name="execute", tool_call_id="c4", content="x", status="error")
    skipped = msg("tool", name="Answer", tool_call_id="c2", content="", status=None)
    update = {
        "model": {"messages": [ai]},
        "tools": {"messages": [tool, err, skipped]},
        "SomeMiddleware.after_model": {"messages": [ai]},
    }
    events = da.update_events(update, frozenset({"Answer"}))
    assert events == [
        ToolCall(
            id="c1",
            name="write",
            native_name="write_file",
            input={"file_path": "/a"},
            builtin=True,
        ),
        ToolCall(
            id="c3", name="add", native_name="add", input={"args": None}, builtin=False
        ),
        ToolResult(id="c1", output="ok", is_error=False),
        ToolResult(id="c4", output="x", is_error=True),
    ]
    assert da.update_events(None, frozenset()) == []
    assert da.update_events({"model": None}, frozenset()) == []


def test_interrupts_and_approval_requests() -> None:
    assert da.interrupts_in({"__interrupt__": ("i",)}) == ["i"]
    assert da.interrupts_in({}) == [] and da.interrupts_in(None) == []
    value = {"action_requests": [{"name": "write_file", "args": {}}, "junk"]}
    assert da.approval_requests(value) == [{"name": "write_file", "args": {}}]
    assert da.approval_requests(None) == []
    assert da.approval_requests({"action_requests": "x"}) == []


def test_decision() -> None:
    assert da.decision(True, "") == {"type": "approve"}
    assert da.decision(False, "no") == {"type": "reject", "message": "no"}
    assert da.decision(False, "")["message"]


class Answer(BaseModel):
    city: str


def test_final_ai_text_and_structured_json() -> None:
    messages = [
        msg("ai", content="first"),
        msg("tool", content="t"),
        msg("ai", content=""),
    ]
    assert da.final_ai_text(messages) == "first"
    assert da.final_ai_text([]) == ""
    assert da.structured_json(None) is None
    assert Answer.model_validate_json(da.structured_json(Answer(city="Paris")))
    assert da.structured_json({"city": "Paris"}) == '{"city": "Paris"}'


def test_config_capabilities_and_validation(tmp_path: Path) -> None:
    config = da.DeepAgentsHarnessConfig()
    assert config.uses_model_endpoint is False
    assert config.capabilities.tool_approval and config.capabilities.history
    assert "ask" in config.capabilities.permission_modes
    config.validate_environment(make_ctx(tmp_path))
    with pytest.raises(ValueError, match="needs model="):
        config.validate_environment(make_ctx(tmp_path, model=None))
    with pytest.raises(OptionsMismatch):
        config.validate_environment(make_ctx(tmp_path, options=CodexOptions()))
    assert "pip install deepagents langchain-litellm" in da.INSTALL_HINT
