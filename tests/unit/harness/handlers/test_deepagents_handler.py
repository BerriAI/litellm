import asyncio
import builtins
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from litellm.harness.context import GatewayTarget, SessionContext
from litellm.harness.errors import HarnessError, HarnessInstallFailed
from litellm.harness.handlers import deepagents_handler as dh
from litellm.harness.sandbox.local import LocalSandbox
from litellm.harness.types import Approval, Harness, Text, ToolCall, ToolResult
from litellm.llms.deepagents.harness.transformation import DeepAgentsHarnessConfig

pytest.importorskip("deepagents")
pytest.importorskip("langchain_litellm")

from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage  # noqa: E402

from litellm.llms.deepagents.harness.sandbox_backend import (  # noqa: E402
    SandboxBackend,
    message_cost,
)

USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


class FakeToolModel(FakeMessagesListChatModel):
    """Canned responses; records the tool names bound on each call."""

    bound: list = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeToolModel":
        names = [getattr(t, "name", None) or t.get("name") for t in tools]
        self.bound.append(sorted(n for n in names if n))
        return self


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
        usage_metadata=USAGE,
    )


def final(text: str) -> AIMessage:
    return AIMessage(content=text, usage_metadata=USAGE)


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch):
    def install(responses: list) -> FakeToolModel:
        model = FakeToolModel(responses=responses, bound=[])
        monkeypatch.setattr(dh, "build_chat_model", lambda ctx, deps: model)
        return model

    return install


def make_ctx(tmp_path: Path, **kwargs: Any) -> SessionContext:
    base: dict[str, Any] = {
        "harness": Harness.DEEPAGENTS,
        "sandbox": LocalSandbox(tmp_path),
        "session_id": f"s-{os.urandom(4).hex()}",
        "model": "gpt-4o-mini",
    }
    return SessionContext(**{**base, **kwargs})


def make_handler() -> dh.DeepAgentsHandler:
    return dh.DeepAgentsHandler(DeepAgentsHarnessConfig())


async def started(ctx: SessionContext) -> dh.DeepAgentsHandler:
    handler = make_handler()
    await handler.start(ctx)
    return handler


async def run_turn(
    handler: dh.DeepAgentsHandler,
    ctx: SessionContext,
    prompt: str,
    approve: bool = True,
) -> list:
    events = []
    async for event in handler.turn(ctx, prompt):
        events.append(event)
        if isinstance(event, Approval):
            event.allow() if approve else event.deny("no")
    return events


async def test_write_then_read_events_and_file(tmp_path: Path, fake_model) -> None:
    fake_model(
        [
            tool_call("write_file", {"file_path": "/hello.txt", "content": "hi"}, "c1"),
            tool_call("read_file", {"file_path": "/hello.txt"}, "c2"),
            final("done"),
        ]
    )
    ctx = make_ctx(tmp_path)
    handler = await started(ctx)
    events = await run_turn(handler, ctx, "write hello.txt with hi")

    calls = [e for e in events if isinstance(e, ToolCall)]
    results = [e for e in events if isinstance(e, ToolResult)]
    assert [(c.name, c.native_name, c.builtin) for c in calls] == [
        ("write", "write_file", True),
        ("read", "read_file", True),
    ]
    assert [r.id for r in results] == ["c1", "c2"]
    assert "hi" in results[1].output
    assert not any(r.is_error for r in results)
    assert "done" in "".join(e.delta for e in events if isinstance(e, Text))
    assert (tmp_path / "hello.txt").read_text() == "hi"
    assert ctx.final_text == "done"
    assert (ctx.input_tokens, ctx.output_tokens, ctx.calls) == (30, 15, 3)
    assert ctx.cost > 0
    history = await handler.history(ctx)
    assert history[0] == {"role": "user", "content": "write hello.txt with hi"}
    assert history[-1]["content"] == "done"


async def test_read_only_hides_write_tools(tmp_path: Path, fake_model) -> None:
    model = fake_model(
        [
            tool_call("write_file", {"file_path": "/x.txt", "content": "no"}, "c1"),
            final("ok"),
        ]
    )
    ctx = make_ctx(tmp_path, permissions="read-only")
    handler = await started(ctx)
    events = await run_turn(handler, ctx, "try to write")

    first = model.bound[0]
    assert "read_file" in first and "ls" in first
    assert not {"write_file", "edit_file", "execute", "delete"} & set(first)
    result = next(e for e in events if isinstance(e, ToolResult))
    assert result.is_error
    assert not (tmp_path / "x.txt").exists()


async def test_disable_tools_uses_normalized_names(tmp_path: Path, fake_model) -> None:
    model = fake_model([final("ok")])
    ctx = make_ctx(tmp_path, disable_tools=["bash", "grep"])
    handler = await started(ctx)
    await run_turn(handler, ctx, "hi")
    assert "execute" not in model.bound[0] and "grep" not in model.bound[0]
    assert "write_file" in model.bound[0]


class Answer(BaseModel):
    city: str


async def test_structured_output(tmp_path: Path, fake_model) -> None:
    fake_model([tool_call("Answer", {"city": "Paris"}, "c1")])
    ctx = make_ctx(tmp_path, output=Answer)
    handler = await started(ctx)
    events = await run_turn(handler, ctx, "capital of France?")
    assert Answer.model_validate_json(ctx.output_json or "") == Answer(city="Paris")
    assert not any(isinstance(e, ToolCall) for e in events)


async def test_custom_tool(tmp_path: Path, fake_model) -> None:
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    fake_model([tool_call("add", {"a": 2, "b": 3}, "c1"), final("5")])
    ctx = make_ctx(tmp_path, tools=[add])
    handler = await started(ctx)
    events = await run_turn(handler, ctx, "2+3")
    call = next(e for e in events if isinstance(e, ToolCall))
    assert (call.name, call.builtin) == ("add", False)
    assert next(e for e in events if isinstance(e, ToolResult)).output == "5"


@pytest.mark.parametrize("approve", [True, False])
async def test_ask_permissions_emit_approval(
    tmp_path: Path, fake_model, approve: bool
) -> None:
    fake_model(
        [
            tool_call("write_file", {"file_path": "/a.txt", "content": "x"}, "c1"),
            final("end"),
        ]
    )
    ctx = make_ctx(tmp_path, permissions="ask")
    handler = await started(ctx)
    events = await run_turn(handler, ctx, "write a", approve=approve)
    approval = next(e for e in events if isinstance(e, Approval))
    assert approval.tool == "write"
    assert approval.input["file_path"] == "/a.txt"
    assert (tmp_path / "a.txt").exists() is approve
    assert ctx.final_text == "end"


async def test_edit_and_execute_through_sandbox(tmp_path: Path, fake_model) -> None:
    (tmp_path / "f.txt").write_text("one two\n")
    fake_model(
        [
            tool_call(
                "edit_file",
                {"file_path": "/f.txt", "old_string": "two", "new_string": "three"},
                "c1",
            ),
            tool_call("execute", {"command": "cat f.txt"}, "c2"),
            final("ok"),
        ]
    )
    ctx = make_ctx(tmp_path)
    handler = await started(ctx)
    events = await run_turn(handler, ctx, "edit")
    calls = [e.name for e in events if isinstance(e, ToolCall)]
    assert calls == ["edit", "bash"]
    results = [e for e in events if isinstance(e, ToolResult)]
    assert "one three" in results[1].output
    assert (tmp_path / "f.txt").read_text() == "one three\n"


async def test_resume_keeps_thread(tmp_path: Path, fake_model) -> None:
    fake_model([final("first"), final("second")])
    ctx = make_ctx(tmp_path)
    handler = await started(ctx)
    await run_turn(handler, ctx, "one")
    native = handler.native_session_id()
    assert native == ctx.session_id

    other = make_handler()
    ctx2 = make_ctx(tmp_path)
    await other.start(ctx2)
    await other.resume(ctx2, native or "")
    await run_turn(other, ctx2, "two")
    history = await other.history(ctx2)
    assert [m["content"] for m in history if m["role"] == "user"] == ["one", "two"]


async def test_skills_copied_and_loaded(tmp_path: Path, fake_model) -> None:
    skill = tmp_path / "src-skills" / "greeter"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: greeter\ndescription: Says hi\n---\nSay hi.\n"
    )
    work = tmp_path / "work"
    work.mkdir()
    fake_model([final("ok")])
    ctx = make_ctx(work, skills=[str(skill)])
    handler = await started(ctx)
    await run_turn(handler, ctx, "hi")
    assert (work / ".deepagents" / "skills" / "greeter" / "SKILL.md").exists()


async def test_turn_and_history_before_start_and_after_stop(
    tmp_path: Path, fake_model
) -> None:
    fake_model([final("ok")])
    ctx = make_ctx(tmp_path)
    handler = make_handler()
    with pytest.raises(HarnessError, match="not started"):
        await run_turn(handler, ctx, "hi")
    await handler.start(ctx)
    await handler.stop(ctx)
    with pytest.raises(HarnessError, match="not started"):
        await handler.history(ctx)


async def test_start_validates_model(tmp_path: Path, fake_model) -> None:
    fake_model([final("ok")])
    with pytest.raises(ValueError):
        await started(make_ctx(tmp_path, model=None))


def test_build_chat_model_uses_chat_model_kwargs(tmp_path: Path) -> None:
    deps = dh.load_deps()
    gw = GatewayTarget(api_base="https://gw.example.com", api_key="sk-virtual")
    model = dh.build_chat_model(make_ctx(tmp_path, gateway=gw), deps)
    assert isinstance(model, deps.chat_litellm)
    assert model.model == "litellm_proxy/gpt-4o-mini"


def test_shared_checkpointer_is_process_wide() -> None:
    deps = dh.load_deps()
    assert dh.shared_checkpointer(deps) is dh.shared_checkpointer(deps)


async def test_sandbox_backend_fs_ops(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("print('hello')\n")
    (tmp_path / "b.txt").write_text("hello world\n")
    backend = SandboxBackend(LocalSandbox(tmp_path), loop=asyncio.get_running_loop())

    ls = await backend.als("/")
    assert {e["path"] for e in ls.entries or []} == {"/b.txt", "/src/"}
    assert (await backend.als("/missing")).error
    globbed = await backend.aglob("*.py")
    assert [m["path"] for m in globbed.matches or []] == ["/src/a.py"]
    grep = await backend.agrep("hello", glob="*.txt")
    assert [(m["path"], m["line"]) for m in grep.matches or []] == [("/b.txt", 1)]
    read = await backend.aread("/b.txt")
    assert read.file_data and read.file_data["content"] == "hello world\n"
    assert (await backend.aread("/nope.txt")).error
    assert (await backend.aread("/../etc/passwd")).error
    edit = await backend.aedit("/b.txt", "hello", "bye")
    assert edit.occurrences == 1
    assert (await backend.aedit("/b.txt", "zzz", "q")).error
    assert (await backend.adelete("/src")).path == "/src"
    assert not (tmp_path / "src").exists()
    assert (
        backend.to_real(str(tmp_path / "b.txt"))
        == str(LocalSandbox(tmp_path).workdir) + "/b.txt"
    )
    sync_ls = await asyncio.to_thread(backend.ls, "/")
    assert [e["path"] for e in sync_ls.entries or []] == ["/b.txt"]

    read_only = SandboxBackend(
        LocalSandbox(tmp_path),
        loop=asyncio.get_running_loop(),
        writable=False,
        allow_execute=False,
    )
    assert (await read_only.awrite("/c.txt", "x")).error
    assert (await read_only.aexecute("ls")).exit_code == 1
    assert not (tmp_path / "c.txt").exists()


def test_message_cost_prefers_reported_and_never_raises() -> None:
    reported = AIMessage(content="", response_metadata={"response_cost": 0.5})
    assert message_cost(reported, "gpt-4o-mini", 1, 1) == 0.5
    assert message_cost(AIMessage(content=""), "not-a-real-model-xyz", 10, 10) == 0.0
    assert message_cost(AIMessage(content=""), "gpt-4o-mini", 1000, 1000) > 0


def test_missing_deps_raise_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("deepagents"):
            raise ImportError("No module named 'deepagents'")
        return real_import(name, *args, **kwargs)

    for mod in [m for m in sys.modules if m.startswith("deepagents")]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(
        HarnessInstallFailed, match="pip install deepagents langchain-litellm"
    ):
        dh.load_deps()


LIVE_BASE = os.environ.get("LITELLM_PROXY_API_BASE", "")
LIVE_KEY = os.environ.get("LITELLM_PROXY_API_KEY", "")


@pytest.mark.skipif(
    not (LIVE_BASE and LIVE_KEY), reason="LITELLM_PROXY_API_BASE / KEY not set"
)
async def test_live_gateway_write_file(tmp_path: Path) -> None:
    model = os.environ.get("HARNESS_DEEPAGENTS_LIVE_MODEL", "claude-haiku-4-5-20251001")
    ctx = make_ctx(
        tmp_path,
        model=model,
        gateway=GatewayTarget(api_base=LIVE_BASE, api_key=LIVE_KEY),
        max_turns=6,
    )
    handler = await started(ctx)
    events = await run_turn(handler, ctx, "write hello.txt with hi")
    assert any(isinstance(e, ToolCall) and e.name == "write" for e in events)
    assert (tmp_path / "hello.txt").read_text().strip() == "hi"
    assert ctx.calls >= 1 and ctx.input_tokens > 0
