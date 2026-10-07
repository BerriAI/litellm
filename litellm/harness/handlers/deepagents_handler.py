"""
In-process handler for Deep Agents.

Deep Agents is a Python library, so there is no process or model endpoint: the handler
builds the agent with a LiteLLM chat model, streams the LangGraph run, turns interrupts into
Approval events and counts usage. Translation lives in
`litellm/llms/deepagents/harness/transformation.py`.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Any

from litellm.harness.context import SessionContext
from litellm.harness.errors import HarnessError, HarnessInstallFailed
from litellm.harness.handlers.base import BaseHarnessHandler
from litellm.harness.handlers.cli_handler import copy_skills
from litellm.harness.options import DeepAgentsOptions
from litellm.harness.types import Approval, Event
from litellm.llms.deepagents.harness.transformation import (
    EXECUTE_TOOLS,
    INSTALL_HINT,
    SKILLS_DIR,
    WRITE_TOOLS,
    TurnState,
    approval_requests,
    blocked_tools,
    chat_model_kwargs,
    decision,
    final_ai_text,
    interrupt_config,
    interrupts_in,
    normalized_tool_name,
    recursion_limit,
    stream_events,
    structured_json,
    update_events,
)

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.types import Command

    from litellm.llms.base_llm.harness.transformation import BaseHarnessConfig

_MODEL_NODE = "model"


@dataclass(frozen=True)
class DeepAgentsDeps:
    """The optional-dependency entrypoints this handler uses."""

    create_deep_agent: Callable[..., CompiledStateGraph]
    chat_litellm: Callable[..., BaseChatModel]
    checkpointer_cls: Callable[[], BaseCheckpointSaver]
    command_cls: Callable[..., Command]
    subagent_defaults: Mapping[str, object]
    convert_to_openai_messages: Any
    backend: ModuleType


def load_deps() -> DeepAgentsDeps:
    """Import deepagents + langchain-litellm, or raise HarnessInstallFailed."""
    try:
        deepagents = importlib.import_module("deepagents")
        subagents = importlib.import_module("deepagents.middleware.subagents")
        chat = importlib.import_module("langchain_litellm")
        memory = importlib.import_module("langgraph.checkpoint.memory")
        lg_types = importlib.import_module("langgraph.types")
        messages = importlib.import_module("langchain_core.messages")
        backend = importlib.import_module("litellm.llms.deepagents.harness.sandbox_backend")
    except ImportError as e:
        raise HarnessInstallFailed(f"{INSTALL_HINT} ({e})") from e
    return DeepAgentsDeps(
        create_deep_agent=deepagents.create_deep_agent,
        chat_litellm=chat.ChatLiteLLM,
        checkpointer_cls=memory.InMemorySaver,
        command_cls=lg_types.Command,
        subagent_defaults=subagents.GENERAL_PURPOSE_SUBAGENT,
        convert_to_openai_messages=messages.convert_to_openai_messages,
        backend=backend,
    )


_SHARED_CHECKPOINTER: dict[str, Any] = {}


def shared_checkpointer(deps: DeepAgentsDeps) -> BaseCheckpointSaver:
    """One in-memory checkpointer per process, so resume() works across sessions in-process."""
    saver = _SHARED_CHECKPOINTER.get("saver")
    if saver is None:
        saver = deps.checkpointer_cls()
        _SHARED_CHECKPOINTER["saver"] = saver
    return saver


def build_chat_model(ctx: SessionContext, deps: DeepAgentsDeps) -> BaseChatModel:
    """The LangChain chat model for this session. Tests monkeypatch this."""
    return deps.chat_litellm(**chat_model_kwargs(ctx))


class DeepAgentsHandler(BaseHarnessHandler):
    def __init__(self, config: BaseHarnessConfig) -> None:
        super().__init__(config)
        self._deps: DeepAgentsDeps | None = None
        self._agent: CompiledStateGraph | None = None
        self._thread_id: str | None = None
        self._skip_tools: frozenset[str] = frozenset()

    async def start(self, ctx: SessionContext) -> None:
        self.config.validate_environment(ctx)
        deps = load_deps()
        self._deps = deps
        blocked = blocked_tools(ctx.permissions, ctx.disable_tools)
        backend = deps.backend.SandboxBackend(
            ctx.sandbox,
            loop=asyncio.get_running_loop(),
            writable=WRITE_TOOLS.isdisjoint(blocked),
            allow_execute=EXECUTE_TOOLS.isdisjoint(blocked),
        )
        self._agent = deps.create_deep_agent(
            model=build_chat_model(ctx, deps),
            tools=list(ctx.tools),
            system_prompt=ctx.instructions,
            middleware=self._middleware(deps, blocked),
            subagents=self._subagents(ctx, deps, blocked),
            skills=await self._install_skills(ctx),
            backend=backend,
            interrupt_on=interrupt_config(ctx.permissions, blocked),
            response_format=ctx.output,
            checkpointer=shared_checkpointer(deps),
        )
        self._skip_tools = frozenset({ctx.output.__name__}) if ctx.output is not None else frozenset()
        if self._thread_id is None:
            self._thread_id = ctx.session_id

    async def stop(self, ctx: SessionContext) -> None:
        self._agent = None

    def native_session_id(self) -> str | None:
        return self._thread_id

    async def resume(self, ctx: SessionContext, native_session_id: str) -> None:
        self._thread_id = native_session_id

    async def history(
        self, ctx: SessionContext
    ) -> list[dict[str, Any]]:  # mutable-ok: BaseHarnessHandler.history API returns OpenAI message dicts
        agent, deps = self._require_agent()
        snapshot = await agent.aget_state(self._run_config(ctx, None))
        messages = (snapshot.values or MappingProxyType({})).get("messages") or ()
        converted: list[dict[str, Any]] = deps.convert_to_openai_messages(  # mutable-ok: LangChain returns a list
            messages
        )
        return converted

    async def turn(self, ctx: SessionContext, prompt: str) -> AsyncIterator[Event]:
        agent, deps = self._require_agent()
        run_config = self._run_config(ctx, deps.backend.UsageCallback(ctx, ctx.model))
        user_message = {"role": "user", "content": prompt}
        payload: dict[str, object] | Command = {"messages": [user_message]}
        while True:
            state = TurnState()
            async for event in self._stream_pass(agent, payload, run_config, state):
                yield event
            if not state.interrupts:
                break
            resume: dict[str, object] = {}
            for interrupt in state.interrupts:
                decisions: list[dict[str, Any]] = []  # mutable-ok: HITL decisions collected across awaited approvals
                for request in approval_requests(getattr(interrupt, "value", None)):
                    approval = Approval(
                        tool=normalized_tool_name(str(request.get("name") or "")),
                        input=dict(request.get("args") or ()),
                    )
                    yield approval
                    decisions.append(decision(*await approval.wait()))
                resume[interrupt.id] = {"decisions": decisions}
            payload = deps.command_cls(resume=resume)
        await self._finish_turn(ctx, agent, run_config)

    async def _stream_pass(
        self,
        agent: CompiledStateGraph,
        payload: dict[str, object] | Command,
        run_config: RunnableConfig,
        state: TurnState,
    ) -> AsyncIterator[Event]:
        async for part in agent.astream(
            payload,
            run_config,
            stream_mode=["messages", "updates"],
        ):
            # A list stream_mode yields (mode, chunk) tuples; LangGraph's overloads don't say so.
            if not isinstance(part, tuple) or len(part) != 2:
                continue
            mode, chunk = part
            if mode == "messages":
                message, meta = chunk
                if isinstance(meta, Mapping) and meta.get("langgraph_node") == _MODEL_NODE:
                    for event in stream_events(message):
                        yield event
            elif mode == "updates":
                state.interrupts = (*state.interrupts, *interrupts_in(chunk))
                for event in update_events(chunk, self._skip_tools):
                    yield event

    async def _finish_turn(self, ctx: SessionContext, agent: CompiledStateGraph, run_config: RunnableConfig) -> None:
        snapshot = await agent.aget_state(run_config)
        values = snapshot.values or MappingProxyType({})
        ctx.final_text = final_ai_text(values.get("messages") or ())
        if ctx.output is not None:
            ctx.output_json = structured_json(values.get("structured_response"))

    def _require_agent(self) -> tuple[CompiledStateGraph, DeepAgentsDeps]:
        if self._agent is None or self._deps is None:
            raise HarnessError("Deep Agents session is not started")
        return self._agent, self._deps

    def _run_config(self, ctx: SessionContext, usage_callback: BaseCallbackHandler | None) -> RunnableConfig:
        run_config: RunnableConfig = {
            "configurable": {"thread_id": self._thread_id or ctx.session_id},
            "recursion_limit": recursion_limit(ctx),
        }
        if usage_callback is not None:
            run_config["callbacks"] = [usage_callback]
        return run_config

    @staticmethod
    def _middleware(deps: DeepAgentsDeps, blocked: frozenset[str]) -> list[object]:  # mutable-ok: deepagents API
        filters = (deps.backend.ToolFilterMiddleware(blocked),) if blocked else ()
        return list(filters)

    def _subagents(
        self, ctx: SessionContext, deps: DeepAgentsDeps, blocked: frozenset[str]
    ) -> list[object]:  # mutable-ok: deepagents create_deep_agent(subagents=) takes a list
        """User subagents, plus a general-purpose one that honours disable_tools when set."""
        options = ctx.options if isinstance(ctx.options, DeepAgentsOptions) else None
        user_subagents = tuple(options.subagents) if options is not None else ()
        has_general = any(
            isinstance(s, Mapping) and s.get("name") == deps.subagent_defaults["name"] for s in user_subagents
        )
        spec = {**deps.subagent_defaults, "middleware": self._middleware(deps, blocked)}
        general = (spec,) if blocked and not has_general else ()
        return [*general, *user_subagents]

    @staticmethod
    async def _install_skills(ctx: SessionContext) -> list[str] | None:  # mutable-ok: deepagents skills= takes a list
        if not ctx.skills:
            return None
        await copy_skills(ctx.sandbox, ctx.skills, f"{ctx.sandbox.workdir}/{SKILLS_DIR}")
        return [f"/{SKILLS_DIR}/"]
