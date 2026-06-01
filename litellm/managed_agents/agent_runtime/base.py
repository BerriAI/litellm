"""
AgentRuntime ABC — drives the LLM tool loop.

Implementations:

  * ``ClaudeSDKAgentRuntime`` — wraps Anthropic's ``claude-agent-sdk``
  * ``LiteLLMAgentRuntime`` — uses ``litellm.acompletion`` for multi-provider

Customers can subclass either to hook into ``before_tool_call`` /
``after_tool_call`` for audit, telemetry, or tool-result rewriting.

The runtime sees ``AgentConfig`` (the static agent definition: model,
system_prompt, tools_config) and ``SessionState`` (the live per-session
state: cwd hint, env_vars, repos), but never the DB row directly. That
keeps the runtime layer pure and testable without a Prisma client.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from litellm.managed_agents.events import Event
from litellm.managed_agents.sandbox.base import Sandbox


@dataclass
class AgentConfig:
    """Static agent config the runtime needs to drive an LLM tool loop.

    Mirrors the columns of ``LiteLLM_Agent`` that matter at runtime; the
    DB row stays in ``Agent`` (the public class), the runtime sees only
    this snapshot. Decoupling the two means a runtime can be unit tested
    with a literal ``AgentConfig(...)`` and no DB at all.
    """

    name: str
    model: str
    system_prompt: Optional[str] = None
    tools_config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
    """Live per-session state.

    ``cwd`` is the working directory for tool execution (provided by the
    sandbox). ``env_vars`` and ``repos`` are exposed so the runtime can
    seed its system prompt with context (e.g. "you have access to repos
    X, Y at /workspace/foo, /workspace/bar").
    """

    session_id: str
    cwd: Optional[str] = None
    env_vars: Dict[str, str] = field(default_factory=dict)
    repos: List[Dict[str, Any]] = field(default_factory=list)


class AgentRuntime(ABC):
    """Drives the LLM tool loop. Yields ``Event`` instances.

    The contract:

      * ``run(prompt, sandbox, session_state, agent_config)`` is an
        async iterator that yields events as the LLM produces them.
      * The runtime decides when to stop (typically on a terminal
        ``run_finished`` event from the LLM).
      * The runtime is responsible for routing tool calls through the
        provided ``sandbox`` (with the documented ClaudeSDKAgentRuntime
        exception — see its docstring).
      * ``before_tool_call`` and ``after_tool_call`` are optional hooks
        subclasses can override to inject behaviour without forking the
        whole runtime. Default implementations are no-ops.
    """

    @abstractmethod
    async def run(
        self,
        prompt: str,
        sandbox: Sandbox,
        session_state: SessionState,
        agent_config: AgentConfig,
    ) -> AsyncIterator[Event]:
        """Drive the LLM tool loop and yield events."""
        # The ``yield`` here makes this an async generator (so the ABC
        # signature matches subclasses). Subclasses MUST implement.
        if False:
            yield Event(type="placeholder", data={})
        raise NotImplementedError

    async def before_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Hook: called right before each tool execution.

        Return the (possibly rewritten) ``tool_input``. Subclasses can
        override to log, validate, or rewrite tool calls before they hit
        the sandbox. Default returns ``tool_input`` unchanged.

        Raise to abort the tool call entirely — the runtime should catch
        and surface the failure as a ``tool_result`` with ``is_error=True``.
        """
        return tool_input

    async def after_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        result: Any,
    ) -> Any:
        """Hook: called right after each tool execution.

        Return the (possibly rewritten) ``result``. Subclasses can
        override to log, redact, or rewrite tool outputs before they get
        shown to the LLM on its next turn. Default returns ``result``
        unchanged.
        """
        return result
