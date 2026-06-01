"""
``litellm.managed_agents`` — Python SDK for spawning and driving managed agents.

Public surface:

  * ``Agent``                — handle for one ``LiteLLM_Agent`` row;
                               spawns sessions.
  * ``Session``              — one running session; ``send(prompt)``
                               drives the runtime and yields events.
  * ``Run``                  — read-only view of one ``LiteLLM_AgentRun``
                               row; ``stream(starting_seq=N)`` replays
                               persisted events.
  * ``Event``                — the dataclass yielded by both runtimes
                               and ``Run.stream``.
  * ``AgentRuntime`` (+ subclasses ``ClaudeSDKAgentRuntime``,
    ``LiteLLMAgentRuntime``) — the LLM tool-loop driver. Subclass to
    add ``before_tool_call`` / ``after_tool_call`` hooks.
  * ``Sandbox`` (+ subclasses ``LocalSandbox``, ``EC2SandboxViaSSM``)
    — where tool calls actually execute.

Typical use::

    from litellm.managed_agents import (
        Agent, LiteLLMAgentRuntime, LocalSandbox,
    )

    agent = await Agent.from_db_row(
        row, db=prisma_client.db,
        runtime=LiteLLMAgentRuntime(),
        sandbox=LocalSandbox(),
    )
    session = await agent.create_session()
    async for event in session.send("hello"):
        print(event.type, event.data)
"""

from litellm.managed_agents.agent import Agent
from litellm.managed_agents.agent_runtime import (
    AgentConfig,
    AgentRuntime,
    ClaudeSDKAgentRuntime,
    LiteLLMAgentRuntime,
    SessionState,
)
from litellm.managed_agents.events import Event
from litellm.managed_agents.run import Run
from litellm.managed_agents.sandbox import (
    EC2SandboxViaSSM,
    LocalSandbox,
    Sandbox,
    ToolResult,
)
from litellm.managed_agents.session import Session

__all__ = [
    "Agent",
    "Session",
    "Run",
    "Event",
    "AgentRuntime",
    "AgentConfig",
    "SessionState",
    "ClaudeSDKAgentRuntime",
    "LiteLLMAgentRuntime",
    "Sandbox",
    "ToolResult",
    "LocalSandbox",
    "EC2SandboxViaSSM",
]
