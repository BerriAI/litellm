"""
ClaudeSDKAgentRuntime — wraps Anthropic's ``claude-agent-sdk``.

This is the default runtime. It delegates the full LLM tool loop (model
choice, tool calling, hooks, MCP, sub-agents) to ``claude-agent-sdk`` and
just translates the SDK's message stream into our ``Event`` shape.

Sandbox interop note (read this before assuming things)
-------------------------------------------------------

``claude-agent-sdk`` runs its built-in tools (``Read``, ``Write``,
``Edit``, ``Bash``, etc.) IN-PROCESS, in its own subprocess. Routing
those through our ``Sandbox`` abstraction would require overriding each
built-in with a custom MCP-backed tool that calls into the sandbox.

For v1 we explicitly do NOT do that: ``ClaudeSDKAgentRuntime`` works
correctly only with ``LocalSandbox`` (the SDK runs locally, sandbox-side
state == process-side state). For remote-execution agents that need to
run on EC2 / Docker, callers should use ``LiteLLMAgentRuntime`` instead,
which fully honors the ``Sandbox`` abstraction. The MCP-backed override
is tracked as a follow-up.

We do still consume ``sandbox.cwd`` though — ``ClaudeAgentOptions(cwd=...)``
controls where the SDK runs its built-in tools, so a ``LocalSandbox``
configured with ``working_dir="/tmp/foo"`` will have the SDK do its
work there. That's enough to support "create file foo.txt" tests.
"""

from typing import Any, AsyncIterator, Dict, Optional

from litellm.managed_agents.agent_runtime.base import (
    AgentConfig,
    AgentRuntime,
    SessionState,
)
from litellm.managed_agents.events import (
    EVENT_TYPE_ASSISTANT_MESSAGE,
    EVENT_TYPE_RUN_FINISHED,
    EVENT_TYPE_SYSTEM,
    EVENT_TYPE_THINKING,
    EVENT_TYPE_TOOL_RESULT,
    EVENT_TYPE_TOOL_USE,
    Event,
)
from litellm.managed_agents.sandbox.base import Sandbox


class ClaudeSDKAgentRuntime(AgentRuntime):
    """Default runtime — wraps ``claude-agent-sdk``.

    Construct with optional overrides; per-run values fall back to the
    ``AgentConfig`` passed to ``run()``. This split lets callers either
    (a) bake everything into the runtime instance and reuse it across
    agents, or (b) construct a fresh runtime per agent that defers
    everything to the agent config.

    ``permission_mode`` defaults to ``"bypassPermissions"`` so the SDK
    doesn't prompt for tool approval — agents are typically invoked
    headlessly. Override to ``"default"`` for interactive flows.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        permission_mode: str = "bypassPermissions",
        max_turns: Optional[int] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.permission_mode = permission_mode
        self.max_turns = max_turns
        self.extra_options = dict(extra_options or {})

    def _build_options(
        self,
        sandbox: Sandbox,
        session_state: SessionState,
        agent_config: AgentConfig,
    ):
        """Translate (config + state + sandbox) into a ClaudeAgentOptions."""
        from claude_agent_sdk import ClaudeAgentOptions

        opts: Dict[str, Any] = {
            "model": self.model or agent_config.model,
            "system_prompt": self.system_prompt or agent_config.system_prompt,
            "permission_mode": self.permission_mode,
        }
        if self.max_turns is not None:
            opts["max_turns"] = self.max_turns

        cwd = sandbox.cwd or session_state.cwd
        if cwd is not None:
            opts["cwd"] = cwd

        if session_state.env_vars:
            opts["env"] = dict(session_state.env_vars)

        # Drop any unset / unsupported keys before constructing.
        opts = {k: v for k, v in opts.items() if v is not None}

        # Caller can layer on raw SDK options too (e.g. ``allowed_tools``,
        # ``hooks``, ``mcp_servers``). These win over our derived defaults.
        opts.update(self.extra_options)

        return ClaudeAgentOptions(**opts)

    async def run(
        self,
        prompt: str,
        sandbox: Sandbox,
        session_state: SessionState,
        agent_config: AgentConfig,
    ) -> AsyncIterator[Event]:
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
            query,
        )

        # Make sure the sandbox is ready (creates tmpdir for LocalSandbox).
        await sandbox.setup()

        options = self._build_options(sandbox, session_state, agent_config)

        async for message in query(prompt=prompt, options=options):
            # AssistantMessage carries an array of content blocks; each
            # block becomes its own event (assistant_message / tool_use /
            # thinking) so consumers can render them individually.
            if isinstance(message, AssistantMessage):
                for block in message.content or []:
                    event = self._block_to_event(block)
                    if event is not None:
                        yield event
            elif isinstance(message, UserMessage):
                # UserMessage from the SDK is the SDK echoing the tool
                # results it just got back (so the LLM saw them on the
                # next turn). We surface those as ``tool_result`` events.
                for block in message.content or []:
                    if isinstance(block, ToolResultBlock):
                        yield Event(
                            type=EVENT_TYPE_TOOL_RESULT,
                            data={
                                "tool_use_id": block.tool_use_id,
                                "output": _stringify(block.content),
                                "is_error": bool(block.is_error),
                            },
                        )
            elif isinstance(message, SystemMessage):
                yield Event(
                    type=EVENT_TYPE_SYSTEM,
                    data={"subtype": message.subtype, "data": message.data},
                )
            elif isinstance(message, ResultMessage):
                yield Event(
                    type=EVENT_TYPE_RUN_FINISHED,
                    data={
                        "result": message.result,
                        "is_error": bool(message.is_error),
                        "stop_reason": message.stop_reason,
                        "num_turns": message.num_turns,
                        "duration_ms": message.duration_ms,
                        "total_cost_usd": message.total_cost_usd,
                    },
                )
                return
            # Other message types (StreamEvent, RateLimitEvent) ignored
            # for now — they're noise for our use case.

    @staticmethod
    def _block_to_event(block: Any) -> Optional[Event]:
        from claude_agent_sdk import TextBlock, ThinkingBlock, ToolUseBlock

        if isinstance(block, TextBlock):
            return Event(
                type=EVENT_TYPE_ASSISTANT_MESSAGE,
                data={"content": block.text},
            )
        if isinstance(block, ToolUseBlock):
            return Event(
                type=EVENT_TYPE_TOOL_USE,
                data={
                    "tool_use_id": block.id,
                    "tool": block.name,
                    "input": block.input,
                },
            )
        if isinstance(block, ThinkingBlock):
            return Event(
                type=EVENT_TYPE_THINKING,
                data={"content": block.thinking},
            )
        return None


def _stringify(content: Any) -> str:
    """Tool results in ``ToolResultBlock`` come as either a string or a
    list of content dicts (the Anthropic SDK shape). Squash to a string
    for the wire payload."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)
