"""
LiteLLMAgentRuntime — drives a manual tool loop via ``litellm.acompletion``.

Why a second runtime when ClaudeSDKAgentRuntime already exists?

  * Multi-provider. ``litellm.acompletion`` speaks Anthropic, OpenAI, Gemini,
    Bedrock, etc. through one unified surface, so the same agent definition
    can run against any provider that supports tool calling.
  * Sandbox-honest. Unlike ``ClaudeSDKAgentRuntime`` (which runs its built-in
    tools in-process), this runtime routes EVERY tool call through
    ``sandbox.execute_tool(...)``. That makes ``EC2SandboxViaSSM`` and any
    future remote-execution sandbox actually work.
  * Hookable. The ``before_tool_call`` / ``after_tool_call`` ABC hooks fire
    around every tool call so subclasses can audit, redact, or rewrite
    without forking the whole loop.

Wire shape: events emitted match the snake_case wire format the integration
branch already serves (``assistant_message`` / ``tool_use`` / ``tool_result``
/ ``run_finished``).

Tool source of truth: tools are configured on the ``AgentConfig`` via
``tools_config``. Two shapes are accepted:

  * ``{"tools": [<openai-tool-dict>, ...]}`` — pre-formatted OpenAI/LiteLLM
    tool definitions. Passed straight through to ``acompletion``.
  * ``[<openai-tool-dict>, ...]`` — bare list, same handling.

If ``tools_config`` is empty / missing, the runtime falls back to the
default LocalSandbox tool surface (Bash/Read/Write/Edit/ls). That keeps
the "create file foo.txt" smoke test working out of the box without
forcing every config to spell out tool schemas.
"""

import json
from typing import Any, AsyncIterator, Dict, List, Optional

import litellm
from litellm.managed_agents.agent_runtime.base import (
    AgentConfig,
    AgentRuntime,
    SessionState,
)
from litellm.managed_agents.events import (
    EVENT_TYPE_ASSISTANT_MESSAGE,
    EVENT_TYPE_RUN_FINISHED,
    EVENT_TYPE_TOOL_RESULT,
    EVENT_TYPE_TOOL_USE,
    Event,
)
from litellm.managed_agents.sandbox.base import Sandbox, ToolResult


# Default upper bound on tool-loop iterations. The LLM gets this many
# turns to call tools before we forcibly stop and yield run_finished.
# Picked to match claude-agent-sdk's default; configurable per-instance.
DEFAULT_MAX_TURNS = 25


# Default tool surface — these mirror what LocalSandbox knows how to
# execute. We hand them to providers that need OpenAI-shaped tool defs
# (which is most of them) when the AgentConfig doesn't supply its own.
_DEFAULT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command in the sandbox working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read the contents of a file inside the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (relative or absolute).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write text to a file inside the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path."},
                    "content": {
                        "type": "string",
                        "description": "Text content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Edit",
            "description": "Replace one occurrence of old_string with new_string in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ls",
            "description": "List entries in a sandbox directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
            },
        },
    },
]


class LiteLLMAgentRuntime(AgentRuntime):
    """Manual tool-loop runtime built on ``litellm.acompletion``.

    Construct with optional overrides; per-run values fall back to the
    ``AgentConfig`` passed to ``run()``. Same split as
    ``ClaudeSDKAgentRuntime`` so callers can either bake everything into
    the runtime instance or defer to the agent config.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        extra_completion_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.extra_completion_kwargs = dict(extra_completion_kwargs or {})

    def _resolve_tools(self, agent_config: AgentConfig) -> List[Dict[str, Any]]:
        """Pick the tool list to send to the LLM.

        Accepts either a ``{"tools": [...]}`` wrapper or a bare list, since
        both shapes show up in real configs. Falls back to the default
        LocalSandbox surface when nothing is configured.
        """
        cfg = agent_config.tools_config
        if isinstance(cfg, dict) and isinstance(cfg.get("tools"), list):
            return list(cfg["tools"])
        if isinstance(cfg, list):
            return list(cfg)
        return list(_DEFAULT_TOOLS)

    def _build_initial_messages(
        self,
        prompt: str,
        agent_config: AgentConfig,
        session_state: SessionState,
    ) -> List[Dict[str, Any]]:
        """Compose the message list passed to the first ``acompletion`` call.

        We seed an extra ``system`` line describing the live session so the
        LLM has cwd/repos context without the agent author having to thread
        it into every prompt.
        """
        messages: List[Dict[str, Any]] = []

        sys_text = self.system_prompt or agent_config.system_prompt
        if sys_text:
            messages.append({"role": "system", "content": sys_text})

        # Session context line — only added when there's something to say.
        ctx_lines: List[str] = []
        if session_state.cwd:
            ctx_lines.append(f"Working directory: {session_state.cwd}")
        if session_state.repos:
            repo_summaries = ", ".join(
                str(r.get("url") or r.get("path") or "?") for r in session_state.repos
            )
            ctx_lines.append(f"Repositories available: {repo_summaries}")
        if ctx_lines:
            messages.append({"role": "system", "content": "\n".join(ctx_lines)})

        messages.append({"role": "user", "content": prompt})
        return messages

    async def _execute_tool_call(
        self,
        sandbox: Sandbox,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> ToolResult:
        """Run a single tool call through the hooks + sandbox.

        Hook failures bubble out as ``ToolResult(is_error=True)`` rather
        than raising, so the LLM sees the failure on its next turn.
        """
        try:
            tool_input = await self.before_tool_call(tool_name, tool_input)
        except Exception as exc:  # noqa: BLE001 — surface to LLM
            return ToolResult(
                output=f"before_tool_call hook raised: {exc}",
                is_error=True,
                metadata={"hook": "before_tool_call"},
            )

        result = await sandbox.execute_tool(tool_name, tool_input)

        try:
            rewritten = await self.after_tool_call(tool_name, tool_input, result)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                output=f"after_tool_call hook raised: {exc}",
                is_error=True,
                metadata={"hook": "after_tool_call"},
            )
        # Hook may return a fully replaced ToolResult or a bare value; only
        # treat the former as authoritative.
        if isinstance(rewritten, ToolResult):
            return rewritten
        return result

    async def run(
        self,
        prompt: str,
        sandbox: Sandbox,
        session_state: SessionState,
        agent_config: AgentConfig,
    ) -> AsyncIterator[Event]:
        await sandbox.setup()

        model = self.model or agent_config.model
        tools = self._resolve_tools(agent_config)
        messages = self._build_initial_messages(prompt, agent_config, session_state)

        completion_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if tools:
            completion_kwargs["tools"] = tools
        completion_kwargs.update(self.extra_completion_kwargs)

        last_text: Optional[str] = None

        for turn in range(self.max_turns):
            response = await litellm.acompletion(**completion_kwargs)

            choice = _first_choice(response)
            if choice is None:
                # Defensive — provider returned no choices, treat as done.
                yield Event(
                    type=EVENT_TYPE_RUN_FINISHED,
                    data={
                        "result": last_text,
                        "is_error": False,
                        "stop_reason": "no_choice",
                        "num_turns": turn + 1,
                    },
                )
                return

            assistant_msg = _choice_message(choice)
            content = _message_content(assistant_msg)
            tool_calls = _message_tool_calls(assistant_msg)

            if content:
                last_text = content
                yield Event(
                    type=EVENT_TYPE_ASSISTANT_MESSAGE,
                    data={"content": content},
                )

            # Append the assistant turn to the conversation BEFORE handling
            # tool calls — the next request needs the assistant's tool_calls
            # array to interleave correctly with tool messages.
            messages.append(_assistant_dict(assistant_msg))

            # No tool call -> the LLM is done. Emit run_finished and stop.
            if not tool_calls:
                yield Event(
                    type=EVENT_TYPE_RUN_FINISHED,
                    data={
                        "result": last_text,
                        "is_error": False,
                        "stop_reason": _choice_finish_reason(choice) or "stop",
                        "num_turns": turn + 1,
                    },
                )
                return

            for call in tool_calls:
                tool_use_id = _tool_call_id(call)
                tool_name = _tool_call_name(call)
                tool_input = _tool_call_arguments(call)

                yield Event(
                    type=EVENT_TYPE_TOOL_USE,
                    data={
                        "tool_use_id": tool_use_id,
                        "tool": tool_name,
                        "input": tool_input,
                    },
                )

                result = await self._execute_tool_call(sandbox, tool_name, tool_input)
                output_str = _stringify(result.output)

                yield Event(
                    type=EVENT_TYPE_TOOL_RESULT,
                    data={
                        "tool_use_id": tool_use_id,
                        "output": output_str,
                        "is_error": result.is_error,
                    },
                )

                # Feed the result back into the conversation so the LLM
                # sees it on the next turn. OpenAI shape: role=tool with a
                # tool_call_id linking back to the assistant's call.
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "name": tool_name,
                        "content": output_str,
                    }
                )

        # Hit max_turns without the LLM signalling done. Yield a terminal
        # event so the caller's run row gets closed — surface as a finished
        # run with stop_reason=max_turns rather than an error (matches the
        # claude-agent-sdk behaviour).
        yield Event(
            type=EVENT_TYPE_RUN_FINISHED,
            data={
                "result": last_text,
                "is_error": False,
                "stop_reason": "max_turns",
                "num_turns": self.max_turns,
            },
        )


# ---------------------------------------------------------------------------
# Response shape adapters.
#
# litellm.acompletion can return either ModelResponse (pydantic-ish) or a
# plain dict depending on caller config and provider. The helpers below
# normalise both shapes so the main loop above doesn't have to care.
# ---------------------------------------------------------------------------


def _first_choice(response: Any) -> Optional[Any]:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        return None
    return choices[0]


def _choice_message(choice: Any) -> Any:
    msg = getattr(choice, "message", None)
    if msg is None and isinstance(choice, dict):
        msg = choice.get("message")
    return msg


def _choice_finish_reason(choice: Any) -> Optional[str]:
    reason = getattr(choice, "finish_reason", None)
    if reason is None and isinstance(choice, dict):
        reason = choice.get("finish_reason")
    return reason


def _message_content(message: Any) -> Optional[str]:
    if message is None:
        return None
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, list):
        # Anthropic-via-litellm sometimes returns content as a list of
        # blocks; concat any text blocks for the assistant_message event.
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p) or None
    if isinstance(content, str) and content:
        return content
    return None


def _message_tool_calls(message: Any) -> List[Any]:
    if message is None:
        return []
    calls = getattr(message, "tool_calls", None)
    if calls is None and isinstance(message, dict):
        calls = message.get("tool_calls")
    return list(calls or [])


def _assistant_dict(message: Any) -> Dict[str, Any]:
    """Serialize the assistant message back into the shape acompletion expects.

    We can't always re-send the raw ModelResponse object — the next call
    needs a plain dict with ``role``, ``content``, and optional
    ``tool_calls`` (each with ``id``, ``type``, ``function``).
    """
    out: Dict[str, Any] = {"role": "assistant"}
    content = _message_content(message) or ""
    out["content"] = content
    calls = _message_tool_calls(message)
    if calls:
        out["tool_calls"] = [_tool_call_to_dict(c) for c in calls]
    return out


def _tool_call_to_dict(call: Any) -> Dict[str, Any]:
    if isinstance(call, dict):
        fn = call.get("function") or {}
        if not isinstance(fn, dict):
            fn = {
                "name": getattr(fn, "name", None),
                "arguments": getattr(fn, "arguments", None),
            }
        return {
            "id": call.get("id"),
            "type": call.get("type", "function"),
            "function": {
                "name": fn.get("name"),
                "arguments": fn.get("arguments", "{}"),
            },
        }
    fn = getattr(call, "function", None)
    return {
        "id": getattr(call, "id", None),
        "type": getattr(call, "type", "function"),
        "function": {
            "name": getattr(fn, "name", None) if fn is not None else None,
            "arguments": getattr(fn, "arguments", "{}") if fn is not None else "{}",
        },
    }


def _tool_call_id(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("id") or "")
    return str(getattr(call, "id", "") or "")


def _tool_call_name(call: Any) -> str:
    if isinstance(call, dict):
        fn = call.get("function") or {}
        if isinstance(fn, dict):
            return str(fn.get("name") or "")
        return str(getattr(fn, "name", "") or "")
    fn = getattr(call, "function", None)
    return str(getattr(fn, "name", "") or "") if fn is not None else ""


def _tool_call_arguments(call: Any) -> Dict[str, Any]:
    """Tool arguments come back as a JSON string — parse to dict.

    Defensive: providers occasionally emit malformed JSON; surface that as
    an empty dict so the loop can still call the sandbox (which will then
    return an error the LLM can see).
    """
    if isinstance(call, dict):
        fn = call.get("function") or {}
        raw = (
            fn.get("arguments")
            if isinstance(fn, dict)
            else getattr(fn, "arguments", None)
        )
    else:
        fn = getattr(call, "function", None)
        raw = getattr(fn, "arguments", None) if fn is not None else None

    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _stringify(output: Any) -> str:
    if isinstance(output, str):
        return output
    if output is None:
        return ""
    try:
        return json.dumps(output)
    except (TypeError, ValueError):
        return str(output)
