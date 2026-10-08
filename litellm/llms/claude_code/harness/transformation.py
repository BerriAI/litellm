"""
Claude Code harness config: `claude -p --output-format stream-json`, once per turn.

Every model call goes to the per-session endpoint with the per-session token. The CLI gets
a private CLAUDE_CONFIG_DIR and only the `user` setting source (that private dir), so
neither the user's login, keychain, nor a repo's `.claude/settings.json` can swap the base
URL or credentials. Verified against Claude Code 2.1.285.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from pydantic import ConfigDict, TypeAdapter

from litellm.harness.errors import HarnessError, OptionsMismatch
from litellm.harness.options import ClaudeCodeOptions
from litellm.harness.types import (
    Capabilities,
    Compaction,
    Event,
    Harness,
    Reasoning,
    Text,
    ToolCall,
    ToolResult,
)
from litellm.llms.base_llm.harness.transformation import (
    BaseCLIHarnessConfig,
    HarnessSessionSetup,
    HarnessTurnError,
    HarnessTurnRequest,
    HarnessTurnResponse,
    event_list,
)
from litellm.llms.base_llm.harness.utils import (
    last_json_object,
    native_tool_names,
    normalize_tool_name,
    stderr_tail_text,
)

if TYPE_CHECKING:
    from litellm.harness.context import SessionContext

CLAUDE_BINARY: Final = "claude"
SYNTHETIC_MODEL: Final = "<synthetic>"
_BLOCK: Final = TypeAdapter(Mapping[object, object], config=ConfigDict(hide_input_in_errors=True))

BASE_COMMAND: Final = ("-p", "--output-format", "stream-json", "--verbose", "--input-format", "text")

PERMISSION_MODES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "read-only": "plan",
        "ask": "default",
        "edit": "acceptEdits",
        "full": "bypassPermissions",
    }
)

NATIVE_TO_NORMALIZED: Final[Mapping[str, str]] = MappingProxyType(
    {
        "Read": "read",
        "Write": "write",
        "Edit": "edit",
        "MultiEdit": "edit",
        "Bash": "bash",
        "Glob": "glob",
        "Grep": "grep",
        "WebSearch": "web_search",
        "LS": "ls",
    }
)

NORMALIZED_TO_NATIVE: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "read": ("Read",),
        "write": ("Write",),
        "edit": ("Edit", "MultiEdit"),
        "bash": ("Bash",),
        "glob": ("Glob",),
        "grep": ("Grep",),
        "web_search": ("WebSearch",),
        "ls": ("LS",),
    }
)

# Env the config owns; ClaudeCodeOptions.env may not override these.
MANAGED_ENV_KEYS: Final = frozenset(
    {
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL",
        "CLAUDE_CONFIG_DIR",
    }
)

# Claude Code settings.json keys LiteLLM manages (or that could reroute model calls or credentials).
MANAGED_CONFIG_KEYS: Final = frozenset(
    {"env", "apiKeyHelper", "model", "permissions", "awsAuthRefresh", "awsCredentialExport", "forceLoginMethod"}
)

STATIC_ENV: Final[Mapping[str, str]] = MappingProxyType(
    {
        "DISABLE_TELEMETRY": "1",
        "DISABLE_ERROR_REPORTING": "1",
        "DISABLE_AUTOUPDATER": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }
)

STRUCTURED_OUTPUT_INSTRUCTION: Final = (
    "When you have finished the task, end your final reply with only a single JSON "
    "object (no code fences, no prose after it) that matches this JSON schema:\n{schema}"
)


@dataclass
class ClaudeCodeStreamState:
    """What the parser has learned from one turn's stream-json output."""

    session_id: str | None = None
    text_parts: list[str] = field(default_factory=list)  # mutable-ok: parser appends text deltas
    result_seen: bool = False
    result_text: str | None = None
    is_error: bool = False
    errors: Sequence[str] = ()
    structured_output: object = None

    @property
    def final_text(self) -> str:
        if self.result_text is not None:
            return self.result_text
        return "".join(self.text_parts)


def stringify_tool_output(content: object) -> str:
    """tool_result content is a string or a list of content blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_stringify_block(block) for block in content)
    return json.dumps(content, ensure_ascii=False)


def _stringify_block(block: object) -> str:
    if isinstance(block, dict) and block.get("type") == "text":
        return str(block.get("text", ""))
    if isinstance(block, str):
        return block
    return json.dumps(block, ensure_ascii=False)


def _message_blocks(event: Mapping[str, object]) -> Sequence[object]:
    message: Final = event.get("message")
    content: Final = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, str):
        return ({"type": "text", "text": content},)  # mutable-ok: JSON content block, like the stream's
    return content if isinstance(content, list) else ()


def _assistant_block_events(block: Mapping[str, Any], state: ClaudeCodeStreamState) -> tuple[Event, ...]:
    kind = block.get("type")
    if kind == "text" and block.get("text"):
        state.text_parts.append(block["text"])
        return (Text(delta=block["text"]),)
    if kind == "thinking" and block.get("thinking"):
        return (Reasoning(delta=block["thinking"]),)
    if kind == "tool_use":
        native = str(block.get("name", ""))
        return (
            ToolCall(
                id=str(block.get("id", "")),
                name=normalize_tool_name(native, NATIVE_TO_NORMALIZED),
                native_name=native,
                input=block.get("input")
                or {},  # mutable-ok: ToolCall.input is a dict field; empty default for a missing input
                builtin=not native.startswith("mcp__"),
            ),
        )
    return ()


def _assistant_events(event: Mapping[str, object], state: ClaudeCodeStreamState) -> Sequence[Event]:
    if event.get("parent_tool_use_id"):
        return event_list()  # subagent traffic
    message: Final = event.get("message")
    if isinstance(message, Mapping) and message.get("model") == SYNTHETIC_MODEL:
        return event_list()  # CLI-generated error text; surfaced via the result event
    blocks: Final = (block for block in _message_blocks(event) if isinstance(block, dict))
    return event_list(*itertools.chain.from_iterable(_assistant_block_events(block, state) for block in blocks))


def _is_tool_result(block: object) -> bool:
    return isinstance(block, dict) and block.get("type") == "tool_result"


def _user_events(event: Mapping[str, object]) -> Sequence[Event]:
    if event.get("parent_tool_use_id"):
        return event_list()
    return event_list(
        *(
            ToolResult(
                id=str(block.get("tool_use_id", "")),
                output=stringify_tool_output(block.get("content")),
                is_error=bool(block.get("is_error", False)),
            )
            for block in map(_BLOCK.validate_python, filter(_is_tool_result, _message_blocks(event)))
        )
    )


def _system_events(event: Mapping[str, object], state: ClaudeCodeStreamState) -> Sequence[Event]:
    subtype = event.get("subtype")
    if subtype == "init" and event.get("session_id"):
        state.session_id = str(event["session_id"])
        return event_list()
    if subtype == "compact_boundary":
        meta: Final = event.get("compact_metadata")
        pre_tokens: Final = meta.get("pre_tokens") if isinstance(meta, Mapping) else None
        return event_list(Compaction(tokens_before=pre_tokens, tokens_after=None))
    return event_list()


def _record_result(event: Mapping[str, Any], state: ClaudeCodeStreamState) -> Sequence[Event]:
    state.result_seen = True
    state.is_error = bool(event.get("is_error", False))
    result = event.get("result")
    state.result_text = result if isinstance(result, str) else None
    state.errors = [str(e) for e in event.get("errors") or ()]  # mutable-ok: mirrors the JSON errors array
    state.structured_output = event.get("structured_output")
    if event.get("session_id"):
        state.session_id = str(event["session_id"])
    return event_list()


def turn_error_message(state: ClaudeCodeStreamState, exit_code: int, stderr_tail: Sequence[str]) -> str | None:
    """None if the turn succeeded, else the message for HarnessTurnError."""
    if exit_code == 0 and state.result_seen and not state.is_error:
        return None
    reason = state.result_text or "; ".join(state.errors)
    if not reason:
        reason = "no result event" if not state.result_seen else "unknown error"
    message = f"claude exited with code {exit_code}: {reason}"
    tail = stderr_tail_text(stderr_tail)
    return f"{message}\nstderr:\n{tail}" if tail else message


def build_system_prompt(instructions: str | None, output_schema: Mapping[str, object] | None) -> str | None:
    schema_part: Final = (
        STRUCTURED_OUTPUT_INSTRUCTION.format(schema=json.dumps(output_schema)) if output_schema is not None else None
    )
    parts: Final = tuple(part for part in (instructions, schema_part) if part)
    return "\n\n".join(parts) if parts else None


class ClaudeCodeHarnessConfig(BaseCLIHarnessConfig):
    harness = Harness.CLAUDE_CODE
    options_type = ClaudeCodeOptions
    capabilities = Capabilities(
        structured_output=True,
        tool_approval=False,
        tool_filtering=True,
        history=False,
        custom_tools=False,
        skills=True,
        resume=True,
        permission_modes=frozenset({"read-only", "edit", "full"}),
    )

    def get_binary(self) -> str:
        return CLAUDE_BINARY

    def get_install_hint(self) -> str:
        return "npm install -g @anthropic-ai/claude-code"

    def validate_environment(self, ctx: SessionContext) -> None:
        options: ClaudeCodeOptions = self.get_options(ctx)
        clashing = sorted(MANAGED_ENV_KEYS.intersection(options.env))
        if clashing:
            raise OptionsMismatch(f"ClaudeCodeOptions.env may not set {', '.join(clashing)}; LiteLLM manages it")
        managed = sorted(MANAGED_CONFIG_KEYS.intersection(options.config))
        if managed:
            raise OptionsMismatch(
                f"ClaudeCodeOptions.config may not set {', '.join(managed)}; "
                "use the matching agent() argument (model=, permissions=) instead"
            )

    def transform_session_setup(self, ctx: SessionContext, private_dir: str) -> HarnessSessionSetup:
        if ctx.endpoint is None or not ctx.endpoint.token:
            raise HarnessError("Claude Code needs the session model endpoint")
        options: ClaudeCodeOptions = self.get_options(ctx)
        model = ctx.model
        # Background calls (titles, summaries) use the same model group, like OpenCode.
        model_env: Final = (
            MappingProxyType({"ANTHROPIC_MODEL": model, "ANTHROPIC_SMALL_FAST_MODEL": model})
            if model
            else MappingProxyType({})
        )
        env: Final = MappingProxyType(
            {
                **options.env,
                **STATIC_ENV,
                "ANTHROPIC_BASE_URL": ctx.sandbox.host_url(ctx.endpoint.port),
                "ANTHROPIC_AUTH_TOKEN": ctx.endpoint.token,
                "ANTHROPIC_API_KEY": "",
                "CLAUDE_CONFIG_DIR": private_dir,
                **model_env,
            }
        )
        return HarnessSessionSetup(
            persisted_dirs=[("projects", "claude_code/projects")],  # mutable-ok: tests compare to a list
            skills_dir="skills",
            env=env,
        )

    def transform_turn_request(
        self,
        ctx: SessionContext,
        setup: HarnessSessionSetup,
        private_dir: str,
        prompt: str,
        native_session_id: str | None,
    ) -> HarnessTurnRequest:
        options: ClaudeCodeOptions = self.get_options(ctx)
        schema = ctx.output.model_json_schema() if ctx.output is not None else None
        system_prompt: Final = build_system_prompt(ctx.instructions, schema)
        disallowed: Final = native_tool_names(ctx.disable_tools, NORMALIZED_TO_NATIVE)
        config: Final = dict(options.config)  # mutable-ok: json.dumps needs a plain dict
        settings: Final = json.dumps(config) if config else None
        argv: Final = (
            CLAUDE_BINARY,
            *BASE_COMMAND,
            "--permission-mode",
            PERMISSION_MODES[ctx.permissions],
            *(("--model", ctx.model) if ctx.model else ()),
            # Only read settings from the private CLAUDE_CONFIG_DIR, never the repo's .claude/.
            "--setting-sources",
            "user",
            *(("--settings", settings) if settings else ()),
            *(("--append-system-prompt", system_prompt) if system_prompt else ()),
            *(("--max-turns", str(ctx.max_turns)) if ctx.max_turns is not None else ()),
            *(("--disallowedTools", ",".join(disallowed)) if disallowed else ()),
            *(("--resume", native_session_id) if native_session_id else ()),
        )
        return HarnessTurnRequest(argv=argv, env=setup.env, stdin=prompt)

    def create_stream_state(self) -> ClaudeCodeStreamState:
        return ClaudeCodeStreamState()

    def transform_stream_line(self, line: Mapping[str, object], state: ClaudeCodeStreamState) -> Sequence[Event]:
        kind = line.get("type")
        if kind == "assistant":
            return _assistant_events(line, state)
        if kind == "user":
            return _user_events(line)
        if kind == "system":
            return _system_events(line, state)
        if kind == "result":
            return _record_result(line, state)
        return event_list()

    def get_native_session_id(self, state: ClaudeCodeStreamState) -> str | None:
        return state.session_id

    def transform_turn_response(
        self,
        ctx: SessionContext,
        state: ClaudeCodeStreamState,
        exit_code: int,
        stderr_tail: Sequence[str],
    ) -> HarnessTurnResponse:
        error = turn_error_message(state, exit_code, stderr_tail)
        if error is not None:
            raise HarnessTurnError(error)
        output_json: str | None = None
        if ctx.output is not None:
            if isinstance(state.structured_output, dict):
                output_json = json.dumps(state.structured_output)
            else:
                output_json = last_json_object(state.final_text)
        return HarnessTurnResponse(final_text=state.final_text, output_json=output_json)
