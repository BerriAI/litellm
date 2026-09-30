"""Claude Code adapter: runs `claude -p --output-format stream-json` once per turn.

Every model call goes to the per-session endpoint with the per-session token. The CLI
gets a private CLAUDE_CONFIG_DIR and only the `user` setting source (which is that
private dir), so neither the user's login, keychain, nor a repo's `.claude/settings.json`
can swap the base URL or credentials.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from litellm._logging import verbose_logger
from litellm.constants import HARNESS_STDERR_TAIL_LINES
from litellm.harness.adapters.base import HarnessAdapter, SessionContext
from litellm.harness.errors import HarnessError, HarnessInstallFailed, OptionsMismatch
from litellm.harness.options import ClaudeCodeOptions
from litellm.harness.sandbox.base import Process, Sandbox
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

CLAUDE_BINARY = "claude"
# Transcripts live outside the per-session CLAUDE_CONFIG_DIR so a later session can resume.
_PERSIST_TRANSCRIPTS_SCRIPT = (
    'd="${HOME:-/tmp}/.cache/litellm-harness/claude_code/projects"; mkdir -p "$d" && ln -sfn "$d" "$1/projects"'
)
SKILL_MANIFEST = "SKILL.md"
SYNTHETIC_MODEL = "<synthetic>"
# TODO(core): move to litellm/constants.py as HARNESS_STREAM_READ_CHUNK_BYTES.
HARNESS_STREAM_READ_CHUNK_BYTES = 64 * 1024

BASE_COMMAND: tuple[str, ...] = (
    "-p",
    "--output-format",
    "stream-json",
    "--verbose",
    "--input-format",
    "text",
)

PERMISSION_MODES: Mapping[str, str] = {
    "read-only": "plan",
    "ask": "default",
    "edit": "acceptEdits",
    "full": "bypassPermissions",
}

NATIVE_TO_NORMALIZED: Mapping[str, str] = {
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

NORMALIZED_TO_NATIVE: Mapping[str, tuple[str, ...]] = {
    "read": ("Read",),
    "write": ("Write",),
    "edit": ("Edit", "MultiEdit"),
    "bash": ("Bash",),
    "glob": ("Glob",),
    "grep": ("Grep",),
    "web_search": ("WebSearch",),
    "ls": ("LS",),
}

# Env the adapter owns; options.env may not override these.
MANAGED_ENV_KEYS: frozenset[str] = frozenset(
    {
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL",
        "CLAUDE_CONFIG_DIR",
    }
)

STATIC_ENV: Mapping[str, str] = {
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "DISABLE_AUTOUPDATER": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}

STRUCTURED_OUTPUT_INSTRUCTION = (
    "When you have finished the task, end your final reply with only a single JSON "
    "object (no code fences, no prose after it) that matches this JSON schema:\n{schema}"
)


@dataclass
class StreamState:
    """What the parser has learned from the stream so far in one turn."""

    session_id: str | None = None
    text_parts: list[str] = field(default_factory=list)
    result_seen: bool = False
    result_text: str | None = None
    is_error: bool = False
    errors: list[str] = field(default_factory=list)
    structured_output: Any | None = None

    @property
    def final_text(self) -> str:
        if self.result_text is not None:
            return self.result_text
        return "".join(self.text_parts)


# ---------------------------------------------------------------------------
# Stream-json parsing (pure)
# ---------------------------------------------------------------------------


def normalize_tool_name(native_name: str) -> str:
    return NATIVE_TO_NORMALIZED.get(native_name, native_name)


def stringify_tool_output(content: Any) -> str:
    """tool_result content is a string or a list of content blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_stringify_block(block) for block in content)
    return json.dumps(content, ensure_ascii=False)


def _stringify_block(block: Any) -> str:
    if isinstance(block, dict) and block.get("type") == "text":
        return str(block.get("text", ""))
    if isinstance(block, str):
        return block
    return json.dumps(block, ensure_ascii=False)


def _message_blocks(event: Mapping[str, Any]) -> list[Any]:
    content = (event.get("message") or {}).get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content if isinstance(content, list) else []


def _assistant_block_event(block: Mapping[str, Any], state: StreamState) -> list[Event]:
    kind = block.get("type")
    if kind == "text" and block.get("text"):
        state.text_parts.append(block["text"])
        return [Text(delta=block["text"])]
    if kind == "thinking" and block.get("thinking"):
        return [Reasoning(delta=block["thinking"])]
    if kind == "tool_use":
        native = str(block.get("name", ""))
        return [
            ToolCall(
                id=str(block.get("id", "")),
                name=normalize_tool_name(native),
                native_name=native,
                input=block.get("input") or {},
                builtin=not native.startswith("mcp__"),
            )
        ]
    return []


def _parse_assistant(event: Mapping[str, Any], state: StreamState) -> list[Event]:
    if event.get("parent_tool_use_id"):
        return []  # subagent traffic
    if (event.get("message") or {}).get("model") == SYNTHETIC_MODEL:
        return []  # CLI-generated error text; surfaced via the result event
    events: list[Event] = []
    for block in _message_blocks(event):
        if isinstance(block, dict):
            events.extend(_assistant_block_event(block, state))
    return events


def _parse_user(event: Mapping[str, Any]) -> list[Event]:
    if event.get("parent_tool_use_id"):
        return []
    return [
        ToolResult(
            id=str(block.get("tool_use_id", "")),
            output=stringify_tool_output(block.get("content")),
            is_error=bool(block.get("is_error", False)),
        )
        for block in _message_blocks(event)
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]


def _parse_system(event: Mapping[str, Any], state: StreamState) -> list[Event]:
    subtype = event.get("subtype")
    if subtype == "init" and event.get("session_id"):
        state.session_id = str(event["session_id"])
        return []
    if subtype == "compact_boundary":
        meta = event.get("compact_metadata") or {}
        return [Compaction(tokens_before=meta.get("pre_tokens"), tokens_after=None)]
    return []


def _parse_result(event: Mapping[str, Any], state: StreamState) -> list[Event]:
    state.result_seen = True
    state.is_error = bool(event.get("is_error", False))
    result = event.get("result")
    state.result_text = result if isinstance(result, str) else None
    state.errors = [str(e) for e in event.get("errors") or []]
    state.structured_output = event.get("structured_output")
    if event.get("session_id"):
        state.session_id = str(event["session_id"])
    return []


def parse_stream_json_line(line: str, state: StreamState) -> list[Event]:
    """Turn one stream-json line into zero or more events; update state in place."""
    stripped = line.strip()
    if not stripped:
        return []
    try:
        event = json.loads(stripped)
    except json.JSONDecodeError:
        return []
    if not isinstance(event, dict):
        return []
    kind = event.get("type")
    if kind == "assistant":
        return _parse_assistant(event, state)
    if kind == "user":
        return _parse_user(event)
    if kind == "system":
        return _parse_system(event, state)
    if kind == "result":
        return _parse_result(event, state)
    return []


def extract_last_json_object(text: str) -> str | None:
    """The last top-level `{...}` in text that parses as a JSON object, re-serialized."""
    decoder = json.JSONDecoder()
    last: str | None = None
    index = text.find("{")
    while index != -1:
        try:
            obj, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index = text.find("{", index + 1)
            continue
        if isinstance(obj, dict):
            last = json.dumps(obj)
        index = text.find("{", end)
    return last


def turn_error_message(state: StreamState, exit_code: int, stderr_tail: Sequence[str]) -> str | None:
    """None if the turn succeeded, else a message for the RuntimeError."""
    if exit_code == 0 and state.result_seen and not state.is_error:
        return None
    reason = state.result_text or "; ".join(state.errors)
    if not reason:
        reason = "no result event" if not state.result_seen else "unknown error"
    message = f"claude exited with code {exit_code}: {reason}"
    tail = "\n".join(line for line in stderr_tail if line.strip())
    return f"{message}\nstderr:\n{tail}" if tail else message


# ---------------------------------------------------------------------------
# Command / env construction (pure)
# ---------------------------------------------------------------------------


def native_disallowed_tools(disable_tools: Sequence[str]) -> list[str]:
    natives: list[str] = []
    for name in disable_tools:
        for native in NORMALIZED_TO_NATIVE.get(name, (name,)):
            if native not in natives:
                natives.append(native)
    return natives


def build_system_prompt(instructions: str | None, output_schema: Mapping[str, Any] | None) -> str | None:
    parts = [instructions] if instructions else []
    if output_schema is not None:
        parts.append(STRUCTURED_OUTPUT_INSTRUCTION.format(schema=json.dumps(output_schema)))
    return "\n\n".join(parts) if parts else None


def build_command(
    binary: str,
    *,
    model: str | None,
    permissions: str,
    system_prompt: str | None,
    max_turns: int | None,
    disable_tools: Sequence[str],
    resume_session_id: str | None,
    isolated_config: bool,
) -> list[str]:
    cmd = [binary, *BASE_COMMAND, "--permission-mode", PERMISSION_MODES[permissions]]
    if model:
        cmd += ["--model", model]
    if isolated_config:
        cmd += ["--setting-sources", "user"]
    if system_prompt:
        cmd += ["--append-system-prompt", system_prompt]
    if max_turns is not None:
        cmd += ["--max-turns", str(max_turns)]
    disallowed = native_disallowed_tools(disable_tools)
    if disallowed:
        cmd += ["--disallowedTools", ",".join(disallowed)]
    if resume_session_id:
        cmd += ["--resume", resume_session_id]
    return cmd


def build_env(
    *,
    base_url: str,
    token: str,
    model: str | None,
    small_model: str | None,
    config_dir: str | None,
    extra_env: Mapping[str, str],
) -> dict[str, str]:
    if not token:
        raise HarnessError("Claude Code needs a session endpoint token")
    clashing = sorted(MANAGED_ENV_KEYS.intersection(extra_env))
    if clashing:
        raise OptionsMismatch(f"ClaudeCodeOptions.env may not set {', '.join(clashing)}; LiteLLM manages it")
    managed: dict[str, str] = {
        **STATIC_ENV,
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_AUTH_TOKEN": token,
        "ANTHROPIC_API_KEY": "",
    }
    if model:
        managed["ANTHROPIC_MODEL"] = model
    if small_model or model:
        managed["ANTHROPIC_SMALL_FAST_MODEL"] = small_model or model or ""
    if config_dir:
        managed["CLAUDE_CONFIG_DIR"] = config_dir
    return {**extra_env, **managed}


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


def read_skill_files(skill_dir: str) -> list[tuple[str, bytes]]:
    """(relative path, bytes) for every file under a local skill folder."""
    root = os.path.realpath(os.fspath(skill_dir))
    if not os.path.isfile(os.path.join(root, SKILL_MANIFEST)):
        raise HarnessError(f"skill folder {skill_dir!r} has no {SKILL_MANIFEST}")
    files: list[tuple[str, bytes]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in sorted(filenames):
            path = os.path.join(dirpath, filename)
            with open(path, "rb") as fh:
                files.append((os.path.relpath(path, root), fh.read()))
    return files


async def copy_skills(sandbox: Sandbox, skills: Sequence[str], skills_root: str) -> None:
    for skill in skills:
        name = os.path.basename(os.path.realpath(os.fspath(skill)))
        for rel, data in await asyncio.to_thread(read_skill_files, skill):
            await sandbox.write(f"{skills_root}/{name}/{rel}", data)


async def private_config_dir(sandbox: Sandbox) -> str | None:
    tempdir = getattr(sandbox, "tempdir", None)
    if tempdir is None:
        return None
    return await tempdir()


async def persist_transcripts(sandbox: Sandbox, config_dir: str) -> None:
    """Keep transcripts outside the per-session config dir so a later session can --resume."""
    run = await sandbox.run(["sh", "-c", _PERSIST_TRANSCRIPTS_SCRIPT, "sh", config_dir])
    if run.exit_code != 0:
        verbose_logger.debug(
            "claude_code: could not persist transcripts, resume across sessions disabled: %s",
            run.stderr.strip(),
        )


# ---------------------------------------------------------------------------
# Process IO
# ---------------------------------------------------------------------------


async def iter_lines(stream: asyncio.StreamReader) -> AsyncIterator[str]:
    """Decoded lines without StreamReader's line-length limit (tool output can be huge)."""
    buffer = b""
    while True:
        chunk = await stream.read(HARNESS_STREAM_READ_CHUNK_BYTES)
        if not chunk:
            break
        buffer += chunk
        *lines, buffer = buffer.split(b"\n")
        for line in lines:
            yield line.decode("utf-8", errors="replace")
    if buffer:
        yield buffer.decode("utf-8", errors="replace")


async def drain_stderr(stream: asyncio.StreamReader, tail: deque[str]) -> None:
    async for line in iter_lines(stream):
        tail.append(line)


async def send_prompt(proc: Process, prompt: str) -> None:
    if proc.stdin is None:
        return
    proc.stdin.write(prompt.encode("utf-8"))
    await proc.stdin.drain()
    proc.stdin.close()


class ClaudeCodeAdapter(HarnessAdapter):
    harness: ClassVar[Harness] = Harness.CLAUDE_CODE
    options_type: ClassVar[type] = ClaudeCodeOptions
    capabilities: ClassVar[Capabilities] = Capabilities(
        structured_output=True,
        tool_approval=False,
        tool_filtering=True,
        history=False,
        custom_tools=False,
        skills=True,
        resume=True,
        permission_modes=frozenset({"read-only", "edit", "full"}),
    )

    def __init__(self) -> None:
        self._binary: str | None = None
        self._env: dict[str, str] = {}
        self._config_dir: str | None = None
        self._session_id: str | None = None
        self._proc: Process | None = None

    async def start(self, ctx: SessionContext) -> None:
        self._binary = await ctx.sandbox.which(CLAUDE_BINARY)
        if not self._binary:
            raise HarnessInstallFailed(
                f"`{CLAUDE_BINARY}` was not found on PATH in the sandbox. Install Claude "
                "Code (npm install -g @anthropic-ai/claude-code) in the sandbox."
            )
        self._config_dir = await private_config_dir(ctx.sandbox)
        if self._config_dir:
            await persist_transcripts(ctx.sandbox, self._config_dir)
        skills_root = f"{self._config_dir}/skills" if self._config_dir else f"{ctx.sandbox.workdir}/.claude/skills"
        await copy_skills(ctx.sandbox, ctx.skills, skills_root)
        self._env = self._session_env(ctx)

    def _session_env(self, ctx: SessionContext) -> dict[str, str]:
        if ctx.endpoint is None:
            raise HarnessError("Claude Code needs the session model endpoint")
        options = self._options(ctx)
        return build_env(
            base_url=ctx.sandbox.host_url(ctx.endpoint.port),
            token=ctx.endpoint.token,
            model=ctx.model,
            small_model=options.small_model,
            config_dir=self._config_dir,
            extra_env=options.env,
        )

    def _options(self, ctx: SessionContext) -> ClaudeCodeOptions:
        if isinstance(ctx.options, ClaudeCodeOptions):
            return ctx.options
        return ClaudeCodeOptions()

    def command(self, ctx: SessionContext) -> list[str]:
        schema = ctx.output.model_json_schema() if ctx.output is not None else None
        return build_command(
            self._binary or CLAUDE_BINARY,
            model=ctx.model,
            permissions=ctx.permissions,
            system_prompt=build_system_prompt(ctx.instructions, schema),
            max_turns=self._options(ctx).max_turns,
            disable_tools=ctx.disable_tools,
            resume_session_id=self._session_id,
            isolated_config=self._config_dir is not None,
        )

    async def turn(self, ctx: SessionContext, prompt: str) -> AsyncIterator[Event]:
        proc = await ctx.sandbox.exec(self.command(ctx), env=self._env)
        self._proc = proc
        tail: deque[str] = deque(maxlen=HARNESS_STDERR_TAIL_LINES)
        stderr_task = asyncio.ensure_future(drain_stderr(proc.stderr, tail))
        state = StreamState()
        try:
            await send_prompt(proc, prompt)
            async for line in iter_lines(proc.stdout):
                for event in parse_stream_json_line(line, state):
                    yield event
                self._session_id = state.session_id or self._session_id
            exit_code = await proc.wait()
            await stderr_task
        finally:
            self._proc = None
            if not stderr_task.done():
                await proc.kill()
                stderr_task.cancel()
        self._finish_turn(ctx, state, exit_code, list(tail))

    def _finish_turn(
        self,
        ctx: SessionContext,
        state: StreamState,
        exit_code: int,
        stderr_tail: list[str],
    ) -> None:
        error = turn_error_message(state, exit_code, stderr_tail)
        if error is not None:
            raise RuntimeError(error)
        ctx.final_text = state.final_text
        if ctx.output is not None:
            ctx.output_json = self._output_json(state)

    def _output_json(self, state: StreamState) -> str | None:
        if isinstance(state.structured_output, dict):
            return json.dumps(state.structured_output)
        return extract_last_json_object(state.final_text)

    async def stop(self, ctx: SessionContext) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            await proc.kill()

    def native_session_id(self) -> str | None:
        return self._session_id

    async def resume(self, ctx: SessionContext, native_session_id: str) -> None:
        self._session_id = native_session_id
