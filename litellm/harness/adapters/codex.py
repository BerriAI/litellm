"""Codex adapter: drives `codex exec --json` (JSONL events) through the session endpoint.

Verified against codex-cli 0.135.0. Every model call goes to a single custom provider
(`litellm`, wire_api=responses) pointing at the per-session endpoint; the bearer token only
travels in the LITELLM_HARNESS_TOKEN env var, never in argv. CODEX_HOME is a private temp
dir in the sandbox so the user's own Codex config and auth are never read.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from litellm._logging import verbose_logger
from litellm.constants import HARNESS_STDERR_TAIL_LINES
from litellm.harness.adapters.base import HarnessAdapter, SessionContext
from litellm.harness.errors import (
    HarnessInstallFailed,
    OptionsMismatch,
    SandboxError,
)
from litellm.harness.options import CodexOptions
from litellm.harness.sandbox.base import Process
from litellm.harness.sandbox.docker import DockerSandbox
from litellm.harness.types import (
    Capabilities,
    Event,
    Harness,
    Reasoning,
    Text,
    ToolCall,
    ToolResult,
)

CODEX_BINARY: Final = "codex"
CODEX_PROVIDER_ID: Final = "litellm"
CODEX_TOKEN_ENV: Final = "LITELLM_HARNESS_TOKEN"
CODEX_SCHEMA_FILENAME: Final = "output_schema.json"
# Session rollouts live outside the per-session CODEX_HOME so a later session can resume.
_PERSIST_SESSIONS_SCRIPT: Final = (
    'd="${HOME:-/tmp}/.cache/litellm-harness/codex/sessions"; mkdir -p "$d" && ln -sfn "$d" "$1/sessions"'
)
# Top-level config keys LiteLLM sets itself; users may not override them via options.config.
MANAGED_CONFIG_KEYS: Final = frozenset(
    {
        "model",
        "model_provider",
        "model_providers",
        "approval_policy",
        "sandbox_mode",
        "mcp_servers",
        "developer_instructions",
        "web_search",
    }
)
_READ_CHUNK_BYTES: Final = 64 * 1024
_BARE_TOML_KEY: Final = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class CodexParseState:
    """Mutable state carried across the JSONL events of one turn."""

    thread_id: str | None = None
    final_text: str = ""
    error: str | None = None
    failed: bool = False
    started: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Pure helpers (unit tested directly)
# ---------------------------------------------------------------------------


def _tool_input(item: Mapping[str, Any]) -> tuple[str, str, dict[str, Any], bool]:
    """(normalized name, native name, input, builtin) for a tool-like item."""
    item_type = item.get("type")
    if item_type == "command_execution":
        return "bash", "command_execution", {"command": item.get("command", "")}, True
    if item_type == "file_change":
        return "edit", "apply_patch", {"changes": list(item.get("changes") or [])}, True
    if item_type == "web_search":
        return "web_search", "web_search", {"query": item.get("query", "")}, True
    server = str(item.get("server") or "")
    tool = str(item.get("tool") or "")
    arguments = item.get("arguments")
    tool_args = arguments if isinstance(arguments, dict) else {"arguments": arguments}
    name = f"{server}.{tool}" if server else tool
    return name, tool, tool_args, False


def _tool_output(item: Mapping[str, Any]) -> tuple[str, bool]:
    """(output text, is_error) for a completed tool-like item."""
    item_type = item.get("type")
    status = item.get("status")
    if item_type == "command_execution":
        exit_code = item.get("exit_code")
        is_error = status == "failed" or (exit_code is not None and exit_code != 0)
        return str(item.get("aggregated_output") or ""), is_error
    if item_type == "file_change":
        lines = [f"{change.get('kind', '')} {change.get('path', '')}".strip() for change in item.get("changes") or []]
        return "\n".join(lines), status == "failed"
    if item_type == "web_search":
        return "", status == "failed"
    error = item.get("error")
    if error:
        message = error.get("message") if isinstance(error, dict) else error
        return str(message), True
    result = item.get("result")
    if result is None:
        return "", status == "failed"
    if isinstance(result, str):
        return result, status == "failed"
    return json.dumps(result), status == "failed"


_TOOL_ITEM_TYPES: Final = frozenset({"command_execution", "file_change", "web_search", "mcp_tool_call"})


def _parse_item(event_type: str, item: Mapping[str, Any], state: CodexParseState) -> list[Event]:
    item_type = item.get("type")
    item_id = str(item.get("id") or "")
    completed = event_type == "item.completed"
    if item_type == "agent_message":
        if not completed:
            return []
        text = str(item.get("text") or "")
        state.final_text = text
        return [Text(delta=text)] if text else []
    if item_type == "reasoning":
        text = str(item.get("text") or "")
        return [Reasoning(delta=text)] if completed and text else []
    if item_type not in _TOOL_ITEM_TYPES:
        return []
    events: list[Event] = []
    if item_id not in state.started:
        state.started.add(item_id)
        name, native_name, tool_input, builtin = _tool_input(item)
        events.append(
            ToolCall(
                id=item_id,
                name=name,
                native_name=native_name,
                input=tool_input,
                builtin=builtin,
            )
        )
    if completed:
        output, is_error = _tool_output(item)
        events.append(ToolResult(id=item_id, output=output, is_error=is_error))
    return events


def parse_codex_event(obj: Mapping[str, Any], state: CodexParseState) -> list[Event]:
    """Map one `codex exec --json` event to harness events, updating state.

    turn.completed usage is ignored on purpose: the session endpoint already accounts it.
    Failures are recorded on state (turn.failed sets state.failed); the caller raises.
    """
    event_type = obj.get("type")
    if event_type == "thread.started":
        thread_id = obj.get("thread_id")
        if thread_id:
            state.thread_id = str(thread_id)
        return []
    if event_type in ("item.started", "item.updated", "item.completed"):
        item = obj.get("item")
        if not isinstance(item, dict):
            return []
        return _parse_item(str(event_type), item, state)
    if event_type == "error":
        state.error = str(obj.get("message") or "codex reported an error")
        return []
    if event_type == "turn.failed":
        error = obj.get("error")
        message = error.get("message") if isinstance(error, dict) else error
        state.error = str(message or state.error or "codex turn failed")
        state.failed = True
        return []
    return []


def strict_json_schema(schema: Any) -> Any:
    """Make a JSON schema acceptable to OpenAI strict structured outputs.

    Every object gets `additionalProperties: false` and all of its properties required,
    recursively (including $defs / definitions, items, anyOf/allOf/oneOf). Keywords that
    strict mode rejects alongside $ref are dropped from $ref nodes.
    """
    if isinstance(schema, list):
        return [strict_json_schema(entry) for entry in schema]
    if not isinstance(schema, dict):
        return schema
    result = {key: strict_json_schema(value) for key, value in schema.items()}
    if "$ref" in result:
        return {"$ref": result["$ref"]}
    result.pop("default", None)
    properties = result.get("properties")
    if result.get("type") == "object" or isinstance(properties, dict):
        props = properties if isinstance(properties, dict) else {}
        result = {
            **result,
            "properties": props,
            "required": list(props.keys()),
            "additionalProperties": False,
        }
    return result


def toml_value(value: Any) -> str:
    """Encode a Python value as a TOML value for `codex -c key=value`."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, Mapping):
        pairs = ", ".join(f"{toml_key(k)} = {toml_value(v)}" for k, v in value.items())
        return "{" + pairs + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    raise OptionsMismatch(f"CodexOptions.config value of type {type(value).__name__} cannot be passed to codex")


def toml_key(key: Any) -> str:
    text = str(key)
    return text if _BARE_TOML_KEY.match(text) else json.dumps(text)


def validate_config(config: Mapping[str, Any]) -> list[str]:
    """Return `-c` override strings for CodexOptions.config, rejecting managed keys."""
    overrides: list[str] = []
    for key, value in config.items():
        dotted = str(key)
        top = dotted.split(".", 1)[0]
        if not dotted or "=" in dotted:
            raise OptionsMismatch(f"Invalid CodexOptions.config key: {dotted!r}")
        if top in MANAGED_CONFIG_KEYS:
            raise OptionsMismatch(
                f"CodexOptions.config[{dotted!r}] is managed by LiteLLM; use the matching run() argument instead"
            )
        overrides.append(f"{dotted}={toml_value(value)}")
    return overrides


def _flag_pairs(flag: str, values: list[str]) -> list[str]:
    argv: list[str] = []
    for value in values:
        argv.extend([flag, value])
    return argv


def _collect_skill_files(skill_dir: str) -> list[tuple[str, bytes]]:
    """(relative path, bytes) for every file under a local skill folder."""
    files: list[tuple[str, bytes]] = []
    for root, _dirs, names in os.walk(skill_dir):
        for name in names:
            path = os.path.join(root, name)
            with open(path, "rb") as fh:
                files.append((os.path.relpath(path, skill_dir), fh.read()))
    return files


def _tail(text: str) -> str:
    return "\n".join(text.strip().splitlines()[-HARNESS_STDERR_TAIL_LINES:])


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class CodexAdapter(HarnessAdapter):
    harness = Harness.CODEX
    options_type = CodexOptions
    capabilities = Capabilities(
        structured_output=True,
        tool_approval=False,
        tool_filtering=False,
        history=False,
        custom_tools=False,
        skills=True,
        resume=True,
        permission_modes=frozenset({"read-only", "full"}),
    )

    def __init__(self) -> None:
        self._thread_id: str | None = None
        self._codex_home: str | None = None
        self._schema_path: str | None = None
        self._config_overrides: list[str] = []
        self._process: Process | None = None

    # -- lifecycle ---------------------------------------------------------

    async def start(self, ctx: SessionContext) -> None:
        if ctx.endpoint is None:
            raise ValueError("Codex adapter needs the session model endpoint")
        options = self._options(ctx)
        self._config_overrides = validate_config(options.config)
        if await ctx.sandbox.which(CODEX_BINARY) is None:
            raise HarnessInstallFailed(
                "`codex` was not found on PATH in the sandbox. Install it with "
                "`npm install -g @openai/codex` (or `brew install codex`)."
            )
        tempdir = getattr(ctx.sandbox, "tempdir", None)
        if tempdir is None:
            raise SandboxError(f"{type(ctx.sandbox).__name__} has no tempdir(); Codex needs a private CODEX_HOME")
        self._codex_home = await tempdir()
        await self._persist_sessions(ctx, self._codex_home)
        await self._install_skills(ctx, self._codex_home)
        if ctx.output is not None:
            schema = strict_json_schema(ctx.output.model_json_schema())
            self._schema_path = f"{self._codex_home}/{CODEX_SCHEMA_FILENAME}"
            await ctx.sandbox.write(self._schema_path, json.dumps(schema).encode("utf-8"))

    async def stop(self, ctx: SessionContext) -> None:
        process, self._process = self._process, None
        if process is not None:
            await process.kill()

    def native_session_id(self) -> str | None:
        return self._thread_id

    async def resume(self, ctx: SessionContext, native_session_id: str) -> None:
        self._thread_id = native_session_id

    # -- turn --------------------------------------------------------------

    async def turn(self, ctx: SessionContext, prompt: str) -> AsyncIterator[Event]:
        argv = self.build_argv(ctx)
        process = await ctx.sandbox.exec(argv, env=self.build_env(ctx), cwd=ctx.sandbox.workdir)
        self._process = process
        stderr_task = asyncio.ensure_future(process.stderr.read())
        state = CodexParseState(thread_id=self._thread_id)
        exit_code: int | None = None
        try:
            await self._send_prompt(process, prompt)
            async for line in self._iter_lines(process.stdout):
                obj = self._decode(line)
                if obj is None:
                    continue
                for event in parse_codex_event(obj, state):
                    yield event
                if state.thread_id:
                    self._thread_id = state.thread_id
            exit_code = await process.wait()
            stderr = (await stderr_task).decode("utf-8", "replace")
        finally:
            if exit_code is None:
                # Consumer stopped early, timed out or errored: don't leave codex running.
                await process.kill()
            if not stderr_task.done():
                stderr_task.cancel()
            self._process = None
        self._finish(ctx, state, exit_code, stderr)

    def _finish(self, ctx: SessionContext, state: CodexParseState, exit_code: int, stderr: str) -> None:
        if state.failed:
            raise RuntimeError(f"codex turn failed: {state.error}")
        if exit_code != 0:
            detail = _tail(stderr) or state.error or "no output"
            raise RuntimeError(f"codex exited with code {exit_code}: {detail}")
        ctx.final_text = state.final_text
        if ctx.output is not None:
            ctx.output_json = state.final_text

    # -- argv / env ----------------------------------------------------------

    def build_argv(self, ctx: SessionContext) -> list[str]:
        options = self._options(ctx)
        if self._thread_id:
            head = ["codex", "exec", "resume", self._thread_id]
        else:
            head = ["codex", "exec"]
        argv = [*head, "--json", "--skip-git-repo-check"]
        if ctx.model:
            argv.extend(["-m", ctx.model])
        argv.extend(_flag_pairs("-c", self._provider_overrides(ctx)))
        argv.extend(self._permission_args(ctx))
        argv.extend(_flag_pairs("-c", self._feature_overrides(ctx, options)))
        argv.extend(_flag_pairs("-c", self._config_overrides))
        if self._schema_path:
            argv.extend(["--output-schema", self._schema_path])
        if not self._thread_id:
            argv.extend(["-C", ctx.sandbox.workdir])
        argv.append("-")
        return argv

    def build_env(self, ctx: SessionContext) -> dict[str, str]:
        if ctx.endpoint is None or self._codex_home is None:
            raise RuntimeError("CodexAdapter.turn() called before start()")
        options = self._options(ctx)
        return {
            **options.env,
            CODEX_TOKEN_ENV: ctx.endpoint.token,
            "CODEX_HOME": self._codex_home,
        }

    def _provider_overrides(self, ctx: SessionContext) -> list[str]:
        if ctx.endpoint is None:
            raise RuntimeError("Codex adapter needs the session model endpoint")
        base_url = ctx.sandbox.host_url(ctx.endpoint.port).rstrip("/") + "/v1"
        prefix = f"model_providers.{CODEX_PROVIDER_ID}"
        return [
            f"model_provider={CODEX_PROVIDER_ID}",
            f"{prefix}.name={CODEX_PROVIDER_ID}",
            f"{prefix}.base_url={toml_value(base_url)}",
            f"{prefix}.env_key={CODEX_TOKEN_ENV}",
            f"{prefix}.wire_api=responses",
            "approval_policy=never",
        ]

    def _permission_args(self, ctx: SessionContext) -> list[str]:
        if ctx.permissions == "read-only":
            mode = "read-only"
        elif isinstance(ctx.sandbox, DockerSandbox):
            return ["--dangerously-bypass-approvals-and-sandbox"]
        else:
            mode = "workspace-write"
        # `codex exec resume` has no --sandbox flag; the config key works for both.
        if self._thread_id:
            return ["-c", f"sandbox_mode={toml_value(mode)}"]
        return ["--sandbox", mode]

    def _feature_overrides(self, ctx: SessionContext, options: CodexOptions) -> list[str]:
        overrides = [f"web_search={'live' if options.web_search else 'disabled'}"]
        if options.reasoning_effort:
            overrides.extend(
                [
                    f"model_reasoning_effort={options.reasoning_effort}",
                    "model_reasoning_summary=auto",
                    "model_supports_reasoning_summaries=true",
                ]
            )
        if ctx.instructions:
            overrides.append(f"developer_instructions={toml_value(ctx.instructions)}")
        return overrides

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _options(ctx: SessionContext) -> CodexOptions:
        options = ctx.options
        if options is None:
            return CodexOptions()
        if not isinstance(options, CodexOptions):
            raise OptionsMismatch(f"Harness.CODEX takes CodexOptions, got {type(options).__name__}")
        return options

    @staticmethod
    async def _persist_sessions(ctx: SessionContext, codex_home: str) -> None:
        run = await ctx.sandbox.run(["sh", "-c", _PERSIST_SESSIONS_SCRIPT, "sh", codex_home])
        if run.exit_code != 0:
            verbose_logger.debug(
                "codex: could not persist session dir, resume across sessions disabled: %s",
                run.stderr.strip(),
            )

    @staticmethod
    async def _install_skills(ctx: SessionContext, codex_home: str) -> None:
        for skill in ctx.skills:
            src = os.path.abspath(os.fspath(skill))
            name = os.path.basename(src.rstrip(os.sep))
            files = await asyncio.to_thread(_collect_skill_files, src)
            for rel_path, data in files:
                dest = f"{codex_home}/skills/{name}/{rel_path.replace(os.sep, '/')}"
                await ctx.sandbox.write(dest, data)

    @staticmethod
    async def _send_prompt(process: Process, prompt: str) -> None:
        if process.stdin is None:
            raise SandboxError("codex process has no stdin")
        process.stdin.write(prompt.encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()

    @staticmethod
    async def _iter_lines(reader: asyncio.StreamReader) -> AsyncIterator[bytes]:
        """Yield newline-delimited lines without StreamReader's 64KiB readline limit."""
        buffer = b""
        while True:
            chunk = await reader.read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                yield line
        if buffer:
            yield buffer

    @staticmethod
    def _decode(line: bytes) -> dict[str, Any] | None:
        text = line.strip()
        if not text:
            return None
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            verbose_logger.debug("codex: skipping non-JSON line: %r", text[:200])
            return None
        return obj if isinstance(obj, dict) else None
