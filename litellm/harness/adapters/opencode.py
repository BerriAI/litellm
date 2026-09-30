"""OpenCode adapter: drives `opencode run --format json` through the session endpoint.

Verified against opencode 1.14.41. Every model call goes to one custom provider (`litellm`,
`@ai-sdk/openai-compatible`, bundled in the binary so nothing is npm-installed for it) whose
baseURL is the per-session endpoint. The config travels in OPENCODE_CONFIG_CONTENT, which
opencode applies after global and project config, so a repo's own opencode.json cannot
redirect model calls. The bearer token is never in argv or env: the config references it
with `{file:<tempdir>/token}`. The prompt is sent on stdin (opencode appends non-TTY stdin
to the message), so it never appears in argv either.

XDG dirs point at a LiteLLM-owned root (`$HOME/.cache/litellm-harness/opencode`) instead of
the user's, so the user's opencode config, auth and sessions are never read. The root is
shared across sessions on purpose: the session DB must outlive a session for resume(), and
the plugin/ripgrep caches then only download once.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from litellm._logging import verbose_logger
from litellm.constants import HARNESS_STDERR_TAIL_LINES
from litellm.harness.adapters.base import HarnessAdapter, SessionContext
from litellm.harness.errors import (
    CapabilityUnsupported,
    HarnessInstallFailed,
    OptionsMismatch,
    SandboxError,
)
from litellm.harness.options import OpenCodeOptions
from litellm.harness.runtime import last_json_object
from litellm.harness.sandbox.base import Process
from litellm.harness.types import (
    Capabilities,
    Event,
    Harness,
    PermissionMode,
    Reasoning,
    Text,
    ToolCall,
    ToolResult,
)

OPENCODE_BINARY: Final = "opencode"
OPENCODE_PROVIDER_ID: Final = "litellm"
OPENCODE_PROVIDER_NPM: Final = "@ai-sdk/openai-compatible"
# A fixed title skips opencode's extra title-generation model call on the first turn.
OPENCODE_SESSION_TITLE: Final = "litellm-harness"
_TOKEN_FILENAME: Final = "token"
_INSTRUCTIONS_FILENAME: Final = "instructions.md"
_SKILLS_DIRNAME: Final = "skills"
_XDG_SUBDIRS: Final = ("config", "data", "state", "cache")
_XDG_ROOT_SCRIPT: Final = (
    'd="${HOME:-/tmp}/.cache/litellm-harness/opencode"; '
    'mkdir -p "$d/config" "$d/data" "$d/state" "$d/cache" && printf %s "$d"'
)
_READ_CHUNK_BYTES: Final = 64 * 1024

# Env that keeps opencode off the network (except the endpoint) and away from ~/.claude.
OPENCODE_ISOLATION_ENV: Final[Mapping[str, str]] = {
    "OPENCODE_DISABLE_AUTOUPDATE": "1",
    "OPENCODE_DISABLE_MODELS_FETCH": "1",
    "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
    "OPENCODE_DISABLE_SHARE": "1",
    "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
    "OPENCODE_DISABLE_CLAUDE_CODE": "1",
    "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
    # Blank (falsy to opencode) so an inherited value can't add config, auth or rules.
    "OPENCODE_CONFIG": "",
    "OPENCODE_CONFIG_DIR": "",
    "OPENCODE_PERMISSION": "",
    "OPENCODE_AUTH_CONTENT": "",
}

# Top-level config keys LiteLLM sets itself (or that could override permissions).
MANAGED_CONFIG_KEYS: Final = frozenset(
    {
        "provider",
        "model",
        "small_model",
        "permission",
        "tools",
        "enabled_providers",
        "disabled_providers",
    }
)
_AGENT_MANAGED_KEYS: Final = frozenset({"permission", "tools", "model"})

# Permission rules per mode. Later keys win in opencode, so disable_tools denies go last.
# opencode run auto-rejects anything left at "ask", so no mode leaves a tool on ask.
PERMISSION_RULES: Final[Mapping[str, Mapping[str, str]]] = {
    "read-only": {"edit": "deny", "bash": "deny", "webfetch": "deny"},
    "edit": {"edit": "allow", "bash": "deny", "webfetch": "allow"},
    "full": {"*": "allow"},
}

# Normalized tool name -> opencode permission names that gate it. opencode gates write,
# edit and apply_patch with the single `edit` permission, so disabling either disables both.
NORMALIZED_TO_NATIVE: Final[Mapping[str, tuple[str, ...]]] = {
    "read": ("read",),
    "write": ("edit",),
    "edit": ("edit",),
    "bash": ("bash",),
    "glob": ("glob",),
    "grep": ("grep",),
    "ls": ("list",),
    "web_search": ("webfetch", "websearch"),
}

NATIVE_TO_NORMALIZED: Final[Mapping[str, str]] = {
    "read": "read",
    "write": "write",
    "edit": "edit",
    "multiedit": "edit",
    "patch": "edit",
    "apply_patch": "edit",
    "bash": "bash",
    "glob": "glob",
    "grep": "grep",
    "list": "ls",
    "webfetch": "web_search",
    "websearch": "web_search",
}

OPENCODE_BUILTIN_TOOLS: Final = frozenset(
    {
        *NATIVE_TO_NORMALIZED,
        "task",
        "todowrite",
        "todoread",
        "skill",
        "invalid",
        "question",
        "lsp",
        "codesearch",
        "plan_enter",
        "plan_exit",
    }
)


@dataclass
class OpenCodeParseState:
    """Mutable state carried across the JSON events of one `opencode run`."""

    session_id: str | None = None
    final_text: str = ""
    error: str | None = None
    step_texts: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure helpers (unit tested directly)
# ---------------------------------------------------------------------------


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def normalize_tool_name(native: str) -> str:
    return NATIVE_TO_NORMALIZED.get(native, native)


def _tool_events(part: Mapping[str, Any]) -> list[Event]:
    native = str(part.get("tool") or "")
    call_id = str(part.get("callID") or part.get("id") or "")
    state = _as_dict(part.get("state"))
    tool_input = state.get("input")
    status = state.get("status")
    call = ToolCall(
        id=call_id,
        name=normalize_tool_name(native),
        native_name=native,
        input=tool_input if isinstance(tool_input, dict) else {"input": tool_input},
        builtin=native in OPENCODE_BUILTIN_TOOLS,
    )
    if status == "error":
        message = str(state.get("error") or state.get("output") or "tool failed")
        return [call, ToolResult(id=call_id, output=message, is_error=True)]
    output = state.get("output")
    text = output if isinstance(output, str) else json.dumps(output)
    # The `invalid` pseudo-tool is how opencode reports a call to an unavailable tool.
    return [call, ToolResult(id=call_id, output=text, is_error=native == "invalid")]


def _error_message(error: Any) -> str:
    if not isinstance(error, dict):
        return str(error or "opencode reported an error")
    data = error.get("data")
    if isinstance(data, dict) and data.get("message"):
        return str(data["message"])
    return str(error.get("name") or "opencode reported an error")


def parse_opencode_event(obj: Mapping[str, Any], state: OpenCodeParseState) -> list[Event]:
    """Map one `opencode run --format json` line to harness events, updating state.

    step_finish token counts are ignored on purpose: the session endpoint accounts usage.
    Errors are recorded on state.error; the caller raises after the process exits.
    """
    session_id = obj.get("sessionID")
    if session_id and state.session_id is None:
        state.session_id = str(session_id)
    event_type = obj.get("type")
    part = _as_dict(obj.get("part"))
    if event_type == "step_start":
        state.step_texts = []
        return []
    if event_type == "text":
        text = str(part.get("text") or "")
        if not text:
            return []
        state.step_texts.append(text)
        state.final_text = "\n\n".join(state.step_texts)
        return [Text(delta=text)]
    if event_type == "reasoning":
        text = str(part.get("text") or "")
        return [Reasoning(delta=text)] if text else []
    if event_type == "tool_use":
        return _tool_events(part)
    if event_type == "error":
        message = _error_message(obj.get("error"))
        state.error = f"{state.error}\n{message}" if state.error else message
        return []
    return []


def validate_user_config(config: Mapping[str, Any]) -> None:
    """Reject OpenCodeOptions.config keys LiteLLM manages (or that bypass permissions)."""
    for key in config:
        if key in MANAGED_CONFIG_KEYS:
            raise OptionsMismatch(
                f"OpenCodeOptions.config[{key!r}] is managed by LiteLLM; use the matching "
                "run() argument (model=, permissions=, disable_tools=) instead"
            )
    for section in ("agent", "mode"):
        entries = config.get(section)
        if entries is None:
            continue
        if not isinstance(entries, Mapping):
            raise OptionsMismatch(f"OpenCodeOptions.config[{section!r}] must be a mapping")
        for name, agent in entries.items():
            managed = _AGENT_MANAGED_KEYS & set(agent or {})
            if managed:
                raise OptionsMismatch(
                    f"OpenCodeOptions.config[{section!r}][{name!r}] sets {sorted(managed)}, "
                    "which LiteLLM manages; use permissions=/disable_tools=/model= instead"
                )


def permission_rules(permissions: PermissionMode, disable_tools: Sequence[str]) -> dict[str, str]:
    """opencode `permission` config for a mode plus denies for disable_tools."""
    if permissions not in PERMISSION_RULES:
        raise CapabilityUnsupported(
            f"Harness.OPENCODE does not support permissions={permissions!r} in v1 "
            f"(supported: {sorted(PERMISSION_RULES)})"
        )
    rules = dict(PERMISSION_RULES[permissions])
    for tool in disable_tools:
        for native in NORMALIZED_TO_NATIVE.get(tool, (tool,)):
            rules.pop(native, None)
            rules[native] = "deny"
    return rules


def build_opencode_config(
    *,
    model: str,
    base_url: str,
    token_path: str,
    permissions: PermissionMode,
    disable_tools: Sequence[str] = (),
    user_config: Mapping[str, Any] | None = None,
    instructions_path: str | None = None,
    skills_path: str | None = None,
) -> dict[str, Any]:
    """The full opencode config: user config underneath, LiteLLM-managed keys on top."""
    user = dict(user_config or {})
    validate_user_config(user)
    qualified = f"{OPENCODE_PROVIDER_ID}/{model}"
    instructions = list(user.get("instructions") or [])
    if instructions_path:
        instructions.append(instructions_path)
    user_skills = _as_dict(user.get("skills"))
    skill_paths = list(user_skills.get("paths") or [])
    if skills_path:
        skill_paths.append(skills_path)
    managed: dict[str, Any] = {
        "provider": {
            OPENCODE_PROVIDER_ID: {
                "npm": OPENCODE_PROVIDER_NPM,
                "name": "LiteLLM",
                "options": {
                    "baseURL": base_url,
                    "apiKey": "{file:" + token_path + "}",
                },
                "models": {model: {}},
            }
        },
        "enabled_providers": [OPENCODE_PROVIDER_ID],
        "model": qualified,
        "small_model": qualified,
        "permission": permission_rules(permissions, disable_tools),
        "autoupdate": False,
        "share": "disabled",
    }
    if instructions:
        managed["instructions"] = instructions
    if skill_paths:
        managed["skills"] = {**user_skills, "paths": skill_paths}
    return {**user, **managed}


def turn_prompt(ctx: SessionContext, prompt: str) -> str:
    """Repeat the schema instruction in the user turn; system instructions alone are too weak."""
    if ctx.output is None:
        return prompt
    return f"{prompt}\n\n{structured_output_instruction(ctx.output.model_json_schema())}"


def structured_output_instruction(schema: Mapping[str, Any]) -> str:
    return (
        "When you have finished, your final message must be a single JSON object that "
        "matches this JSON schema, with no other text before or after it:\n"
        f"{json.dumps(schema)}"
    )


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


class OpenCodeAdapter(HarnessAdapter):
    harness = Harness.OPENCODE
    options_type = OpenCodeOptions
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

    def __init__(self) -> None:
        self._session_id: str | None = None
        self._model: str | None = None
        self._config_json: str | None = None
        self._xdg_root: str | None = None
        self._process: Process | None = None

    # -- lifecycle ---------------------------------------------------------

    async def start(self, ctx: SessionContext) -> None:
        if ctx.endpoint is None:
            raise ValueError("OpenCode adapter needs the session model endpoint")
        options = self._options(ctx)
        validate_user_config(options.config)
        model = ctx.model or ctx.endpoint.model
        if not model:
            raise ValueError("Harness.OPENCODE needs model= (a gateway or litellm model)")
        if await ctx.sandbox.which(OPENCODE_BINARY) is None:
            raise HarnessInstallFailed(
                "`opencode` was not found on PATH in the sandbox. Install it with "
                "`npm install -g opencode-ai` (or `brew install sst/tap/opencode`)."
            )
        tempdir = getattr(ctx.sandbox, "tempdir", None)
        if tempdir is None:
            raise SandboxError(f"{type(ctx.sandbox).__name__} has no tempdir(); OpenCode needs a private config dir")
        private_dir: str = await tempdir()
        token_path = f"{private_dir}/{_TOKEN_FILENAME}"
        await ctx.sandbox.write(token_path, ctx.endpoint.token.encode("utf-8"))
        instructions_path = await self._write_instructions(ctx, private_dir)
        skills_path = await self._install_skills(ctx, private_dir)
        config = build_opencode_config(
            model=model,
            base_url=ctx.sandbox.host_url(ctx.endpoint.port).rstrip("/") + "/v1",
            token_path=token_path,
            permissions=ctx.permissions,
            disable_tools=ctx.disable_tools,
            user_config=options.config,
            instructions_path=instructions_path,
            skills_path=skills_path,
        )
        self._model = model
        self._config_json = json.dumps(config)
        self._xdg_root = await self._resolve_xdg_root(ctx)

    async def stop(self, ctx: SessionContext) -> None:
        process, self._process = self._process, None
        if process is not None:
            await process.kill()

    def native_session_id(self) -> str | None:
        return self._session_id

    async def resume(self, ctx: SessionContext, native_session_id: str) -> None:
        self._session_id = native_session_id

    # -- turn --------------------------------------------------------------

    async def turn(self, ctx: SessionContext, prompt: str) -> AsyncIterator[Event]:
        argv = self.build_argv(ctx)
        process = await ctx.sandbox.exec(argv, env=self.build_env(ctx), cwd=ctx.sandbox.workdir)
        self._process = process
        stderr_task = asyncio.ensure_future(process.stderr.read())
        state = OpenCodeParseState(session_id=self._session_id)
        exit_code: int | None = None
        stderr = ""
        try:
            await self._send_prompt(process, turn_prompt(ctx, prompt))
            async for line in self._iter_lines(process.stdout):
                obj = self._decode(line)
                if obj is None:
                    continue
                for event in parse_opencode_event(obj, state):
                    yield event
                if state.session_id:
                    self._session_id = state.session_id
            exit_code = await process.wait()
            stderr = (await stderr_task).decode("utf-8", "replace")
        finally:
            if exit_code is None:
                # Consumer stopped early, timed out or errored: don't leave opencode running.
                await process.kill()
            if not stderr_task.done():
                stderr_task.cancel()
            self._process = None
        self._finish(ctx, state, exit_code, stderr)

    def _finish(
        self,
        ctx: SessionContext,
        state: OpenCodeParseState,
        exit_code: int | None,
        stderr: str,
    ) -> None:
        if state.error:
            raise RuntimeError(f"opencode turn failed: {state.error}")
        if exit_code != 0:
            raise RuntimeError(f"opencode exited with code {exit_code}: {_tail(stderr) or 'no output'}")
        ctx.final_text = state.final_text
        if ctx.output is not None:
            ctx.output_json = last_json_object(state.final_text)

    # -- argv / env ----------------------------------------------------------

    def build_argv(self, ctx: SessionContext) -> list[str]:
        if self._model is None:
            raise RuntimeError("OpenCodeAdapter.turn() called before start()")
        options = self._options(ctx)
        argv = [
            OPENCODE_BINARY,
            "run",
            "--format",
            "json",
            "--thinking",
            "-m",
            f"{OPENCODE_PROVIDER_ID}/{self._model}",
        ]
        if options.agent:
            argv.extend(["--agent", options.agent])
        if self._session_id:
            argv.extend(["--session", self._session_id])
        else:
            argv.extend(["--title", OPENCODE_SESSION_TITLE])
        return argv

    def build_env(self, ctx: SessionContext) -> dict[str, str]:
        if self._config_json is None or self._xdg_root is None:
            raise RuntimeError("OpenCodeAdapter.turn() called before start()")
        options = self._options(ctx)
        xdg = {f"XDG_{sub.upper()}_HOME": f"{self._xdg_root}/{sub}" for sub in _XDG_SUBDIRS}
        return {
            **OPENCODE_ISOLATION_ENV,
            **options.env,
            **xdg,
            "OPENCODE_CONFIG_CONTENT": self._config_json,
        }

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _options(ctx: SessionContext) -> OpenCodeOptions:
        options = ctx.options
        if options is None:
            return OpenCodeOptions()
        if not isinstance(options, OpenCodeOptions):
            raise OptionsMismatch(f"Harness.OPENCODE takes OpenCodeOptions, got {type(options).__name__}")
        return options

    @staticmethod
    async def _resolve_xdg_root(ctx: SessionContext) -> str:
        """Persistent LiteLLM-owned XDG root in the sandbox, else a fresh tempdir."""
        run = await ctx.sandbox.run(["sh", "-c", _XDG_ROOT_SCRIPT])
        root = run.stdout.strip()
        if run.exit_code == 0 and root:
            return root
        verbose_logger.debug(
            "opencode: no persistent XDG root, resume across sessions disabled: %s",
            run.stderr.strip(),
        )
        tempdir = getattr(ctx.sandbox, "tempdir")
        fallback: str = await tempdir()
        return fallback

    @staticmethod
    async def _write_instructions(ctx: SessionContext, private_dir: str) -> str | None:
        sections: list[str] = []
        if ctx.instructions:
            sections.append(ctx.instructions)
        if ctx.output is not None:
            sections.append(structured_output_instruction(ctx.output.model_json_schema()))
        if not sections:
            return None
        path = f"{private_dir}/{_INSTRUCTIONS_FILENAME}"
        await ctx.sandbox.write(path, "\n\n".join(sections).encode("utf-8"))
        return path

    @staticmethod
    async def _install_skills(ctx: SessionContext, private_dir: str) -> str | None:
        """Copy skill folders to <private_dir>/skills/<name>/ (loaded via skills.paths)."""
        if not ctx.skills:
            return None
        root = f"{private_dir}/{_SKILLS_DIRNAME}"
        for skill in ctx.skills:
            src = os.path.abspath(os.fspath(skill))
            name = os.path.basename(src.rstrip(os.sep))
            files = await asyncio.to_thread(_collect_skill_files, src)
            for rel_path, data in files:
                dest = f"{root}/{name}/{rel_path.replace(os.sep, '/')}"
                await ctx.sandbox.write(dest, data)
        return root

    @staticmethod
    async def _send_prompt(process: Process, prompt: str) -> None:
        if process.stdin is None:
            raise SandboxError("opencode process has no stdin")
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
            verbose_logger.debug("opencode: skipping non-JSON line: %r", text[:200])
            return None
        return obj if isinstance(obj, dict) else None
