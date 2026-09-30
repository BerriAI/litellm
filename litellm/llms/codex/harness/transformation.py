"""
Codex harness config: `codex exec --json` (JSONL events), once per turn.

Every model call goes to one custom provider (`litellm`, wire_api=responses) pointing at the
per-session endpoint. The bearer token only travels in the LITELLM_HARNESS_TOKEN env var,
never in argv. CODEX_HOME is the private session dir so the user's own Codex config and
auth are never read. Verified against codex-cli 0.135.0.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from litellm.harness.errors import HarnessError, OptionsMismatch
from litellm.harness.options import CodexOptions
from litellm.harness.types import (
    Capabilities,
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
)
from litellm.llms.base_llm.harness.utils import stderr_tail_text, strict_json_schema

if TYPE_CHECKING:
    from litellm.harness.context import SessionContext

CODEX_BINARY: Final = "codex"
CODEX_PROVIDER_ID: Final = "litellm"
CODEX_TOKEN_ENV: Final = "LITELLM_HARNESS_TOKEN"
CODEX_SCHEMA_FILENAME: Final = "output_schema.json"
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
_BARE_TOML_KEY: Final = re.compile(r"^[A-Za-z0-9_-]+$")
_TOOL_ITEM_TYPES: Final = frozenset({"command_execution", "file_change", "web_search", "mcp_tool_call"})


@dataclass
class CodexStreamState:
    """What the parser has learned from one turn's JSONL events."""

    thread_id: str | None = None
    final_text: str = ""
    error: str | None = None
    failed: bool = False
    started: set[str] = field(default_factory=set)


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
        lines = [f"{c.get('kind', '')} {c.get('path', '')}".strip() for c in item.get("changes") or []]
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


def _item_events(event_type: str, item: Mapping[str, Any], state: CodexStreamState) -> list[Event]:
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
        events.append(ToolCall(id=item_id, name=name, native_name=native_name, input=tool_input, builtin=builtin))
    if completed:
        output, is_error = _tool_output(item)
        events.append(ToolResult(id=item_id, output=output, is_error=is_error))
    return events


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


def config_overrides(config: Mapping[str, Any]) -> list[str]:
    """`-c` override strings for CodexOptions.config, rejecting managed keys."""
    overrides: list[str] = []
    for key, value in config.items():
        dotted = str(key)
        if not dotted or "=" in dotted:
            raise OptionsMismatch(f"Invalid CodexOptions.config key: {dotted!r}")
        if dotted.split(".", 1)[0] in MANAGED_CONFIG_KEYS:
            raise OptionsMismatch(
                f"CodexOptions.config[{dotted!r}] is managed by LiteLLM; use the matching agent() argument instead"
            )
        overrides.append(f"{dotted}={toml_value(value)}")
    return overrides


def _flag_pairs(flag: str, values: Sequence[str]) -> list[str]:
    argv: list[str] = []
    for value in values:
        argv.extend([flag, value])
    return argv


class CodexHarnessConfig(BaseCLIHarnessConfig):
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

    def get_binary(self) -> str:
        return CODEX_BINARY

    def get_install_hint(self) -> str:
        return "npm install -g @openai/codex (or brew install codex)"

    def validate_environment(self, ctx: SessionContext) -> None:
        options: CodexOptions = self.get_options(ctx)
        config_overrides(options.config)

    def transform_session_setup(self, ctx: SessionContext, private_dir: str) -> HarnessSessionSetup:
        if ctx.endpoint is None:
            raise HarnessError("Codex needs the session model endpoint")
        options: CodexOptions = self.get_options(ctx)
        files: dict[str, bytes] = {}
        if ctx.output is not None:
            schema = strict_json_schema(ctx.output.model_json_schema())
            files[CODEX_SCHEMA_FILENAME] = json.dumps(schema).encode("utf-8")
        return HarnessSessionSetup(
            files=files,
            persisted_dirs=[("sessions", "codex/sessions")],
            skills_dir="skills",
            env={**options.env, CODEX_TOKEN_ENV: ctx.endpoint.token, "CODEX_HOME": private_dir},
        )

    def transform_turn_request(
        self,
        ctx: SessionContext,
        setup: HarnessSessionSetup,
        private_dir: str,
        prompt: str,
        native_session_id: str | None,
    ) -> HarnessTurnRequest:
        if ctx.endpoint is None:
            raise HarnessError("Codex needs the session model endpoint")
        options: CodexOptions = self.get_options(ctx)
        head = [CODEX_BINARY, "exec", "resume", native_session_id] if native_session_id else [CODEX_BINARY, "exec"]
        argv = [*head, "--json", "--skip-git-repo-check"]
        if ctx.model:
            argv += ["-m", ctx.model]
        argv += _flag_pairs("-c", self._provider_overrides(ctx))
        argv += self._permission_args(ctx, native_session_id)
        argv += _flag_pairs("-c", self._feature_overrides(ctx, options))
        argv += _flag_pairs("-c", config_overrides(options.config))
        if CODEX_SCHEMA_FILENAME in setup.files:
            argv += ["--output-schema", f"{private_dir}/{CODEX_SCHEMA_FILENAME}"]
        if not native_session_id:
            argv += ["-C", ctx.sandbox.workdir]
        argv.append("-")
        return HarnessTurnRequest(argv=argv, env=setup.env, stdin=prompt, cwd=ctx.sandbox.workdir)

    def _provider_overrides(self, ctx: SessionContext) -> list[str]:
        assert ctx.endpoint is not None
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

    @staticmethod
    def _permission_args(ctx: SessionContext, native_session_id: str | None) -> list[str]:
        if ctx.permissions == "read-only":
            mode = "read-only"
        elif getattr(ctx.sandbox, "is_container", False):
            # The container is already the boundary; nested sandboxing fails in containers.
            return ["--dangerously-bypass-approvals-and-sandbox"]
        else:
            mode = "workspace-write"
        # `codex exec resume` has no --sandbox flag; the config key works for both.
        if native_session_id:
            return ["-c", f"sandbox_mode={toml_value(mode)}"]
        return ["--sandbox", mode]

    @staticmethod
    def _feature_overrides(ctx: SessionContext, options: CodexOptions) -> list[str]:
        overrides = [f"web_search={'live' if options.web_search else 'disabled'}"]
        if options.reasoning_effort:
            overrides += [
                f"model_reasoning_effort={options.reasoning_effort}",
                "model_reasoning_summary=auto",
                "model_supports_reasoning_summaries=true",
            ]
        if ctx.instructions:
            overrides.append(f"developer_instructions={toml_value(ctx.instructions)}")
        return overrides

    def create_stream_state(self) -> CodexStreamState:
        return CodexStreamState()

    def transform_stream_line(self, line: Mapping[str, Any], state: CodexStreamState) -> list[Event]:
        """turn.completed usage is ignored on purpose: the session endpoint accounts it."""
        event_type = line.get("type")
        if event_type == "thread.started":
            if line.get("thread_id"):
                state.thread_id = str(line["thread_id"])
            return []
        if event_type in ("item.started", "item.updated", "item.completed"):
            item = line.get("item")
            return _item_events(str(event_type), item, state) if isinstance(item, dict) else []
        if event_type == "error":
            state.error = str(line.get("message") or "codex reported an error")
            return []
        if event_type == "turn.failed":
            error = line.get("error")
            message = error.get("message") if isinstance(error, dict) else error
            state.error = str(message or state.error or "codex turn failed")
            state.failed = True
        return []

    def get_native_session_id(self, state: CodexStreamState) -> str | None:
        return state.thread_id

    def transform_turn_response(
        self,
        ctx: SessionContext,
        state: CodexStreamState,
        exit_code: int,
        stderr_tail: Sequence[str],
    ) -> HarnessTurnResponse:
        if state.failed:
            raise HarnessTurnError(f"codex turn failed: {state.error}")
        if exit_code != 0:
            detail = stderr_tail_text(stderr_tail) or state.error or "no output"
            raise HarnessTurnError(f"codex exited with code {exit_code}: {detail}")
        output_json = state.final_text if ctx.output is not None else None
        return HarnessTurnResponse(final_text=state.final_text, output_json=output_json)
