"""
pi harness config: `pi --mode json` (JSONL session events), once per turn.

Every model call goes to one custom provider (`litellm`, api=openai-completions) declared in
the models.json of a LiteLLM-owned PI_CODING_AGENT_DIR, so the user's own pi settings, auth,
extensions, MCP servers and skills are never read. PiOptions.config is pi's own config:
`mcpServers` goes to that dir's mcp.json and every other key to its settings.json, minus the
keys LiteLLM manages. The provider's apiKey is the `$`-reference
pi resolves from LITELLM_HARNESS_TOKEN, so the token is never in argv or on disk. pi exits 0
after a failed provider call, so failure is read from the last assistant message instead.
Verified against @earendil-works/pi-coding-agent 1.1.0.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.harness.errors import CapabilityUnsupported, HarnessError, OptionsMismatch
from litellm.harness.options import PiOptions, PiThinkingLevel
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
from litellm.llms.base_llm.harness.transformation import (
    BaseCLIHarnessConfig,
    HarnessSessionSetup,
    HarnessTurnError,
    HarnessTurnRequest,
    HarnessTurnResponse,
    event_list,
)
from litellm.llms.base_llm.harness.utils import (
    build_instructions,
    last_json_object,
    native_tool_names,
    normalize_tool_name,
    stderr_tail_text,
    turn_prompt,
)

if TYPE_CHECKING:
    from litellm.harness.context import SessionContext

PI_BINARY: Final = "pi"
PI_PROVIDER_ID: Final = "litellm"
PI_TOKEN_ENV: Final = "LITELLM_HARNESS_TOKEN"
PI_CONFIG_DIR_ENV: Final = "PI_CODING_AGENT_DIR"
AGENT_DIRNAME: Final = "agent"
SESSIONS_DIRNAME: Final = "sessions"
SKILLS_DIRNAME: Final = "skills"
INSTRUCTIONS_FILENAME: Final = "instructions.md"
MODELS_FILENAME: Final = f"{AGENT_DIRNAME}/models.json"
SETTINGS_FILENAME: Final = f"{AGENT_DIRNAME}/settings.json"
MCP_FILENAME: Final = f"{AGENT_DIRNAME}/mcp.json"
MCP_CONFIG_KEY: Final = "mcpServers"
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
MCP_TOOL_PATTERN: Final = "mcp__*"
EXPOSURE_TOOLS: Final = (("codemode", "codemode"), ("codemode-deferred", "codemode"), ("deferred", "tool_search"))

# settings.json keys that would reroute model calls, bypass permissions=, or load code as the host user.
MANAGED_CONFIG_KEYS: Final = frozenset(
    {
        "defaultProvider",
        "defaultModel",
        "enabledModels",
        "defaultThinkingLevel",
        "defaultTools",
        "defaultProjectTrust",
        "sessionDir",
        "httpProxy",
        "packages",
        "extensions",
        "skills",
    }
)

MANAGED_ENV_KEYS: Final = frozenset({PI_CONFIG_DIR_ENV, PI_TOKEN_ENV})

PI_ISOLATION_ENV: Final[Mapping[str, str]] = MappingProxyType(
    {
        "PI_OFFLINE": "1",
        "PI_SKIP_VERSION_CHECK": "1",
        "PI_TELEMETRY": "0",
    }
)

PERMISSION_TOOLS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "read-only": ("read", "grep", "find", "ls"),
        "edit": ("read", "edit", "write", "grep", "find", "ls"),
        "full": ("read", "bash", "edit", "write", "grep", "find", "ls"),
    }
)

NORMALIZED_TO_NATIVE: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "read": ("read",),
        "write": ("write",),
        "edit": ("edit",),
        "bash": ("bash", "powershell"),
        "glob": ("find",),
        "grep": ("grep",),
        "ls": ("ls",),
    }
)

NATIVE_TO_NORMALIZED: Final[Mapping[str, str]] = MappingProxyType(
    {
        "read": "read",
        "write": "write",
        "edit": "edit",
        "bash": "bash",
        "powershell": "bash",
        "find": "glob",
        "grep": "grep",
        "ls": "ls",
    }
)

PI_THINKING_LEVEL_MAP: Final[Mapping[str, str]] = MappingProxyType({"xhigh": "xhigh", "max": "max"})

PI_BUILTIN_TOOLS: Final = frozenset({*NATIVE_TO_NORMALIZED, "codemode", "tool_search"})
FAILED_STOP_REASONS: Final = frozenset({"error", "aborted"})


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class _McpServer(_Frozen):
    exposure: str = "codemode"
    tool_exposure: Mapping[str, str] = Field(default_factory=lambda: MappingProxyType({}), alias="toolExposure")

    def exposures(self) -> frozenset[str]:
        return frozenset({self.exposure, *self.tool_exposure.values()})


_MCP_SERVERS: Final = TypeAdapter(Mapping[str, _McpServer])


class _ContentBlock(_Frozen):
    type: str
    text: str = ""


class _Role(_Frozen):
    role: str = ""


class _AssistantMessage(_Frozen):
    content: tuple[_ContentBlock, ...] = ()
    stop_reason: str | None = Field(default=None, alias="stopReason")
    error_message: str | None = Field(default=None, alias="errorMessage")


class _ToolOutput(_Frozen):
    content: tuple[_ContentBlock, ...] = ()


class _ToolStart(_Frozen):
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    args: Mapping[str, object] = Field(default_factory=lambda: MappingProxyType({}))


class _ToolEnd(_Frozen):
    tool_call_id: str = Field(alias="toolCallId")
    result: _ToolOutput | str | None = None
    is_error: bool = Field(default=False, alias="isError")


class _MessageUpdate(_Frozen):
    type: str
    delta: str = ""


@dataclass
class PiStreamState:
    """What the parser has learned from one `pi --mode json` run."""

    session_id: str | None = None
    final_text: str = ""
    stop_reason: str | None = None
    error: str | None = None

    def record_session(self, session_id: object) -> None:
        if isinstance(session_id, str) and self.session_id is None:
            self.session_id = session_id

    def record_message_end(self, message: object) -> None:
        if _Role.model_validate(message).role != "assistant":
            return
        assistant: Final = _AssistantMessage.model_validate(message)
        self.final_text = _text_of(assistant.content)
        self.stop_reason = assistant.stop_reason
        self.error = assistant.error_message

    def mark_aborted(self) -> None:
        self.stop_reason = "aborted"


def _text_of(blocks: Sequence[_ContentBlock]) -> str:
    return "".join(block.text for block in blocks if block.type == "text")


def _tool_output_text(result: _ToolOutput | str | None) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    return _text_of(result.content)


def _message_update_events(update: object) -> Sequence[Event]:
    try:
        event: Final = _MessageUpdate.model_validate(update)
    except ValidationError:
        return event_list()
    if not event.delta:
        return event_list()
    if event.type == "text_delta":
        return event_list(Text(delta=event.delta))
    if event.type == "thinking_delta":
        return event_list(Reasoning(delta=event.delta))
    return event_list()


def _tool_start_events(line: Mapping[str, object]) -> Sequence[Event]:
    start: Final = _ToolStart.model_validate(line)
    return event_list(
        ToolCall(
            id=start.tool_call_id,
            name=normalize_tool_name(start.tool_name, NATIVE_TO_NORMALIZED),
            native_name=start.tool_name,
            input=start.args,
            builtin=start.tool_name in PI_BUILTIN_TOOLS,
        )
    )


def _tool_end_events(line: Mapping[str, object]) -> Sequence[Event]:
    end: Final = _ToolEnd.model_validate(line)
    return event_list(ToolResult(id=end.tool_call_id, output=_tool_output_text(end.result), is_error=end.is_error))


def tool_args(
    permissions: PermissionMode, disable_tools: Sequence[str], mcp_tools: Sequence[str] = ()
) -> tuple[str, ...]:
    """`--tools` allowlist for a permission mode and MCP servers, plus `--exclude-tools` for disable_tools."""
    if permissions not in PERMISSION_TOOLS:
        raise CapabilityUnsupported(
            f"Harness.PI does not support permissions={permissions!r} (supported: {sorted(PERMISSION_TOOLS)})"
        )
    denied: Final = native_tool_names(disable_tools, NORMALIZED_TO_NATIVE)
    exclude: Final = ("--exclude-tools", ",".join(denied)) if denied else ()
    return ("--tools", ",".join((*PERMISSION_TOOLS[permissions], *mcp_tools)), *exclude)


def mcp_tool_entries(config: Mapping[str, object]) -> tuple[str, ...]:
    """`--tools` entries that keep configured MCP servers reachable.

    A plain-name `--tools` list drops MCP tools in pi 1.1.0, so `mcp__*` is listed whenever servers
    are configured, plus the tool each exposure is reached through: `codemode` (pi's default) or
    `tool_search` for `deferred`.
    """
    servers: Final = _MCP_SERVERS.validate_python(config.get(MCP_CONFIG_KEY) or {})
    if not servers:
        return ()
    exposures: Final = frozenset(itertools.chain.from_iterable(server.exposures() for server in servers.values()))
    reach: Final = tuple(tool for exposure, tool in EXPOSURE_TOOLS if exposure in exposures)
    return (MCP_TOOL_PATTERN, *reach)


def session_model(ctx: SessionContext) -> str:
    """The model pi is pointed at, from agent(model=) or the session endpoint."""
    model: Final = ctx.model or (ctx.endpoint.model if ctx.endpoint else None)
    if not model:
        raise ValueError("Harness.PI needs model= (a gateway model group or litellm model)")
    return model


def build_models_json(model: str, base_url: str, thinking: PiThinkingLevel | None) -> str:
    # pi clamps --thinking to "off" unless the model declares reasoning, and offers xhigh/max only when
    # thinkingLevelMap names them. Declaring reasoning without a requested level would make pi send its
    # default reasoning_effort on every call, so it is declared only when a level is asked for
    reasoning: Final = (
        MappingProxyType({"reasoning": True, "thinkingLevelMap": PI_THINKING_LEVEL_MAP})
        if thinking is not None and thinking != "off"
        else MappingProxyType({})
    )
    provider: Final = MappingProxyType(
        {
            "baseUrl": base_url,
            "api": "openai-completions",
            "apiKey": f"${PI_TOKEN_ENV}",
            "models": (MappingProxyType({"id": model, **reasoning}),),
        }
    )
    return json.dumps({"providers": {PI_PROVIDER_ID: provider}}, default=_plain_json)


def validate_user_config(config: Mapping[str, object]) -> None:
    """Reject PiOptions.config keys LiteLLM manages, and a malformed mcpServers."""
    managed: Final = sorted(MANAGED_CONFIG_KEYS.intersection(config))
    if managed:
        raise OptionsMismatch(
            f"PiOptions.config may not set {', '.join(managed)}; LiteLLM manages it. Use the matching "
            "agent() argument (model=, permissions=, disable_tools=, skills=) or PiOptions.thinking instead"
        )
    try:
        _MCP_SERVERS.validate_python(config.get(MCP_CONFIG_KEY) or {})
    except ValidationError as e:
        raise OptionsMismatch(
            f"PiOptions.config[{MCP_CONFIG_KEY!r}] must map each server name to its config: {e}"
        ) from e


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return _JSON_OBJECT.validate_python(value)
    raise TypeError(f"{type(value).__name__} in PiOptions.config is not JSON serializable")


def config_files(config: Mapping[str, object]) -> tuple[tuple[str, bytes], ...]:
    """settings.json for every key but mcpServers, and mcp.json for mcpServers; each only when non-empty."""
    settings: Final = MappingProxyType({key: value for key, value in config.items() if key != MCP_CONFIG_KEY})
    servers: Final = config.get(MCP_CONFIG_KEY)
    settings_file: Final = ((SETTINGS_FILENAME, json.dumps(settings, default=_plain_json)),) if settings else ()
    mcp_file: Final = ((MCP_FILENAME, json.dumps({MCP_CONFIG_KEY: servers}, default=_plain_json)),) if servers else ()
    return tuple((path, text.encode("utf-8")) for path, text in (*settings_file, *mcp_file))


class PiHarnessConfig(BaseCLIHarnessConfig[PiOptions, PiStreamState]):
    harness = Harness.PI
    options_type = PiOptions
    capabilities = Capabilities(
        structured_output=True,
        tool_approval=False,
        tool_filtering=True,
        history=False,
        custom_tools=False,
        skills=True,
        resume=True,
        permission_modes=frozenset(PERMISSION_TOOLS),
    )

    def get_binary(self) -> str:
        return PI_BINARY

    def get_install_hint(self) -> str:
        return "npm install -g @earendil-works/pi-coding-agent"

    def validate_environment(self, ctx: SessionContext) -> None:
        options: Final = self.get_options(ctx)
        clashing: Final = sorted(MANAGED_ENV_KEYS.intersection(options.env))
        if clashing:
            raise OptionsMismatch(f"PiOptions.env may not set {', '.join(clashing)}; LiteLLM manages it")
        validate_user_config(options.config)
        tool_args(ctx.permissions, ctx.disable_tools, mcp_tool_entries(options.config))

    def transform_session_setup(self, ctx: SessionContext, private_dir: str) -> HarnessSessionSetup:
        if ctx.endpoint is None:
            raise HarnessError("pi needs the session model endpoint")
        model: Final = session_model(ctx)
        options: Final = self.get_options(ctx)
        base_url: Final = ctx.sandbox.host_url(ctx.endpoint.port).rstrip("/") + "/v1"
        models_file: Final = (MODELS_FILENAME, build_models_json(model, base_url, options.thinking).encode("utf-8"))
        instructions: Final = build_instructions(ctx)
        instructions_file: Final = (
            ((INSTRUCTIONS_FILENAME, instructions.encode("utf-8")),) if instructions is not None else ()
        )
        return HarnessSessionSetup(
            files=MappingProxyType(dict((models_file, *config_files(options.config), *instructions_file))),
            persisted_dirs=((SESSIONS_DIRNAME, "pi/sessions"),),
            skills_dir=SKILLS_DIRNAME,
            env=MappingProxyType(
                {
                    **PI_ISOLATION_ENV,
                    **options.env,
                    PI_CONFIG_DIR_ENV: f"{private_dir}/{AGENT_DIRNAME}",
                    PI_TOKEN_ENV: ctx.endpoint.token,
                }
            ),
        )

    def transform_turn_request(
        self,
        ctx: SessionContext,
        setup: HarnessSessionSetup,
        private_dir: str,
        prompt: str,
        native_session_id: str | None,
    ) -> HarnessTurnRequest:
        options: Final = self.get_options(ctx)
        # --no-approve: a repo's .pi/extensions would otherwise run as the host user at startup.
        # --no-skills: ~/.agents/skills is discovered outside PI_CODING_AGENT_DIR.
        argv: Final = (
            PI_BINARY,
            "--mode",
            "json",
            "--no-approve",
            "--no-skills",
            "--provider",
            PI_PROVIDER_ID,
            "--model",
            session_model(ctx),
            "--session-dir",
            f"{private_dir}/{SESSIONS_DIRNAME}",
            *(("--session", native_session_id) if native_session_id else ()),
            *tool_args(ctx.permissions, ctx.disable_tools, mcp_tool_entries(options.config)),
            *(("--thinking", options.thinking) if options.thinking else ()),
            *(("--skill", f"{private_dir}/{SKILLS_DIRNAME}") if ctx.skills else ()),
            *(
                ("--append-system-prompt", f"{private_dir}/{INSTRUCTIONS_FILENAME}")
                if INSTRUCTIONS_FILENAME in setup.files
                else ()
            ),
        )
        # The prompt goes on stdin; pi prepends piped stdin to the (empty) first message.
        return HarnessTurnRequest(argv=argv, env=setup.env, stdin=turn_prompt(ctx, prompt), cwd=ctx.sandbox.workdir)

    def create_stream_state(self) -> PiStreamState:
        return PiStreamState()

    def transform_stream_line(self, line: Mapping[str, object], state: PiStreamState) -> Sequence[Event]:
        """Usage on message_update is ignored on purpose: the session endpoint accounts it."""
        event_type: Final = line.get("type")
        if event_type == "session":
            state.record_session(line.get("id"))
            return event_list()
        if event_type == "message_update":
            return _message_update_events(line.get("assistantMessageEvent"))
        if event_type == "message_end":
            state.record_message_end(line.get("message"))
            return event_list()
        if event_type == "tool_execution_start":
            return _tool_start_events(line)
        if event_type == "tool_execution_end":
            return _tool_end_events(line)
        if event_type == "agent_settled" and line.get("aborted") is True:
            state.mark_aborted()
        return event_list()

    def get_native_session_id(self, state: PiStreamState) -> str | None:
        return state.session_id

    def transform_turn_response(
        self,
        ctx: SessionContext,
        state: PiStreamState,
        exit_code: int,
        stderr_tail: Sequence[str],
    ) -> HarnessTurnResponse:
        if state.stop_reason in FAILED_STOP_REASONS:
            raise HarnessTurnError(f"pi turn failed: {state.error or state.stop_reason}")
        if exit_code != 0:
            raise HarnessTurnError(f"pi exited with code {exit_code}: {stderr_tail_text(stderr_tail) or 'no output'}")
        output_json: Final = last_json_object(state.final_text) if ctx.output is not None else None
        return HarnessTurnResponse(final_text=state.final_text, output_json=output_json)
