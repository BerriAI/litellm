"""
OpenCode harness config: `opencode run --format json`, once per turn.

Every model call goes to one custom provider (`litellm`, `@ai-sdk/openai-compatible`,
bundled in the binary) whose baseURL is the per-session endpoint. The config travels in
OPENCODE_CONFIG_CONTENT, which opencode applies after global and project config, so a
repo's own opencode.json cannot redirect model calls. The token is never in argv or env:
the config references it with `{file:<private_dir>/token}`. XDG dirs point at a persisted
LiteLLM-owned root so the user's opencode config and auth are never read, and the session
DB outlives a session for resume. Verified against opencode 1.14.41.

opencode files each session under the folder it ran in and only streams a resumed session's
events from that folder, so resuming one filed elsewhere (by an older LiteLLM that left PWD
unset, or from another sandbox) hangs. The first turn after a resume runs `opencode export` in
the sandbox, and when the session is filed elsewhere it imports a copy under fresh ids, filed
under the sandbox folder, and continues the copy. Import overwrites the folder of a session id it
already has, so reusing the ids would break the original wherever it was saved. Before 1.15.6,
import files a session under the folder named in its JSON, so the copy names the sandbox folder.
A failed or timed-out export or import fails that turn instead of hanging, and the next turn
tries again. Verified against opencode 1.14.41, 1.15.5, 1.15.6 and 1.19.3.
"""

from __future__ import annotations

import itertools
import json
import secrets
import string
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from litellm.harness.errors import CapabilityUnsupported, HarnessError, OptionsMismatch
from litellm.harness.options import OpenCodeOptions
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
    HarnessSessionCopy,
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

OPENCODE_BINARY: Final = "opencode"
OPENCODE_PROVIDER_ID: Final = "litellm"
OPENCODE_PROVIDER_NPM: Final = "@ai-sdk/openai-compatible"
# A fixed title skips opencode's extra title-generation model call on the first turn.
OPENCODE_SESSION_TITLE: Final = "litellm-harness"
TOKEN_FILENAME: Final = "token"
RESUME_COPY_FILENAME: Final = "resume-session.json"
ID_ORDERED_PREFIX_LENGTH: Final = 16
ID_RANDOM_LENGTH: Final = 14
ID_RANDOM_ALPHABET: Final = string.ascii_letters + string.digits
INSTRUCTIONS_FILENAME: Final = "instructions.md"
XDG_DIRNAME: Final = "xdg"
XDG_SUBDIRS: Final = ("config", "data", "state", "cache")

# Env that keeps opencode off the network (except the endpoint) and away from ~/.claude.
OPENCODE_ISOLATION_ENV: Final[Mapping[str, str]] = MappingProxyType(
    {
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
)

MANAGED_CONFIG_KEYS: Final = frozenset(
    {
        "provider",
        "model",
        "small_model",
        "permission",
        "tools",
        "enabled_providers",
        "disabled_providers",
        # plugins run arbitrary code as the host user; runs always use --pure
        "plugin",
    }
)
AGENT_MANAGED_KEYS: Final = frozenset({"permission", "tools", "model"})

# Later keys win in opencode, so disable_tools denies go last. `opencode run` auto-rejects
# anything left at "ask", so no mode leaves a tool on ask.
PERMISSION_RULES: Final[Mapping[str, Mapping[str, str]]] = MappingProxyType(
    {
        "read-only": MappingProxyType({"edit": "deny", "bash": "deny", "webfetch": "deny"}),
        "edit": MappingProxyType({"edit": "allow", "bash": "deny", "webfetch": "allow"}),
        "full": MappingProxyType({"*": "allow"}),
    }
)

# opencode gates write, edit and apply_patch with the single `edit` permission.
NORMALIZED_TO_NATIVE: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "read": ("read",),
        "write": ("edit",),
        "edit": ("edit",),
        "bash": ("bash",),
        "glob": ("glob",),
        "grep": ("grep",),
        "ls": ("list",),
        "web_search": ("webfetch", "websearch"),
    }
)

NATIVE_TO_NORMALIZED: Final[Mapping[str, str]] = MappingProxyType(
    {
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
)

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
class OpenCodeStreamState:
    """What the parser has learned from one `opencode run`."""

    session_id: str | None = None
    final_text: str = ""
    error: str | None = None
    step_texts: Sequence[str] = ()


class ExportedPart(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str


class ExportedMessageInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str


class ExportedMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    info: ExportedMessageInfo
    parts: tuple[ExportedPart, ...]


class ExportedSessionInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    directory: str


class ExportedSession(BaseModel):
    """The ids and folder of an `opencode export`; every other field is copied as is."""

    model_config = ConfigDict(frozen=True)

    info: ExportedSessionInfo
    messages: tuple[ExportedMessage, ...]

    def ids(self) -> Iterator[str]:
        yield self.info.id
        for message in self.messages:
            yield message.info.id
            yield from (part.id for part in message.parts)


EXPORT_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


def fresh_id(old: str) -> str:
    """Keeps the type prefix and time-ordered part of an opencode id, so the copy sorts like the original."""
    suffix: Final = "".join(secrets.choice(ID_RANDOM_ALPHABET) for _ in range(ID_RANDOM_LENGTH))
    return f"{old[:ID_ORDERED_PREFIX_LENGTH]}{suffix}"


def with_id_fields_replaced(value: JsonValue, fresh: Mapping[str, str], key: str = "") -> JsonValue:
    match value:
        case str() if key == "id" or key.endswith("ID"):
            return fresh.get(value, value)
        case list():
            return [with_id_fields_replaced(item, fresh, key) for item in value]
        case dict():
            return {name: with_id_fields_replaced(item, fresh, name) for name, item in value.items()}
        case _:
            return value


def filed_under(exported: JsonValue, workdir: str) -> JsonValue:
    match exported:
        case {"info": dict() as info}:
            return {**exported, "info": {**info, "directory": workdir}}
        case _:
            return exported


def session_copy(exported: str, path: str, workdir: str) -> HarnessSessionCopy | None:
    """A copy of an exported session filed under workdir, or None when the session is already filed there."""
    try:
        raw: Final = EXPORT_JSON.validate_json(exported)
        session: Final = ExportedSession.model_validate(raw)
    except ValidationError as exc:
        raise HarnessError(f"opencode export did not print a session: {exc.errors()[0]['msg']}") from exc
    if session.info.directory == workdir:
        return None
    fresh: Final = MappingProxyType({old: fresh_id(old) for old in session.ids()})
    return HarnessSessionCopy(
        session_id=fresh[session.info.id],
        path=path,
        data=json.dumps(filed_under(with_id_fields_replaced(raw, fresh), workdir)).encode("utf-8"),
        import_command=(OPENCODE_BINARY, "--pure", "import", path),
    )


def _as_dict(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else MappingProxyType({})


def _tool_events(part: Mapping[str, object]) -> Sequence[Event]:
    native = str(part.get("tool") or "")
    call_id = str(part.get("callID") or part.get("id") or "")
    state = _as_dict(part.get("state"))
    tool_input = state.get("input")
    call = ToolCall(
        id=call_id,
        name=normalize_tool_name(native, NATIVE_TO_NORMALIZED),
        native_name=native,
        input=tool_input if isinstance(tool_input, dict) else MappingProxyType({"input": tool_input}),
        builtin=native in OPENCODE_BUILTIN_TOOLS,
    )
    if state.get("status") == "error":
        message = str(state.get("error") or state.get("output") or "tool failed")
        return event_list(call, ToolResult(id=call_id, output=message, is_error=True))
    output = state.get("output")
    text = output if isinstance(output, str) else json.dumps(output)
    # The `invalid` pseudo-tool is how opencode reports a call to an unavailable tool.
    return event_list(call, ToolResult(id=call_id, output=text, is_error=native == "invalid"))


def _error_message(error: object) -> str:
    if not isinstance(error, dict):
        return str(error or "opencode reported an error")
    data = error.get("data")
    if isinstance(data, dict) and data.get("message"):
        return str(data["message"])
    return str(error.get("name") or "opencode reported an error")


def validate_user_config(config: Mapping[str, object]) -> None:
    """Reject OpenCodeOptions.config keys LiteLLM manages (or that bypass permissions)."""
    for key in config:
        if key in MANAGED_CONFIG_KEYS:
            raise OptionsMismatch(
                f"OpenCodeOptions.config[{key!r}] is managed by LiteLLM; use the matching "
                "agent() argument (model=, permissions=, disable_tools=) instead"
            )
    for section in ("agent", "mode"):
        entries = config.get(section)
        if entries is None:
            continue
        if not isinstance(entries, Mapping):
            raise OptionsMismatch(f"OpenCodeOptions.config[{section!r}] must be a mapping")
        for name, agent in entries.items():
            managed = AGENT_MANAGED_KEYS & frozenset(agent or ())
            if managed:
                raise OptionsMismatch(
                    f"OpenCodeOptions.config[{section!r}][{name!r}] sets {sorted(managed)}, "
                    "which LiteLLM manages; use permissions=/disable_tools=/model= instead"
                )


def permission_rules(permissions: PermissionMode, disable_tools: Sequence[str]) -> Mapping[str, str]:
    """opencode `permission` config for a mode plus denies for disable_tools."""
    if permissions not in PERMISSION_RULES:
        raise CapabilityUnsupported(
            f"Harness.OPENCODE does not support permissions={permissions!r} (supported: {sorted(PERMISSION_RULES)})"
        )
    denied: Final = native_tool_names(disable_tools, NORMALIZED_TO_NATIVE)
    # Denies go last (later keys win in opencode), so drop them from the mode rules first.
    kept: Final = ((key, value) for key, value in PERMISSION_RULES[permissions].items() if key not in denied)
    rules: Final = itertools.chain(kept, ((native, "deny") for native in denied))
    return dict(rules)  # mutable-ok: opencode config JSON


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
) -> Mapping[str, object]:
    """The full opencode config: user config underneath, LiteLLM-managed keys on top."""
    user: Final = user_config or MappingProxyType({})
    validate_user_config(user)
    qualified = f"{OPENCODE_PROVIDER_ID}/{model}"
    extra_instructions: Final = (instructions_path,) if instructions_path else ()
    instructions: Final = [*(user.get("instructions") or ()), *extra_instructions]  # mutable-ok: opencode config JSON
    user_skills = _as_dict(user.get("skills"))
    extra_skills: Final = (skills_path,) if skills_path else ()
    skill_paths: Final = [*(user_skills.get("paths") or ()), *extra_skills]  # mutable-ok: opencode config JSON
    options: Final = {"baseURL": base_url, "apiKey": "{file:" + token_path + "}"}  # mutable-ok: opencode config JSON
    models: Final[dict[str, Mapping[str, object]]] = {model: {}}  # mutable-ok: opencode config JSON
    provider: Final = {  # mutable-ok: opencode config JSON
        "npm": OPENCODE_PROVIDER_NPM,
        "name": "LiteLLM",
        "options": options,
        "models": models,
    }
    managed: Final = {  # mutable-ok: opencode config JSON
        "provider": {OPENCODE_PROVIDER_ID: provider},  # mutable-ok: opencode config JSON
        "enabled_providers": [OPENCODE_PROVIDER_ID],  # mutable-ok: opencode config JSON
        "model": qualified,
        "small_model": qualified,
        "permission": permission_rules(permissions, disable_tools),
        "autoupdate": False,
        "share": "disabled",
    }
    skills: Final = {**user_skills, "paths": skill_paths}  # mutable-ok: opencode config JSON
    optional: Final = (("instructions", instructions), ("skills", skills if skill_paths else None))
    present: Final = ((key, value) for key, value in optional if value)
    return {**user, **managed, **dict(present)}  # mutable-ok: opencode config JSON


class OpenCodeHarnessConfig(BaseCLIHarnessConfig):
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

    def get_binary(self) -> str:
        return OPENCODE_BINARY

    def get_install_hint(self) -> str:
        return "npm install -g opencode-ai (or brew install sst/tap/opencode)"

    def validate_environment(self, ctx: SessionContext) -> None:
        options: OpenCodeOptions = self.get_options(ctx)
        validate_user_config(options.config)

    def transform_session_setup(self, ctx: SessionContext, private_dir: str) -> HarnessSessionSetup:
        if ctx.endpoint is None:
            raise HarnessError("OpenCode needs the session model endpoint")
        model = ctx.model or ctx.endpoint.model
        if not model:
            raise ValueError("Harness.OPENCODE needs model= (a gateway model group or litellm model)")
        options: OpenCodeOptions = self.get_options(ctx)
        instructions = build_instructions(ctx)
        token: Final = ctx.endpoint.token.encode("utf-8")
        files: Final = (
            MappingProxyType({TOKEN_FILENAME: token, INSTRUCTIONS_FILENAME: instructions.encode("utf-8")})
            if instructions is not None
            else MappingProxyType({TOKEN_FILENAME: token})
        )
        config = build_opencode_config(
            model=model,
            base_url=ctx.sandbox.host_url(ctx.endpoint.port).rstrip("/") + "/v1",
            token_path=f"{private_dir}/{TOKEN_FILENAME}",
            permissions=ctx.permissions,
            disable_tools=ctx.disable_tools,
            user_config=options.config,
            instructions_path=f"{private_dir}/{INSTRUCTIONS_FILENAME}" if instructions is not None else None,
            skills_path=f"{private_dir}/skills" if ctx.skills else None,
        )
        xdg: Final = MappingProxyType(
            {f"XDG_{sub.upper()}_HOME": f"{private_dir}/{XDG_DIRNAME}/{sub}" for sub in XDG_SUBDIRS}
        )
        return HarnessSessionSetup(
            files=files,
            persisted_dirs=((XDG_DIRNAME, "opencode"),),
            skills_dir="skills",
            env=MappingProxyType(
                {**OPENCODE_ISOLATION_ENV, **options.env, **xdg, "OPENCODE_CONFIG_CONTENT": json.dumps(config)}
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
        options: OpenCodeOptions = self.get_options(ctx)
        model = ctx.model or (ctx.endpoint.model if ctx.endpoint else None)
        # --pure: never load plugins. A repo's .opencode/plugin/*.js would otherwise run as the
        # host user at startup, before any tool permission applies.
        argv: Final = (
            OPENCODE_BINARY,
            "run",
            "--pure",
            "--format",
            "json",
            "--thinking",
            "-m",
            f"{OPENCODE_PROVIDER_ID}/{model}",
            *(("--agent", options.agent) if options.agent else ()),
            *(("--session", native_session_id) if native_session_id else ("--title", OPENCODE_SESSION_TITLE)),
        )
        # The prompt goes on stdin; opencode appends non-TTY stdin to the message.
        return HarnessTurnRequest(argv=argv, env=setup.env, stdin=turn_prompt(ctx, prompt), cwd=ctx.sandbox.workdir)

    def transform_resume_export_command(self, native_session_id: str) -> Sequence[str]:
        return (OPENCODE_BINARY, "--pure", "export", native_session_id)

    def transform_resume_copy(self, exported: str, private_dir: str, workdir: str) -> HarnessSessionCopy | None:
        return session_copy(exported, f"{private_dir}/{RESUME_COPY_FILENAME}", workdir)

    def create_stream_state(self) -> OpenCodeStreamState:
        return OpenCodeStreamState()

    def transform_stream_line(self, line: Mapping[str, object], state: OpenCodeStreamState) -> Sequence[Event]:
        """step_finish token counts are ignored on purpose: the session endpoint accounts usage."""
        session_id = line.get("sessionID")
        if session_id and state.session_id is None:
            state.session_id = str(session_id)
        event_type = line.get("type")
        part = _as_dict(line.get("part"))
        if event_type == "step_start":
            state.step_texts = ()
            return event_list()
        if event_type == "text":
            text = str(part.get("text") or "")
            if not text:
                return event_list()
            state.step_texts = (*state.step_texts, text)
            state.final_text = "\n\n".join(state.step_texts)
            return event_list(Text(delta=text))
        if event_type == "reasoning":
            text = str(part.get("text") or "")
            return event_list(Reasoning(delta=text)) if text else event_list()
        if event_type == "tool_use":
            return _tool_events(part)
        if event_type == "error":
            message = _error_message(line.get("error"))
            state.error = f"{state.error}\n{message}" if state.error else message
        return event_list()

    def get_native_session_id(self, state: OpenCodeStreamState) -> str | None:
        return state.session_id

    def transform_turn_response(
        self,
        ctx: SessionContext,
        state: OpenCodeStreamState,
        exit_code: int,
        stderr_tail: Sequence[str],
    ) -> HarnessTurnResponse:
        # opencode exits 0 after an `error` event, so check state first.
        if state.error:
            raise HarnessTurnError(f"opencode turn failed: {state.error}")
        if exit_code != 0:
            raise HarnessTurnError(
                f"opencode exited with code {exit_code}: {stderr_tail_text(stderr_tail) or 'no output'}"
            )
        output_json = last_json_object(state.final_text) if ctx.output is not None else None
        return HarnessTurnResponse(final_text=state.final_text, output_json=output_json)
