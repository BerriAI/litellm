import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import click
import requests
from pydantic import BaseModel, TypeAdapter, ValidationError

from .auth import CliContextObj, context_secret_vault, get_stored_api_key, login
from .cmd_quoting import quote_for_cmd

ANTHROPIC_BASE_URL_ENV: Final = "ANTHROPIC_BASE_URL"
ANTHROPIC_AUTH_TOKEN_ENV: Final = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_API_KEY_ENV: Final = "ANTHROPIC_API_KEY"
ENABLE_TOOL_SEARCH_ENV: Final = "ENABLE_TOOL_SEARCH"
ENABLE_TOOL_SEARCH_VALUE: Final = "true"
ENABLE_GATEWAY_MODEL_DISCOVERY_ENV: Final = "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"
ENABLE_GATEWAY_MODEL_DISCOVERY_VALUE: Final = "1"
OPENAI_BASE_URL_ENV: Final = "OPENAI_BASE_URL"
OPENAI_API_KEY_ENV: Final = "OPENAI_API_KEY"
OPENCODE_CONFIG_CONTENT_ENV: Final = "OPENCODE_CONFIG_CONTENT"
OPENCODE_PROVIDER_ID: Final = "litellm"
OPENCODE_PROVIDER_NAME: Final = "LiteLLM"
OPENCODE_PROVIDER_NPM: Final = "@ai-sdk/openai-compatible"

_SKIP_VERIFY_FLAG: Final = "--skip-verify"

PROFILE_ANTHROPIC: Final = "anthropic"
PROFILE_OPENAI: Final = "openai"

_KNOWN_AGENTS: Final[dict[str, tuple[str, frozenset[str]]]] = {
    "claude": ("Claude Code", frozenset({PROFILE_ANTHROPIC})),
    "codex": ("Codex", frozenset({PROFILE_OPENAI})),
    "opencode": ("OpenCode", frozenset({PROFILE_OPENAI})),
}

_INSTALL_DOCS: Final[dict[str, str]] = {
    "claude": "https://docs.claude.com/en/docs/claude-code/setup",
    "codex": "https://developers.openai.com/codex/cli",
    "opencode": "https://opencode.ai/docs",
}

CODEX_PROXY_PROVIDER: Final = "litellm"


class AgentRunError(Exception):
    """Raised for any user-actionable failure while preparing to run an agent."""


def agent_profile(command: str) -> tuple[str, frozenset[str]]:
    """Return the (display name, env profiles) for a wrapped command.

    Known agents map to the API family they speak. Anything else gets both
    families so it works regardless of which env vars the tool reads.
    """
    base: Final = os.path.basename(command)
    if base in _KNOWN_AGENTS:
        return _KNOWN_AGENTS[base]
    return base, frozenset({PROFILE_ANTHROPIC, PROFILE_OPENAI})


def build_agent_env(
    base_env: Mapping[str, str],
    base_url: str,
    api_key: str,
    profiles: frozenset[str],
) -> dict[str, str]:
    """Return a copy of base_env wired to route the agent through the proxy.

    Anthropic clients (Claude Code) append /v1/messages to ANTHROPIC_BASE_URL,
    so it stays the bare proxy root; OpenAI clients (Codex, OpenCode) expect the
    /v1 suffix on OPENAI_BASE_URL. ANTHROPIC_API_KEY is dropped so a stray
    Anthropic key cannot win over the bearer token we set. ENABLE_TOOL_SEARCH
    defaults to true because Claude Code turns tool search off when
    ANTHROPIC_BASE_URL is not a first-party Anthropic host; a value already in
    the environment is left alone. CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY
    defaults to 1 so Claude Code (v2.1.129+) fills its /model picker from the
    proxy's /v1/models; likewise left alone when already set.
    """
    env: Final = dict(base_env)
    root: Final = base_url.rstrip("/")
    if PROFILE_ANTHROPIC in profiles:
        env[ANTHROPIC_BASE_URL_ENV] = root
        env[ANTHROPIC_AUTH_TOKEN_ENV] = api_key
        env.pop(ANTHROPIC_API_KEY_ENV, None)
        if ENABLE_TOOL_SEARCH_ENV not in env:
            env[ENABLE_TOOL_SEARCH_ENV] = ENABLE_TOOL_SEARCH_VALUE
        if ENABLE_GATEWAY_MODEL_DISCOVERY_ENV not in env:
            env[ENABLE_GATEWAY_MODEL_DISCOVERY_ENV] = ENABLE_GATEWAY_MODEL_DISCOVERY_VALUE
    if PROFILE_OPENAI in profiles:
        env[OPENAI_BASE_URL_ENV] = root + "/v1"
        env[OPENAI_API_KEY_ENV] = api_key
    return env


def _codex_proxy_args(base_url: str) -> list[str]:
    """Codex `-c` overrides that point it at the proxy.

    Codex ignores OPENAI_BASE_URL (it always dials api.openai.com), so the env
    profile alone cannot route it. It does honor a custom provider, so define one
    inline; supports_websockets=false forces the HTTP/SSE Responses transport
    because the proxy does not speak the Responses WebSocket protocol. The key is
    read from OPENAI_API_KEY, which build_agent_env already exports.
    """
    root: Final = base_url.rstrip("/") + "/v1"
    provider: Final = f"model_providers.{CODEX_PROXY_PROVIDER}"
    return [
        "-c",
        f'model_provider="{CODEX_PROXY_PROVIDER}"',
        "-c",
        f'{provider}.name="LiteLLM proxy"',
        "-c",
        f'{provider}.base_url="{root}"',
        "-c",
        f'{provider}.env_key="{OPENAI_API_KEY_ENV}"',
        "-c",
        f'{provider}.wire_api="responses"',
        "-c",
        f"{provider}.supports_websockets=false",
    ]


_PROXY_ARGS: Final[dict[str, Callable[[str], list[str]]]] = {
    "codex": _codex_proxy_args,
}


def agent_launch_args(command: str, base_url: str) -> list[str]:
    """Extra CLI args an agent needs to actually honor the proxy.

    Claude Code and OpenCode respect the exported env vars, so they get nothing
    here; Codex needs its provider pointed via config overrides.
    """
    builder: Final = _PROXY_ARGS.get(os.path.basename(command))
    return builder(base_url) if builder else []


class ListedModel(BaseModel):
    """The fields of a /v1/models entry that an OpenCode model entry is built from."""

    id: str
    mode: str | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None


class _ModelListing(BaseModel):
    data: tuple[ListedModel, ...]


_MODEL_LISTING: Final = TypeAdapter(_ModelListing)
_OPENCODE_CHAT_MODES: Final[frozenset[str]] = frozenset({"chat", "responses"})
_NO_EXTRA_ENV: Final[Mapping[str, str]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ModelSyncSkipped:
    reason: str


class _OpenCodeLimit(BaseModel):
    context: int
    output: int


class _OpenCodeModel(BaseModel):
    name: str
    limit: _OpenCodeLimit | None = None


class _OpenCodeProviderOptions(BaseModel):
    baseURL: str
    apiKey: str


class _OpenCodeProvider(BaseModel):
    npm: str
    name: str
    options: _OpenCodeProviderOptions
    models: Mapping[str, _OpenCodeModel]


class _OpenCodeConfig(BaseModel):
    provider: Mapping[str, _OpenCodeProvider]


def _opencode_model_entry(model: ListedModel) -> _OpenCodeModel:
    if model.max_input_tokens is None or model.max_output_tokens is None:
        return _OpenCodeModel(name=model.id)
    return _OpenCodeModel(
        name=model.id, limit=_OpenCodeLimit(context=model.max_input_tokens, output=model.max_output_tokens)
    )


def opencode_provider_config(base_url: str, models: Sequence[ListedModel]) -> str:
    """OPENCODE_CONFIG_CONTENT declaring the proxy as OpenCode provider `litellm`.

    One model entry per chat-capable /v1/models row (mode chat, responses, or
    unknown), so OpenCode's model picker mirrors what the key can call. The key
    is read back through {env:OPENAI_API_KEY}, which build_agent_env exports, so
    it never lands in the config text. OpenCode merges this inline config over
    the user's own files, leaving unrelated keys and providers untouched.
    """
    chat_models: Final = tuple(m for m in models if m.mode is None or m.mode in _OPENCODE_CHAT_MODES)
    provider: Final = _OpenCodeProvider(
        npm=OPENCODE_PROVIDER_NPM,
        name=OPENCODE_PROVIDER_NAME,
        options=_OpenCodeProviderOptions(
            baseURL=base_url.rstrip("/") + "/v1",
            apiKey=f"{{env:{OPENAI_API_KEY_ENV}}}",
        ),
        models=MappingProxyType({m.id: _opencode_model_entry(m) for m in chat_models}),
    )
    config: Final = _OpenCodeConfig(provider=MappingProxyType({OPENCODE_PROVIDER_ID: provider}))
    return config.model_dump_json(exclude_none=True)


def opencode_model_sync_env(
    base_env: Mapping[str, str],
    base_url: str,
    api_key: str,
    *,
    get: Callable[..., requests.Response] = requests.get,
) -> Mapping[str, str] | ModelSyncSkipped:
    """Env addition that hands OpenCode the proxy's model list, or why it was skipped.

    Fetches /v1/models with the key and packs it into OPENCODE_CONFIG_CONTENT.
    An OPENCODE_CONFIG_CONTENT already in the environment is left alone, and a
    failed fetch is reported rather than raised: OpenCode still launches on the
    plain OPENAI_* env, just without a synced model list.
    """
    if OPENCODE_CONFIG_CONTENT_ENV in base_env:
        return ModelSyncSkipped(f"{OPENCODE_CONFIG_CONTENT_ENV} is already set")
    url: Final = base_url.rstrip("/") + "/v1/models"
    try:
        resp: Final = get(url, headers=MappingProxyType({"Authorization": f"Bearer {api_key}"}), timeout=10)
    except requests.RequestException as e:
        return ModelSyncSkipped(f"could not reach {url}: {e}")
    if resp.status_code != 200:
        return ModelSyncSkipped(f"{url} returned HTTP {resp.status_code}")
    try:
        listing: Final = _MODEL_LISTING.validate_json(resp.content)
    except ValidationError:
        return ModelSyncSkipped(f"{url} returned an unexpected body")
    return MappingProxyType({OPENCODE_CONFIG_CONTENT_ENV: opencode_provider_config(base_url, listing.data)})


def agent_model_sync_env(
    command: str,
    base_env: Mapping[str, str],
    base_url: str,
    api_key: str,
    skip_verify: bool,
    *,
    get: Callable[..., requests.Response] = requests.get,
) -> Mapping[str, str] | ModelSyncSkipped:
    """Extra env an agent needs to see the proxy's model list.

    Only OpenCode needs one: Claude Code discovers models through
    CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY and Codex takes the model by name.
    skip_verify means the caller wants no pre-launch proxy call at all, so the
    listing is skipped too rather than hanging on an offline proxy.
    """
    if os.path.basename(command) != "opencode":
        return _NO_EXTRA_ENV
    if skip_verify:
        return ModelSyncSkipped(f"{_SKIP_VERIFY_FLAG} was passed")
    return opencode_model_sync_env(base_env, base_url, api_key, get=get)


def verify_proxy_key(
    base_url: str,
    api_key: str,
    *,
    get: Callable[..., requests.Response] = requests.get,
) -> None:
    """Probe the proxy with the key so bad creds fail here, not inside the agent.

    Raises AgentRunError when the proxy is unreachable or rejects the key. Other
    non-2xx responses are tolerated; the agent's own call is the real test.
    """
    url: Final = base_url.rstrip("/") + "/v1/models"
    try:
        resp: Final = get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
    except requests.RequestException as e:
        raise AgentRunError(
            f"Could not reach the LiteLLM proxy at {base_url.rstrip('/')}: {e}. "
            "Is it running, and is --base-url (or LITELLM_PROXY_URL) correct?"
        )
    if resp.status_code in (401, 403):
        raise AgentRunError(
            f"LiteLLM rejected your key (HTTP {resp.status_code}). "
            "Run `lite login` to refresh it, or pass a valid --api-key."
        )


_WINDOWS_SHIM_SUFFIXES: Final[frozenset[str]] = frozenset({".cmd", ".bat"})
_CMD_LINE_BREAKS: Final = ("\r", "\n")


def _windows_command(path: str, args: Sequence[str]) -> str | tuple[str, ...]:
    """Build what CreateProcess runs, routing batch shims through cmd.exe.

    npm installs Claude Code as `claude.cmd`, which PATHEXT lets shutil.which
    resolve but CreateProcess refuses to run (WinError 193), so a shim has to go
    through the command processor. cmd.exe does not follow the C runtime quoting
    that subprocess would apply to an argument list, and it would split on `&` or
    `|` in a forwarded argument, so the shim case is emitted as one verbatim
    command line with every token quoted. Every switch is load-bearing: `/s`
    makes cmd strip only the outer pair, leaving each token quoted and its
    metacharacters inert, `/e:on` keeps the command extensions that the percent
    guard is built out of, `/v:off` keeps `!` from expanding, and `/d` keeps a
    machine's AutoRun commands out of the launch. argv[0] carries the
    caller-facing name on POSIX; Windows needs the resolved path there.

    Raises AgentRunError for an argument holding a line break, which cmd would
    read as the end of the command line and silently drop the rest of.
    """
    rest: Final = tuple(args[1:])
    if os.path.splitext(path)[1].lower() not in _WINDOWS_SHIM_SUFFIXES:
        return (path, *rest)
    if any(brk in token for token in rest for brk in _CMD_LINE_BREAKS):
        raise AgentRunError(
            f"Cannot pass an argument containing a line break to `{os.path.basename(path)}` on "
            "Windows: cmd.exe ends the command line there, so the agent would silently lose it."
        )
    inner: Final = " ".join(quote_for_cmd(token) for token in (path, *rest))
    return f'cmd.exe /d /e:on /v:off /s /c "{inner}"'


def _spawn_and_wait(command: str | Sequence[str], env: Mapping[str, str]) -> int:
    return subprocess.run(command, env=dict(env), check=False).returncode


def _replace_process(
    path: str,
    args: Sequence[str],
    env: Mapping[str, str],
    *,
    execvpe: Callable[..., None] = os.execvpe,
) -> None:
    execvpe(path, list(args), dict(env))


def _hand_off(
    path: str,
    args: Sequence[str],
    env: Mapping[str, str],
    *,
    platform: str = sys.platform,
    replace: Callable[[str, Sequence[str], Mapping[str, str]], None] = _replace_process,
    spawn: Callable[[str | Sequence[str], Mapping[str, str]], int] = _spawn_and_wait,
) -> None:
    """Replace this process with the agent; on Windows, run it as a child instead.

    os.exec* has no process-replacement semantics on Windows: the C runtime
    spawns a detached child and terminates the parent, so the shell reclaims the
    console and the agent's TUI never gets one. Windows therefore waits on the
    child and exits with its status.
    """
    if platform.startswith("win"):
        raise SystemExit(spawn(_windows_command(path, args), env))
    replace(path, list(args), dict(env))


def _restore_controlling_terminal() -> None:
    """Reattach the controlling terminal to stdin before handing off to the agent.

    Completing the browser SSO login can leave stdin detached from the terminal,
    which makes a TUI agent like Claude Code start in non-interactive mode and
    exit immediately. Reopening /dev/tty onto fd 0 gives the agent a live
    terminal; when stdin is still a tty (no login happened) this is a no-op.
    """
    if sys.stdin.isatty():
        return
    try:
        fd: Final = os.open("/dev/tty", os.O_RDONLY)
    except OSError:
        return
    try:
        os.dup2(fd, 0)
    finally:
        os.close(fd)


def _warn(message: str) -> None:
    click.echo(message, err=True)


def run_agent(
    base_url: str,
    api_key: str,
    command: Sequence[str],
    *,
    skip_verify: bool = False,
    base_env: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
    verify: Callable[[str, str], None] = verify_proxy_key,
    sync_models: Callable[[str, Mapping[str, str], str, str, bool], Mapping[str, str] | ModelSyncSkipped] = (
        agent_model_sync_env
    ),
    warn: Callable[[str], None] = _warn,
    launcher: Callable[[str, Sequence[str], Mapping[str, str]], None] = _hand_off,
    reattach_terminal: Callable[[], None] | None = None,
) -> None:
    """Validate, wire the environment, and hand off to the agent.

    On success this never returns: POSIX replaces the current process, Windows
    waits on the agent and exits with its status. Raises AgentRunError for
    missing binaries, an unreachable proxy, or a rejected key. The model list is
    synced only once the key check passed, so an unreachable proxy costs one
    timeout rather than two, and --skip-verify keeps the launch fully offline.
    reattach_terminal, when given, runs just before handoff to restore stdin.
    """
    if not command:
        raise AgentRunError("Nothing to run.")

    display_name, profiles = agent_profile(command[0])
    binary: Final = which(command[0])
    if binary is None:
        docs: Final = _INSTALL_DOCS.get(os.path.basename(command[0]))
        hint: Final = f" Install it first: {docs}" if docs else ""
        raise AgentRunError(f"Could not find `{command[0]}` on your PATH.{hint}")

    if not skip_verify:
        verify(base_url, api_key)

    env_before_sync: Final = base_env if base_env is not None else os.environ
    synced: Final = sync_models(command[0], env_before_sync, base_url, api_key, skip_verify)
    if isinstance(synced, ModelSyncSkipped):
        warn(f"litellm: not syncing {display_name} models from the proxy: {synced.reason}")

    env: Final = MappingProxyType(
        {
            **build_agent_env(env_before_sync, base_url, api_key, profiles),
            **(_NO_EXTRA_ENV if isinstance(synced, ModelSyncSkipped) else synced),
        }
    )
    extra_args: Final = agent_launch_args(command[0], base_url)
    if reattach_terminal is not None:
        reattach_terminal()
    launcher(binary, [command[0], *extra_args, *command[1:]], env)


def _is_interactive() -> bool:
    return sys.stdin.isatty()


def resolve_api_key(ctx: click.Context) -> str:
    ctx_obj: Final[CliContextObj] = ctx.obj
    base_url: Final = ctx_obj["base_url"]
    api_key = ctx_obj.get("api_key")
    if api_key:
        return api_key

    if not _is_interactive():
        raise click.ClickException(
            "No LiteLLM key found. Set LITELLM_PROXY_API_KEY (or pass --api-key) for "
            "non-interactive use, or run `lite login` from a terminal."
        )

    click.echo("No LiteLLM credentials found; starting login...")
    ctx.invoke(login)
    api_key = get_stored_api_key(expected_base_url=base_url, vault=context_secret_vault(ctx))
    if not api_key:
        raise click.ClickException("Login did not produce an API key; cannot start the agent.")
    return api_key


_SKIP_VERIFY_HELP: Final = "Skip the pre-launch key check against the proxy."


def _launch(ctx: click.Context, binary: str, args: Sequence[str], *, skip_verify: bool) -> None:
    ctx_obj: Final[CliContextObj] = ctx.obj
    base_url: Final = ctx_obj["base_url"]
    started_interactive: Final = _is_interactive()
    api_key: Final = resolve_api_key(ctx)

    display_name, _ = agent_profile(binary)
    click.echo(f"litellm: routing {display_name} through proxy at {base_url.rstrip('/')}")

    try:
        run_agent(
            base_url,
            api_key,
            [binary, *args],
            skip_verify=skip_verify,
            reattach_terminal=(_restore_controlling_terminal if started_interactive else None),
        )
    except AgentRunError as e:
        raise click.ClickException(str(e))


def _make_agent_command(binary: str, display_name: str) -> click.Command:
    @click.command(
        name=binary,
        context_settings={"ignore_unknown_options": True},
        short_help=f"Run {display_name} through your LiteLLM proxy",
    )
    @click.option("--skip-verify", is_flag=True, default=False, help=_SKIP_VERIFY_HELP)
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    @click.pass_context
    def _command(ctx: click.Context, skip_verify: bool, args: Sequence[str]) -> None:
        _launch(ctx, binary, list(args), skip_verify=skip_verify)

    _command.help = (
        f"Run {display_name} routed through your LiteLLM proxy.\n\n"
        f"Logs in with LiteLLM if needed, verifies your key against the proxy, "
        f"exports the env vars {binary} reads, then hands off. Any arguments are "
        f"forwarded to `{binary}`."
    )
    return _command


def agent_commands() -> tuple[click.Command, ...]:
    """Build one top-level command per known agent, e.g. `lite claude`."""
    return tuple(_make_agent_command(binary, name) for binary, (name, _profiles) in _KNOWN_AGENTS.items())


__all__ = [
    "AgentRunError",
    "ListedModel",
    "ModelSyncSkipped",
    "agent_commands",
    "agent_launch_args",
    "agent_model_sync_env",
    "agent_profile",
    "build_agent_env",
    "opencode_model_sync_env",
    "opencode_provider_config",
    "resolve_api_key",
    "run_agent",
    "verify_proxy_key",
]
