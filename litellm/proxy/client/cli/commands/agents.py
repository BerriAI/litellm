import os
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Final

import click
import requests

from .auth import get_stored_api_key, login

ANTHROPIC_BASE_URL_ENV: Final = "ANTHROPIC_BASE_URL"
ANTHROPIC_AUTH_TOKEN_ENV: Final = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_API_KEY_ENV: Final = "ANTHROPIC_API_KEY"
OPENAI_BASE_URL_ENV: Final = "OPENAI_BASE_URL"
OPENAI_API_KEY_ENV: Final = "OPENAI_API_KEY"

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
    Anthropic key cannot win over the bearer token we set.
    """
    env: Final = dict(base_env)
    root: Final = base_url.rstrip("/")
    if PROFILE_ANTHROPIC in profiles:
        env[ANTHROPIC_BASE_URL_ENV] = root
        env[ANTHROPIC_AUTH_TOKEN_ENV] = api_key
        env.pop(ANTHROPIC_API_KEY_ENV, None)
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


def _exec(path: str, args: Sequence[str], env: Mapping[str, str]) -> None:
    os.execvpe(path, list(args), dict(env))


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


def run_agent(
    base_url: str,
    api_key: str,
    command: Sequence[str],
    *,
    skip_verify: bool = False,
    base_env: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
    verify: Callable[[str, str], None] = verify_proxy_key,
    launcher: Callable[[str, Sequence[str], Mapping[str, str]], None] = _exec,
    reattach_terminal: Callable[[], None] | None = None,
) -> None:
    """Validate, wire the environment, and hand off to the agent.

    On success this replaces the current process and never returns. Raises
    AgentRunError for missing binaries, an unreachable proxy, or a rejected key.
    reattach_terminal, when given, runs just before handoff to restore stdin.
    """
    if not command:
        raise AgentRunError("Nothing to run.")

    _, profiles = agent_profile(command[0])
    binary: Final = which(command[0])
    if binary is None:
        docs: Final = _INSTALL_DOCS.get(os.path.basename(command[0]))
        hint: Final = f" Install it first: {docs}" if docs else ""
        raise AgentRunError(f"Could not find `{command[0]}` on your PATH.{hint}")

    if not skip_verify:
        verify(base_url, api_key)

    env: Final = build_agent_env(
        base_env if base_env is not None else os.environ,
        base_url,
        api_key,
        profiles,
    )
    extra_args: Final = agent_launch_args(command[0], base_url)
    if reattach_terminal is not None:
        reattach_terminal()
    launcher(binary, [command[0], *extra_args, *command[1:]], env)


def _is_interactive() -> bool:
    return sys.stdin.isatty()


def resolve_api_key(ctx: click.Context) -> str:
    base_url: Final = ctx.obj["base_url"]
    api_key = ctx.obj.get("api_key")
    if api_key:
        return api_key

    if not _is_interactive():
        raise click.ClickException(
            "No LiteLLM key found. Set LITELLM_PROXY_API_KEY (or pass --api-key) for "
            "non-interactive use, or run `lite login` from a terminal."
        )

    click.echo("No LiteLLM credentials found; starting login...")
    ctx.invoke(login)
    api_key = get_stored_api_key(expected_base_url=base_url)
    if not api_key:
        raise click.ClickException("Login did not produce an API key; cannot start the agent.")
    return api_key


_SKIP_VERIFY_HELP: Final = "Skip the pre-launch key check against the proxy."


def _launch(ctx: click.Context, binary: str, args: Sequence[str], *, skip_verify: bool) -> None:
    base_url: Final = ctx.obj["base_url"]
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


def agent_commands() -> list[click.Command]:
    """Build one top-level command per known agent, e.g. `lite claude`."""
    return [_make_agent_command(binary, name) for binary, (name, _profiles) in _KNOWN_AGENTS.items()]


__all__ = [
    "AgentRunError",
    "agent_commands",
    "agent_launch_args",
    "agent_profile",
    "build_agent_env",
    "resolve_api_key",
    "run_agent",
    "verify_proxy_key",
]
