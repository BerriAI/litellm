import os
import shutil
import sys
from typing import Callable, Dict, FrozenSet, Mapping, Optional, Sequence, Tuple

import click
import requests

from .auth import get_stored_api_key, login

ANTHROPIC_BASE_URL_ENV = "ANTHROPIC_BASE_URL"
ANTHROPIC_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
ANTHROPIC_MODEL_ENV = "ANTHROPIC_MODEL"
ANTHROPIC_SMALL_FAST_MODEL_ENV = "ANTHROPIC_SMALL_FAST_MODEL"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

PROFILE_ANTHROPIC = "anthropic"
PROFILE_OPENAI = "openai"

_KNOWN_AGENTS: Dict[str, Tuple[str, FrozenSet[str]]] = {
    "claude": ("Claude Code", frozenset({PROFILE_ANTHROPIC})),
    "codex": ("Codex", frozenset({PROFILE_OPENAI})),
    "opencode": ("OpenCode", frozenset({PROFILE_OPENAI})),
}

_INSTALL_DOCS: Dict[str, str] = {
    "claude": "https://docs.claude.com/en/docs/claude-code/setup",
    "codex": "https://developers.openai.com/codex/cli",
    "opencode": "https://opencode.ai/docs",
}


class AgentRunError(Exception):
    """Raised for any user-actionable failure while preparing to run an agent."""


def agent_profile(command: str) -> Tuple[str, FrozenSet[str]]:
    """Return the (display name, env profiles) for a wrapped command.

    Known agents map to the API family they speak. Anything else gets both
    families so it works regardless of which env vars the tool reads.
    """
    base = os.path.basename(command)
    if base in _KNOWN_AGENTS:
        return _KNOWN_AGENTS[base]
    return base, frozenset({PROFILE_ANTHROPIC, PROFILE_OPENAI})


def build_agent_env(
    base_env: Mapping[str, str],
    base_url: str,
    api_key: str,
    profiles: FrozenSet[str],
    *,
    model: Optional[str] = None,
    small_fast_model: Optional[str] = None,
) -> Dict[str, str]:
    """Return a copy of base_env wired to route the agent through the proxy.

    Anthropic clients (Claude Code) append /v1/messages to ANTHROPIC_BASE_URL,
    so it stays the bare proxy root; OpenAI clients (Codex, OpenCode) expect the
    /v1 suffix on OPENAI_BASE_URL. ANTHROPIC_API_KEY is dropped so a stray
    Anthropic key cannot win over the bearer token we set.
    """
    env = dict(base_env)
    root = base_url.rstrip("/")
    if PROFILE_ANTHROPIC in profiles:
        env[ANTHROPIC_BASE_URL_ENV] = root
        env[ANTHROPIC_AUTH_TOKEN_ENV] = api_key
        env.pop(ANTHROPIC_API_KEY_ENV, None)
        if model:
            env[ANTHROPIC_MODEL_ENV] = model
        if small_fast_model:
            env[ANTHROPIC_SMALL_FAST_MODEL_ENV] = small_fast_model
    if PROFILE_OPENAI in profiles:
        env[OPENAI_BASE_URL_ENV] = root + "/v1"
        env[OPENAI_API_KEY_ENV] = api_key
    return env


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
    url = base_url.rstrip("/") + "/v1/models"
    try:
        resp = get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
    except requests.RequestException as e:
        raise AgentRunError(
            f"Could not reach the LiteLLM proxy at {base_url.rstrip('/')}: {e}. "
            "Is it running, and is --base-url (or LITELLM_PROXY_URL) correct?"
        )
    if resp.status_code in (401, 403):
        raise AgentRunError(
            f"LiteLLM rejected your key (HTTP {resp.status_code}). "
            "Run `litellm-proxy login` to refresh it, or pass a valid --api-key."
        )


def _exec(path: str, args: Sequence[str], env: Mapping[str, str]) -> None:
    os.execvpe(path, list(args), dict(env))


def run_agent(
    base_url: str,
    api_key: str,
    command: Sequence[str],
    *,
    model: Optional[str] = None,
    small_fast_model: Optional[str] = None,
    skip_verify: bool = False,
    base_env: Optional[Mapping[str, str]] = None,
    which: Callable[[str], Optional[str]] = shutil.which,
    verify: Callable[[str, str], None] = verify_proxy_key,
    launcher: Callable[[str, Sequence[str], Mapping[str, str]], None] = _exec,
) -> None:
    """Validate, wire the environment, and hand off to the agent.

    On success this replaces the current process and never returns. Raises
    AgentRunError for missing binaries, an unreachable proxy, or a rejected key.
    """
    if not command:
        raise AgentRunError("Nothing to run. Try `litellm-proxy run -- claude`.")

    _, profiles = agent_profile(command[0])
    binary = which(command[0])
    if binary is None:
        docs = _INSTALL_DOCS.get(os.path.basename(command[0]))
        hint = f" Install it first: {docs}" if docs else ""
        raise AgentRunError(f"Could not find `{command[0]}` on your PATH.{hint}")

    if not skip_verify:
        verify(base_url, api_key)

    env = build_agent_env(
        base_env if base_env is not None else os.environ,
        base_url,
        api_key,
        profiles,
        model=model,
        small_fast_model=small_fast_model,
    )
    launcher(binary, command, env)


def _is_interactive() -> bool:
    return sys.stdin.isatty()


def _resolve_api_key(ctx: click.Context) -> str:
    base_url = ctx.obj["base_url"]
    api_key = ctx.obj.get("api_key")
    if api_key:
        return api_key

    if not _is_interactive():
        raise click.ClickException(
            "No LiteLLM key found. Set LITELLM_PROXY_API_KEY (or pass --api-key) for "
            "non-interactive use, or run `litellm-proxy login` from a terminal."
        )

    click.echo("No LiteLLM credentials found; starting login...")
    ctx.invoke(login)
    api_key = get_stored_api_key(expected_base_url=base_url)
    if not api_key:
        raise click.ClickException(
            "Login did not produce an API key; cannot start the agent."
        )
    return api_key


_MODEL_HELP = (
    "Model the agent should request, exported as ANTHROPIC_MODEL for Claude Code. "
    "Must resolve on your proxy. OpenAI-style agents take the model via their own flag."
)
_SMALL_FAST_MODEL_HELP = (
    "Background-task model for Claude Code, exported as ANTHROPIC_SMALL_FAST_MODEL."
)
_SKIP_VERIFY_HELP = "Skip the pre-launch key check against the proxy."


@click.command(name="run", context_settings={"ignore_unknown_options": True})
@click.option("--model", "-m", default=None, help=_MODEL_HELP)
@click.option("--small-fast-model", default=None, help=_SMALL_FAST_MODEL_HELP)
@click.option("--skip-verify", is_flag=True, default=False, help=_SKIP_VERIFY_HELP)
@click.argument("command", nargs=-1, type=click.UNPROCESSED, required=True)
@click.pass_context
def run(
    ctx: click.Context,
    model: Optional[str],
    small_fast_model: Optional[str],
    skip_verify: bool,
    command: Sequence[str],
):
    """Run a coding agent wired to your LiteLLM proxy.

    Everything after `--` is the command to execute, e.g.
    `litellm-proxy run -- claude`, `litellm-proxy run -- codex`, or
    `litellm-proxy run -- opencode`. Logs in with LiteLLM if needed, checks the
    key against the proxy, exports the env vars the agent reads, then hands off.
    """
    command = list(command)
    base_url = ctx.obj["base_url"]
    api_key = _resolve_api_key(ctx)

    display_name, _ = agent_profile(command[0])
    click.echo(
        f"litellm: routing {display_name} through proxy at {base_url.rstrip('/')}"
    )

    try:
        run_agent(
            base_url,
            api_key,
            command,
            model=model,
            small_fast_model=small_fast_model,
            skip_verify=skip_verify,
        )
    except AgentRunError as e:
        raise click.ClickException(str(e))


@click.command(name="claude-code", context_settings={"ignore_unknown_options": True})
@click.option("--model", "-m", default=None, help=_MODEL_HELP)
@click.option("--small-fast-model", default=None, help=_SMALL_FAST_MODEL_HELP)
@click.option("--skip-verify", is_flag=True, default=False, help=_SKIP_VERIFY_HELP)
@click.argument("claude_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def claude_code(
    ctx: click.Context,
    model: Optional[str],
    small_fast_model: Optional[str],
    skip_verify: bool,
    claude_args: Sequence[str],
):
    """Shortcut for `litellm-proxy run -- claude`. Extra args are forwarded to claude."""
    ctx.invoke(
        run,
        model=model,
        small_fast_model=small_fast_model,
        skip_verify=skip_verify,
        command=("claude", *claude_args),
    )


__all__ = [
    "run",
    "claude_code",
    "run_agent",
    "build_agent_env",
    "verify_proxy_key",
    "agent_profile",
]
