import os
import shutil
from typing import Callable, Dict, Mapping, Optional, Sequence

import click

from .auth import get_stored_api_key, login

ANTHROPIC_BASE_URL_ENV = "ANTHROPIC_BASE_URL"
ANTHROPIC_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
ANTHROPIC_MODEL_ENV = "ANTHROPIC_MODEL"
ANTHROPIC_SMALL_FAST_MODEL_ENV = "ANTHROPIC_SMALL_FAST_MODEL"

CLAUDE_BINARY = "claude"
CLAUDE_INSTALL_DOCS = "https://docs.claude.com/en/docs/claude-code/setup"


class ClaudeCodeNotFoundError(Exception):
    """Raised when the Claude Code `claude` executable cannot be located."""


def build_claude_env(
    base_env: Mapping[str, str],
    base_url: str,
    api_key: str,
    model: Optional[str] = None,
    small_fast_model: Optional[str] = None,
) -> Dict[str, str]:
    """Return a copy of base_env wired to route Claude Code through the LiteLLM proxy.

    Claude Code sends requests to ``{ANTHROPIC_BASE_URL}/v1/messages`` with the
    LiteLLM key as a bearer token. ANTHROPIC_API_KEY is dropped so a stray
    Anthropic key in the environment cannot win over the proxy token.
    """
    env = dict(base_env)
    env[ANTHROPIC_BASE_URL_ENV] = base_url.rstrip("/")
    env[ANTHROPIC_AUTH_TOKEN_ENV] = api_key
    env.pop(ANTHROPIC_API_KEY_ENV, None)
    if model:
        env[ANTHROPIC_MODEL_ENV] = model
    if small_fast_model:
        env[ANTHROPIC_SMALL_FAST_MODEL_ENV] = small_fast_model
    return env


def _exec_claude(claude_path: str, args: Sequence[str], env: Mapping[str, str]) -> None:
    os.execvpe(claude_path, [claude_path, *args], dict(env))


def launch_claude_code(
    base_url: str,
    api_key: str,
    *,
    model: Optional[str] = None,
    small_fast_model: Optional[str] = None,
    claude_args: Sequence[str] = (),
    base_env: Optional[Mapping[str, str]] = None,
    which: Callable[[str], Optional[str]] = shutil.which,
    launcher: Callable[[str, Sequence[str], Mapping[str, str]], None] = _exec_claude,
) -> None:
    """Wire the Anthropic environment to the LiteLLM proxy and hand off to `claude`.

    On success this replaces the current process with Claude Code and never
    returns. Raises ClaudeCodeNotFoundError when `claude` is not on PATH.
    """
    claude_path = which(CLAUDE_BINARY)
    if claude_path is None:
        raise ClaudeCodeNotFoundError(
            f"Could not find the `{CLAUDE_BINARY}` executable on your PATH. "
            f"Install Claude Code first: {CLAUDE_INSTALL_DOCS}"
        )

    env = build_claude_env(
        base_env if base_env is not None else os.environ,
        base_url,
        api_key,
        model,
        small_fast_model,
    )
    launcher(claude_path, claude_args, env)


@click.command(
    name="claude-code",
    context_settings={"ignore_unknown_options": True},
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Model Claude Code should request (sets ANTHROPIC_MODEL). Must exist on your proxy.",
)
@click.option(
    "--small-fast-model",
    default=None,
    help="Model for background tasks (sets ANTHROPIC_SMALL_FAST_MODEL). Must exist on your proxy.",
)
@click.argument("claude_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def claude_code(
    ctx: click.Context,
    model: Optional[str],
    small_fast_model: Optional[str],
    claude_args: Sequence[str],
):
    """Launch Claude Code pointed at your LiteLLM proxy.

    Logs in with LiteLLM if needed, exports the Anthropic environment variables
    Claude Code reads, then hands off to the `claude` binary. Any extra
    arguments are forwarded to `claude` (use `--` to pass flags claude owns,
    e.g. `litellm-proxy claude-code -- --resume`).
    """
    base_url = ctx.obj["base_url"]
    api_key = ctx.obj.get("api_key")

    if not api_key:
        click.echo("No LiteLLM credentials found; starting login...")
        ctx.invoke(login)
        api_key = get_stored_api_key(expected_base_url=base_url)

    if not api_key:
        raise click.ClickException(
            "Could not obtain a LiteLLM API key. Run `litellm-proxy login` and try again."
        )

    click.echo(f"Launching Claude Code through LiteLLM proxy at {base_url.rstrip('/')}")

    try:
        launch_claude_code(
            base_url,
            api_key,
            model=model,
            small_fast_model=small_fast_model,
            claude_args=claude_args,
        )
    except ClaudeCodeNotFoundError as e:
        raise click.ClickException(str(e))


__all__ = ["claude_code", "launch_claude_code", "build_claude_env"]
