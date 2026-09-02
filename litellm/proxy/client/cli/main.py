# stdlib imports

# third party imports
from typing import Final

import click

from litellm._version import version as litellm_version
from litellm.proxy.client.health import HealthManagementClient

from .commands.agents import agent_commands
from .commands.auth import auth_group, context_secret_vault, get_stored_api_key, login, logout, whoami
from .commands.autoroute.commands import autoroute_group
from .commands.chat import chat
from .commands.config import config_commands, get_config_value, hidden_command_names
from .commands.credentials import credentials
from .commands.encryption import encryption
from .commands.http import http
from .commands.keys import keys
from .commands.model_groups import model_groups

# local imports
from .commands.models import models
from .commands.teams import teams
from .commands.up import down, up
from .commands.users import users
from .interface import interactive_shell


def print_version(base_url: str, api_key: str | None):
    """Print CLI and server version info."""
    click.echo(f"LiteLLM Proxy CLI Version: {litellm_version}")
    if base_url:
        click.echo(f"LiteLLM Proxy Server URL: {base_url}")
    try:
        health_client: Final = HealthManagementClient(base_url=base_url, api_key=api_key)
        server_version: Final = health_client.get_server_version()
        if server_version:
            click.echo(f"LiteLLM Proxy Server Version: {server_version}")
        else:
            click.echo("LiteLLM Proxy Server Version: (unavailable)")
    except Exception as e:
        click.echo(f"Could not retrieve server version: {e}")


class HideConfiguredCommandsGroup(click.Group):
    """Group that omits operator-hidden commands from listings, still running them.

    Deployments hand `lite` to users who should only see a curated subset of
    commands (`lite config set hidden_commands codex,opencode`). Filtering the
    listing rather than dropping the commands keeps anyone's existing scripts
    working.
    """

    def list_commands(self, ctx: click.Context) -> list[str]:
        hidden: Final = hidden_command_names()
        return [name for name in super().list_commands(ctx) if name not in hidden]


@click.group(cls=HideConfiguredCommandsGroup, invoke_without_command=True)
@click.option(
    "--version",
    "-v",
    "show_version",
    is_flag=True,
    help="Show the LiteLLM Proxy CLI and server version and exit.",
)
@click.option(
    "--base-url",
    envvar="LITELLM_PROXY_URL",
    show_envvar=True,
    default=None,
    show_default="base_url from `lite config`, else http://localhost:4000",
    help="Base URL of the LiteLLM proxy server",
)
@click.option(
    "--api-key",
    envvar="LITELLM_PROXY_API_KEY",
    show_envvar=True,
    help="API key for authentication",
)
@click.pass_context
def cli(ctx: click.Context, show_version: bool, base_url: str | None, api_key: str | None) -> None:
    """LiteLLM Proxy CLI - Manage your LiteLLM proxy server"""
    ctx.ensure_object(dict)

    stored_base_url: Final = get_config_value("base_url")
    base_url_provided: Final = base_url is not None

    # Normalize once here so every downstream command (login, agents, http, ...) can safely
    # do f"{base_url}/some/path" without producing a double slash.
    base_url = ((stored_base_url or "http://localhost:4000") if base_url is None else base_url).rstrip("/")

    # If no API key provided via flag or environment variable, try to load from saved token.
    # Pass base_url so we only use the stored key when it was issued for this server.
    api_key_from_token_file: Final = api_key is None
    resolved_api_key: Final = (
        get_stored_api_key(expected_base_url=base_url, vault=context_secret_vault(ctx))
        if api_key_from_token_file
        else api_key
    )

    ctx.obj["base_url"] = base_url
    ctx.obj["api_key"] = resolved_api_key
    ctx.obj["api_key_from_token_file"] = api_key_from_token_file
    # `--base-url` defaults to localhost:4000 for local dev convenience, but
    # apiKeyHelper is invoked bare (no flags) -- commands that must work
    # unattended (print-token) need to tell "user didn't say" apart from
    # "user said localhost:4000 on purpose" so they can fall back to
    # whatever server the stored token was actually issued for. A base_url
    # saved via `lite config set` counts as the user saying it.
    ctx.obj["base_url_explicit"] = base_url_provided or bool(stored_base_url)

    if show_version:
        print_version(base_url, resolved_api_key)
        ctx.exit()

    # If no subcommand was invoked, start interactive mode
    if ctx.invoked_subcommand is None:
        interactive_shell(ctx)


@cli.command()
@click.pass_context
def version(ctx: click.Context):
    """Show the LiteLLM Proxy CLI and server version."""
    print_version(ctx.obj.get("base_url"), ctx.obj.get("api_key"))


# Add authentication commands as top-level commands
cli.add_command(login)
cli.add_command(logout)
cli.add_command(whoami)
# Add the auth command group (e.g. `lite auth print-token`, used as Claude Code's apiKeyHelper)
cli.add_command(auth_group, name="auth")
# Add the models command group
cli.add_command(models)
# Add the credentials command group
cli.add_command(credentials)
# Add the encryption migration command group
cli.add_command(encryption)
# Add the chat command group
cli.add_command(chat)
# Add the http command group
cli.add_command(http)
# Add the keys command group
cli.add_command(keys)
# Add the teams command group
cli.add_command(teams)
# Add the users command group
cli.add_command(users)
# Add a top-level command per coding agent (claude, codex, opencode, ...)
for agent_command in agent_commands():
    cli.add_command(agent_command)
# Add the up/down commands (route Claude Code through the local LiteLLM proxy)
cli.add_command(up)
cli.add_command(down)
# Add the model-groups command group (discover models your key can access)
cli.add_command(model_groups)
# Add the autoroute command group (QA auto-routing against your real proxy)
cli.add_command(autoroute_group, name="autoroute")
cli.add_command(config_commands)


if __name__ == "__main__":
    cli()
