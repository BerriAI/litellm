# stdlib imports
import os
from typing import Optional

# third party imports
import click
from click import ParameterSource

from litellm._version import version as litellm_version
from litellm.litellm_core_utils.cli_token_utils import get_stored_base_url
from litellm.proxy.client.health import HealthManagementClient

from .commands.agents import agent_commands
from .commands.auth import get_stored_api_key, login, logout, whoami
from .commands.chat import chat
from .commands.credentials import credentials
from .commands.http import http
from .commands.keys import keys

# local imports
from .commands.models import models
from .commands.teams import teams
from .commands.users import users
from .interface import interactive_shell

DEFAULT_BASE_URL = "http://localhost:4000"


def print_version(base_url: str, api_key: Optional[str]):
    """Print CLI and server version info."""
    click.echo(f"LiteLLM Proxy CLI Version: {litellm_version}")
    if base_url:
        click.echo(f"LiteLLM Proxy Server URL: {base_url}")
    try:
        health_client = HealthManagementClient(base_url=base_url, api_key=api_key)
        server_version = health_client.get_server_version()
        if server_version:
            click.echo(f"LiteLLM Proxy Server Version: {server_version}")
        else:
            click.echo("LiteLLM Proxy Server Version: (unavailable)")
    except Exception as e:
        click.echo(f"Could not retrieve server version: {e}")


def _eager_version_base_url(ctx: click.Context) -> str:
    """Resolve the base URL for `--version`, which can fire before --base-url is parsed."""
    return (
        ctx.params.get("base_url")
        or os.environ.get("LITELLM_PROXY_URL")
        or get_stored_base_url()
        or DEFAULT_BASE_URL
    )


def _print_version_callback(
    ctx: click.Context, param: click.Parameter, value: bool
) -> None:
    if not value or ctx.resilient_parsing:
        return
    print_version(_eager_version_base_url(ctx), ctx.params.get("api_key"))
    ctx.exit()


@click.group(invoke_without_command=True)
@click.option(
    "--version",
    "-v",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    help="Show the LiteLLM Proxy CLI and server version and exit.",
    callback=_print_version_callback,
)
@click.option(
    "--base-url",
    envvar="LITELLM_PROXY_URL",
    show_envvar=True,
    default=DEFAULT_BASE_URL,
    help="Base URL of the LiteLLM proxy server",
)
@click.option(
    "--api-key",
    envvar="LITELLM_PROXY_API_KEY",
    show_envvar=True,
    help="API key for authentication",
)
@click.pass_context
def cli(ctx: click.Context, base_url: str, api_key: Optional[str]) -> None:
    """LiteLLM Proxy CLI - Manage your LiteLLM proxy server"""
    ctx.ensure_object(dict)

    base_url_is_default = (
        ctx.get_parameter_source("base_url") == ParameterSource.DEFAULT
    )
    if base_url_is_default:
        stored_base_url = get_stored_base_url()
        if stored_base_url:
            base_url = stored_base_url
            base_url_is_default = False

    # If no API key provided via flag or environment variable, try to load from saved token.
    # Pass base_url so we only use the stored key when it was issued for this server.
    if api_key is None:
        api_key = get_stored_api_key(expected_base_url=base_url)

    ctx.obj["base_url"] = base_url
    ctx.obj["base_url_is_default"] = base_url_is_default
    ctx.obj["api_key"] = api_key

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
# Add the models command group
cli.add_command(models)
# Add the credentials command group
cli.add_command(credentials)
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


if __name__ == "__main__":
    cli()
