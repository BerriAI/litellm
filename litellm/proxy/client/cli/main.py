# stdlib imports
import sys
from typing import Optional

# third party imports
import click

from litellm._version import version as litellm_version
from litellm.proxy.client.health import HealthManagementClient

from .banner import show_banner
from .commands.auth import get_stored_api_key, login, logout, whoami
from .commands.chat import chat
from .commands.credentials import credentials
from .commands.http import http
from .commands.keys import keys

# local imports
from .commands.models import models
from .commands.users import users


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



def show_commands():
    """Display available commands."""
    commands = [
        ("login", "Authenticate with the LiteLLM proxy server"),
        ("logout", "Clear stored authentication"),
        ("whoami", "Show current authentication status"),
        ("models", "Manage and view model configurations"),
        ("credentials", "Manage API credentials"),
        ("chat", "Interactive chat with models"),
        ("http", "Make HTTP requests to the proxy"),
        ("keys", "Manage API keys"),
        ("users", "Manage users"),
        ("version", "Show version information"),
        ("help", "Show this help message"),
        ("quit", "Exit the interactive session"),
    ]
    
    click.echo("Available commands:")
    for cmd, description in commands:
        click.echo(f"  {cmd:<15} {description}")
    click.echo()


def interactive_shell(ctx: click.Context):
    """Run the interactive shell."""
    show_banner()
    
    # Show server connection info
    base_url = ctx.obj.get("base_url")
    click.secho(f"Connected to LiteLLM server: {base_url}\n", fg="green")
    
    show_commands()
    
    while True:
        try:
            # Show prompt
            user_input = click.prompt("> ", prompt_suffix="", show_default=False).strip()
            
            if not user_input:
                continue
                
            # Handle special commands
            if user_input.lower() in ["exit", "quit"]:
                click.echo("Goodbye!")
                break
            elif user_input.lower() == "help":
                show_commands()
                continue
            elif user_input.lower() == "clear":
                click.clear()
                show_banner()
                show_commands()
                continue
            
            # Parse command and arguments
            parts = user_input.split()
            command = parts[0]
            args = parts[1:] if len(parts) > 1 else []
            
            # Check if command exists
            if command not in cli.commands:
                click.echo(f"Unknown command: {command}")
                click.echo("Type 'help' to see available commands.")
                continue
            
            # Execute the command
            try:
                # Create a new argument list for click to parse
                sys.argv = ["litellm-proxy"] + [command] + args
                
                # Get the command object and invoke it
                cmd = cli.commands[command]
                
                # Create a new context for the subcommand
                with ctx.scope():
                    cmd.main(
                        args,
                        parent=ctx,
                        standalone_mode=False
                    )
                    
            except click.ClickException as e:
                e.show()
            except click.Abort:
                click.echo("Command aborted.")
            except SystemExit:
                # Prevent the interactive shell from exiting on command errors
                pass
            except Exception as e:
                click.echo(f"Error executing command: {e}")
                
        except (KeyboardInterrupt, EOFError):
            click.echo("\nGoodbye!")
            break
        except Exception as e:
            click.echo(f"Error: {e}")


@click.group(invoke_without_command=True)
@click.option(
    "--version", "-v", is_flag=True, is_eager=True, expose_value=False,
    help="Show the LiteLLM Proxy CLI and server version and exit.",
    callback=lambda ctx, param, value: (
        print_version(
            ctx.params.get("base_url") or "http://localhost:4000",
            ctx.params.get("api_key")
        )
        or ctx.exit()
    ) if value and not ctx.resilient_parsing else None,
)
@click.option(
    "--base-url",
    envvar="LITELLM_PROXY_URL",
    show_envvar=True,
    default="http://localhost:4000",
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

    # If no API key provided via flag or environment variable, try to load from saved token
    if api_key is None:
        api_key = get_stored_api_key()

    ctx.obj["base_url"] = base_url
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
# Add the users command group
cli.add_command(users)


if __name__ == "__main__":
    cli()
