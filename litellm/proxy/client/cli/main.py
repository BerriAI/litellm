# stdlib imports
from typing import Optional

# third party imports
import click

# local imports
from .commands.models import models
from .commands.credentials import credentials
from .commands.chat import chat


@click.group()
@click.option(
    "--base-url",
    envvar="LITELLM_PROXY_URL",
    default="http://localhost:4000",
    help="Base URL of the LiteLLM proxy server",
)
@click.option(
    "--api-key",
    envvar="LITELLM_PROXY_API_KEY",
    help="API key for authentication",
)
@click.pass_context
def cli(ctx: click.Context, base_url: str, api_key: Optional[str]) -> None:
    """LiteLLM Proxy CLI - Manage your LiteLLM proxy server"""
    ctx.ensure_object(dict)
    ctx.obj["base_url"] = base_url
    ctx.obj["api_key"] = api_key


# Add the models command group
cli.add_command(models)
# Add the credentials command group
cli.add_command(credentials)
# Add the chat command group
cli.add_command(chat)


if __name__ == "__main__":
    cli()
