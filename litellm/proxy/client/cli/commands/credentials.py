import json
from typing import Final, Literal

import click
import requests
import rich
from rich.table import Table

from ...credentials import CredentialsManagementClient


@click.group()
def credentials():
    """Manage credentials for the LiteLLM proxy server"""


@credentials.command()
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format (table or json)",
)
@click.pass_context
def list(ctx: click.Context, output_format: Literal["table", "json"]):
    """List all credentials"""
    client: Final = CredentialsManagementClient(ctx.obj["base_url"], ctx.obj["api_key"])
    response: Final = client.list()
    assert isinstance(response, dict)

    if output_format == "json":
        rich.print_json(data=response)
    else:  # table format
        table: Final = Table(title="Credentials")

        # Add columns
        table.add_column("Credential Name", style="cyan")
        table.add_column("Custom LLM Provider", style="green")

        # Add rows
        for cred in response.get("credentials", []):
            info = cred.get("credential_info", {})
            table.add_row(
                str(cred.get("credential_name", "")),
                str(info.get("custom_llm_provider", "")),
            )

        rich.print(table)


@credentials.command()
@click.argument("credential_name")
@click.option(
    "--info",
    type=str,
    help="JSON string containing credential info",
    required=True,
)
@click.option(
    "--values",
    type=str,
    help="JSON string containing credential values",
    required=True,
)
@click.pass_context
def create(ctx: click.Context, credential_name: str, info: str, values: str):
    """Create a new credential"""
    client: Final = CredentialsManagementClient(ctx.obj["base_url"], ctx.obj["api_key"])
    try:
        credential_info: Final = json.loads(info)
        credential_values: Final = json.loads(values)
    except json.JSONDecodeError as e:
        raise click.BadParameter(f"Invalid JSON: {e}")

    try:
        response: Final = client.create(credential_name, credential_info, credential_values)
        rich.print_json(data=response)
    except requests.exceptions.HTTPError as e:
        click.echo(f"Error: HTTP {e.response.status_code}", err=True)
        try:
            error_body: Final = e.response.json()
            rich.print_json(data=error_body)
        except json.JSONDecodeError:
            click.echo(e.response.text, err=True)
        raise click.Abort()


@credentials.command()
@click.argument("credential_name")
@click.pass_context
def delete(ctx: click.Context, credential_name: str):
    """Delete a credential by name"""
    client: Final = CredentialsManagementClient(ctx.obj["base_url"], ctx.obj["api_key"])
    try:
        response: Final = client.delete(credential_name)
        rich.print_json(data=response)
    except requests.exceptions.HTTPError as e:
        click.echo(f"Error: HTTP {e.response.status_code}", err=True)
        try:
            error_body: Final = e.response.json()
            rich.print_json(data=error_body)
        except json.JSONDecodeError:
            click.echo(e.response.text, err=True)
        raise click.Abort()


@credentials.command()
@click.argument("credential_name")
@click.pass_context
def get(ctx: click.Context, credential_name: str):
    """Get a credential by name"""
    client: Final = CredentialsManagementClient(ctx.obj["base_url"], ctx.obj["api_key"])
    response: Final = client.get(credential_name)
    rich.print_json(data=response)
