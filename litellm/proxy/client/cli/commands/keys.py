import json
from typing import Literal, Optional

import click
import rich
import requests
from rich.table import Table

from ...keys import KeysManagementClient


@click.group()
def keys():
    """Manage API keys for the LiteLLM proxy server"""
    pass


@keys.command()
@click.option("--page", type=int, help="Page number for pagination")
@click.option("--size", type=int, help="Number of items per page")
@click.option("--user-id", type=str, help="Filter keys by user ID")
@click.option("--team-id", type=str, help="Filter keys by team ID")
@click.option("--organization-id", type=str, help="Filter keys by organization ID")
@click.option("--key-hash", type=str, help="Filter by specific key hash")
@click.option("--key-alias", type=str, help="Filter by key alias")
@click.option("--return-full-object", is_flag=True, default=True, help="Return the full key object")
@click.option("--include-team-keys", is_flag=True, help="Include team keys in the response")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format (table or json)",
)
@click.pass_context
def list(
    ctx: click.Context,
    page: Optional[int],
    size: Optional[int],
    user_id: Optional[str],
    team_id: Optional[str],
    organization_id: Optional[str],
    key_hash: Optional[str],
    key_alias: Optional[str],
    include_team_keys: bool,
    output_format: Literal["table", "json"],
    return_full_object: bool,
):
    """List all API keys"""
    client = KeysManagementClient(ctx.obj["base_url"], ctx.obj["api_key"])
    response = client.list(
        page=page,
        size=size,
        user_id=user_id,
        team_id=team_id,
        organization_id=organization_id,
        key_hash=key_hash,
        key_alias=key_alias,
        return_full_object=return_full_object,
        include_team_keys=include_team_keys,
    )
    assert isinstance(response, dict)

    if output_format == "json":
        rich.print_json(data=response)
    else:
        rich.print(f"Showing {len(response.get('keys', []))} keys out of {response.get('total_count', 0)}")
        table = Table(title="API Keys")
        table.add_column("Key Hash", style="cyan")
        table.add_column("Alias", style="green")
        table.add_column("User ID", style="magenta")
        table.add_column("Team ID", style="yellow")
        table.add_column("Spend", style="red")
        for key in response.get("keys", []):
            table.add_row(
                str(key.get("token", "")),
                str(key.get("key_alias", "")),
                str(key.get("user_id", "")),
                str(key.get("team_id", "")),
                str(key.get("spend", "")),
            )
        rich.print(table)


@keys.command()
@click.option("--models", type=str, help="Comma-separated list of allowed models")
@click.option("--aliases", type=str, help="JSON string of model alias mappings")
@click.option("--spend", type=float, help="Maximum spend limit for this key")
@click.option("--duration", type=str, help="Duration for which the key is valid (e.g. '24h', '7d')")
@click.option("--key-alias", type=str, help="Alias/name for the key")
@click.option("--team-id", type=str, help="Team ID to associate the key with")
@click.option("--user-id", type=str, help="User ID to associate the key with")
@click.option("--budget-id", type=str, help="Budget ID to associate the key with")
@click.option("--config", type=str, help="JSON string of additional configuration parameters")
@click.pass_context
def generate(
    ctx: click.Context,
    models: Optional[str],
    aliases: Optional[str],
    spend: Optional[float],
    duration: Optional[str],
    key_alias: Optional[str],
    team_id: Optional[str],
    user_id: Optional[str],
    budget_id: Optional[str],
    config: Optional[str],
):
    """Generate a new API key"""
    client = KeysManagementClient(ctx.obj["base_url"], ctx.obj["api_key"])
    try:
        models_list = [m.strip() for m in models.split(",")] if models else None
        aliases_dict = json.loads(aliases) if aliases else None
        config_dict = json.loads(config) if config else None
    except json.JSONDecodeError as e:
        raise click.BadParameter(f"Invalid JSON: {str(e)}")
    try:
        response = client.generate(
            models=models_list,
            aliases=aliases_dict,
            spend=spend,
            duration=duration,
            key_alias=key_alias,
            team_id=team_id,
            user_id=user_id,
            budget_id=budget_id,
            config=config_dict,
        )
        rich.print_json(data=response)
    except requests.exceptions.HTTPError as e:
        click.echo(f"Error: HTTP {e.response.status_code}", err=True)
        try:
            error_body = e.response.json()
            rich.print_json(data=error_body)
        except json.JSONDecodeError:
            click.echo(e.response.text, err=True)
        raise click.Abort()


@keys.command()
@click.option("--keys", type=str, help="Comma-separated list of API keys to delete")
@click.option("--key-aliases", type=str, help="Comma-separated list of key aliases to delete")
@click.pass_context
def delete(ctx: click.Context, keys: Optional[str], key_aliases: Optional[str]):
    """Delete API keys by key or alias"""
    client = KeysManagementClient(ctx.obj["base_url"], ctx.obj["api_key"])
    keys_list = [k.strip() for k in keys.split(",")] if keys else None
    aliases_list = [a.strip() for a in key_aliases.split(",")] if key_aliases else None
    try:
        response = client.delete(keys=keys_list, key_aliases=aliases_list)
        rich.print_json(data=response)
    except requests.exceptions.HTTPError as e:
        click.echo(f"Error: HTTP {e.response.status_code}", err=True)
        try:
            error_body = e.response.json()
            rich.print_json(data=error_body)
        except json.JSONDecodeError:
            click.echo(e.response.text, err=True)
        raise click.Abort()
