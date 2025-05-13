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


@keys.command()
@click.option("--key", required=True, type=str, help="The key to update")
@click.option("--key-alias", type=str, help="Alias/name for the key")
@click.option("--user-id", type=str, help="User ID to associate the key with")
@click.option("--team-id", type=str, help="Team ID to associate the key with")
@click.option("--budget-id", type=str, help="Budget ID to associate the key with")
@click.option("--models", type=str, help="Comma-separated list of allowed models")
@click.option("--tags", type=str, help="Comma-separated list of tags")
@click.option("--enforced-params", type=str, help="Comma-separated list of enforced params")
@click.option("--spend", type=float, help="Amount spent by key")
@click.option("--max-budget", type=float, help="Max budget for key")
@click.option("--model-max-budget", type=str, help="JSON string of model-specific budgets")
@click.option("--budget-duration", type=str, help="Budget reset period (e.g. '30d', '1h')")
@click.option("--max-parallel-requests", type=int, help="Rate limit for parallel requests")
@click.option("--metadata", type=str, help="JSON string of metadata for key")
@click.option("--tpm-limit", type=int, help="Tokens per minute limit")
@click.option("--rpm-limit", type=int, help="Requests per minute limit")
@click.option("--model-rpm-limit", type=str, help="JSON string of model-specific RPM limits")
@click.option("--model-tpm-limit", type=str, help="JSON string of model-specific TPM limits")
@click.option("--allowed-cache-controls", type=str, help="Comma-separated list of allowed cache controls")
@click.option("--duration", type=str, help="Key validity duration (e.g. '30d', '1h')")
@click.option("--permissions", type=str, help="JSON string of key-specific permissions")
@click.option("--guardrails", type=str, help="Comma-separated list of guardrails")
@click.option("--blocked", is_flag=True, help="Whether the key is blocked")
@click.option("--aliases", type=str, help="JSON string of model aliases for the key")
@click.option("--config", type=str, help="JSON string of key-specific config")
@click.option("--temp-budget-increase", type=float, help="Temporary budget increase for the key")
@click.option("--temp-budget-expiry", type=str, help="Expiry time for the temporary budget increase")
@click.option("--allowed-routes", type=str, help="Comma-separated list of allowed routes for the key")
@click.pass_context
def update(
    ctx: click.Context,
    key: str,
    key_alias: Optional[str],
    user_id: Optional[str],
    team_id: Optional[str],
    budget_id: Optional[str],
    models: Optional[str],
    tags: Optional[str],
    enforced_params: Optional[str],
    spend: Optional[float],
    max_budget: Optional[float],
    model_max_budget: Optional[str],
    budget_duration: Optional[str],
    max_parallel_requests: Optional[int],
    metadata: Optional[str],
    tpm_limit: Optional[int],
    rpm_limit: Optional[int],
    model_rpm_limit: Optional[str],
    model_tpm_limit: Optional[str],
    allowed_cache_controls: Optional[str],
    duration: Optional[str],
    permissions: Optional[str],
    guardrails: Optional[str],
    blocked: bool,
    aliases: Optional[str],
    config: Optional[str],
    temp_budget_increase: Optional[float],
    temp_budget_expiry: Optional[str],
    allowed_routes: Optional[str],
):
    """Update an existing API key"""
    client = KeysManagementClient(ctx.obj["base_url"], ctx.obj["api_key"])
    try:
        models_list = [m.strip() for m in models.split(",")] if models else None
        tags_list = [t.strip() for t in tags.split(",")] if tags else None
        enforced_params_list = [e.strip() for e in enforced_params.split(",")] if enforced_params else None
        allowed_cache_controls_list = [a.strip() for a in allowed_cache_controls.split(",")] if allowed_cache_controls else None
        guardrails_list = [g.strip() for g in guardrails.split(",")] if guardrails else None
        allowed_routes_list = [r.strip() for r in allowed_routes.split(",")] if allowed_routes else None
        model_max_budget_dict = json.loads(model_max_budget) if model_max_budget else None
        metadata_dict = json.loads(metadata) if metadata else None
        model_rpm_limit_dict = json.loads(model_rpm_limit) if model_rpm_limit else None
        model_tpm_limit_dict = json.loads(model_tpm_limit) if model_tpm_limit else None
        permissions_dict = json.loads(permissions) if permissions else None
        aliases_dict = json.loads(aliases) if aliases else None
        config_dict = json.loads(config) if config else None
    except json.JSONDecodeError as e:
        raise click.BadParameter(f"Invalid JSON: {str(e)}")
    try:
        response = client.update(
            key=key,
            key_alias=key_alias,
            user_id=user_id,
            team_id=team_id,
            budget_id=budget_id,
            models=models_list,
            tags=tags_list,
            enforced_params=enforced_params_list,
            spend=spend,
            max_budget=max_budget,
            model_max_budget=model_max_budget_dict,
            budget_duration=budget_duration,
            soft_budget=soft_budget,
            max_parallel_requests=max_parallel_requests,
            metadata=metadata_dict,
            tpm_limit=tpm_limit,
            rpm_limit=rpm_limit,
            model_rpm_limit=model_rpm_limit_dict,
            model_tpm_limit=model_tpm_limit_dict,
            allowed_cache_controls=allowed_cache_controls_list,
            duration=duration,
            permissions=permissions_dict,
            send_invite_email=send_invite_email,
            guardrails=guardrails_list,
            blocked=blocked,
            aliases=aliases_dict,
            config=config_dict,
            temp_budget_increase=temp_budget_increase,
            temp_budget_expiry=temp_budget_expiry,
            allowed_routes=allowed_routes_list,
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
