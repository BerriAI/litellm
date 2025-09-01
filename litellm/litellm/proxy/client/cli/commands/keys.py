import json
from datetime import datetime
from typing import Literal, Optional, List, Dict, Any

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


def _parse_created_since_filter(created_since: Optional[str]) -> Optional[datetime]:
    """Parse and validate the created_since date filter."""
    if not created_since:
        return None

    try:
        # Support formats: YYYY-MM-DD_HH:MM or YYYY-MM-DD
        if "_" in created_since:
            return datetime.strptime(created_since, "%Y-%m-%d_%H:%M")
        else:
            return datetime.strptime(created_since, "%Y-%m-%d")
    except ValueError:
        click.echo(f"Error: Invalid date format '{created_since}'. Use YYYY-MM-DD_HH:MM or YYYY-MM-DD", err=True)
        raise click.Abort()


def _fetch_all_keys_with_pagination(source_client: KeysManagementClient, source_base_url: str) -> List[Dict[str, Any]]:
    """Fetch all keys from source instance using pagination."""
    click.echo(f"Fetching keys from source server: {source_base_url}")
    source_keys = []
    page = 1
    page_size = 100  # Use a larger page size to minimize API calls

    while True:
        source_response = source_client.list(return_full_object=True, page=page, size=page_size)
        # source_client.list() returns Dict[str, Any] when return_request is False (default)
        assert isinstance(source_response, dict), "Expected dict response from list API"
        page_keys = source_response.get("keys", [])

        if not page_keys:
            break

        source_keys.extend(page_keys)
        click.echo(f"Fetched page {page}: {len(page_keys)} keys")

        # Check if we got fewer keys than the page size, indicating last page
        if len(page_keys) < page_size:
            break

        page += 1

    return source_keys


def _filter_keys_by_created_since(
    source_keys: List[Dict[str, Any]], created_since_dt: Optional[datetime], created_since: str
) -> List[Dict[str, Any]]:
    """Filter keys by created_since date if specified."""
    if not created_since_dt:
        return source_keys

    filtered_keys = []
    for key in source_keys:
        key_created_at = key.get("created_at")
        if key_created_at:
            # Parse the key's created_at timestamp
            if isinstance(key_created_at, str):
                if "T" in key_created_at:
                    key_dt = datetime.fromisoformat(key_created_at.replace("Z", "+00:00"))
                else:
                    key_dt = datetime.fromisoformat(key_created_at)

                # Convert to naive datetime for comparison (assuming UTC)
                if key_dt.tzinfo:
                    key_dt = key_dt.replace(tzinfo=None)

                if key_dt >= created_since_dt:
                    filtered_keys.append(key)

    click.echo(f"Filtered {len(source_keys)} keys to {len(filtered_keys)} keys created since {created_since}")
    return filtered_keys


def _display_dry_run_table(source_keys: List[Dict[str, Any]]) -> None:
    """Display a table of keys that would be imported in dry-run mode."""
    click.echo("\n--- DRY RUN MODE ---")
    table = Table(title="Keys that would be imported")
    table.add_column("Key Alias", style="green")
    table.add_column("User ID", style="magenta")
    table.add_column("Created", style="cyan")

    for key in source_keys:
        created_at = key.get("created_at", "")
        # Format the timestamp if it exists
        if created_at:
            # Try to parse and format the timestamp for better readability
            if isinstance(created_at, str):
                # Handle common timestamp formats
                if "T" in created_at:
                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    created_at = dt.strftime("%Y-%m-%d %H:%M")

        table.add_row(str(key.get("key_alias", "")), str(key.get("user_id", "")), str(created_at))
    rich.print(table)


def _prepare_key_import_data(key: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare key data for import by extracting relevant fields."""
    import_data = {}

    # Copy relevant fields if they exist
    for field in ["models", "aliases", "spend", "key_alias", "team_id", "user_id", "budget_id", "config"]:
        if key.get(field):
            import_data[field] = key[field]

    return import_data


def _import_keys_to_destination(
    source_keys: List[Dict[str, Any]], dest_client: KeysManagementClient
) -> tuple[int, int]:
    """Import each key to the destination instance and return counts."""
    imported_count = 0
    failed_count = 0

    for key in source_keys:
        try:
            # Prepare key data for import
            import_data = _prepare_key_import_data(key)

            # Generate the key in destination instance
            response = dest_client.generate(**import_data)
            click.echo(f"Generated key: {response}")
            # The generate method returns JSON data directly, not a Response object
            imported_count += 1

            key_alias = key.get("key_alias", "N/A")
            click.echo(f"✓ Imported key: {key_alias}")

        except Exception as e:
            failed_count += 1
            key_alias = key.get("key_alias", "N/A")
            click.echo(f"✗ Failed to import key {key_alias}: {str(e)}", err=True)

    return imported_count, failed_count


@keys.command(name="import")
@click.option(
    "--source-base-url", required=True, help="Base URL of the source LiteLLM proxy server to import keys from"
)
@click.option("--source-api-key", help="API key for authentication to the source server")
@click.option("--dry-run", is_flag=True, help="Show what would be imported without actually importing")
@click.option(
    "--created-since", help="Only import keys created after this date/time (format: YYYY-MM-DD_HH:MM or YYYY-MM-DD)"
)
@click.pass_context
def import_keys(
    ctx: click.Context, source_base_url: str, source_api_key: Optional[str], dry_run: bool, created_since: Optional[str]
):
    """Import API keys from another LiteLLM instance"""
    # Parse created_since filter if provided
    created_since_dt = _parse_created_since_filter(created_since)

    # Create clients for both source and destination
    source_client = KeysManagementClient(source_base_url, source_api_key)
    dest_client = KeysManagementClient(ctx.obj["base_url"], ctx.obj["api_key"])

    try:
        # Get all keys from source instance with pagination
        source_keys = _fetch_all_keys_with_pagination(source_client, source_base_url)

        # Filter keys by created_since if specified
        if created_since:
            source_keys = _filter_keys_by_created_since(source_keys, created_since_dt, created_since)

        if not source_keys:
            click.echo("No keys found in source instance.")
            return

        click.echo(f"Found {len(source_keys)} keys in source instance.")

        if dry_run:
            _display_dry_run_table(source_keys)
            return

        # Import each key
        imported_count, failed_count = _import_keys_to_destination(source_keys, dest_client)

        # Summary
        click.echo("\nImport completed:")
        click.echo(f"  Successfully imported: {imported_count}")
        click.echo(f"  Failed to import: {failed_count}")
        click.echo(f"  Total keys processed: {len(source_keys)}")

    except requests.exceptions.HTTPError as e:
        click.echo(f"Error: HTTP {e.response.status_code}", err=True)
        try:
            error_body = e.response.json()
            rich.print_json(data=error_body)
        except json.JSONDecodeError:
            click.echo(e.response.text, err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()
