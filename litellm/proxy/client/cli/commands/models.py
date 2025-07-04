# stdlib imports
from datetime import datetime
import re
from typing import Optional, Literal, Any
import yaml
from dataclasses import dataclass
from collections import defaultdict

# third party imports
import click
import rich

# local imports
from ... import Client


@dataclass
class ModelYamlInfo:
    model_name: str
    model_params: dict[str, Any]
    model_info: dict[str, Any]
    model_id: str
    access_groups: list[str]
    provider: str

    @property
    def access_groups_str(self) -> str:
        return ", ".join(self.access_groups) if self.access_groups else ""


def _get_model_info_obj_from_yaml(model: dict[str, Any]) -> ModelYamlInfo:
    """Extract model info from a model dict and return as ModelYamlInfo dataclass."""
    model_name: str = model["model_name"]
    model_params: dict[str, Any] = model["litellm_params"]
    model_info: dict[str, Any] = model.get("model_info", {})
    model_id: str = model_params["model"]
    access_groups = model_info.get("access_groups", [])
    provider = model_id.split("/", 1)[0] if "/" in model_id else model_id
    return ModelYamlInfo(
        model_name=model_name,
        model_params=model_params,
        model_info=model_info,
        model_id=model_id,
        access_groups=access_groups,
        provider=provider,
    )


def format_iso_datetime_str(iso_datetime_str: Optional[str]) -> str:
    """Format an ISO format datetime string to human-readable date with minute resolution."""
    if not iso_datetime_str:
        return ""
    try:
        # Parse ISO format datetime string
        dt = datetime.fromisoformat(iso_datetime_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(iso_datetime_str)


def format_timestamp(timestamp: Optional[int]) -> str:
    """Format a Unix timestamp (integer) to human-readable date with minute resolution."""
    if timestamp is None:
        return ""
    try:
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(timestamp)


def format_cost_per_1k_tokens(cost: Optional[float]) -> str:
    """Format a per-token cost to cost per 1000 tokens."""
    if cost is None:
        return ""
    try:
        # Convert string to float if needed
        cost_float = float(cost)
        # Multiply by 1000 and format to 4 decimal places
        return f"${cost_float * 1000:.4f}"
    except (TypeError, ValueError):
        return str(cost)


def create_client(ctx: click.Context) -> Client:
    """Helper function to create a client from context."""
    return Client(base_url=ctx.obj["base_url"], api_key=ctx.obj["api_key"])


@click.group()
def models() -> None:
    """Manage models on your LiteLLM proxy server"""
    pass


@models.command("list")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format (table or json)",
)
@click.pass_context
def list_models(ctx: click.Context, output_format: Literal["table", "json"]) -> None:
    """List all available models"""
    client = create_client(ctx)
    models_list = client.models.list()
    assert isinstance(models_list, list)

    if output_format == "json":
        rich.print_json(data=models_list)
    else:  # table format
        table = rich.table.Table(title="Available Models")

        # Add columns based on the data structure
        table.add_column("ID", style="cyan")
        table.add_column("Object", style="green")
        table.add_column("Created", style="magenta")
        table.add_column("Owned By", style="yellow")

        # Add rows
        for model in models_list:
            created = model.get("created")
            # Convert string timestamp to integer if needed
            if isinstance(created, str) and created.isdigit():
                created = int(created)

            table.add_row(
                str(model.get("id", "")),
                str(model.get("object", "model")),
                format_timestamp(created) if isinstance(created, int) else format_iso_datetime_str(created),
                str(model.get("owned_by", "")),
            )

        rich.print(table)


@models.command("add")
@click.argument("model-name")
@click.option(
    "--param",
    "-p",
    multiple=True,
    help="Model parameters in key=value format (can be specified multiple times)",
)
@click.option(
    "--info",
    "-i",
    multiple=True,
    help="Model info in key=value format (can be specified multiple times)",
)
@click.pass_context
def add_model(ctx: click.Context, model_name: str, param: tuple[str, ...], info: tuple[str, ...]) -> None:
    """Add a new model to the proxy"""
    # Convert parameters from key=value format to dict
    model_params = dict(p.split("=", 1) for p in param)
    model_info = dict(i.split("=", 1) for i in info) if info else None

    client = create_client(ctx)
    result = client.models.new(
        model_name=model_name,
        model_params=model_params,
        model_info=model_info,
    )
    rich.print_json(data=result)


@models.command("delete")
@click.argument("model-id")
@click.pass_context
def delete_model(ctx: click.Context, model_id: str) -> None:
    """Delete a model from the proxy"""
    client = create_client(ctx)
    result = client.models.delete(model_id=model_id)
    rich.print_json(data=result)


@models.command("get")
@click.option("--id", "model_id", help="ID of the model to retrieve")
@click.option("--name", "model_name", help="Name of the model to retrieve")
@click.pass_context
def get_model(ctx: click.Context, model_id: Optional[str], model_name: Optional[str]) -> None:
    """Get information about a specific model"""
    if not model_id and not model_name:
        raise click.UsageError("Either --id or --name must be provided")

    client = create_client(ctx)
    result = client.models.get(model_id=model_id, model_name=model_name)
    rich.print_json(data=result)


@models.command("info")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format (table or json)",
)
@click.option(
    "--columns",
    "columns",
    default="public_model,upstream_model,updated_at",
    help="Comma-separated list of columns to display. Valid columns: public_model, upstream_model, credential_name, created_at, updated_at, id, input_cost, output_cost. Default: public_model,upstream_model,updated_at",
)
@click.pass_context
def get_models_info(ctx: click.Context, output_format: Literal["table", "json"], columns: str) -> None:
    """Get detailed information about all models"""
    client = create_client(ctx)
    models_info = client.models.info()
    assert isinstance(models_info, list)

    if output_format == "json":
        rich.print_json(data=models_info)
    else:  # table format
        table = rich.table.Table(title="Models Information")

        # Define all possible columns with their configurations
        column_configs: dict[str, dict[str, Any]] = {
            "public_model": {
                "header": "Public Model",
                "style": "cyan",
                "get_value": lambda m: str(m.get("model_name", "")),
            },
            "upstream_model": {
                "header": "Upstream Model",
                "style": "green",
                "get_value": lambda m: str(m.get("litellm_params", {}).get("model", "")),
            },
            "credential_name": {
                "header": "Credential Name",
                "style": "yellow",
                "get_value": lambda m: str(m.get("litellm_params", {}).get("litellm_credential_name", "")),
            },
            "created_at": {
                "header": "Created At",
                "style": "magenta",
                "get_value": lambda m: format_iso_datetime_str(m.get("model_info", {}).get("created_at")),
            },
            "updated_at": {
                "header": "Updated At",
                "style": "magenta",
                "get_value": lambda m: format_iso_datetime_str(m.get("model_info", {}).get("updated_at")),
            },
            "id": {
                "header": "ID",
                "style": "blue",
                "get_value": lambda m: str(m.get("model_info", {}).get("id", "")),
            },
            "input_cost": {
                "header": "Input Cost",
                "style": "green",
                "justify": "right",
                "get_value": lambda m: format_cost_per_1k_tokens(m.get("model_info", {}).get("input_cost_per_token")),
            },
            "output_cost": {
                "header": "Output Cost",
                "style": "green",
                "justify": "right",
                "get_value": lambda m: format_cost_per_1k_tokens(m.get("model_info", {}).get("output_cost_per_token")),
            },
        }

        # Add requested columns
        requested_columns = [col.strip() for col in columns.split(",")]
        for col_name in requested_columns:
            if col_name in column_configs:
                config = column_configs[col_name]
                table.add_column(config["header"], style=config["style"], justify=config.get("justify", "left"))
            else:
                click.echo(f"Warning: Unknown column '{col_name}'", err=True)

        # Add rows with only the requested columns
        for model in models_info:
            row_values = []
            for col_name in requested_columns:
                if col_name in column_configs:
                    row_values.append(column_configs[col_name]["get_value"](model))
            if row_values:
                table.add_row(*row_values)

        rich.print(table)


@models.command("update")
@click.argument("model-id")
@click.option(
    "--param",
    "-p",
    multiple=True,
    help="Model parameters in key=value format (can be specified multiple times)",
)
@click.option(
    "--info",
    "-i",
    multiple=True,
    help="Model info in key=value format (can be specified multiple times)",
)
@click.pass_context
def update_model(ctx: click.Context, model_id: str, param: tuple[str, ...], info: tuple[str, ...]) -> None:
    """Update an existing model's configuration"""
    # Convert parameters from key=value format to dict
    model_params = dict(p.split("=", 1) for p in param)
    model_info = dict(i.split("=", 1) for i in info) if info else None

    client = create_client(ctx)
    result = client.models.update(
        model_id=model_id,
        model_params=model_params,
        model_info=model_info,
    )
    rich.print_json(data=result)


def _filter_model(model, model_regex, access_group_regex):
    model_name = model.get("model_name")
    model_params = model.get("litellm_params")
    model_info = model.get("model_info", {})
    if not model_name or not model_params:
        return False
    model_id = model_params.get("model")
    if not model_id or not isinstance(model_id, str):
        return False
    if model_regex and not model_regex.search(model_id):
        return False
    access_groups = model_info.get("access_groups", [])
    if access_group_regex:
        if not isinstance(access_groups, list):
            return False
        if not any(isinstance(group, str) and access_group_regex.search(group) for group in access_groups):
            return False
    return True


def _print_models_table(added_models: list[ModelYamlInfo], table_title: str):
    if not added_models:
        return
    table = rich.table.Table(title=table_title)
    table.add_column("Model Name", style="cyan")
    table.add_column("Upstream Model", style="green")
    table.add_column("Access Groups", style="magenta")
    for m in added_models:
        table.add_row(m.model_name, m.model_id, m.access_groups_str)
    rich.print(table)


def _print_summary_table(provider_counts):
    summary_table = rich.table.Table(title="Model Import Summary")
    summary_table.add_column("Provider", style="cyan")
    summary_table.add_column("Count", style="green")

    for provider, count in provider_counts.items():
        summary_table.add_row(str(provider), str(count))

    total = sum(provider_counts.values())
    summary_table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]")

    rich.print(summary_table)


def get_model_list_from_yaml_file(yaml_file: str) -> list[dict[str, Any]]:
    """Load and validate the model list from a YAML file."""
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
    if not data or "model_list" not in data:
        raise click.ClickException("YAML file must contain a 'model_list' key with a list of models.")
    model_list = data["model_list"]
    if not isinstance(model_list, list):
        raise click.ClickException("'model_list' must be a list of model definitions.")
    return model_list


def _get_filtered_model_list(model_list, only_models_matching_regex, only_access_groups_matching_regex):
    """Return a list of models that pass the filter criteria."""
    model_regex = re.compile(only_models_matching_regex) if only_models_matching_regex else None
    access_group_regex = re.compile(only_access_groups_matching_regex) if only_access_groups_matching_regex else None
    return [model for model in model_list if _filter_model(model, model_regex, access_group_regex)]


def _import_models_get_table_title(dry_run: bool) -> str:
    if dry_run:
        return "Models that would be imported if [yellow]--dry-run[/yellow] was not provided"
    else:
        return "Models Imported"


@models.command("import")
@click.argument("yaml_file", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--dry-run", is_flag=True, help="Show what would be imported without making any changes.")
@click.option(
    "--only-models-matching-regex",
    default=None,
    help="Only import models where litellm_params.model matches the given regex.",
)
@click.option(
    "--only-access-groups-matching-regex",
    default=None,
    help="Only import models where at least one item in model_info.access_groups matches the given regex.",
)
@click.pass_context
def import_models(
    ctx: click.Context,
    yaml_file: str,
    dry_run: bool,
    only_models_matching_regex: Optional[str],
    only_access_groups_matching_regex: Optional[str],
) -> None:
    """Import models from a YAML file and add them to the proxy."""
    provider_counts: dict[str, int] = defaultdict(int)
    added_models: list[ModelYamlInfo] = []
    model_list = get_model_list_from_yaml_file(yaml_file)
    filtered_model_list = _get_filtered_model_list(
        model_list, only_models_matching_regex, only_access_groups_matching_regex
    )

    if not dry_run:
        client = create_client(ctx)

    for model in filtered_model_list:
        model_info_obj = _get_model_info_obj_from_yaml(model)
        if not dry_run:
            try:
                client.models.new(
                    model_name=model_info_obj.model_name,
                    model_params=model_info_obj.model_params,
                    model_info=model_info_obj.model_info,
                )
            except Exception:
                pass  # For summary, ignore errors
        added_models.append(model_info_obj)
        provider_counts[model_info_obj.provider] += 1

    table_title = _import_models_get_table_title(dry_run)
    _print_models_table(added_models, table_title)
    _print_summary_table(provider_counts)
