# stdlib imports
from typing import Optional, Literal, Any
from datetime import datetime

# third party imports
import click
import rich

# local imports
from ... import Client


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
