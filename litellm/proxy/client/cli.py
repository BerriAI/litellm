import click
import json
from typing import Optional, Literal
from .client import Client
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from datetime import datetime


def create_client(ctx: click.Context) -> Client:
    """Helper function to create a client from context."""
    return Client(base_url=ctx.obj["base_url"], api_key=ctx.obj["api_key"])


def format_iso_datetime_str(iso_datetime_str: Optional[str]) -> str:
    """Format an ISO format datetime string to human-readable date with minute resolution."""
    if not iso_datetime_str:
        return ""
    try:
        # Parse ISO format datetime string
        dt = datetime.fromisoformat(iso_datetime_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(iso_datetime_str)


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
    ctx.obj["console"] = Console()


@cli.group()
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
    models = client.models.list()
    console = ctx.obj["console"]

    if output_format == "json":
        # Create syntax highlighted JSON without line numbers
        json_str = json.dumps(models, indent=2)
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
        console.print(syntax)
    else:  # table format
        table = Table(title="Available Models")
        
        # Add columns based on the data structure
        table.add_column("ID", style="cyan")
        table.add_column("Object", style="green")
        table.add_column("Created", style="magenta")
        table.add_column("Owned By", style="yellow")

        # Add rows
        for model in models:
            table.add_row(
                str(model.get("id", "")),
                str(model.get("object", "model")),
                format_iso_datetime_str(model.get("created")),
                str(model.get("owned_by", ""))
            )

        console.print(table)


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
    click.echo(json.dumps(result, indent=2))


@models.command("delete")
@click.argument("model-id")
@click.pass_context
def delete_model(ctx: click.Context, model_id: str) -> None:
    """Delete a model from the proxy"""
    client = create_client(ctx)
    result = client.models.delete(model_id=model_id)
    click.echo(json.dumps(result, indent=2))


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
    click.echo(json.dumps(result, indent=2))


@models.command("info")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format (table or json)",
)
@click.pass_context
def get_models_info(ctx: click.Context, output_format: Literal["table", "json"]) -> None:
    """Get detailed information about all models"""
    client = create_client(ctx)
    models_info = client.models.info()
    console = ctx.obj["console"]

    if output_format == "json":
        # Create syntax highlighted JSON without line numbers
        json_str = json.dumps(models_info, indent=2)
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
        console.print(syntax)
    else:  # table format
        table = Table(title="Models Information")
        
        # Add columns based on the data structure
        table.add_column("Public Model", style="cyan")
        table.add_column("Upstream Model", style="green")
        table.add_column("Credential Name", style="yellow")
        table.add_column("Created At", style="magenta")
        table.add_column("Updated At", style="magenta")
        table.add_column("ID", style="blue")
        table.add_column("Input Cost", style="green", justify="right")
        table.add_column("Output Cost", style="green", justify="right")

        # Add rows
        for model in models_info:
            input_cost = model.get("model_info", {}).get("input_cost_per_token")
            output_cost = model.get("model_info", {}).get("output_cost_per_token")
            
            table.add_row(
                str(model.get("model_name", "")),
                str(model.get("litellm_params", {}).get("model", "")),
                str(model.get("litellm_params", {}).get("litellm_credential_name", "")),
                format_iso_datetime_str(model.get("model_info", {}).get("created_at")),
                format_iso_datetime_str(model.get("model_info", {}).get("updated_at")),
                str(model.get("model_info", {}).get("id", "")),
                format_cost_per_1k_tokens(input_cost),
                format_cost_per_1k_tokens(output_cost)
            )

        console.print(table)


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
    click.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli() 