from typing import Literal

import click
import rich
import rich.table

from ... import Client


def create_client(ctx: click.Context) -> Client:
    return Client(base_url=ctx.obj["base_url"], api_key=ctx.obj["api_key"])


@click.group(name="model-groups")
def model_groups() -> None:
    """Inspect model groups your key can access on the proxy"""


@model_groups.command("list")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format (table or json)",
)
@click.pass_context
def list_model_groups(ctx: click.Context, output_format: Literal["table", "json"]) -> None:
    """List model groups accessible to your key, with mode and pricing"""
    client = create_client(ctx)
    groups = client.model_groups.info()
    if not isinstance(groups, list):
        raise click.ClickException(
            f"Unexpected response from /model_group/info: expected a list, got {type(groups).__name__}"
        )

    if output_format == "json":
        rich.print_json(data=groups)
        return

    table = rich.table.Table(title="Accessible Model Groups")
    table.add_column("Model", style="cyan")
    table.add_column("Mode", style="green")
    table.add_column("Input $/token", style="yellow")
    table.add_column("Output $/token", style="yellow")

    for group in groups:
        table.add_row(
            str(group.get("model_group", "")),
            str(group.get("mode", "chat")),
            str(group.get("input_cost_per_token", "")),
            str(group.get("output_cost_per_token", "")),
        )
    rich.print(table)


__all__ = ["model_groups"]
