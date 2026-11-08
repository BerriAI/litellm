from collections.abc import Mapping
from typing import Final, Literal

import click
import rich
import rich.table

from ... import Client
from ._cli_context import cli_context_values


def create_client(ctx: click.Context) -> Client:
    context: Final = cli_context_values(ctx)
    return Client(base_url=context["base_url"], api_key=context["api_key"])


def _rendered_field(group: Mapping[str, object], key: str, default: str) -> str:
    """The rendered value of one model group field, or ``default`` when the group omits it."""
    return str(group.get(key, default))


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
    client: Final = create_client(ctx)
    groups: Final = client.model_groups.info()
    if not isinstance(groups, list):
        raise click.ClickException(
            f"Unexpected response from /model_group/info: expected a list, got {type(groups).__name__}"
        )

    if output_format == "json":
        rich.print_json(data=groups)
        return

    table: Final = rich.table.Table(title="Accessible Model Groups")
    table.add_column("Model", style="cyan")
    table.add_column("Mode", style="green")
    table.add_column("Input $/token", style="yellow")
    table.add_column("Output $/token", style="yellow")

    for group in groups:
        table.add_row(
            _rendered_field(group, "model_group", ""),
            _rendered_field(group, "mode", "chat"),
            _rendered_field(group, "input_cost_per_token", ""),
            _rendered_field(group, "output_cost_per_token", ""),
        )
    rich.print(table)


__all__ = ["model_groups"]
