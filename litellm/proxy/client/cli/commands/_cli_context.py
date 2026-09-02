from typing import Final

import click
from typing_extensions import ReadOnly, TypedDict


class CliContextValues(TypedDict):
    """Values the top-level CLI group stores on the click context."""

    base_url: ReadOnly[str]
    api_key: ReadOnly[str | None]


_UNSET_CLI_CONTEXT: Final[CliContextValues] = {"base_url": "", "api_key": None}


def cli_context_values(ctx: click.Context) -> CliContextValues:
    values: Final[CliContextValues] = getattr(ctx, "obj", _UNSET_CLI_CONTEXT)
    return values
