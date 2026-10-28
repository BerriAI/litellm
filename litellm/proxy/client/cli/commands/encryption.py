"""CLI commands for the at-rest credential encryption migration."""

from typing import Final

import click
import rich
from pydantic import TypeAdapter

from litellm.litellm_core_utils.core_helpers import as_str_mapping

from ...http_client import HTTPClient
from ._cli_context import cli_context_values

_DECODED_JSON: Final = TypeAdapter(object)


@click.group()
def encryption():
    """Migrate at-rest credentials to AES-256-GCM and attest residual state."""


@encryption.command(name="migrate")
@click.option(
    "--check",
    "check_only",
    is_flag=True,
    default=False,
    help="Read-only residual scan (no writes). Reports legacy values remaining.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Run the full migration walkers without writing any changes.",
)
@click.pass_context
def migrate(ctx: click.Context, check_only: bool, dry_run: bool):
    """Re-encrypt at-rest credentials into the AES-256-GCM (v2:gcm:) format.

    Requires the proxy to be started with
    ``general_settings.encryption_algorithm: aes-256-gcm``. Idempotent and
    resumable; safe to re-run after an interruption.

    Examples:
        lite encryption migrate --check   # attestation scan, no writes
        lite encryption migrate           # perform the migration
    """
    context: Final = cli_context_values(ctx)
    client: Final = HTTPClient(context["base_url"], context["api_key"])

    response: Final = _DECODED_JSON.validate_python(
        client.request("GET", "/credentials/migrate-encryption/check")
        if check_only
        else client.request(
            "POST",
            "/credentials/migrate-encryption",
            json={},
            params={"dry_run": "true"} if dry_run else None,
        )
    )

    rich.print_json(data=response)

    report_value: Final = (as_str_mapping(response) or {}).get("report", {})
    report: Final = as_str_mapping(report_value)
    if report is None:
        raise click.ClickException(f"Unexpected report in the proxy response: {report_value!r}")
    residual: Final = report.get("residual_legacy")
    if residual is None:
        return
    if not isinstance(residual, int | float):
        raise click.ClickException(f"Unexpected residual_legacy in the proxy response: {residual!r}")
    if residual > 0:
        rich.print(f"[yellow]Residual legacy values remaining: {residual}[/yellow]")
    elif residual == 0:
        rich.print("[green]No legacy values remaining (residual_legacy == 0).[/green]")
