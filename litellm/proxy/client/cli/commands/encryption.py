"""CLI commands for the at-rest credential encryption migration."""

from typing import Final, TypedDict

import click
import rich
from pydantic import ConfigDict, TypeAdapter, with_config
from typing_extensions import ReadOnly

from ...http_client import HTTPClient
from .auth import cli_context_obj


@with_config(ConfigDict(extra="allow", strict=True))
class _MigrationReport(TypedDict, total=False):
    residual_legacy: ReadOnly[int | float | None]


@with_config(ConfigDict(extra="allow", strict=True))
class _MigrationResponse(TypedDict, total=False):
    report: ReadOnly[_MigrationReport]


_JSON_OBJECT: Final = TypeAdapter(object)
_MIGRATION_RESPONSE: Final = TypeAdapter(_MigrationResponse)


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
    context: Final = cli_context_obj(ctx)
    base_url: Final = context["base_url"]
    if "api_key" not in context:
        raise KeyError("api_key")
    client: Final = HTTPClient(base_url, context["api_key"])

    response: Final[object] = _JSON_OBJECT.validate_python(
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

    residual: Final = _residual_legacy(response)
    if residual is not None and residual > 0:
        rich.print(f"[yellow]Residual legacy values remaining: {residual}[/yellow]")
    elif residual == 0:
        rich.print("[green]No legacy values remaining (residual_legacy == 0).[/green]")


def _residual_legacy(response: object) -> int | float | None:
    if not isinstance(response, dict):
        return None
    payload: Final = _MIGRATION_RESPONSE.validate_python(response)
    report: Final = payload.get("report", {})
    return report.get("residual_legacy")
