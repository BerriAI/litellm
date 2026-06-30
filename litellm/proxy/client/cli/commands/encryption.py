"""CLI commands for the at-rest credential encryption migration."""

import click
import rich

from ...http_client import HTTPClient


@click.group()
def encryption():
    """Migrate at-rest credentials to AES-256-GCM and attest residual state."""
    pass


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
    resumable — safe to re-run after an interruption.

    Examples:
        litellm-proxy encryption migrate --check   # attestation scan, no writes
        litellm-proxy encryption migrate           # perform the migration
    """
    client = HTTPClient(ctx.obj["base_url"], ctx.obj["api_key"])

    if check_only:
        response = client.request("GET", "/credentials/migrate-encryption/check")
    else:
        response = client.request(
            "POST",
            "/credentials/migrate-encryption",
            json={},
            params={"dry_run": "true"} if dry_run else None,
        )

    rich.print_json(data=response)

    report = response.get("report", {}) if isinstance(response, dict) else {}
    residual = report.get("residual_legacy")
    if residual is not None and residual > 0:
        rich.print(f"[yellow]Residual legacy values remaining: {residual}[/yellow]")
    elif residual == 0:
        rich.print("[green]No legacy values remaining (residual_legacy == 0).[/green]")
