"""CLI commands for the at-rest credential encryption migration."""

from types import MappingProxyType
from typing import Final

import click
import rich

from ...http_client import HTTPClient


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
@click.option(
    "--mode",
    type=click.Choice(("algorithm", "salt-key")),
    default="algorithm",
    show_default=True,
    help="'algorithm' re-encrypts into AES-256-GCM; 'salt-key' re-encrypts under the active salt key.",
)
@click.pass_context
def migrate(ctx: click.Context, check_only: bool, dry_run: bool, mode: str):
    """Re-encrypt at-rest credentials, by algorithm or under a rotated salt key.

    ``--mode algorithm`` (the default) moves values into the AES-256-GCM
    (v2:gcm:) format and requires the proxy to be started with
    ``general_settings.encryption_algorithm: aes-256-gcm``.

    ``--mode salt-key`` re-encrypts values that still decrypt only under a
    retired salt key. Restart the proxy with the new key in ``LITELLM_SALT_KEY``
    and the retired one(s) in ``LITELLM_SALT_KEY_PREVIOUS`` (comma-separated), run
    this, then drop ``LITELLM_SALT_KEY_PREVIOUS`` once ``--check`` reports
    ``residual_legacy: 0`` and an empty ``unreadable_locations``. Virtual keys are
    SHA-256 hashes rather than salt-key ciphertext, so they keep working
    throughout and never need regenerating.

    Both modes are idempotent and resumable; safe to re-run after an
    interruption.

    Examples:
        litellm-proxy encryption migrate --check              # attestation scan, no writes
        litellm-proxy encryption migrate                     # perform the migration
        litellm-proxy encryption migrate --mode salt-key     # rotate the salt key
    """
    client: Final = HTTPClient(ctx.obj["base_url"], ctx.obj["api_key"])

    if check_only:
        response = client.request(
            "GET",
            "/credentials/migrate-encryption/check",
            params=MappingProxyType({"mode": mode}),
        )
    else:
        response = client.request(
            "POST",
            "/credentials/migrate-encryption",
            json={},
            params=MappingProxyType({"mode": mode, "dry_run": str(dry_run).lower()}),
        )

    rich.print_json(data=response)

    report: Final = response.get("report", {}) if isinstance(response, dict) else {}
    residual: Final = report.get("residual_legacy")
    if residual is not None and residual > 0:
        rich.print(f"[yellow]Residual legacy values remaining: {residual}[/yellow]")
    elif residual == 0:
        rich.print("[green]No legacy values remaining (residual_legacy == 0).[/green]")
