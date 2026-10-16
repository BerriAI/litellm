from __future__ import annotations

from pathlib import Path
from typing import Final

import click

from .runner import DriftError, inspect_drift


def drift_command(repo_root: Path) -> click.Command:
    def inspect(target: str, manifest: Path, since: str | None, output: Path | None) -> int:
        manifest_path: Final = repo_root / manifest
        try:
            overwrites_manifest: Final = output is not None and (
                output.resolve() == manifest_path.resolve()
                or (output.exists() and manifest_path.exists() and output.samefile(manifest_path))
            )
        except (OSError, RuntimeError) as error:
            raise click.ClickException(f"Cannot resolve drift output or manifest path, {error}") from error
        if overwrites_manifest:
            raise click.ClickException("Output path must differ from the drift manifest")
        report: Final = inspect_drift(repo_root, manifest_path, target_ref=target, since_ref=since)
        if isinstance(report, DriftError):
            raise click.ClickException(report.message)
        rendered: Final = report.model_dump_json(indent=2)
        if output is None:
            click.echo(rendered)
            return 0
        try:
            output.write_text(f"{rendered}\n", encoding="utf-8")
        except OSError as error:
            raise click.ClickException(f"Cannot write drift report to {output}, {error}") from error
        return 0

    return click.Command(
        "drift",
        params=[
            click.Option(("--target",), default="HEAD", show_default=True, help="Git revision to inspect"),
            click.Option(
                ("--manifest",),
                default="tests/rust-python-harness/strategies/drift/drift.json",
                type=click.Path(path_type=Path, dir_okay=False),
                show_default=True,
                help="Manifest path, relative to the repository root",
            ),
            click.Option(("--since",), help="Exploratory comparison revision, overriding the checkpoint"),
            click.Option(
                ("--output",),
                type=click.Path(path_type=Path, dir_okay=False),
                help="Write JSON to this path instead of stdout",
            ),
        ],
        callback=inspect,
        help="Inventory Python and Rust changes. Exit zero means a report was generated, without asserting parity.",
    )
