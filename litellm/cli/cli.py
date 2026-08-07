"""``litellm-cli`` — parent group for the offline SDK subcommands.

The four offline subcommands (``litellm-doctor``, ``litellm-cost-estimate``,
``litellm-token-count``, ``litellm-config-validate``) ship as flat
entry points in ``pyproject.toml``. This module groups them under a
single ``litellm-cli`` parent so operators can run ``litellm-cli
doctor``, ``litellm-cli cost-estimate --help``, and so on.

Subcommands are registered with lazy imports: if a subcommand file is
not present in the current install, the entry is skipped silently
and the user sees a click "unknown command" error if they try to
invoke it. This keeps the parent self-contained regardless of which
of the four subcommand PRs have been merged.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

import click


_CLI_HELP = "Offline SDK utilities for litellm."

# Subcommand name -> import path. The import path resolves to a module
# that exposes a `cli` Click command. Keep this in sync with the
# entry points in pyproject.toml.
_SUB_COMMANDS: tuple[tuple[str, str], ...] = (
    ("doctor", "litellm.cli.doctor"),
    ("cost-estimate", "litellm.cli.cost_estimate"),
    ("token-count", "litellm.cli.token_count"),
    ("config-validate", "litellm.cli.config_validate"),
)


def _try_add_subcommand(group: click.Group, name: str, module_path: str) -> str | None:
    """Attempt to import and register a subcommand. Returns a status string for tests."""
    try:
        module: ModuleType = import_module(module_path)
    except ImportError:
        return f"missing:{module_path}"
    subcommand = getattr(module, "cli", None)
    if not isinstance(subcommand, click.Command):
        return f"no-cli-attr:{module_path}"
    group.add_command(subcommand, name=name)
    return f"registered:{module_path}"


@click.group(help=_CLI_HELP)
def cli() -> None:
    """Offline SDK utilities for litellm."""


for _name, _module_path in _SUB_COMMANDS:
    _try_add_subcommand(cli, _name, _module_path)


def main() -> None:
    """Entry point for ``python -m litellm.cli``."""
    cli()


if __name__ == "__main__":
    main()
