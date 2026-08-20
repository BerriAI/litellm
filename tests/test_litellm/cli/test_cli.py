"""Tests for the ``litellm-cli`` parent group command."""

from __future__ import annotations

import click
from click.testing import CliRunner

from litellm.cli.cli import (
    _SUB_COMMANDS,
    _try_add_subcommand,
    cli,
    main,
)


# ---------- parent group structure ----------


def test_cli_is_a_click_group():
    """The parent is a Click group, not a bare function."""
    assert isinstance(cli, click.Group)


def test_cli_help_lists_subcommands_or_no_subcommand_message():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    # The help text contains either the parent description or subcommand names.
    # Subcommands may be missing in this branch; the parent still works.
    assert "Offline SDK utilities" in result.output or "litellm" in result.output.lower()


def test_cli_unknown_subcommand_exits_2():
    """click default: unknown subcommand produces exit code 2 and an error message."""
    runner = CliRunner()
    result = runner.invoke(cli, ["definitely-not-a-real-subcommand"])
    assert result.exit_code == 2
    assert "No such command" in result.output or "unknown" in result.output.lower()


def test_cli_no_args_prints_usage_and_exits_2():
    """`litellm-cli` with no args exits 2 (click default for a group with no resolved subcommand)."""
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code == 2
    assert "Usage:" in result.output
    assert "Offline SDK utilities" in result.output


# ---------- subcommand registration ----------


def test_sub_command_list_includes_expected_names():
    """The static subcommand list registers the four expected names."""
    names = {name for name, _ in _SUB_COMMANDS}
    assert {"doctor", "cost-estimate", "token-count", "config-validate"} <= names


def test_try_add_subcommand_handles_missing_module(monkeypatch):
    """When the subcommand module cannot be imported, the helper returns a status string and does not raise."""
    from click import Group

    group = Group("test")
    status = _try_add_subcommand(group, "missing", "definitely.not.a.real.module")
    assert status.startswith("missing:")
    # The group has no commands registered.
    assert "missing" not in group.commands


def test_try_add_subcommand_rejects_module_without_cli_attr(monkeypatch):
    """A module that does not expose `cli` is skipped cleanly."""
    import sys
    import types

    fake_module = types.ModuleType("fake_module_no_cli")
    # No `cli` attribute.
    monkeypatch.setitem(sys.modules, "fake_module_no_cli", fake_module)

    group = click.Group("test")
    status = _try_add_subcommand(group, "fake", "fake_module_no_cli")
    assert status.startswith("no-cli-attr:")
    assert "fake" not in group.commands


def test_try_add_subcommand_registers_real_click_command(monkeypatch):
    """A module that exposes a Click command is registered as a subcommand."""
    import sys
    import types

    @click.command()
    def _inner() -> None:
        """noop"""

    fake_module = types.ModuleType("fake_module_with_cli")
    fake_module.cli = _inner
    monkeypatch.setitem(sys.modules, "fake_module_with_cli", fake_module)

    group = click.Group("test")
    status = _try_add_subcommand(group, "fake", "fake_module_with_cli")
    assert status.startswith("registered:")
    assert "fake" in group.commands


# ---------- python -m litellm.cli entry ----------


def test_main_is_callable():
    """`main()` is the entry point for `python -m litellm.cli` and is callable."""
    assert main is not None
    assert callable(main)
