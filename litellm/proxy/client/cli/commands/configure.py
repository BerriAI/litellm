"""Commands for saved Claude Code and Codex gateway setup."""

import sys
from pathlib import Path
from typing import Final

import click
from InquirerPy import inquirer
from pydantic import BaseModel

from .auth import CliContextObj
from .claude_settings import (
    ClaudeSettingsError,
    UnconfigureOutcome,
    preflight_claude_settings,
    settings_file_owners,
    unconfigure_claude_settings,
)
from .codex_settings import CodexSettingsError, unconfigure_codex_settings
from .config import normalize_base_url
from .configure_profiles import (
    TARGETS,
    Target,
    forget_saved_setup,
    read_saved_setup,
    receipt_path_for,
    settings_path_for,
    setup_locks,
    setup_profile_path,
)
from .configure_setup import (
    MODEL_OPTION_HELP,
    ConnectionSettings,
    configure_targets,
    interactive_configure,
    pick_targets,
    resolve_credential,
)


class _ConnectionOptions(BaseModel):
    api_key: str | None = None
    gateway_url: str | None = None


def _connection_settings(ctx: click.Context, api_key: str | None, gateway_url: str | None) -> CliContextObj:
    """The context object a subcommand runs with: its own --api-key / --gateway-url over the group's, over `lite`'s."""
    ctx_obj: Final = ConnectionSettings.model_validate(ctx.find_object(object))
    group: Final = (
        _ConnectionOptions.model_validate(ctx.parent.params)
        if ctx.parent is not None and ctx.parent.command.name in ("configure", "reconfigure")
        else _ConnectionOptions()
    )
    key: Final = api_key if api_key is not None else group.api_key
    url: Final = gateway_url if gateway_url is not None else group.gateway_url
    normalized: Final = normalize_base_url(url if url is not None else ctx_obj.base_url)
    connection: Final[CliContextObj] = {
        "base_url": normalized.removesuffix("/v1"),
        "base_url_explicit": url is not None or ctx_obj.base_url_explicit,
        "api_key": key if key is not None else ctx_obj.api_key,
        "api_key_from_token_file": False if key is not None else ctx_obj.api_key_from_token_file,
    }
    return connection


def _connection_context(ctx: click.Context, settings: CliContextObj) -> click.Context:
    return click.Context(ctx.command, parent=ctx.parent, obj=settings)


def _require_terminal(command: str) -> None:
    if sys.stdin.isatty():
        return
    raise click.ClickException(
        f"`lite {command}` asks questions, so it needs a terminal. Non-interactively, run "
        f"`lite {command} claude --api-key <key> --model <model>` or "
        f"`lite {command} codex --api-key <key> --model <model>`"
    )


def _configure_group(ctx: click.Context, api_key: str | None, gateway_url: str | None, edit: bool) -> None:
    if ctx.invoked_subcommand is not None:
        return
    settings: Final = _connection_settings(ctx, api_key, gateway_url)
    connection: Final = _connection_context(ctx, settings)
    with setup_locks(TARGETS):
        saved_targets: Final[tuple[Target, ...]] = tuple(
            target for target in TARGETS if read_saved_setup(target) is not None
        )
        if saved_targets and not edit:
            configure_targets(connection, saved_targets)
            return
        _require_terminal("reconfigure" if edit else "configure")
        selected: Final = pick_targets(saved_targets or TARGETS, edit=edit)
        if not selected:
            return
        if edit:
            configure_targets(connection, selected, interactive=True, edit_connection=True)
            return
        prompted: Final = (
            settings
            if settings.get("base_url_explicit")
            else _connection_settings(connection, None, click.prompt("Gateway URL", default=settings["base_url"]))
        )
        configure_targets(_connection_context(connection, prompted), selected, interactive=True)


@click.group(name="configure", invoke_without_command=True)
@click.option("--api-key", default=None, help="Long-lived LiteLLM virtual key to save for the selected agents.")
@click.option("--gateway-url", "--base-url", default=None, help="Gateway URL, including any deployment path prefix.")
@click.pass_context
def configure_group(ctx: click.Context, api_key: str | None, gateway_url: str | None) -> None:
    """Apply saved setup, or choose agents and models on the first run."""
    _configure_group(ctx, api_key, gateway_url, False)


@click.group(name="reconfigure", invoke_without_command=True)
@click.option("--api-key", default=None, help="Long-lived LiteLLM virtual key to save for the selected agents.")
@click.option("--gateway-url", "--base-url", default=None, help="Gateway URL, including any deployment path prefix.")
@click.pass_context
def reconfigure_group(ctx: click.Context, api_key: str | None, gateway_url: str | None) -> None:
    """Edit saved gateway, key and model choices, using current choices as defaults."""
    _configure_group(ctx, api_key, gateway_url, True)


def _configure_target(
    ctx: click.Context,
    target: Target,
    api_key: str | None,
    gateway_url: str | None,
    model: str | None,
    default_model: bool = False,
    *,
    edit: bool = False,
) -> None:
    settings: Final = _connection_settings(ctx, api_key, gateway_url)
    interactive: Final = edit and model is None and not default_model
    if interactive:
        _require_terminal("reconfigure")
    with setup_locks((target,)):
        configure_targets(
            _connection_context(ctx, settings),
            (target,),
            model=model,
            default_model=default_model,
            interactive=interactive,
            edit_connection=interactive,
        )


@configure_group.command(name="claude")
@click.option("--api-key", default=None, help="Long-lived LiteLLM virtual key, or reuse the saved key.")
@click.option("--gateway-url", "--base-url", default=None, help="Gateway URL, or reuse the saved gateway.")
@click.option("--model", default=None, help=MODEL_OPTION_HELP)
@click.option(
    "--default-model", is_flag=True, help="Stop pinning a starting model; let Claude Code choose its default."
)
@click.pass_context
def configure_claude(
    ctx: click.Context,
    api_key: str | None,
    gateway_url: str | None,
    model: str | None,
    default_model: bool,
) -> None:
    """Apply Claude Code's saved setup, or save the supplied settings."""
    _configure_target(ctx, "claude", api_key, gateway_url, model, default_model)


@configure_group.command(name="codex")
@click.option("--api-key", default=None, help="Long-lived LiteLLM virtual key, or reuse the saved key.")
@click.option("--gateway-url", "--base-url", default=None, help="Gateway URL, or reuse the saved gateway.")
@click.option("--model", default=None, help="Gateway model to start on; required only for first-time setup.")
@click.pass_context
def configure_codex(ctx: click.Context, api_key: str | None, gateway_url: str | None, model: str | None) -> None:
    """Apply Codex's saved setup, or save the supplied settings."""
    _configure_target(ctx, "codex", api_key, gateway_url, model)


@reconfigure_group.command(name="claude")
@click.option("--api-key", default=None, help="Long-lived LiteLLM virtual key, or reuse the saved key.")
@click.option("--gateway-url", "--base-url", default=None, help="Gateway URL, or reuse the saved gateway.")
@click.option("--model", default=None, help=MODEL_OPTION_HELP)
@click.option(
    "--default-model", is_flag=True, help="Stop pinning a starting model; let Claude Code choose its default."
)
@click.pass_context
def reconfigure_claude(
    ctx: click.Context,
    api_key: str | None,
    gateway_url: str | None,
    model: str | None,
    default_model: bool,
) -> None:
    """Edit Claude Code setup, or supply --model / --default-model to apply directly."""
    _configure_target(ctx, "claude", api_key, gateway_url, model, default_model, edit=True)


@reconfigure_group.command(name="codex")
@click.option("--api-key", default=None, help="Long-lived LiteLLM virtual key, or reuse the saved key.")
@click.option("--gateway-url", "--base-url", default=None, help="Gateway URL, or reuse the saved gateway.")
@click.option("--model", default=None, help="Gateway model to start on; omit to open the setup wizard.")
@click.pass_context
def reconfigure_codex(ctx: click.Context, api_key: str | None, gateway_url: str | None, model: str | None) -> None:
    """Edit Codex setup, or supply --model to apply directly."""
    _configure_target(ctx, "codex", api_key, gateway_url, model, edit=True)


def _disconnect(target: Target, forget: bool) -> None:
    settings_path: Final = settings_path_for(target)
    state_path: Final = receipt_path_for(target, settings_path)
    profile: Final = setup_profile_path(target, settings_path)
    try:
        if not state_path.exists():
            click.echo(f"No {target} undo receipt at {state_path}; nothing to undo. Agent settings were not changed.")
            if settings_path.exists():
                click.echo(
                    f"Cannot confirm disconnection. Check {settings_path} and remove any remaining gateway "
                    "connection and key manually.",
                    err=True,
                )
        elif target == "claude":
            preflight_claude_settings(settings_path)
            outcome: Final = unconfigure_claude_settings(settings_path, state_path, settings_file_owners(settings_path))
            _report_unconfigure(settings_path, state_path, outcome)
        else:
            codex_outcome: Final = unconfigure_codex_settings(settings_path)
            if codex_outcome.file_removed:
                click.echo(f"Removed {settings_path}; it held only settings created by `lite configure codex`.")
            elif codex_outcome.restored:
                click.echo(f"Restored in {settings_path}: {', '.join(codex_outcome.restored)}.")
            else:
                click.echo(f"Nothing in {settings_path} was still ours to restore.")
            if codex_outcome.kept:
                click.echo(f"Left as you changed them since: {', '.join(codex_outcome.kept)}.")
    except (ClaudeSettingsError, CodexSettingsError) as error:
        raise click.ClickException(str(error)) from error
    if forget:
        forget_saved_setup(target)
        click.echo(f"Forgot saved {target} setup, including its saved key.")
    elif profile.exists():
        click.echo(f"Saved setup retained. Run `lite configure {target}` to apply it again.")


@click.group(name="unconfigure", invoke_without_command=True)
@click.option("--forget", is_flag=True, help="Also delete saved setups and their keys.")
@click.pass_context
def unconfigure_group(ctx: click.Context, forget: bool) -> None:
    """Disconnect agents while retaining saved setup for `lite configure`."""
    if ctx.invoked_subcommand is not None:
        return
    with setup_locks(TARGETS):
        for target in TARGETS:
            _disconnect(target, forget)


class _UnconfigureOptions(BaseModel):
    forget: bool = False


def _unconfigure_target(ctx: click.Context, target: Target, forget: bool) -> None:
    parent: Final = _UnconfigureOptions.model_validate(ctx.parent.params) if ctx.parent else _UnconfigureOptions()
    with setup_locks((target,)):
        _disconnect(target, forget or parent.forget)


@unconfigure_group.command(name="claude")
@click.option("--forget", is_flag=True, help="Also delete the saved Claude Code setup and key.")
@click.pass_context
def unconfigure_claude(ctx: click.Context, forget: bool) -> None:
    """Restore Claude Code settings, including those applied by `lite login --config-claude`."""
    _unconfigure_target(ctx, "claude", forget)


@unconfigure_group.command(name="codex")
@click.option("--forget", is_flag=True, help="Also delete the saved Codex setup and key.")
@click.pass_context
def unconfigure_codex(ctx: click.Context, forget: bool) -> None:
    """Restore Codex settings still holding what configure wrote."""
    _unconfigure_target(ctx, "codex", forget)


def _report_unconfigure(settings_path: Path, state_path: Path, outcome: UnconfigureOutcome) -> None:
    """Say what unconfigure did, naming only keys whose value it changed."""
    if outcome.file_removed:
        click.echo(
            f"No settings file remains at {settings_path}; it held nothing but `lite configure claude`'s own keys."
        )
    elif outcome.restored:
        click.echo(f"Restored in {settings_path}: {', '.join(outcome.restored)}.")
    else:
        click.echo(f"Nothing in {settings_path} was still ours to restore.")
    if outcome.kept:
        click.echo(f"Left as you changed them since: {', '.join(outcome.kept)}.")
    if outcome.withheld:
        click.echo(
            "Left removed, since the file now points at a different server than they were issued for: "
            + "; ".join(f"{item.key} (captured with {item.endpoint})" for item in outcome.withheld)
            + f". They stay in {state_path}: point env.ANTHROPIC_BASE_URL back and run `lite unconfigure claude` "
            "again to put them back, or delete that file to drop them."
        )


__all__ = (
    "configure_group",
    "inquirer",
    "interactive_configure",
    "reconfigure_group",
    "resolve_credential",
    "unconfigure_group",
)
