"""`lite configure claude` and `lite unconfigure claude`: persistent Claude Code wiring, undoable."""

import os
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import click
from InquirerPy import inquirer
from InquirerPy.base.control import Choice

from .auth import CliContextObj, context_secret_vault, get_stored_api_key
from .claude_settings import (
    STARTING_MODEL_ROLE,
    ApiKeyHelper,
    ClaudeCredential,
    ClaudeSettingsError,
    ModelChoice,
    StartOn,
    StaticToken,
    UnconfigureOutcome,
    UnpinModel,
    claude_settings_path,
    configure_claude_settings,
    configure_state_path,
    refuse_while_owned,
    resolve_api_key_helper,
    settings_file_owners,
    unconfigure_claude_settings,
)
from .pi import ListingFailure, PiSyncError, fetch_model_ids
from .up import ensure_fresh_login

_LISTED_MODELS_SHOWN: Final = 20
_CLAUDE_TARGET: Final = "claude"
_TARGETS: Final = ((_CLAUDE_TARGET, "Claude Code (CLI)"),)
_KEEP_DEFAULT_MODEL: Final = "Keep Claude Code's own default"
_CLAUDE_CODE_PICKER_FILTER: Final = re.compile(r"claude|anthropic", re.IGNORECASE)
_MODEL_OPTION_HELP: Final = (
    f"Proxy model to set as {STARTING_MODEL_ROLE}. Must be listed on /v1/models for the key; without it, "
    "Claude Code keeps its own default and a pin an earlier configure made is let go of. Nothing pins Claude "
    "Code's sub-agent or background tiers; `lite autoroute up` is the mode that does."
)


def resolve_credential(ctx: click.Context, api_key: str | None) -> tuple[ClaudeCredential, str]:
    """The credential to write and the key to check the proxy with.

    An explicit key (--api-key, `lite --api-key`, LITELLM_PROXY_API_KEY) is long-lived and goes
    into settings.json as a static token. Without one, the stored `lite login` credential is used
    the way `lite login --config-claude` uses it, through apiKeyHelper, since it expires within a
    day and renews in place there; a missing or stale login is refreshed first, as `lite up` does.
    """
    ctx_obj: Final[CliContextObj] = ctx.obj
    explicit: Final = api_key or (None if ctx_obj.get("api_key_from_token_file") else ctx_obj.get("api_key"))
    if explicit:
        return StaticToken(explicit), explicit
    base_url: Final = ctx_obj["base_url"]
    ensure_fresh_login(ctx)
    stored: Final = get_stored_api_key(expected_base_url=base_url, vault=context_secret_vault(ctx))
    if not stored:
        raise ClaudeSettingsError("Login did not produce a usable token.")
    return ApiKeyHelper(resolve_api_key_helper(base_url)), stored


def _start(ctx: click.Context, api_key: str | None) -> tuple[ClaudeCredential, tuple[str, ...]]:
    """Every configure path begins the same way: the local ownership check first, so a `lite up`
    session is refused before any login prompt or request, then the credential, then the listing."""
    settings_path: Final = claude_settings_path(os.environ)
    try:
        refuse_while_owned(settings_path, settings_file_owners(settings_path))
        credential, key = resolve_credential(ctx, api_key)
    except ClaudeSettingsError as e:
        raise click.ClickException(str(e))
    return credential, _listed_models(ctx.obj["base_url"], key)


def _listing_error(base_url: str, error: PiSyncError) -> str:
    """The hint that fits how the listing failed: only an unreachable proxy gets the "is it running" question."""
    if error.kind is ListingFailure.REJECTED:
        return f"LiteLLM rejected your key (HTTP {error.status}). Run `lite login` to refresh it, or pass a valid --api-key."
    if error.kind is ListingFailure.UNREACHABLE:
        return f"{error.message} Is the proxy at {base_url} running, and is --base-url (or LITELLM_PROXY_URL) correct?"
    if error.kind is ListingFailure.EMPTY:
        return f"{error.message} Claude Code would have nothing to run; give the key access to at least one model."
    return f"{error.message} The proxy at {base_url} answered, so check that it is a LiteLLM proxy and is healthy."


def _listed_models(base_url: str, key: str) -> tuple[str, ...]:
    listed: Final = fetch_model_ids(base_url, key)
    if isinstance(listed, PiSyncError):
        raise click.ClickException(_listing_error(base_url, listed))
    return listed


def _model_choice(model: str | None) -> ModelChoice:
    return StartOn(model) if model is not None else UnpinModel()


def _apply_claude(ctx: click.Context, credential: ClaudeCredential, listed: Sequence[str], model: str | None) -> None:
    ctx_obj: Final[CliContextObj] = ctx.obj
    base_url: Final = ctx_obj["base_url"]
    if model is not None and model not in listed:
        shown: Final = ", ".join(listed[:_LISTED_MODELS_SHOWN])
        more: Final = f", and {len(listed) - _LISTED_MODELS_SHOWN} more" if len(listed) > _LISTED_MODELS_SHOWN else ""
        raise click.ClickException(
            f"{model!r} is not served by {base_url} for this key. /v1/models lists: {shown}{more}."
        )
    settings_path: Final = claude_settings_path(os.environ)
    try:
        configure_claude_settings(
            base_url,
            credential,
            _model_choice(model),
            settings_path,
            configure_state_path(settings_path),
            settings_file_owners(settings_path),
        )
    except ClaudeSettingsError as e:
        raise click.ClickException(str(e))
    in_picker: Final = sum(1 for listed_model in listed if _CLAUDE_CODE_PICKER_FILTER.search(listed_model))
    click.echo(f"Configured Claude Code: {settings_path} now routes through {base_url}.")
    click.echo(
        "Credential: your virtual key, stored in the file as ANTHROPIC_AUTH_TOKEN."
        if isinstance(credential, StaticToken)
        else "Credential: your `lite login`, read through apiKeyHelper on every request, so a later login renews it."
    )
    click.echo(
        f"Starting model: {model} ({STARTING_MODEL_ROLE}); switch any time with /model."
        if model is not None
        else "Starting model: not pinned (Claude Code's default, or a model you set yourself); switch with /model, or "
        "pass --model to start on a proxy model."
    )
    click.echo(
        f"/model will list {in_picker} of the proxy's {len(listed)} models (Claude Code shows only ids containing "
        "'claude' or 'anthropic')."
    )
    click.echo("Start `claude` from any terminal. Undo with `lite unconfigure claude`.")
    if isinstance(credential, StaticToken) and settings_path.is_symlink():
        click.echo(
            f"Note: {settings_path} is a symlink to {settings_path.resolve()}, so your key now lives in "
            "that file; keep it out of version control.",
            err=True,
        )


def _pick_targets() -> tuple[str, ...]:
    picked: Final = inquirer.checkbox(
        message="Which agents should route through LiteLLM?",
        choices=[Choice(value, name=label, enabled=True) for value, label in _TARGETS],
        validate=lambda chosen: len(chosen) > 0,
        invalid_message="Pick at least one.",
    ).execute()
    return tuple(str(value) for value in picked)


def _pick_model(listed: Sequence[str]) -> str | None:
    picked: Final = inquirer.fuzzy(
        message="Model Claude Code starts on (type to filter; /model switches any time):",
        choices=[_KEEP_DEFAULT_MODEL, *listed],
    ).execute()
    return None if picked == _KEEP_DEFAULT_MODEL else str(picked)


def interactive_configure(
    ctx: click.Context,
    pick_targets: Callable[[], tuple[str, ...]] = _pick_targets,
    pick_model: Callable[[Sequence[str]], str | None] = _pick_model,
) -> None:
    """`lite configure` with no agent named: ask which agents to wire and which model to pin."""
    targets: Final = pick_targets()
    if _CLAUDE_TARGET not in targets:
        return
    credential, listed = _start(ctx, None)
    _apply_claude(ctx, credential, listed, pick_model(listed))


@click.group(name="configure", invoke_without_command=True)
@click.pass_context
def configure_group(ctx: click.Context) -> None:
    """Persistently route a coding agent through your LiteLLM proxy.

    With no agent named, asks which agents to wire and which proxy model to pin.
    """
    if ctx.invoked_subcommand is not None:
        return
    if not sys.stdin.isatty():
        raise click.ClickException(
            "`lite configure` asks questions, so it needs a terminal. Non-interactively, run "
            "`lite configure claude --api-key <key> --model <model>`."
        )
    interactive_configure(ctx)


@click.group(name="unconfigure")
def unconfigure_group() -> None:
    """Undo `lite configure` for a coding agent."""


@configure_group.command(name="claude")
@click.option(
    "--api-key",
    "api_key",
    default=None,
    help="Long-lived LiteLLM virtual key written into Claude Code's settings. Defaults to the `lite --api-key` / "
    "LITELLM_PROXY_API_KEY value; with neither, your `lite login` credential is used through apiKeyHelper.",
)
@click.option("--model", default=None, help=_MODEL_OPTION_HELP)
@click.pass_context
def configure_claude(ctx: click.Context, api_key: str | None, model: str | None) -> None:
    """Route every Claude Code session through your LiteLLM proxy until `lite unconfigure claude`.

    Patches ~/.claude/settings.json in place: the proxy URL, your credential (a virtual key as a
    static token, or your `lite login` through apiKeyHelper), and gateway model discovery so
    /model lists the proxy's models; --model picks the one Claude Code starts on. Every other
    setting is kept, and what changed is recorded so `lite unconfigure claude` can put it back.
    Assumes the proxy is already running.
    """
    credential, listed = _start(ctx, api_key)
    _apply_claude(ctx, credential, listed, model)


@unconfigure_group.command(name="claude")
def unconfigure_claude() -> None:
    """Return Claude Code's settings to what they were before `lite configure claude`.

    Also undoes `lite login --config-claude`. Only keys still holding what configure wrote are
    put back; anything you changed since is left as it is and named in the output.
    """
    settings_path: Final = claude_settings_path(os.environ)
    state_path: Final = configure_state_path(settings_path)
    try:
        outcome: Final = unconfigure_claude_settings(settings_path, state_path, settings_file_owners(settings_path))
    except ClaudeSettingsError as e:
        raise click.ClickException(str(e))
    _report_unconfigure(settings_path, state_path, outcome)


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


__all__ = ("configure_group", "interactive_configure", "resolve_credential", "unconfigure_group")
