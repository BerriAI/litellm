"""Persistent Claude Code and Codex gateway configuration."""

import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import click
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from pydantic import BaseModel

from litellm.proxy.common_utils.model_listing_utils import (
    CLAUDE_CODE_CLIENT,
    CLAUDE_CODE_PICKER_PATTERN,
    GATEWAY_CLIENT_HEADER,
)

from .agents import codex_config_path
from .auth import CliContextObj
from .claude_settings import (
    STARTING_MODEL_ROLE,
    ClaudeSettingsError,
    ModelChoice,
    StartOn,
    StaticToken,
    UnconfigureOutcome,
    UnpinModel,
    claude_settings_path,
    configure_claude_settings,
    configure_state_path,
    preflight_claude_settings,
    settings_file_owners,
    unconfigure_claude_settings,
)
from .codex_settings import (
    CodexSettingsError,
    configure_codex_settings,
    preflight_codex_settings,
    unconfigure_codex_settings,
)
from .config import normalize_base_url
from .pi import ListedModel, ListingFailure, PiSyncError, fetch_model_listing

_LISTED_MODELS_SHOWN: Final = 20
_CLAUDE_TARGET: Final = "claude"
_CODEX_TARGET: Final = "codex"
_TARGETS: Final = ((_CLAUDE_TARGET, "Claude Code (CLI)"), (_CODEX_TARGET, "Codex (CLI)"))
_KEEP_DEFAULT_MODEL: Final = "Keep Claude Code's own default"
_CLAUDE_CODE_VIEW: Final = MappingProxyType(
    {"anthropic-version": "2023-06-01", GATEWAY_CLIENT_HEADER: CLAUDE_CODE_CLIENT}
)
_MODEL_OPTION_HELP: Final = (
    f"Proxy model to set as {STARTING_MODEL_ROLE}. Must be listed on /v1/models for the key; without it, "
    "Claude Code keeps its own default and a pin an earlier configure made is let go of. Nothing pins Claude "
    "Code's sub-agent or background tiers; `lite autoroute up` is the mode that does."
)


def resolve_credential(ctx: click.Context, api_key: str | None) -> StaticToken:
    """The long-lived key written into settings.json: --api-key, `lite --api-key` or LITELLM_PROXY_API_KEY.

    A `lite login` credential is never written: it expires within a day, and keeping it fresh would mean
    Claude Code running `lite` through `apiKeyHelper` on every credential refresh.
    """
    ctx_obj: Final[CliContextObj] = ctx.obj
    explicit: Final = api_key or (None if ctx_obj.get("api_key_from_token_file") else ctx_obj.get("api_key"))
    if not explicit:
        raise ClaudeSettingsError(
            "`lite configure` needs a long-lived virtual key: pass --api-key, `lite --api-key`, or set "
            "LITELLM_PROXY_API_KEY. Your `lite login` credential expires within a day, so it is not written "
            "into agent settings."
        )
    if not explicit.strip() or any(ord(char) <= 32 or ord(char) == 127 for char in explicit):
        raise ClaudeSettingsError("The virtual key must not be blank or contain whitespace or control characters.")
    return StaticToken(explicit)


@dataclass(frozen=True, slots=True)
class _Listing:
    models: tuple[ListedModel, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(model.id for model in self.models)


def _preflight(target: str) -> None:
    try:
        if target == _CLAUDE_TARGET:
            preflight_claude_settings(claude_settings_path(os.environ))
        else:
            preflight_codex_settings(codex_config_path(os.environ))
    except (ClaudeSettingsError, CodexSettingsError) as e:
        raise click.ClickException(str(e)) from e


def _start(ctx: click.Context, api_key: str | None, target: str = _CLAUDE_TARGET) -> tuple[StaticToken, _Listing]:
    _preflight(target)
    try:
        credential: Final = resolve_credential(ctx, api_key)
    except ClaudeSettingsError as e:
        raise click.ClickException(str(e))
    return credential, _listed_models(ctx.obj["base_url"], credential.token, target)


def _listing_error(base_url: str, error: PiSyncError, target: str) -> str:
    """The hint that fits how the listing failed: only an unreachable proxy gets the "is it running" question."""
    if error.kind is ListingFailure.REJECTED:
        return f"LiteLLM rejected your key (HTTP {error.status}). Pass a valid --api-key."
    if error.kind is ListingFailure.UNREACHABLE:
        return (
            f"Could not connect. Is the proxy at {base_url} running, and is --base-url (or LITELLM_PROXY_URL) correct?"
        )
    if error.kind is ListingFailure.EMPTY:
        name: Final = "Claude Code" if target == _CLAUDE_TARGET else "Codex"
        return f"{error.message} {name} would have nothing to run; give the key access to at least one model."
    return f"The proxy at {base_url} answered, so check that it is a LiteLLM proxy and is healthy."


def _listed_models(base_url: str, key: str, target: str = _CLAUDE_TARGET) -> _Listing:
    listed: Final = fetch_model_listing(
        base_url, key, headers=_CLAUDE_CODE_VIEW if target == _CLAUDE_TARGET else MappingProxyType({})
    )
    if isinstance(listed, PiSyncError):
        raise click.ClickException(_listing_error(base_url, listed, target))
    return _Listing(listed)


def _starting_model(model: str, listing: _Listing) -> str | None:
    source: Final = next((listed.id for listed in listing.models if listed.source_model == model), None)
    return source or next((listed.id for listed in listing.models if listed.id == model), None)


def _model_choice(model: str | None) -> ModelChoice:
    return StartOn(model) if model is not None else UnpinModel()


def _validated_model(model: str | None, listing: _Listing, base_url: str) -> str | None:
    starting: Final = _starting_model(model, listing) if model is not None else None
    if model is not None and starting is None:
        shown: Final = ", ".join(listing.ids[:_LISTED_MODELS_SHOWN])
        raise click.ClickException(f"{model!r} is not served by {base_url} for this key. /v1/models lists: {shown}.")
    return starting


def _apply_claude(ctx: click.Context, credential: StaticToken, listing: _Listing, model: str | None) -> None:
    ctx_obj: Final[CliContextObj] = ctx.obj
    base_url: Final = ctx_obj["base_url"]
    listed: Final = listing.ids
    starting: Final = _validated_model(model, listing, base_url)
    settings_path: Final = claude_settings_path(os.environ)
    try:
        configure_claude_settings(
            base_url,
            credential,
            _model_choice(starting),
            settings_path,
            configure_state_path(settings_path),
            settings_file_owners(settings_path),
        )
    except ClaudeSettingsError as e:
        raise click.ClickException(str(e))
    in_picker: Final = sum(1 for listed_model in listed if CLAUDE_CODE_PICKER_PATTERN.search(listed_model))
    click.echo(f"Configured Claude Code: {settings_path} now routes through {base_url}.")

    click.echo("Credential: your virtual key, stored in the file as ANTHROPIC_AUTH_TOKEN.")
    click.echo(
        f"Starting model: {starting} ({STARTING_MODEL_ROLE}); switch any time with /model."
        if starting is not None
        else "Starting model: not pinned (Claude Code's default, or a model you set yourself); switch with /model, or "
        "pass --model to start on a proxy model. Without a pin, a resumed session re-sends the model its transcript "
        "recorded, which behind a raw-model auto-router is the tier model."
    )
    click.echo(
        f"/model will list all {len(listed)} of the proxy's models."
        if in_picker == len(listed)
        else f"/model will list {in_picker} of the proxy's {len(listed)} models: Claude Code shows only ids containing "
        "'claude' or 'anthropic', and this proxy does not list the rest under such names."
    )
    click.echo("Start `claude` from any terminal. Undo with `lite unconfigure claude`.")
    if settings_path.is_symlink():
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
        default=listed[0] if listed else _KEEP_DEFAULT_MODEL,
    ).execute()
    return None if picked == _KEEP_DEFAULT_MODEL else str(picked)


def _pick_codex_model(listed: Sequence[str]) -> str:
    choices: Final = list(listed)  # mutable-ok: InquirerPy's choices parameter requires a list
    return str(inquirer.fuzzy(message="Model Codex starts on (type to filter):", choices=choices).execute())


def _apply_codex(ctx: click.Context, credential: StaticToken, listing: _Listing, model: str) -> None:
    base_url: Final[str] = ctx.obj["base_url"]
    _validated_model(model, listing, base_url)
    settings_path: Final = codex_config_path(os.environ)
    try:
        configure_codex_settings(base_url, credential.token, model, settings_path)
    except CodexSettingsError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"Configured Codex: {settings_path} now routes through {base_url}.")
    click.echo(f"Starting model: {model}. Credential: your virtual key, stored in the private provider settings.")
    click.echo("Start `codex` from any terminal. Undo with `lite unconfigure codex`.")
    if settings_path.is_symlink():
        click.echo(f"Note: your key now lives in {settings_path.resolve()}; keep it out of version control.", err=True)


@dataclass(frozen=True, slots=True)
class _Setup:
    target: str
    listing: _Listing
    model: str | None


def _choose_setup(
    ctx: click.Context,
    target: str,
    credential: StaticToken,
    pick_model: Callable[[Sequence[str]], str | None],
    pick_codex_model: Callable[[Sequence[str]], str],
) -> _Setup:
    base_url: Final[str] = ctx.obj["base_url"]
    listing: Final = _listed_models(base_url, credential.token, target)
    model: Final = (
        pick_model(tuple(item.source_model or item.id for item in listing.models))
        if target == _CLAUDE_TARGET
        else pick_codex_model(listing.ids)
    )
    _validated_model(model, listing, base_url)
    return _Setup(target, listing, model)


def interactive_configure(
    ctx: click.Context,
    pick_targets: Callable[[], tuple[str, ...]] = _pick_targets,
    pick_model: Callable[[Sequence[str]], str | None] = _pick_model,
    pick_codex_model: Callable[[Sequence[str]], str] = _pick_codex_model,
) -> None:
    """`lite configure` with no agent named: ask which agents to wire and which model to pin."""
    targets: Final = pick_targets()
    if not targets:
        return
    for target in targets:
        _preflight(target)
    try:
        credential: Final = resolve_credential(ctx, None)
    except ClaudeSettingsError as e:
        raise click.ClickException(str(e)) from e
    setups: Final = tuple(_choose_setup(ctx, target, credential, pick_model, pick_codex_model) for target in targets)
    for setup in setups:
        if setup.target == _CLAUDE_TARGET:
            _apply_claude(ctx, credential, setup.listing, setup.model)
        elif setup.model is not None:
            _apply_codex(ctx, credential, setup.listing, setup.model)


class _ConnectionOptions(BaseModel):
    api_key: str | None = None
    gateway_url: str | None = None


def _connection_context(ctx: click.Context, api_key: str | None, gateway_url: str | None) -> click.Context:
    ctx_obj: Final[CliContextObj] = ctx.obj
    group: Final = (
        _ConnectionOptions.model_validate(ctx.parent.params)
        if ctx.parent is not None and ctx.parent.command.name == "configure"
        else _ConnectionOptions()
    )
    key: Final = api_key if api_key is not None else group.api_key
    url: Final = gateway_url if gateway_url is not None else group.gateway_url
    normalized: Final = normalize_base_url(url if url is not None else ctx_obj["base_url"])
    connection: Final[CliContextObj] = {
        **ctx_obj,
        "base_url": normalized.removesuffix("/v1"),
        "base_url_explicit": url is not None or ctx_obj.get("base_url_explicit", False),
        "api_key": key if key is not None else ctx_obj.get("api_key"),
        "api_key_from_token_file": False if key is not None else ctx_obj.get("api_key_from_token_file", False),
    }
    return click.Context(ctx.command, parent=ctx.parent, obj=connection)


@click.group(name="configure", invoke_without_command=True)
@click.option("--api-key", default=None, help="Long-lived LiteLLM virtual key to store in the selected agents.")
@click.option(
    "--gateway-url", "--base-url", default=None, help="Gateway URL; defaults to `lite --base-url` / LITELLM_PROXY_URL."
)
@click.pass_context
def configure_group(ctx: click.Context, api_key: str | None, gateway_url: str | None) -> None:
    """Persistently route a coding agent through your LiteLLM proxy.

    With no agent named, asks which agents to wire and which proxy model to pin.
    """
    if ctx.invoked_subcommand is not None:
        return
    connection: Final = _connection_context(ctx, api_key, gateway_url)
    if not sys.stdin.isatty():
        raise click.ClickException(
            "`lite configure` asks questions, so it needs a terminal. Non-interactively, run "
            "`lite configure claude --api-key <key> --model <model>` or "
            "`lite configure codex --api-key <key> --model <model>`."
        )
    prompted: Final = (
        connection
        if connection.obj.get("base_url_explicit")
        else _connection_context(connection, None, click.prompt("Gateway URL", default=connection.obj["base_url"]))
    )
    interactive_configure(prompted)


@click.group(name="unconfigure")
def unconfigure_group() -> None:
    """Undo `lite configure` for a coding agent."""


@configure_group.command(name="claude")
@click.option(
    "--api-key",
    "api_key",
    default=None,
    help="Long-lived LiteLLM virtual key written into Claude Code's settings. Defaults to the `lite --api-key` / "
    "LITELLM_PROXY_API_KEY value; required, since a `lite login` credential expires within a day.",
)
@click.option("--model", default=None, help=_MODEL_OPTION_HELP)
@click.option("--gateway-url", "--base-url", default=None, help="Gateway URL, including any deployment path prefix.")
@click.pass_context
def configure_claude(ctx: click.Context, api_key: str | None, model: str | None, gateway_url: str | None) -> None:
    """Route every Claude Code session through your LiteLLM proxy until `lite unconfigure claude`.

    Patches ~/.claude/settings.json in place: the proxy URL, your virtual key as a static token,
    and gateway model discovery so /model lists the proxy's models; --model picks the one Claude
    Code starts on and resumes with. Every other
    setting is kept, and what changed is recorded so `lite unconfigure claude` can put it back.
    Assumes the proxy is already running.
    """
    connection: Final = _connection_context(ctx, api_key, gateway_url)
    credential, listing = _start(connection, api_key)
    _apply_claude(connection, credential, listing, model)


@configure_group.command(name="codex")
@click.option("--api-key", default=None, help="Long-lived LiteLLM virtual key to store in Codex's user config.")
@click.option("--gateway-url", "--base-url", default=None, help="Gateway URL, including any deployment path prefix.")
@click.option("--model", required=True, help="Gateway model Codex starts on, as listed by /v1/models for your key.")
@click.pass_context
def configure_codex(ctx: click.Context, api_key: str | None, gateway_url: str | None, model: str) -> None:
    """Route plain `codex` through the gateway until `lite unconfigure codex`."""
    connection: Final = _connection_context(ctx, api_key, gateway_url)
    credential, listing = _start(connection, api_key, _CODEX_TARGET)
    _apply_codex(connection, credential, listing, model)


@unconfigure_group.command(name="codex")
def unconfigure_codex() -> None:
    """Restore only Codex settings still holding what configure wrote."""
    settings_path: Final = codex_config_path(os.environ)
    try:
        outcome: Final = unconfigure_codex_settings(settings_path)
    except CodexSettingsError as e:
        raise click.ClickException(str(e)) from e
    if outcome.file_removed:
        click.echo(f"Removed {settings_path}; it held only settings created by `lite configure codex`.")
    elif outcome.restored:
        click.echo(f"Restored in {settings_path}: {', '.join(outcome.restored)}.")
    else:
        click.echo(f"Nothing in {settings_path} was still ours to restore.")
    if outcome.kept:
        click.echo(f"Left as you changed them since: {', '.join(outcome.kept)}.")


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
