"""Persistent Claude Code and Codex gateway configuration."""

import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Final

import click
import requests
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm.proxy.common_utils.model_listing_utils import (
    CLAUDE_CODE_CLIENT,
    CLAUDE_CODE_PICKER_PATTERN,
    GATEWAY_CLIENT_HEADER,
)

from .agents import codex_config_path
from .claude_settings import (
    STARTING_MODEL_ROLE,
    ClaudeSettingsError,
    ModelChoice,
    StartOn,
    StaticToken,
    UnpinModel,
    claude_settings_path,
    configure_claude_settings,
    configure_state_path,
    preflight_claude_settings,
    settings_file_owners,
)
from .codex_settings import (
    CodexSettingsError,
    configure_codex_settings,
    preflight_codex_settings,
)
from .config import normalize_base_url
from .configure_profiles import (
    TARGETS,
    SavedSetup,
    Target,
    read_saved_setup,
    save_setup,
    settings_path_for,
    setup_locks,
)
from .pi import ListedModel, ListingFailure, PiSyncError, fetch_model_listing

_LISTED_MODELS_SHOWN: Final = 20
_CLAUDE_TARGET: Final = "claude"
_CODEX_TARGET: Final = "codex"
_TARGETS: Final = ((_CLAUDE_TARGET, "Claude Code (CLI)"), (_CODEX_TARGET, "Codex (CLI)"))
_KEEP_DEFAULT_MODEL: Final = "Keep Claude Code's own default"
_CLAUDE_CODE_VIEW: Final = MappingProxyType(
    {"anthropic-version": "2023-06-01", GATEWAY_CLIENT_HEADER: CLAUDE_CODE_CLIENT}
)
MODEL_OPTION_HELP: Final = (
    f"Proxy model to set as {STARTING_MODEL_ROLE}. Must be listed on /v1/models for the key; omission keeps "
    "the saved choice. Use --default-model to stop pinning a model. Nothing pins Claude "
    "Code's sub-agent or background tiers; `lite autoroute start` is the mode that does."
)
_TARGET_SELECTION: Final = TypeAdapter(tuple[Target, ...])
_MODEL_SELECTION: Final = TypeAdapter(str)


class ConnectionSettings(BaseModel):
    base_url: str
    base_url_explicit: bool = False
    api_key: str | None = None
    api_key_from_token_file: bool = False


def resolve_credential(ctx: click.Context, api_key: str | None) -> StaticToken:
    """The long-lived key written into settings.json: --api-key, `lite --api-key` or LITELLM_PROXY_API_KEY.

    A `lite login` credential is never written: it expires within a day, and keeping it fresh would mean
    Claude Code running `lite` through `apiKeyHelper` on every credential refresh.
    """
    ctx_obj: Final = ConnectionSettings.model_validate(ctx.find_object(object))
    explicit: Final = api_key if api_key is not None else (None if ctx_obj.api_key_from_token_file else ctx_obj.api_key)
    if explicit is None:
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


def _preflight(target: Target) -> None:
    try:
        if target == _CLAUDE_TARGET:
            preflight_claude_settings(claude_settings_path(os.environ))
        else:
            preflight_codex_settings(codex_config_path(os.environ))
    except (ClaudeSettingsError, CodexSettingsError) as e:
        raise click.ClickException(str(e)) from e


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


def _fetch_models(base_url: str, key: str, target: Target) -> tuple[ListedModel, ...] | PiSyncError:
    return fetch_model_listing(
        base_url,
        key,
        get=partial(requests.get, allow_redirects=False),
        headers=_CLAUDE_CODE_VIEW if target == _CLAUDE_TARGET else MappingProxyType({}),
    )


def _connection_listing(
    ctx: click.Context,
    base_url: str,
    credential: StaticToken,
    target: Target,
    repair: bool,
) -> tuple[StaticToken, _Listing]:
    listed: Final = _fetch_models(base_url, credential.token, target)
    if not isinstance(listed, PiSyncError):
        return credential, _Listing(listed)
    if not repair or listed.kind is not ListingFailure.REJECTED:
        raise click.ClickException(_listing_error(base_url, listed, target))
    replacement: Final = click.prompt("Replacement virtual key", hide_input=True, show_default=False)
    try:
        refreshed: Final = resolve_credential(ctx, replacement)
    except ClaudeSettingsError as error:
        raise click.ClickException(str(error)) from error
    retried: Final = _fetch_models(base_url, refreshed.token, target)
    if isinstance(retried, PiSyncError):
        raise click.ClickException(_listing_error(base_url, retried, target))
    return refreshed, _Listing(retried)


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


def _apply_claude(base_url: str, credential: StaticToken, listing: _Listing, model: str | None) -> None:
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


def _has_targets(chosen: Sequence[object]) -> bool:
    return bool(chosen)


def pick_targets(defaults: tuple[Target, ...] = ("claude", "codex"), *, edit: bool = False) -> tuple[Target, ...]:
    choices: Final = [Choice(value, name=label, enabled=value in defaults) for value, label in _TARGETS]
    picked: Final = _TARGET_SELECTION.validate_python(
        inquirer.checkbox(
            message="Which agents should be edited? Unselected agents keep their current setup"
            if edit
            else "Which agents should route through LiteLLM?",
            choices=choices,
            validate=_has_targets,
            invalid_message="Pick at least one.",
        ).execute()
    )
    return tuple(target for target in TARGETS if target in picked)


def _pick_model(listed: Sequence[str], default: str | None = None) -> str | None:
    choices: Final = [_KEEP_DEFAULT_MODEL, *listed]
    picked: Final = _MODEL_SELECTION.validate_python(
        inquirer.fuzzy(
            message="Model Claude Code starts on (type to filter; /model switches any time):",
            choices=choices,
            default=default if default in listed else _KEEP_DEFAULT_MODEL,
        ).execute()
    )
    return None if picked == _KEEP_DEFAULT_MODEL else picked


def _pick_codex_model(listed: Sequence[str], default: str | None = None) -> str:
    choices: Final = list(listed)
    return _MODEL_SELECTION.validate_python(
        inquirer.fuzzy(
            message="Model Codex starts on (type to filter):",
            choices=choices,
            default=default if default in listed else listed[0],
        ).execute()
    )


def _apply_codex(base_url: str, credential: StaticToken, listing: _Listing, model: str) -> None:
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
class PreparedSetup:
    saved: SavedSetup
    listing: _Listing


def _connection(ctx: click.Context, saved: SavedSetup | None) -> tuple[str, StaticToken]:
    settings: Final = ConnectionSettings.model_validate(ctx.find_object(object))
    base_url: Final = settings.base_url if saved is None or settings.base_url_explicit else saved.base_url
    supplied_key: Final = None if settings.api_key_from_token_file else settings.api_key
    reusable_key: Final = saved.api_key if saved is not None and saved.base_url == base_url else None
    try:
        credential: Final = resolve_credential(ctx, supplied_key if supplied_key is not None else reusable_key)
    except ClaudeSettingsError as error:
        if saved is not None and saved.base_url != base_url and supplied_key is None:
            raise click.ClickException(
                "The gateway changed. Pass --api-key for the new gateway; the saved key was not used"
            ) from error
        raise click.ClickException(str(error)) from error
    return base_url, credential


def _prompt_connection(ctx: click.Context, target: Target, saved: SavedSetup | None) -> tuple[str, StaticToken]:
    settings: Final = ConnectionSettings.model_validate(ctx.find_object(object))
    default_url: Final = settings.base_url if saved is None or settings.base_url_explicit else saved.base_url
    base_url: Final = normalize_base_url(
        click.prompt(f"{target.capitalize()} gateway URL", default=default_url)
    ).removesuffix("/v1")
    supplied_key: Final = None if settings.api_key_from_token_file else settings.api_key
    kept_key: Final = (
        supplied_key
        if supplied_key is not None
        else (saved.api_key if saved is not None and saved.base_url == base_url else None)
    )
    entered: Final = click.prompt(
        "Virtual key (press Enter to keep the current key)" if kept_key is not None else "Virtual key",
        default="" if kept_key is not None else None,
        show_default=False,
        hide_input=True,
    )
    key: Final[str | None] = entered or kept_key
    try:
        return base_url, resolve_credential(ctx, key)
    except ClaudeSettingsError as error:
        raise click.ClickException(str(error)) from error


def _prepare(
    ctx: click.Context,
    target: Target,
    saved: SavedSetup | None,
    model: str | None,
    default_model: bool,
    *,
    interactive: bool = False,
    edit_connection: bool = False,
    pick_model: Callable[[Sequence[str]], str | None] | None = None,
    pick_codex_model: Callable[[Sequence[str]], str] | None = None,
) -> PreparedSetup:
    default: Final = None if default_model else (model if model is not None else (saved.model if saved else None))
    if target == "codex" and default is None and not interactive:
        raise click.UsageError("Missing option '--model'. First-time Codex setup needs a starting model")
    base_url, credential = _prompt_connection(ctx, target, saved) if edit_connection else _connection(ctx, saved)
    repair: Final = saved is not None and not interactive and sys.stdin.isatty()
    active_credential, listing = _connection_listing(ctx, base_url, credential, target, repair)
    repair_model: Final = repair and default is not None and _starting_model(default, listing) is None
    source_names: Final = tuple(item.source_model or item.id for item in listing.models)
    chosen: Final = (
        (pick_model(source_names) if pick_model is not None else _pick_model(source_names, default))
        if (interactive or repair_model) and target == "claude"
        else (
            pick_codex_model(listing.ids) if pick_codex_model is not None else _pick_codex_model(listing.ids, default)
        )
        if interactive or repair_model
        else default
    )
    if target == "codex" and chosen is None:
        raise click.ClickException("First-time Codex setup needs --model. Run `lite configure` for the model picker")
    validated: Final = _validated_model(chosen, listing, base_url)
    saved_model: Final = (
        next(item.source_model or item.id for item in listing.models if item.id == validated)
        if target == "claude" and validated is not None
        else chosen
    )
    try:
        profile: Final = SavedSetup(
            target=target,
            settings_path=str(settings_path_for(target).resolve()),
            base_url=base_url,
            api_key=active_credential.token,
            model=saved_model,
        )
    except ValidationError as error:
        raise click.ClickException("Invalid gateway setup; no settings were changed") from error
    return PreparedSetup(profile, listing)


def _apply(setup: PreparedSetup) -> None:
    saved: Final = setup.saved
    save_setup(saved)
    try:
        if saved.target == "claude":
            _apply_claude(saved.base_url, StaticToken(saved.api_key), setup.listing, saved.model)
        elif saved.model is not None:
            _apply_codex(saved.base_url, StaticToken(saved.api_key), setup.listing, saved.model)
    except click.ClickException as error:
        raise click.ClickException(
            f"{error.format_message()} {saved.target.capitalize()} setup was saved. "
            f"Run `lite configure {saved.target}` to retry applying it"
        ) from error
    click.echo(
        f"Setup saved. Edit with `lite reconfigure {saved.target}`; "
        f"remove saved settings and key with `lite unconfigure {saved.target} --forget`."
    )


def configure_targets(
    ctx: click.Context,
    targets: tuple[Target, ...],
    *,
    model: str | None = None,
    default_model: bool = False,
    interactive: bool = False,
    edit_connection: bool = False,
    pick_model: Callable[[Sequence[str]], str | None] | None = None,
    pick_codex_model: Callable[[Sequence[str]], str] | None = None,
) -> None:
    if model is not None and default_model:
        raise click.UsageError("--model and --default-model cannot be used together")
    for target in targets:
        _preflight(target)
    setups: Final = tuple(
        _prepare(
            ctx,
            target,
            read_saved_setup(target),
            model,
            default_model,
            interactive=interactive,
            edit_connection=edit_connection,
            pick_model=pick_model,
            pick_codex_model=pick_codex_model,
        )
        for target in targets
    )
    for setup in setups:
        _apply(setup)


def interactive_configure(
    ctx: click.Context,
    pick_targets: Callable[[], tuple[str, ...]] = pick_targets,
    pick_model: Callable[[Sequence[str]], str | None] | None = None,
    pick_codex_model: Callable[[Sequence[str]], str] | None = None,
) -> None:
    """Configure selected agents, retaining injectable pickers for embedders."""
    selected: Final = pick_targets()
    targets: Final[tuple[Target, ...]] = tuple(target for target in TARGETS if target in selected)
    if not targets:
        return
    with setup_locks(targets):
        configure_targets(ctx, targets, interactive=True, pick_model=pick_model, pick_codex_model=pick_codex_model)
