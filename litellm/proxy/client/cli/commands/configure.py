"""`lite configure [claude|codex]` and `lite unconfigure [claude|codex]`: persistent agent wiring, undoable."""

import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import click
from InquirerPy import inquirer
from InquirerPy.base.control import Choice

from litellm.proxy.common_utils.model_listing_utils import (
    CLAUDE_CODE_CLIENT,
    CLAUDE_CODE_PICKER_PATTERN,
    GATEWAY_CLIENT_HEADER,
)

from .agent_config import AgentConfigError, fingerprint, read_configure_receipt
from .agents import is_interactive
from .auth import CliContextObj, context_secret_vault, get_stored_api_key
from .claude_settings import (
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
    print_token_command,
    refuse_while_owned,
    resolve_api_key_helper,
    settings_file_owners,
    unconfigure_claude_settings,
)
from .codex_settings import (
    codex_config_path,
    codex_configure_state_path,
    configure_codex_config,
    unconfigure_codex_config,
)
from .configured_launch import launch_configured_agents
from .pi import ListedModel, ListingFailure, PiSyncError, fetch_model_ids, fetch_model_limits, fetch_model_listing
from .up import UpError, ensure_fresh_login

_LISTED_MODELS_SHOWN: Final = 20
_CLAUDE_TARGET: Final = "claude"
_CODEX_TARGET: Final = "codex"
_TARGETS: Final = ((_CLAUDE_TARGET, "Claude Code (CLI)"), (_CODEX_TARGET, "Codex (CLI and app)"))
_TARGET_LABELS: Final[Mapping[str, str]] = MappingProxyType(dict(_TARGETS))
_KEEP_DEFAULT_MODEL: Final = "Keep the agent's own default"
_CLAUDE_CODE_VIEW: Final = MappingProxyType(
    {"anthropic-version": "2023-06-01", GATEWAY_CLIENT_HEADER: CLAUDE_CODE_CLIENT}
)
_MODEL_OPTION_HELP: Final = (
    "Proxy model to set as Claude Code's starting model. Must be listed on /v1/models for the key; without it, "
    "Claude Code keeps its own default and a pin an earlier configure made is let go of. Nothing pins Claude "
    "Code's sub-agent or background tiers; `lite autoroute up` is the mode that does."
)
_API_KEY_HELP: Final = (
    "Long-lived LiteLLM virtual key written into the agent's config. Defaults to the `lite --api-key` / "
    "LITELLM_PROXY_API_KEY value. Codex can instead use your `lite login` credential through `lite auth print-token`."
)
_LAUNCH_HELP: Final = "Start the agent right after configuring it."


@dataclass(frozen=True, slots=True)
class _ClaudeListing:
    models: tuple[ListedModel, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(model.id for model in self.models)


@dataclass(frozen=True, slots=True)
class _Session:
    base_url: str
    credential: ClaudeCredential
    key: str
    started_interactive: bool

    def launch(self, agents: Sequence[str]) -> None:
        launch_configured_agents(agents, started_interactive=self.started_interactive)


def _explicit_key(ctx: click.Context, api_key: str | None) -> str | None:
    ctx_obj: Final[CliContextObj] = ctx.obj
    return api_key or (None if ctx_obj.get("api_key_from_token_file") else ctx_obj.get("api_key"))


def _claude_credential(ctx: click.Context, api_key: str | None) -> StaticToken:
    settings_path: Final = claude_settings_path(os.environ)
    refuse_while_owned(settings_path, settings_file_owners(settings_path))
    explicit: Final = _explicit_key(ctx, api_key)
    if explicit is None:
        raise ClaudeSettingsError(
            "`lite configure claude` needs a long-lived virtual key: pass --api-key, `lite --api-key`, or set "
            "LITELLM_PROXY_API_KEY. Your `lite login` credential expires within a day, so it is not written "
            "into Claude Code's settings."
        )
    return StaticToken(explicit)


def _codex_credential(ctx: click.Context, api_key: str | None) -> tuple[ClaudeCredential, str]:
    explicit: Final = _explicit_key(ctx, api_key)
    if explicit is not None:
        return StaticToken(explicit), explicit
    try:
        ensure_fresh_login(ctx, reader="Codex's provider auth command reads this token on every request")
    except UpError as e:
        raise ClaudeSettingsError(str(e)) from e
    base_url: Final = ctx.obj["base_url"]
    stored: Final = get_stored_api_key(expected_base_url=base_url, vault=context_secret_vault(ctx))
    if stored is None:
        raise ClaudeSettingsError("Login did not produce a usable token.")
    return ApiKeyHelper(resolve_api_key_helper(base_url)), stored


def _listing_error(base_url: str, error: PiSyncError, agent: str) -> str:
    if error.kind is ListingFailure.REJECTED:
        return f"LiteLLM rejected your key (HTTP {error.status}). Pass a valid --api-key."
    if error.kind is ListingFailure.UNREACHABLE:
        return f"{error.message} Is the proxy at {base_url} running, and is --base-url (or LITELLM_PROXY_URL) correct?"
    if error.kind is ListingFailure.EMPTY:
        return f"{error.message} {agent} would have nothing to run; give the key access to at least one model."
    return f"{error.message} The proxy at {base_url} answered, so check that it is a LiteLLM proxy and is healthy."


def _claude_listing(base_url: str, key: str) -> _ClaudeListing:
    listed: Final = fetch_model_listing(base_url, key, headers=_CLAUDE_CODE_VIEW)
    if isinstance(listed, PiSyncError):
        raise click.ClickException(_listing_error(base_url, listed, "Claude Code"))
    return _ClaudeListing(listed)


def _codex_listing(base_url: str, key: str) -> tuple[str, ...]:
    listed: Final = fetch_model_ids(base_url, key)
    if isinstance(listed, PiSyncError):
        raise click.ClickException(_listing_error(base_url, listed, "Codex"))
    return listed


def _session(ctx: click.Context, api_key: str | None, agent: str) -> _Session:
    started_interactive: Final = is_interactive()
    try:
        if agent == _CLAUDE_TARGET:
            credential: Final = _claude_credential(ctx, api_key)
            return _Session(ctx.obj["base_url"], credential, credential.token, started_interactive)
        codex_credential, key = _codex_credential(ctx, api_key)
        return _Session(ctx.obj["base_url"], codex_credential, key, started_interactive)
    except (AgentConfigError, ClaudeSettingsError) as e:
        raise click.ClickException(str(e)) from e


def _model_choice(model: str | None) -> ModelChoice:
    return StartOn(model) if model is not None else UnpinModel()


def _require_listed(base_url: str, listed: Sequence[str], model: str | None) -> ModelChoice:
    if model is None:
        return UnpinModel()
    if model in listed:
        return StartOn(model)
    shown: Final = ", ".join(listed[:_LISTED_MODELS_SHOWN])
    more: Final = f", and {len(listed) - _LISTED_MODELS_SHOWN} more" if len(listed) > _LISTED_MODELS_SHOWN else ""
    raise click.ClickException(f"{model!r} is not served by {base_url} for this key. /v1/models lists: {shown}{more}.")


def _apply_claude(session: _Session, listing: _ClaudeListing, model: str | None) -> None:
    source: Final = (
        next((item.id for item in listing.models if item.source_model == model), None) if model is not None else None
    )
    choice: Final = _model_choice(source or model)
    if model is not None and source is None and model not in listing.ids:
        _require_listed(session.base_url, listing.ids, model)
    settings_path: Final = claude_settings_path(os.environ)
    try:
        configure_claude_settings(
            session.base_url,
            session.credential,
            choice,
            settings_path,
            configure_state_path(settings_path),
            settings_file_owners(settings_path),
        )
    except ClaudeSettingsError as e:
        raise click.ClickException(str(e))
    in_picker: Final = sum(1 for item in listing.ids if CLAUDE_CODE_PICKER_PATTERN.search(item))
    starting: Final = source or model
    click.echo(f"Configured Claude Code: {settings_path} now routes through {session.base_url}.")
    click.echo("Credential: your virtual key, stored in the file as ANTHROPIC_AUTH_TOKEN.")
    click.echo(
        f"Starting model: {starting} (the /model picker's default row, the model Claude Code starts and resumes on); switch any time with /model."
        if starting is not None
        else "Starting model: not pinned (Claude Code's default, or a model you set yourself); switch with /model, or pass --model to start on a proxy model."
    )
    click.echo(
        f"/model will list all {len(listing.ids)} of the proxy's models."
        if in_picker == len(listing.ids)
        else f"/model will list {in_picker} of the proxy's {len(listing.ids)} models: Claude Code shows only ids containing 'claude' or 'anthropic', and this proxy does not list the rest under such names."
    )
    click.echo("Start `claude` from any terminal. Undo with `lite unconfigure claude`.")
    if isinstance(session.credential, StaticToken) and settings_path.is_symlink():
        click.echo(
            f"Note: {settings_path} is a symlink to {settings_path.resolve()}, so your key now lives in "
            "that file; keep it out of version control.",
            err=True,
        )


def _pin_released(state_path: Path, choice: ModelChoice) -> bool:
    if not isinstance(choice, UnpinModel):
        return False
    earlier: Final = read_configure_receipt(state_path)
    root: Final = earlier.sections.get("") if earlier is not None else None
    return root is not None and "model" in root.written and root.written["model"] != fingerprint(root.previous["model"])


def _apply_codex(session: _Session, listed: tuple[str, ...], model: str | None) -> None:
    choice: Final = _require_listed(session.base_url, listed, model)
    config_path: Final = codex_config_path(os.environ)
    state_path: Final = codex_configure_state_path(config_path)
    try:
        released: Final = _pin_released(state_path, choice)
        limits: Final = fetch_model_limits(session.base_url, session.key) if model is not None else MappingProxyType({})
        context_window: Final = limits[model].context_window if model is not None and model in limits else None
        configure_codex_config(
            session.base_url,
            session.credential,
            lambda: print_token_command(session.base_url),
            choice,
            context_window,
            config_path,
            state_path,
        )
    except (AgentConfigError, ClaudeSettingsError) as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"Configured Codex: {config_path} now routes through {session.base_url}.")
    click.echo(
        "Credential: your virtual key, stored in the file as a bearer token."
        if isinstance(session.credential, StaticToken)
        else "Credential: your `lite login`, which Codex reads through `lite auth print-token` on demand."
    )
    window: Final = f", context window {context_window} tokens from the proxy" if context_window is not None else ""
    click.echo(
        f"Model: {model}{window}."
        if model is not None
        else "Model: the pin an earlier configure made is released, back to Codex's own default id."
        if released
        else "Model: not pinned (Codex's own default id)."
    )
    click.echo("Start `codex` from any terminal or the Codex app. Undo with `lite unconfigure codex`.")


def _target_choices() -> tuple[Choice, ...]:
    return tuple(Choice(value, name=label, enabled=value == _CLAUDE_TARGET) for value, label in _TARGETS)


def _pick_targets() -> tuple[str, ...]:
    picked: Final = inquirer.checkbox(
        message="Which agents should route through LiteLLM?",
        choices=_target_choices(),
        validate=lambda chosen: len(chosen) > 0,
        invalid_message="Pick at least one.",
    ).execute()
    return tuple(str(value) for value in picked)


def _pick_model(agent: str, listed: Sequence[str]) -> str | None:
    picked: Final = inquirer.fuzzy(
        message=f"Model {_TARGET_LABELS[agent]} starts on (type to filter):",
        choices=[_KEEP_DEFAULT_MODEL, *listed],  # mutable-ok: InquirerPy takes a list
    ).execute()
    return None if picked == _KEEP_DEFAULT_MODEL else str(picked)


def _confirm_launch(agent: str) -> bool:
    answer: Final[object] = inquirer.confirm(message=f"Start {_TARGET_LABELS[agent]} now?", default=True).execute()
    return answer is True


def interactive_configure(
    ctx: click.Context,
    pick_targets: Callable[[], tuple[str, ...]] = _pick_targets,
    pick_model: Callable[[str, Sequence[str]], str | None] = _pick_model,
    confirm_launch: Callable[[str], bool] = _confirm_launch,
) -> None:
    targets: Final = tuple(target for target in pick_targets() if target in (_CLAUDE_TARGET, _CODEX_TARGET))
    if not targets:
        return
    session: Final = _session(ctx, None, _CLAUDE_TARGET if _CLAUDE_TARGET in targets else _CODEX_TARGET)
    if _CLAUDE_TARGET in targets:
        claude_listing: Final = _claude_listing(session.base_url, session.key)
        _apply_claude(
            session,
            claude_listing,
            pick_model(_CLAUDE_TARGET, tuple(model.source_model or model.id for model in claude_listing.models)),
        )
    if _CODEX_TARGET in targets:
        codex_models: Final = _codex_listing(session.base_url, session.key)
        _apply_codex(session, codex_models, pick_model(_CODEX_TARGET, codex_models))
    selected: Final = tuple(target for target in targets if confirm_launch(target))
    if selected:
        session.launch(selected)


def _configure_context(ctx: click.Context, _param: click.Parameter, api_key: str | None) -> None:
    ctx.ensure_object(dict)
    if api_key is not None:
        ctx.obj["api_key"] = api_key  # rebind-ok: Click option callback populates its command context
        ctx.obj["api_key_from_token_file"] = False  # rebind-ok: Click option callback owns this credential marker


@click.group(name="configure", invoke_without_command=True)
@click.option("--api-key", "api_key", default=None, help=_API_KEY_HELP, expose_value=False, callback=_configure_context)
@click.pass_context
def configure_group(ctx: click.Context) -> None:
    """Persistently route a coding agent through your LiteLLM proxy."""
    if ctx.invoked_subcommand is not None:
        return
    if not sys.stdin.isatty():
        raise click.ClickException(
            "`lite configure` asks questions, so it needs a terminal. Non-interactively, run "
            "`lite configure --api-key <key> claude --model <model>` or `lite configure --api-key <key> codex ...`."
        )
    interactive_configure(ctx)


@click.group(name="unconfigure")
def unconfigure_group() -> None:
    """Undo `lite configure` for a coding agent."""


@configure_group.command(name="claude")
@click.option("--api-key", "api_key", default=None, help=_API_KEY_HELP)
@click.option("--model", default=None, help=_MODEL_OPTION_HELP)
@click.option("--launch", is_flag=True, default=False, help=_LAUNCH_HELP)
@click.pass_context
def configure_claude(ctx: click.Context, api_key: str | None, model: str | None, launch: bool) -> None:
    """Route every Claude Code session through your LiteLLM proxy until `lite unconfigure claude`."""
    session: Final = _session(ctx, api_key, _CLAUDE_TARGET)
    _apply_claude(session, _claude_listing(session.base_url, session.key), model)
    if launch:
        session.launch((_CLAUDE_TARGET,))


@configure_group.command(name="codex")
@click.option("--api-key", "api_key", default=None, help=_API_KEY_HELP)
@click.option("--model", default=None, help="Proxy model Codex uses. Must be listed on /v1/models for the key.")
@click.option("--launch", is_flag=True, default=False, help=_LAUNCH_HELP)
@click.pass_context
def configure_codex(ctx: click.Context, api_key: str | None, model: str | None, launch: bool) -> None:
    """Route Codex through your LiteLLM proxy until `lite unconfigure codex`."""
    session: Final = _session(ctx, api_key, _CODEX_TARGET)
    _apply_codex(session, _codex_listing(session.base_url, session.key), model)
    if launch:
        session.launch((_CODEX_TARGET,))


@unconfigure_group.command(name="claude")
def unconfigure_claude() -> None:
    """Return Claude Code's settings to what they were before `lite configure claude`."""
    settings_path: Final = claude_settings_path(os.environ)
    state_path: Final = configure_state_path(settings_path)
    try:
        outcome: Final = unconfigure_claude_settings(settings_path, state_path, settings_file_owners(settings_path))
    except ClaudeSettingsError as e:
        raise click.ClickException(str(e))
    _report_unconfigure(settings_path, outcome, state_path)


@unconfigure_group.command(name="codex")
def unconfigure_codex() -> None:
    """Return Codex's config to what it was before `lite configure codex`."""
    config_path: Final = codex_config_path(os.environ)
    try:
        outcome: Final = unconfigure_codex_config(config_path, codex_configure_state_path(config_path))
    except (AgentConfigError, ClaudeSettingsError) as e:
        raise click.ClickException(str(e)) from e
    _report_unconfigure(config_path, outcome, codex_configure_state_path(config_path))


def _report_unconfigure(path: Path, outcome: UnconfigureOutcome, state_path: Path | None = None) -> None:
    if outcome.file_removed:
        click.echo(f"No settings file remains at {path}; it held nothing but `lite configure`'s own keys.")
    elif outcome.restored:
        click.echo(f"Restored in {path}: {', '.join(outcome.restored)}.")
    else:
        click.echo(f"Nothing in {path} was still ours to restore.")
    if outcome.kept:
        click.echo(f"Left as you changed them since: {', '.join(outcome.kept)}.")
    if outcome.withheld:
        click.echo(
            "Left removed, since the file now points at a different server than they were issued for: "
            + "; ".join(item.key + f" (captured with {item.endpoint})" for item in outcome.withheld)
            + f". They stay in {state_path}: point the endpoint back and run `lite unconfigure` again."
        )


__all__ = ("configure_group", "interactive_configure", "unconfigure_group")
