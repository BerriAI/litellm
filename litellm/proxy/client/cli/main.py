import functools
import importlib
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from typing import Final, Literal, Protocol

import click

from litellm._version import version as litellm_version
from litellm.litellm_core_utils.cli_token_utils import load_cli_token

from .commands.auth import (
    cli_context_obj,
    context_secret_vault,
    get_stored_api_key,
    keychain_unreadable_notice,
)
from .commands.config import get_config_value, hidden_command_names


class _CliContextObjSlots(Protocol):
    def __setitem__(
        self,
        key: Literal["base_url", "api_key", "api_key_from_token_file", "base_url_explicit"],
        value: str | bool | None,
        /,
    ) -> None: ...


class _WritableCliContextView(Protocol):
    @property
    def obj(self) -> _CliContextObjSlots: ...


def _writable_cli_context_obj(view: _WritableCliContextView) -> _CliContextObjSlots:
    return view.obj


def print_version(base_url: str, api_key: str | None):
    """Print CLI and server version info."""
    click.echo(f"LiteLLM Proxy CLI Version: {litellm_version}")
    if base_url:
        click.echo(f"LiteLLM Proxy Server URL: {base_url}")
    from litellm.proxy.client.health import HealthManagementClient

    try:
        health_client: Final = HealthManagementClient(base_url=base_url, api_key=api_key)
        server_version: Final = health_client.get_server_version()
        if server_version:
            click.echo(f"LiteLLM Proxy Server Version: {server_version}")
        else:
            click.echo("LiteLLM Proxy Server Version: (unavailable)")
    except Exception as e:
        click.echo(f"Could not retrieve server version: {e}")


class HideConfiguredCommandsGroup(click.Group):
    """Group that omits operator-hidden commands from listings, still running them.

    Deployments hand `lite` to users who should only see a curated subset of
    commands (`lite config set hidden_commands codex,opencode`). Filtering the
    listing rather than dropping the commands keeps anyone's existing scripts
    working.
    """

    def list_commands(self, ctx: click.Context) -> list[str]:
        hidden: Final = hidden_command_names()
        return [name for name in super().list_commands(ctx) if name not in hidden]


class LazyCommands(MutableMapping[str, click.Command]):
    """Subcommands by name, each imported only when Click looks it up, so `lite <cmd>` loads just that command"""

    def __init__(self, loaders: Mapping[str, Callable[[], click.Command]]) -> None:
        self._loaders = dict(loaders)

    def __getitem__(self, name: str) -> click.Command:
        return self._loaders[name]()

    def __setitem__(self, name: str, command: click.Command) -> None:
        self._loaders[name] = lambda: command

    def __delitem__(self, name: str) -> None:
        del self._loaders[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._loaders)

    def __len__(self) -> int:
        return len(self._loaders)


def _command_in(module: str, attribute: str) -> Callable[[], click.Command]:
    @functools.cache
    def load() -> click.Command:
        match getattr(importlib.import_module(module, __package__), attribute):
            case click.Command() as command:
                return command
            case _:
                raise TypeError(f"{module}.{attribute} is not a click command")

    return load


def _agent_command(name: str) -> Callable[[], click.Command]:
    @functools.cache
    def load() -> click.Command:
        from .commands.agents import agent_commands

        return next(command for command in agent_commands() if command.name == name)

    return load


AGENT_COMMAND_NAMES: Final = ("claude", "codex", "opencode", "pi")

LITE_COMMANDS: Final = LazyCommands(
    {
        "login": _command_in(".commands.auth", "login"),
        "logout": _command_in(".commands.auth", "logout"),
        "whoami": _command_in(".commands.auth", "whoami"),
        "auth": _command_in(".commands.auth", "auth_group"),
        "models": _command_in(".commands.models", "models"),
        "credentials": _command_in(".commands.credentials", "credentials"),
        "encryption": _command_in(".commands.encryption", "encryption"),
        "chat": _command_in(".commands.chat", "chat"),
        "http": _command_in(".commands.http", "http"),
        "debug": _command_in(".commands.debug", "debug"),
        "keys": _command_in(".commands.keys", "keys"),
        "teams": _command_in(".commands.teams", "teams"),
        "users": _command_in(".commands.users", "users"),
        **{name: _agent_command(name) for name in AGENT_COMMAND_NAMES},
        "up": _command_in(".commands.up", "up"),
        "down": _command_in(".commands.up", "down"),
        "model-groups": _command_in(".commands.model_groups", "model_groups"),
        "autoroute": _command_in(".commands.autoroute.commands", "autoroute_group"),
        "config": _command_in(".commands.config", "config_commands"),
        "configure": _command_in(".commands.configure", "configure_group"),
        "reconfigure": _command_in(".commands.configure", "reconfigure_group"),
        "unconfigure": _command_in(".commands.configure", "unconfigure_group"),
    }
)


@click.group(cls=HideConfiguredCommandsGroup, invoke_without_command=True, commands=LITE_COMMANDS)
@click.option(
    "--version",
    "-v",
    "show_version",
    is_flag=True,
    help="Show the LiteLLM Proxy CLI and server version and exit.",
)
@click.option(
    "--base-url",
    envvar="LITELLM_PROXY_URL",
    show_envvar=True,
    default=None,
    show_default="base_url from `lite config`, else the `lite login` server for a bare `lite` with no key, else http://localhost:4000",
    help="Base URL of the LiteLLM proxy server",
)
@click.option(
    "--api-key",
    envvar="LITELLM_PROXY_API_KEY",
    show_envvar=True,
    help="API key for authentication",
)
@click.pass_context
def cli(ctx: click.Context, show_version: bool, base_url: str | None, api_key: str | None) -> None:
    """LiteLLM Proxy CLI - Manage your LiteLLM proxy server"""
    ctx.ensure_object(dict)

    stored_base_url: Final = get_config_value("base_url")
    base_url_provided: Final = base_url is not None
    shell: Final = ctx.invoked_subcommand is None and not show_version
    login: Final = (
        load_cli_token(vault=context_secret_vault(ctx))
        if shell and base_url is None and not stored_base_url and api_key is None
        else None
    )
    login_base_url: Final = login.base_url if login is not None and login.key else None
    if login is not None and login.base_url and not login.key:
        click.secho(
            keychain_unreadable_notice(
                context_secret_vault(ctx), login_command=f"lite --base-url {login.base_url} login"
            ),
            fg="yellow",
            err=True,
        )

    # Normalize once here so every downstream command (login, agents, http, ...) can safely
    # do f"{base_url}/some/path" without producing a double slash.
    base_url = (
        (stored_base_url or login_base_url or "http://localhost:4000") if base_url is None else base_url
    ).rstrip("/")

    # If no API key provided via flag or environment variable, try to load from saved token.
    # Pass base_url so we only use the stored key when it was issued for this server.
    setup_command: Final = ctx.invoked_subcommand in ("configure", "reconfigure", "unconfigure")
    api_key_from_token_file: Final = api_key is None and not setup_command
    resolved_api_key: Final = (
        get_stored_api_key(expected_base_url=base_url, vault=context_secret_vault(ctx))
        if api_key_from_token_file
        else api_key
    )

    ctx_obj: Final = _writable_cli_context_obj(ctx)
    ctx_obj["base_url"] = base_url
    ctx_obj["api_key"] = resolved_api_key
    ctx_obj["api_key_from_token_file"] = api_key_from_token_file
    # `--base-url` defaults to localhost:4000 for local dev convenience, but
    # apiKeyHelper is invoked bare (no flags) -- commands that must work
    # unattended (print-token) need to tell "user didn't say" apart from
    # "user said localhost:4000 on purpose" so they can fall back to
    # whatever server the stored token was actually issued for. A base_url
    # saved via `lite config set` counts as the user saying it.
    ctx_obj["base_url_explicit"] = base_url_provided or (bool(stored_base_url) and not setup_command)

    if show_version:
        print_version(base_url, resolved_api_key)
        ctx.exit()

    # If no subcommand was invoked, start interactive mode
    if ctx.invoked_subcommand is None:
        from .interface import interactive_shell

        interactive_shell(ctx, cli)


@cli.command()
@click.pass_context
def version(ctx: click.Context):
    """Show the LiteLLM Proxy CLI and server version."""
    ctx_obj: Final = cli_context_obj(ctx)
    print_version(ctx_obj.get("base_url"), ctx_obj.get("api_key"))


LITELLM_PROXY_DEPRECATION_NOTICE: Final = (
    "The `litellm-proxy` command is deprecated and will be removed in a future release; "
    "run `lite` instead, it takes the same commands and options."
)


def litellm_proxy_cli() -> None:
    click.secho(LITELLM_PROXY_DEPRECATION_NOTICE, err=True, fg="yellow")
    cli()


if __name__ == "__main__":
    cli()
