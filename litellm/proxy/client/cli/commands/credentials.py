import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

import click
import requests
import rich
from rich.table import Table
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.models.credentials import UserConnectionPollResponse, UserConnectionPollStatus

from ...credentials import CredentialsManagementClient
from ...exceptions import UnauthorizedError
from ._cli_context import cli_context_values

_SLOW_DOWN_STEP_SECONDS: Final = 5
_CONNECT_FAILURES: Final[MappingProxyType[UserConnectionPollStatus, str]] = MappingProxyType(
    {
        "expired": "The code expired before it was approved. Run the command again for a new code",
        "denied": "The GitHub login was denied",
        "no_copilot_seat": "That GitHub account has no GitHub Copilot seat",
    }
)


@dataclass(frozen=True, slots=True)
class PollClock:
    now: Callable[[], float]
    sleep: Callable[[float], None]


SYSTEM_CLOCK: Final = PollClock(now=time.monotonic, sleep=time.sleep)


class _CredentialInfo(TypedDict):
    custom_llm_provider: ReadOnly[NotRequired[str]]


class _CredentialItem(TypedDict):
    credential_name: ReadOnly[NotRequired[str]]
    display_name: ReadOnly[NotRequired[str | None]]
    source: ReadOnly[NotRequired[str]]
    credential_info: ReadOnly[NotRequired[_CredentialInfo]]


class _CredentialsListView(TypedDict):
    credentials: ReadOnly[Sequence[_CredentialItem]]


class _JsonObjectView(TypedDict):
    value: ReadOnly[dict[str, object]]


class _JsonBodyView(TypedDict):
    body: ReadOnly[object]


def _print_json(data: object) -> None:
    rich.print_json(data=data)


@click.group()
def credentials():
    """Manage credentials for the LiteLLM proxy server"""


@credentials.command()
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format (table or json)",
)
@click.pass_context
def list(ctx: click.Context, output_format: Literal["table", "json"]):
    """List all credentials"""
    context: Final = cli_context_values(ctx)
    client: Final = CredentialsManagementClient(context["base_url"], context["api_key"])
    response: Final = client.list()
    assert isinstance(response, dict)

    if output_format == "json":
        _print_json(response)
    else:  # table format
        table: Final = Table(title="Credentials")

        # Add columns
        table.add_column("Credential Name", style="cyan")
        table.add_column("Display Name")
        table.add_column("Source")
        table.add_column("Custom LLM Provider", style="green")

        # Add rows
        listed: Final[_CredentialsListView] = {"credentials": response.get("credentials", [])}
        for cred in listed["credentials"]:
            info = cred.get("credential_info", {})
            table.add_row(
                str(cred.get("credential_name", "")),
                cred.get("display_name") or "",
                str(cred.get("source", "")),
                str(info.get("custom_llm_provider", "")),
            )

        rich.print(table)


@credentials.command()
@click.argument("credential_name")
@click.option(
    "--info",
    type=str,
    help="JSON string containing credential info",
    required=True,
)
@click.option(
    "--values",
    type=str,
    help="JSON string containing credential values",
    required=True,
)
@click.option("--display-name", type=str, default=None, help="Optional label shown in the UI")
@click.pass_context
def create(ctx: click.Context, credential_name: str, info: str, values: str, display_name: str | None) -> None:
    """Create a new credential"""
    context: Final = cli_context_values(ctx)
    client: Final = CredentialsManagementClient(context["base_url"], context["api_key"])
    try:
        credential_info: Final[_JsonObjectView] = {"value": json.loads(info)}
        credential_values: Final[_JsonObjectView] = {"value": json.loads(values)}
    except json.JSONDecodeError as e:
        raise click.BadParameter(f"Invalid JSON: {e}")

    try:
        response: Final = client.create(
            credential_name, credential_info["value"], credential_values["value"], display_name=display_name
        )
        _print_json(response)
    except requests.exceptions.HTTPError as e:
        click.echo(f"Error: HTTP {e.response.status_code}", err=True)
        try:
            error_body: Final[_JsonBodyView] = {"body": e.response.json()}
            _print_json(error_body["body"])
        except json.JSONDecodeError:
            click.echo(e.response.text, err=True)
        raise click.Abort()


@credentials.command()
@click.argument("credential_name")
@click.option("--display-name", type=str, default=None, help="New label shown in the UI")
@click.option("--clear-display-name", is_flag=True, help="Remove the label so the UI shows the credential name")
@click.pass_context
def update(ctx: click.Context, credential_name: str, display_name: str | None, clear_display_name: bool) -> None:
    """Change a credential's display name. The credential name itself cannot change"""
    if (display_name is None) == (not clear_display_name):
        raise click.UsageError("Pass exactly one of --display-name or --clear-display-name")
    context: Final = cli_context_values(ctx)
    client: Final = CredentialsManagementClient(context["base_url"], context["api_key"])
    try:
        response: Final = client.update_display_name(credential_name, display_name)
        _print_json(response)
    except requests.exceptions.HTTPError as e:
        click.echo(f"Error: HTTP {e.response.status_code}", err=True)
        try:
            error_body: Final[_JsonBodyView] = {"body": e.response.json()}
            _print_json(error_body["body"])
        except json.JSONDecodeError:
            click.echo(e.response.text, err=True)
        raise click.Abort()


@credentials.command()
@click.argument("credential_name")
@click.pass_context
def delete(ctx: click.Context, credential_name: str):
    """Delete a credential by name"""
    context: Final = cli_context_values(ctx)
    client: Final = CredentialsManagementClient(context["base_url"], context["api_key"])
    try:
        response: Final = client.delete(credential_name)
        _print_json(response)
    except requests.exceptions.HTTPError as e:
        click.echo(f"Error: HTTP {e.response.status_code}", err=True)
        try:
            error_body: Final[_JsonBodyView] = {"body": e.response.json()}
            _print_json(error_body["body"])
        except json.JSONDecodeError:
            click.echo(e.response.text, err=True)
        raise click.Abort()


@credentials.command()
@click.argument("credential_name")
@click.pass_context
def get(ctx: click.Context, credential_name: str):
    """Get a credential by name"""
    context: Final = cli_context_values(ctx)
    client: Final = CredentialsManagementClient(context["base_url"], context["api_key"])
    response: Final = client.get(credential_name)
    _print_json(response)


def _next_poll_interval(poll: UserConnectionPollResponse, current: int) -> int:
    if poll.interval is not None:
        return poll.interval
    return current + _SLOW_DOWN_STEP_SECONDS if poll.status == "slow_down" else current


def _poll_until_settled(
    client: CredentialsManagementClient,
    credential_name: str,
    flow_handle: str,
    first_interval: int,
    expires_at: float,
    clock: PollClock,
) -> UserConnectionPollResponse:
    interval = first_interval  # rebind-ok: GitHub can ask for a longer wait on any poll
    while clock.now() + interval < expires_at:
        clock.sleep(interval)
        if (poll := client.poll_user_connection(credential_name, flow_handle)).status not in ("pending", "slow_down"):
            return poll
        interval = _next_poll_interval(poll, interval)
    return UserConnectionPollResponse(status="expired")


def connect_user_credential(
    client: CredentialsManagementClient,
    credential_name: str,
    clock: PollClock,
) -> UserConnectionPollResponse:
    """Run the GitHub device login for the calling key's user and return the final poll result.

    Polling stops once the next poll would land after the code expires, since the proxy rejects polls on an expired
    code. The countdown starts before the code is requested, so it never runs past the proxy's own."""
    requested_at: Final = clock.now()
    started: Final = client.start_user_connection(credential_name)
    click.echo(f"Go to {started.verification_uri} and enter code {started.user_code}")
    click.echo("Waiting for approval...")
    expires_at: Final = requested_at + started.expires_in
    return _poll_until_settled(client, credential_name, started.flow_handle, started.interval, expires_at, clock)


def _report_http_error(response: requests.Response) -> None:
    click.echo(f"Error: HTTP {response.status_code}", err=True)
    try:
        error_body: Final[_JsonBodyView] = {"body": response.json()}
        _print_json(error_body["body"])
    except json.JSONDecodeError:
        click.echo(response.text, err=True)


@credentials.command()
@click.argument("credential_name")
@click.pass_context
def connect(ctx: click.Context, credential_name: str) -> None:
    """Connect your own GitHub Copilot account to a per-user credential, without the UI"""
    context: Final = cli_context_values(ctx)
    client: Final = CredentialsManagementClient(context["base_url"], context["api_key"])
    try:
        result: Final = connect_user_credential(client, credential_name, SYSTEM_CLOCK)
    except UnauthorizedError as e:
        if isinstance(e.orig_exception, requests.exceptions.HTTPError) and e.orig_exception.response is not None:
            _report_http_error(e.orig_exception.response)
        raise click.Abort()
    except requests.exceptions.HTTPError as e:
        _report_http_error(e.response)
        raise click.Abort()
    except requests.exceptions.RequestException as e:
        click.echo(
            f"Error: could not reach the proxy at {context['base_url']} ({type(e).__name__}). Run the command again",
            err=True,
        )
        raise click.Abort()
    if result.status != "connected":
        click.echo(f"Error: {_CONNECT_FAILURES.get(result.status, result.status)}", err=True)
        raise click.exceptions.Exit(1)
    click.echo(f"Connected as {result.github_login} for credential '{credential_name}'")
