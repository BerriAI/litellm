import json
import re
import shlex
import threading
import time
import webbrowser
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, NoReturn, Protocol
from urllib.parse import urlparse

import click
import requests
import rich
from pydantic import BaseModel, ConfigDict
from rich.table import Table
from typing_extensions import NotRequired, ReadOnly, TypedDict

from ...credentials import (
    CredentialsManagementClient,
    UserConnection,
    UserConnectionPoll,
    UserConnectionStart,
)
from ...exceptions import UnauthorizedError
from ._cli_context import cli_context_values

SLOW_DOWN_INCREMENT_SECONDS: Final = 5
GITHUB_DEVICE_LOGIN_URL: Final = "https://github.com/login/device"


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


class _ProxyErrorMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str
    type: str | None = None
    param: str | None = None
    code: str | None = None


class _FastAPIErrorDetail(BaseModel):
    model_config = ConfigDict(frozen=True)

    error: str


class _ProxyErrorBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    detail: str | _FastAPIErrorDetail | None = None
    error: _ProxyErrorMessage | None = None


class UserConnectionClient(Protocol):
    def start_user_connection(self, credential_name: str) -> UserConnectionStart: ...

    def poll_user_connection(self, credential_name: str, flow_handle: str) -> UserConnectionPoll: ...


@dataclass(frozen=True, slots=True)
class UserConnectionConnected:
    status: Literal["connected"]
    github_login: str | None


@dataclass(frozen=True, slots=True)
class UserConnectionFailed:
    status: Literal["failed"]
    reason: Literal["expired", "denied", "no_copilot_seat"]
    message: str


def _safe_verification_uri(uri: str) -> str:
    try:
        parsed: Final = urlparse(uri)
        host: Final = parsed.hostname
        has_credentials: Final = parsed.username is not None or parsed.password is not None
        has_no_port: Final = parsed.port is None
        host_is_allowed: Final = host == "github.com" or (
            host is not None and re.fullmatch(r"[a-z0-9-]+\.ghe\.com", host) is not None
        )
        if parsed.scheme == "https" and not has_credentials and has_no_port and host_is_allowed:
            return uri
    except ValueError:
        pass
    return GITHUB_DEVICE_LOGIN_URL


def _user_connection_failure(
    reason: Literal["expired", "denied", "no_copilot_seat"],
    credential_name: str,
) -> UserConnectionFailed:
    command: Final = f"lite credentials connect {shlex.quote(credential_name)}"
    match reason:
        case "expired":
            return UserConnectionFailed(
                status="failed",
                reason=reason,
                message=f"The device code expired before it was approved. Run `{command}` again to get a new code",
            )
        case "denied":
            return UserConnectionFailed(
                status="failed",
                reason=reason,
                message=f"The request was denied on GitHub. Run `{command}` again if this was a mistake",
            )
        case "no_copilot_seat":
            return UserConnectionFailed(
                status="failed",
                reason=reason,
                message="This GitHub account does not have GitHub Copilot access, so nothing was saved",
            )


class _ConnectionPoller:
    def __init__(
        self,
        client: UserConnectionClient,
        credential_name: str,
        flow_handle: str,
        interval: int,
        deadline: float,
        sleep: Callable[[float], None],
        clock: Callable[[], float],
    ) -> None:
        self._client = client
        self._credential_name = credential_name
        self._flow_handle = flow_handle
        self._interval = interval
        self._deadline = deadline
        self._sleep = sleep
        self._clock = clock

    def poll(self) -> UserConnectionPoll | None:
        if self._clock() >= self._deadline:
            return None
        self._sleep(self._interval)
        if self._clock() >= self._deadline:
            return None
        response: Final = self._client.poll_user_connection(self._credential_name, self._flow_handle)
        if response.status == "slow_down":
            self._interval = (
                response.interval if response.interval is not None else self._interval + SLOW_DOWN_INCREMENT_SECONDS
            )
        return response


def run_user_connection(
    client: UserConnectionClient,
    credential_name: str,
    *,
    open_browser: bool,
    launch_browser: Callable[[str], object],
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    echo: Callable[[str], None],
) -> UserConnectionConnected | UserConnectionFailed:
    start: Final = client.start_user_connection(credential_name)
    verification_uri: Final = _safe_verification_uri(start.verification_uri)
    echo(
        f"To connect GitHub Copilot for credential {credential_name!r}, open {verification_uri} "
        f"and enter code {start.user_code}"
    )
    minutes: Final = max(1, round(start.expires_in / 60))
    minute_unit: Final = "minute" if minutes == 1 else "minutes"
    echo(f"Waiting for approval. The code expires in {minutes} {minute_unit}")
    if open_browser:
        launch_browser(verification_uri)
    poller: Final = _ConnectionPoller(
        client,
        credential_name,
        start.flow_handle,
        start.interval,
        clock() + start.expires_in,
        sleep,
        clock,
    )
    for response in iter(poller.poll, None):
        match response.status:
            case "pending" | "slow_down":
                continue
            case "connected":
                if response.github_login is not None:
                    echo(f"Connected GitHub Copilot for credential {credential_name!r} as {response.github_login}")
                else:
                    echo(f"Connected GitHub Copilot for credential {credential_name!r}")
                return UserConnectionConnected(status="connected", github_login=response.github_login)
            case "expired" | "denied" | "no_copilot_seat":
                return _user_connection_failure(response.status, credential_name)
    return _user_connection_failure("expired", credential_name)


def _launch_browser(uri: str) -> object:
    browser_thread: Final = threading.Thread(target=webbrowser.open, args=(uri,), daemon=True)
    browser_thread.start()
    return browser_thread


def _http_error_detail(error: requests.exceptions.HTTPError) -> str:
    response: Final = error.response
    if response is None:
        return str(error)[:500]
    try:
        body: Final = _ProxyErrorBody.model_validate(response.json())
    except ValueError:
        return response.text[:500]
    if body.error is not None:
        return body.error.message[:500]
    detail: Final = body.detail
    if isinstance(detail, str):
        return detail[:500]
    if detail is not None:
        return detail.error[:500]
    return response.text[:500]


def _raise_http_error(error: requests.exceptions.HTTPError) -> NoReturn:
    response: Final = error.response
    status: Final = response.status_code if response is not None else "unknown"
    raise click.ClickException(f"HTTP {status}: {_http_error_detail(error)}") from error


def _raise_proxy_error(
    error: requests.exceptions.HTTPError | requests.exceptions.ConnectionError | UnauthorizedError,
) -> NoReturn:
    if isinstance(error, UnauthorizedError):
        original: Final = error.orig_exception
        if isinstance(original, requests.exceptions.HTTPError):
            _raise_http_error(original)
        raise click.ClickException(str(error)) from error
    if isinstance(error, requests.exceptions.HTTPError):
        _raise_http_error(error)
    raise click.ClickException(f"Could not connect to the LiteLLM proxy: {error}") from error


def _print_json(data: object) -> None:
    rich.print_json(data=data)


def _render_table(
    title: str,
    columns: tuple[tuple[str, str | None], ...],
    rows: Sequence[tuple[str, ...]],
) -> None:
    table: Final = Table(title=title)
    for column_name, style in columns:
        if style is None:
            table.add_column(column_name)
        else:
            table.add_column(column_name, style=style)
    for row in rows:
        table.add_row(*row)
    rich.print(table)


def _credential_table_row(credential: _CredentialItem) -> tuple[str, str, str, str]:
    info: Final = credential.get("credential_info", {})
    return (
        str(credential.get("credential_name", "")),
        credential.get("display_name") or "",
        str(credential.get("source", "")),
        str(info.get("custom_llm_provider", "")),
    )


def _connection_table_row(connection: UserConnection) -> tuple[str, str, str, str]:
    return (
        connection.credential_name,
        "yes" if connection.connected else "no",
        connection.github_login or "",
        connection.connected_at or "",
    )


@click.group()
def credentials():
    """Manage credentials for the LiteLLM proxy server"""


@credentials.command()
@click.argument("credential_name")
@click.option("--no-browser", is_flag=True, help="Do not open the verification page in a browser")
@click.pass_context
def connect(ctx: click.Context, credential_name: str, no_browser: bool) -> None:
    """Connect a GitHub Copilot account to a per-user credential"""
    context: Final = cli_context_values(ctx)
    client: Final = CredentialsManagementClient(context["base_url"], context["api_key"])
    try:
        result: Final = run_user_connection(
            client,
            credential_name,
            open_browser=not no_browser,
            launch_browser=_launch_browser,
            sleep=time.sleep,
            clock=time.monotonic,
            echo=click.echo,
        )
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, UnauthorizedError) as error:
        _raise_proxy_error(error)
    if isinstance(result, UserConnectionFailed):
        raise click.ClickException(result.message)


@credentials.command()
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format (table or json)",
)
@click.pass_context
def connections(ctx: click.Context, output_format: Literal["table", "json"]) -> None:
    """List GitHub Copilot connections for the authenticated user"""
    context: Final = cli_context_values(ctx)
    client: Final = CredentialsManagementClient(context["base_url"], context["api_key"])
    try:
        response: Final = client.list_user_connections()
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, UnauthorizedError) as error:
        _raise_proxy_error(error)
    if output_format == "json":
        _print_json(response.model_dump(mode="json"))
        return
    rows: Final = tuple(_connection_table_row(connection) for connection in response.connections)
    _render_table(
        "GitHub Copilot connections",
        (("Credential Name", "cyan"), ("Connected", None), ("GitHub Login", None), ("Connected At", None)),
        rows,
    )


@credentials.command()
@click.argument("credential_name")
@click.pass_context
def disconnect(ctx: click.Context, credential_name: str) -> None:
    """Disconnect a GitHub Copilot account from a per-user credential"""
    context: Final = cli_context_values(ctx)
    client: Final = CredentialsManagementClient(context["base_url"], context["api_key"])
    try:
        client.delete_user_connection(credential_name)
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, UnauthorizedError) as error:
        _raise_proxy_error(error)
    click.echo(f"Disconnected GitHub Copilot for credential {credential_name!r}")


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
    else:
        listed: Final[_CredentialsListView] = {"credentials": response.get("credentials", [])}
        rows: Final = tuple(_credential_table_row(credential) for credential in listed["credentials"])
        _render_table(
            "Credentials",
            (
                ("Credential Name", "cyan"),
                ("Display Name", None),
                ("Source", None),
                ("Custom LLM Provider", "green"),
            ),
            rows,
        )


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
