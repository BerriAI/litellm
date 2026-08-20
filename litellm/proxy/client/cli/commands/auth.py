import sys
import time
import webbrowser
from collections.abc import Callable, Mapping
from typing import Any, Final
from urllib.parse import urlencode

import click
import requests
from rich.console import Console
from rich.table import Table
from typing_extensions import NotRequired, ReadOnly, TypedDict, assert_never

from litellm.constants import CLI_JWT_EXPIRATION_HOURS
from litellm.litellm_core_utils.cli_keyring import (
    DISABLE_KEYRING_ENV_VAR,
    SYSTEM_KEYRING,
    KeyringDisabled,
    KeyringDiscardsWrites,
    KeyringNotInstalled,
    KeyringUnreachable,
    SecretErased,
    SecretFound,
    SecretMissing,
    SecretStored,
    SecretStranded,
    SecretVault,
)
from litellm.litellm_core_utils.cli_token_utils import (
    CliTokenRecord,
    CredentialNotCleared,
    CredentialNotRecorded,
    CredentialNotSaved,
    SecretSave,
    clear_cli_token,
    get_cli_token_file_path,
    is_cli_token_fresh,
    load_cli_token,
    save_cli_token,
)

from .claude_settings import (
    CLAUDE_SETTINGS_PATH,
    SETTINGS_FILE_OWNERS,
    ClaudeSettingsError,
    write_claude_settings,
)
from .pkce_login import (
    Http,
    PkceFailure,
    RevocationUnavailable,
    fresh_api_key,
    pkce_token_record,
    revoke_stored_credential,
    run_pkce_login,
)


class CliTokenData(TypedDict):
    base_url: str
    key: str
    user_id: str
    user_email: str
    user_role: str
    auth_header_name: str
    jwt_token: str
    timestamp: float
    expires_at: ReadOnly[NotRequired[float]]
    refresh_token: ReadOnly[NotRequired[str]]
    client_id: ReadOnly[NotRequired[str]]
    token_endpoint: ReadOnly[NotRequired[str]]
    revocation_endpoint: ReadOnly[NotRequired[str]]
    resource: ReadOnly[NotRequired[str]]
    team_id: ReadOnly[NotRequired[str | None]]


class CliTeam(TypedDict, total=False):
    team_id: str | None
    team_alias: str | None
    models: list[str]
    max_budget: float | None


class CliContextObj(TypedDict):
    base_url: str
    base_url_explicit: NotRequired[bool]
    secret_vault: NotRequired[ReadOnly[SecretVault]]
    api_key: ReadOnly[NotRequired[str | None]]
    api_key_from_token_file: ReadOnly[NotRequired[bool]]


class CliPollData(TypedDict, total=False):
    status: str
    key: str
    user_id: str
    teams: list[str]
    team_details: object
    requires_team_selection: bool
    team_id: str


class CliPollRequestKwargs(TypedDict, total=False):
    timeout: int
    headers: dict[str, str]


class CliSsoStartData(TypedDict):
    login_id: str
    poll_secret: str
    user_code: str


class CliAuthResult(TypedDict):
    api_key: str
    user_id: str | None
    teams: list[str]
    team_id: str | None


KEYRING_INSTALL_HINT: Final = "pip install 'litellm[cli]'"

KEYRING_ENABLE_HINT: Final = "keyring --enable (or unset PYTHON_KEYRING_BACKEND)"

STRANDED_CREDENTIAL_MESSAGE: Final = (
    "Logged out locally, but your credential is still in the OS keychain and could not be removed."
)

UNCHECKED_KEYCHAIN_MESSAGE: Final = (
    "Logged out locally, but your OS keychain could not be checked, so a credential stored there by "
    "an earlier login may still be usable."
)


def storage_notice(outcome: SecretSave) -> str:
    """Tell the user where the credential ended up, and how to get keychain storage if it did not."""
    path: Final = get_cli_token_file_path()
    match outcome:
        case SecretStored():
            return "Credential stored in your OS keychain."
        case KeyringNotInstalled():
            return (
                f"Credential stored in {path} (owner-only). "
                f"For OS keychain storage, install the keyring package with: {KEYRING_INSTALL_HINT}"
            )
        case KeyringDisabled():
            return f"Keychain storage is off ({DISABLE_KEYRING_ENV_VAR}). Credential stored in {path} (owner-only)."
        case KeyringUnreachable():
            return f"No OS keychain available. Credential stored in {path} (owner-only)."
        case KeyringDiscardsWrites():
            return (
                f"Your keyring backend keeps nothing it is given, so the credential was stored in {path} "
                f"(owner-only) instead. For OS keychain storage, run: {KEYRING_ENABLE_HINT}"
            )
        case CredentialNotSaved(detail=detail):
            return (
                f"Signed in, but the credential could not be saved to {path}: {detail}. "
                "Any login you already had is untouched. Run 'lite login' again once that path is "
                "writable, or 'lite logout' to clear whatever is stored now."
            )
        case CredentialNotRecorded():
            return (
                f"Signed in, and the credential is in your OS keychain, but {path} could not be "
                "replaced, so it still describes your previous login and may still hold its "
                "credential. Run 'lite login' again once that path is writable, or 'lite logout' "
                "to clear both."
            )


def keychain_unreadable_notice(vault: SecretVault) -> str:
    """Explain why the secret half of a stored login cannot be produced, and what fixes it"""
    match vault.read():
        case KeyringNotInstalled():
            return (
                "Your credential is in your OS keychain, which this install cannot read without the "
                f"keyring package. Install it with: {KEYRING_INSTALL_HINT}, or run 'lite login' to start over."
            )
        case KeyringDisabled():
            return (
                f"Your credential is in your OS keychain, which {DISABLE_KEYRING_ENV_VAR} is blocking. "
                "Unset it, or run 'lite login' to start over."
            )
        case KeyringUnreachable():
            return (
                "Your credential is in your OS keychain, which could not be read. Unlock it, or run "
                "'lite login' to start over."
            )
        case SecretFound() | SecretMissing():
            return "Your credential could not be read from your OS keychain. Run 'lite login' to start over."


def context_secret_vault(ctx: click.Context) -> SecretVault:
    """Where this invocation reads and writes secret material; injectable through ctx.obj for tests"""
    ctx_obj: Final[CliContextObj | None] = ctx.obj
    if ctx_obj is None:
        return SYSTEM_KEYRING
    return ctx_obj.get("secret_vault") or SYSTEM_KEYRING


def load_token(*, vault: SecretVault = SYSTEM_KEYRING) -> Mapping[str, object] | None:
    """The stored credential as a plain mapping, with the secret resolved out of the vault.

    The PKCE renewal and revocation helpers read records by field name, so this is the
    shape they get; the keychain split lives underneath, in `load_cli_token`.
    """
    record: Final = load_cli_token(vault=vault)
    return None if record is None else record.model_dump(exclude_none=True)


def save_token(record: CliTokenData, *, vault: SecretVault = SYSTEM_KEYRING) -> SecretSave:
    """Store a credential the PKCE layer produced, secret in the vault and the rest on disk"""
    return save_cli_token(CliTokenRecord(**record), vault=vault)


def _renewal_saver(vault: SecretVault) -> Callable[[CliTokenData], None]:
    """Persist a silently renewed credential, and say on stderr when no store would keep it.

    A renewal rotates the refresh token, so a rotation that is never stored logs this
    machine out on the next command; the user hears about it rather than guessing.
    """

    def save(record: CliTokenData) -> None:
        outcome: Final = save_token(record, vault=vault)
        if isinstance(outcome, (CredentialNotSaved, CredentialNotRecorded)):
            _warn(storage_notice(outcome))

    return save


def _renewal_reader(vault: SecretVault) -> Callable[[], Mapping[str, object] | None]:
    """Re-read the record mid-renewal, so a rotation a sibling `lite` process saved is seen"""

    def reload() -> Mapping[str, object] | None:
        return load_token(vault=vault)

    return reload


def get_stored_api_key(
    expected_base_url: str | None = None,
    *,
    vault: SecretVault = SYSTEM_KEYRING,
) -> str | None:
    """Get the stored API key.

    If expected_base_url is provided, the key is only returned when it was
    originally issued for that URL. This prevents credential leakage when the
    CLI is pointed at a different (possibly malicious) server. A key obtained by
    ``lite login --pkce`` is refreshed here once it nears expiry.
    """
    token_data: Final = load_token(vault=vault)
    if token_data is None:
        return None
    if expected_base_url is not None and token_data.get("base_url") != expected_base_url.rstrip("/"):
        return None
    return fresh_api_key(
        token_data,
        _renewal_saver(vault),
        requests.Session(),
        reload=_renewal_reader(vault),
        warn=_warn,
    )


def _warn(message: str) -> None:
    click.echo(message, err=True)


def _login_command(renews: bool) -> str:
    return "lite login --pkce" if renews else "lite login"


# Team selection utilities
def display_teams_table(teams: list[CliTeam]) -> None:
    """Display teams in a formatted table"""
    console: Final = Console()

    if not teams:
        console.print("No teams found for your user.")
        return

    table: Final = Table(title="Available Teams")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Team Alias", style="magenta")
    table.add_column("Team ID", style="green")
    table.add_column("Models", style="yellow")
    table.add_column("Max Budget", style="blue")

    for i, team in enumerate(teams):
        team_alias = team.get("team_alias") or "N/A"
        team_id = team.get("team_id", "N/A")
        models = team.get("models", [])
        max_budget = team.get("max_budget")

        # Format models list
        if models:
            if len(models) > 3:
                models_str = ", ".join(models[:3]) + f" (+{len(models) - 3} more)"
            else:
                models_str = ", ".join(models)
        else:
            models_str = "All models"

        # Format budget
        budget_str = f"${max_budget}" if max_budget else "Unlimited"

        table.add_row(str(i + 1), team_alias, team_id, models_str, budget_str)

    console.print(table)


def get_key_input():
    """Get a single key input from the user (cross-platform)"""
    try:
        if sys.platform == "win32":
            import msvcrt

            key = msvcrt.getch()
            if key == b"\xe0":  # Arrow keys on Windows
                key = msvcrt.getch()
                if key == b"H":  # Up arrow
                    return "up"
                elif key == b"P":  # Down arrow
                    return "down"
            elif key == b"\r":  # Enter key
                return "enter"
            elif key == b"\x1b":  # Escape key
                return "escape"
            elif key == b"q":
                return "quit"
            return None
        else:
            import termios
            import tty

            fd: Final = sys.stdin.fileno()
            old_settings: Final = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                key = sys.stdin.read(1)

                if key == "\x1b":  # Escape sequence
                    key += sys.stdin.read(2)
                    if key == "\x1b[A":  # Up arrow
                        return "up"
                    elif key == "\x1b[B":  # Down arrow
                        return "down"
                    elif key == "\x1b":  # Just escape
                        return "escape"
                elif key == "\r" or key == "\n":  # Enter key
                    return "enter"
                elif key == "q":
                    return "quit"
                return None
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except ImportError:
        # Fallback to simple input if termios/msvcrt not available
        return None


def display_interactive_team_selection(teams: list[dict[str, Any]], selected_index: int = 0) -> None:
    """Display teams with one highlighted for selection"""
    console: Final = Console()

    # Clear the screen using Rich's method
    console.clear()

    console.print("Select a Team (Use up/down arrows, Enter to select, 'q' to skip):\n")

    for i, team in enumerate(teams):
        team_alias = team.get("team_alias") or "N/A"
        team_id = team.get("team_id", "N/A")
        models: list[str] = team.get("models", [])
        max_budget = team.get("max_budget")

        # Format models list
        if models:
            if len(models) > 3:
                models_str = ", ".join(models[:3]) + f" (+{len(models) - 3} more)"
            else:
                models_str = ", ".join(models)
        else:
            models_str = "All models"

        # Format budget
        budget_str = f"${max_budget}" if max_budget else "Unlimited"

        # Highlight the selected item
        if i == selected_index:
            console.print(f"> [bold cyan]{team_alias}[/bold cyan] ({team_id})")
            console.print(f"   Models: [yellow]{models_str}[/yellow]")
            console.print(f"   Budget: [blue]{budget_str}[/blue]\n")
        else:
            console.print(f"  [dim]{team_alias}[/dim] ({team_id})")
            console.print(f"   Models: [dim]{models_str}[/dim]")
            console.print(f"   Budget: [dim]{budget_str}[/dim]\n")


def prompt_team_selection(teams: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Interactive team selection with arrow keys"""
    if not teams:
        return None

    selected_index = 0

    try:
        # Check if we can use interactive mode
        if not sys.stdin.isatty():
            # Fallback to simple selection for non-interactive environments
            return prompt_team_selection_fallback(teams)

        while True:
            display_interactive_team_selection(teams, selected_index)

            key = get_key_input()

            if key == "up":
                selected_index = (selected_index - 1) % len(teams)
            elif key == "down":
                selected_index = (selected_index + 1) % len(teams)
            elif key == "enter":
                selected_team = teams[selected_index]
                # Clear screen and show selection
                console = Console()
                console.clear()
                click.echo(f"Selected team: {selected_team.get('team_alias', 'N/A')} ({selected_team.get('team_id')})")
                return selected_team
            elif key == "quit" or key == "escape":
                # Clear screen
                console = Console()
                console.clear()
                click.echo("Team selection skipped.")
                return None
            elif key is None:
                # If we can't get key input, fall back to simple selection
                return prompt_team_selection_fallback(teams)

    except KeyboardInterrupt:
        console = Console()
        console.clear()
        click.echo("\nTeam selection cancelled.")
        return None
    except Exception:
        # If interactive mode fails, fall back to simple selection
        return prompt_team_selection_fallback(teams)


def prompt_team_selection_fallback(
    teams: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Fallback team selection for non-interactive environments"""
    if not teams:
        return None

    while True:
        try:
            prompt_response: str = click.prompt(
                "\nSelect a team by entering the index number (or 'skip' to continue without a team)",
                type=str,
            )
            choice = prompt_response.strip()

            if choice.lower() == "skip":
                return None

            index = int(choice) - 1
            if 0 <= index < len(teams):
                selected_team = teams[index]
                click.echo(
                    f"\nSelected team: {selected_team.get('team_alias', 'N/A')} ({selected_team.get('team_id')})"
                )
                return selected_team
            else:
                click.echo(f"Invalid selection. Please enter a number between 1 and {len(teams)}")
        except ValueError:
            click.echo("Invalid input. Please enter a number or 'skip'")
        except KeyboardInterrupt:
            click.echo("\nTeam selection cancelled.")
            return None


def _response_error_detail(response: requests.Response) -> str | None:
    try:
        body: Final[dict[str, object] | list[object] | str | int | float | bool | None] = response.json()
    except ValueError:
        return None
    detail: Final = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str) and detail:
        return detail
    return None


def _polling_error_message(response: requests.Response) -> str:
    detail: Final = _response_error_detail(response)
    if detail:
        return f"Polling error: HTTP {response.status_code}: {detail}"
    return f"Polling error: HTTP {response.status_code}"


def _is_permanent_polling_error(status_code: int) -> bool:
    return 400 <= status_code < 500 and status_code != 429


# Polling-based authentication - no local server needed
def _poll_for_ready_data(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    total_timeout: int = 300,
    poll_interval: int = 2,
    request_timeout: int = 10,
    pending_message: str | None = None,
    pending_log_every: int = 10,
    other_status_message: str | None = None,
    other_status_log_every: int = 10,
    http_error_log_every: int = 10,
    connection_error_log_every: int = 10,
) -> CliPollData | None:
    for attempt in range(total_timeout // poll_interval):
        try:
            request_kwargs: CliPollRequestKwargs = {"timeout": request_timeout}
            if headers is not None:
                request_kwargs["headers"] = headers
            response = requests.get(url, **request_kwargs)
            if response.status_code == 200:
                data: CliPollData = response.json()
                status = data.get("status")
                if status == "ready":
                    return data
                if status == "pending":
                    if pending_message and pending_log_every > 0 and attempt % pending_log_every == 0:
                        click.echo(pending_message)
                elif other_status_message and other_status_log_every > 0 and attempt % other_status_log_every == 0:
                    click.echo(other_status_message)
            elif _is_permanent_polling_error(response.status_code):
                detail = _response_error_detail(response)
                raise ValueError(
                    f"The proxy rejected the login session with HTTP {response.status_code}"
                    + (f": {detail}" if detail else f" and no error detail (from {url})")
                )
            elif http_error_log_every > 0 and attempt % http_error_log_every == 0:
                click.echo(_polling_error_message(response))
        except requests.RequestException as e:
            if connection_error_log_every > 0 and attempt % connection_error_log_every == 0:
                click.echo(f"Connection error (will retry): {e}")
        time.sleep(poll_interval)
    return None


def _normalize_teams(teams: object, team_details: object) -> list[CliTeam]:
    """If team_details are a

    Args:
        teams (_type_): _description_
        team_details (_type_): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(team_details, list) and team_details:
        return [
            {
                "team_id": i.get("team_id") or i.get("id"),
                "team_alias": i.get("team_alias"),
            }
            for i in team_details
            if isinstance(i, dict) and (i.get("team_id") or i.get("id"))
        ]
    if isinstance(teams, list):
        return [{"team_id": str(t), "team_alias": None} for t in teams]
    return []


def _start_cli_sso_flow(base_url: str) -> CliSsoStartData:
    start_url: Final = f"{base_url}/sso/cli/start"
    try:
        response: Final = requests.post(start_url, timeout=10)
    except requests.RequestException as e:
        raise ValueError(
            f"Could not reach the proxy at {start_url}: {e}. "
            "Check that the proxy is running and that --base-url points at it."
        ) from e

    if response.status_code in (404, 405):
        raise ValueError(
            f"POST {start_url} returned HTTP {response.status_code}. "
            "Either --base-url is wrong, or the proxy is older than this CLI and does not support "
            "the CLI SSO login flow; upgrade the proxy or use a CLI version that matches it."
        )
    if response.status_code != 200:
        detail: Final = _response_error_detail(response)
        raise ValueError(
            f"Starting CLI login failed: HTTP {response.status_code} from {start_url}"
            + (f": {detail}" if detail else "")
        )

    try:
        data: Final[CliSsoStartData] = response.json()
    except ValueError:
        content_type: Final = response.headers.get("content-type", "unknown")
        raise ValueError(
            f"The proxy returned a non-JSON response from {start_url} (content-type: {content_type}). "
            "A proxy, load balancer, or auth gateway in front of LiteLLM may be intercepting the request. "
            f"Response starts with: {response.text[:200]!r}"
        )

    required_fields: Final[tuple[str, ...]] = ("login_id", "poll_secret", "user_code")
    missing_fields: Final = tuple(field for field in required_fields if not isinstance(data.get(field), str))
    if missing_fields:
        raise ValueError(
            f"The response from {start_url} is missing required field(s): {', '.join(missing_fields)}. "
            "The proxy version may not match this CLI; upgrade whichever is older."
        )
    return data


def _get_cli_sso_poll_headers(poll_secret: str) -> dict[str, str]:
    return {"x-litellm-cli-poll-secret": poll_secret}


def _poll_for_authentication(base_url: str, key_id: str, poll_secret: str) -> CliAuthResult | None:
    """
    Poll the server for authentication completion and handle team selection.

    Returns:
        Dictionary with authentication data if successful, None otherwise
    """
    poll_url: Final = f"{base_url}/sso/cli/poll/{key_id}"
    data: Final = _poll_for_ready_data(
        poll_url,
        headers=_get_cli_sso_poll_headers(poll_secret),
        pending_message="Still waiting for authentication...",
    )
    if not data:
        return None
    if data.get("requires_team_selection"):
        teams = data.get("teams", [])
        team_details: Final = data.get("team_details")
        user_id = data.get("user_id")
        normalized_teams: Final[list[CliTeam]] = _normalize_teams(teams, team_details)
        if not normalized_teams:
            click.echo("Warning: No teams available for selection.")
            return None

        # User has multiple teams - let them select
        jwt_with_team: Final = _handle_team_selection_during_polling(
            base_url=base_url,
            key_id=key_id,
            poll_secret=poll_secret,
            teams=normalized_teams,
        )

        # Use the team-specific JWT if selection succeeded
        if jwt_with_team:
            return {
                "api_key": jwt_with_team,
                "user_id": user_id,
                "teams": teams,
                "team_id": None,  # Set by server in JWT
            }

        click.echo("Team selection cancelled or JWT generation failed.")
        return None

    # JWT is ready (single team or team already selected)
    api_key: Final = data.get("key")
    user_id = data.get("user_id")
    teams = data.get("teams", [])
    team_id: Final = data.get("team_id")

    # Show which team was assigned
    if team_id and len(teams) == 1:
        click.echo(f"\nAutomatically assigned to team: {team_id}")

    if api_key:
        return {
            "api_key": api_key,
            "user_id": user_id,
            "teams": teams,
            "team_id": team_id,
        }

    return None


def _handle_team_selection_during_polling(
    base_url: str, key_id: str, poll_secret: str, teams: list[CliTeam]
) -> str | None:
    """
    Handle team selection and re-poll with selected team_id.

    Args:
        teams: List of team IDs (strings)

    Returns:
        The JWT token with the selected team, or None if selection was skipped
    """
    if not teams:
        click.echo("No teams found. You can create or join teams using the web interface.")
        return None

    click.echo("\n" + "=" * 60)
    click.echo("Select a team for your CLI session...")

    team_id: Final = _render_and_prompt_for_team_selection(teams)

    if not team_id:
        click.echo("No team selected.")
        return None

    click.echo(f"\nGenerating JWT for team: {team_id}")

    poll_url: Final = f"{base_url}/sso/cli/poll/{key_id}?team_id={team_id}"
    data: Final = _poll_for_ready_data(
        poll_url,
        headers=_get_cli_sso_poll_headers(poll_secret),
        pending_message="Still waiting for team authentication...",
        other_status_message="Waiting for team authentication to complete...",
        http_error_log_every=10,
    )
    if not data:
        return None
    jwt_token: Final = data.get("key")
    if jwt_token:
        click.echo(f"Successfully generated JWT for team: {team_id}")
        return jwt_token

    return None


def _render_and_prompt_for_team_selection(teams: list[CliTeam]) -> str | None:
    """Render teams table and prompt user for a team selection.

    Returns the selected team_id as a string, or None if selection was
    cancelled or skipped without any teams available.
    """
    # Display teams as a simple list, but prefer showing aliases where
    # available while still keeping the underlying IDs intact.
    console: Final = Console()
    table: Final = Table(title="Available Teams")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Team Name", style="magenta")
    table.add_column("Team ID", style="green")

    for i, team in enumerate(teams):
        team_id = str(team.get("team_id"))
        team_alias = team.get("team_alias") or team_id
        table.add_row(str(i + 1), team_alias, team_id)

    console.print(table)

    # Simple selection
    while True:
        try:
            prompt_response: str = click.prompt(
                "\nSelect a team by entering the index number (or 'skip' to use first team)",
                type=str,
            )
            choice = prompt_response.strip()

            if choice.lower() == "skip":
                # Default to the first team's ID if the user skips an
                # explicit selection.
                if teams:
                    first_team = teams[0]
                    return str(first_team.get("team_id"))
                return None

            index = int(choice) - 1
            if 0 <= index < len(teams):
                selected_team = teams[index]
                team_id = str(selected_team.get("team_id"))
                team_alias = selected_team.get("team_alias") or team_id
                click.echo(f"\nSelected team: {team_alias} ({team_id})")
                return team_id

            click.echo(f"Invalid selection. Please enter a number between 1 and {len(teams)}")
        except ValueError:
            click.echo("Invalid input. Please enter a number or 'skip'")
        except KeyboardInterrupt:
            click.echo("\nTeam selection cancelled.")
            return None


def _configure_claude_code(base_url: str) -> None:
    """Point Claude Code at base_url by patching ~/.claude/settings.json."""
    try:
        write_claude_settings(base_url, CLAUDE_SETTINGS_PATH, SETTINGS_FILE_OWNERS)
    except ClaudeSettingsError as e:
        raise click.ClickException(f"Logged in, but could not configure Claude Code: {e}")
    click.echo(f"\nConfigured Claude Code: {CLAUDE_SETTINGS_PATH} now routes through {base_url.rstrip('/')}.")
    click.echo("Your other Claude Code settings were left untouched. Restart Claude Code to pick this up.")


def _finish_login(base_url: str, api_key: str, config_claude: bool, stored: SecretSave) -> None:
    from litellm.proxy.client.cli.interface import show_commands

    click.echo("\nLogin successful!")
    click.echo(f"JWT Token: {api_key[:20]}...")
    click.echo(storage_notice(stored))
    if isinstance(stored, (CredentialNotSaved, CredentialNotRecorded)):
        return
    click.echo("You can now use the CLI without specifying --api-key")
    if config_claude:
        _configure_claude_code(base_url)
    click.echo("\n" + "=" * 60)
    show_commands()


def _replace_stored_token(record: CliTokenData, http: Http, vault: SecretVault) -> SecretSave:
    previous: Final = load_token(vault=vault)
    stored: Final = save_token(record, vault=vault)
    if previous is None or isinstance(stored, CredentialNotSaved):
        return stored
    revocation: Final = revoke_stored_credential(previous, http)
    if revocation is not None:
        click.echo(
            f"Could not revoke the previous login's refresh token on the proxy ({revocation.reason}); "
            "it expires on its own."
        )
    return stored


def _pkce_login(base_url: str, config_claude: bool, vault: SecretVault) -> None:
    http: Final = requests.Session()
    credential: Final = run_pkce_login(base_url, http, echo=click.echo)
    if isinstance(credential, PkceFailure):
        click.echo(f"Authentication failed: {credential.reason}")
        return
    stored: Final = _replace_stored_token(pkce_token_record(base_url, credential), http, vault)
    _finish_login(base_url, credential.access_token, config_claude, stored)


@click.command(name="login")
@click.option(
    "--config-claude",
    is_flag=True,
    default=False,
    help=(
        "After logging in, update ~/.claude/settings.json so Claude Code routes through this proxy. "
        "Unrelated settings are preserved."
    ),
)
@click.option(
    "--pkce",
    is_flag=True,
    default=False,
    help=(
        "Sign in with OAuth authorization code + PKCE through your system browser (loopback redirect), "
        "with a refresh token that renews the key automatically. Requires a proxy that serves "
        "/.well-known/litellm-cli-auth."
    ),
)
@click.pass_context
def login(ctx: click.Context, config_claude: bool, pkce: bool) -> None:
    """Login to LiteLLM proxy using SSO authentication"""
    from litellm.constants import LITELLM_CLI_SOURCE_IDENTIFIER

    ctx_obj: Final[CliContextObj] = ctx.obj
    base_url: Final = ctx_obj["base_url"]

    try:
        if pkce:
            _pkce_login(base_url, config_claude, context_secret_vault(ctx))
            return
        cli_sso_flow: Final = _start_cli_sso_flow(base_url=base_url)
        key_id: Final = cli_sso_flow["login_id"]
        poll_secret: Final = cli_sso_flow["poll_secret"]
        user_code: Final = cli_sso_flow["user_code"]

        sso_url = f"{base_url}/sso/key/generate?" + urlencode({"source": LITELLM_CLI_SOURCE_IDENTIFIER, "key": key_id})

        click.echo(f"Opening browser to: {sso_url}")
        click.echo("Please complete the SSO authentication in your browser...")
        click.echo(f"Verification code: {user_code}")
        click.echo(f"Session ID: {key_id}")

        # Open browser
        webbrowser.open(sso_url)

        # Poll for authentication completion
        click.echo("Waiting for authentication...")

        auth_result: Final = _poll_for_authentication(base_url=base_url, key_id=key_id, poll_secret=poll_secret)

        if auth_result:
            api_key: Final = auth_result["api_key"]
            user_id: Final = auth_result["user_id"]

            # Save token data. base_url is stored so we can verify origin
            # before reusing the key on a subsequent CLI invocation.
            stored: Final = _replace_stored_token(
                {
                    "base_url": base_url.rstrip("/"),
                    "key": api_key,
                    "user_id": user_id or "cli-user",
                    "user_email": "unknown",
                    "user_role": "cli",
                    "auth_header_name": "Authorization",
                    "jwt_token": "",
                    "timestamp": time.time(),
                },
                requests.Session(),
                context_secret_vault(ctx),
            )

            _finish_login(base_url, api_key, config_claude, stored)
            return
        else:
            click.echo("Authentication timed out. Please try again.")
            click.echo(
                "The proxy never reported the browser sign-in as finished. If you did complete it, "
                "check the proxy logs for /sso/callback errors and confirm SSO is configured on the proxy."
            )
            return

    except KeyboardInterrupt:
        click.echo("\nAuthentication cancelled by user.")
        return
    except click.ClickException:
        # Login itself already succeeded; only the post-login step failed, so this
        # must not be relabelled as an authentication failure by the handler below.
        raise
    except Exception as e:
        click.echo(f"Authentication failed: {e}")
        return


@click.command(name="logout")
@click.pass_context
def logout(ctx: click.Context):
    """Logout and clear stored authentication"""
    vault: Final = context_secret_vault(ctx)
    token_data: Final = load_token(vault=vault)
    revocation: Final = revoke_stored_credential(token_data, requests.Session()) if token_data is not None else None
    match revocation:
        case RevocationUnavailable(reason=reason):
            raise click.ClickException(
                f"The proxy could not record the revocation ({reason}). Nothing was cleared; "
                "run `lite logout` again shortly."
            )
        case PkceFailure(reason=reason):
            click.echo(f"Could not revoke the refresh token on the proxy ({reason}); it expires on its own.")
        case None:
            pass
        case _:
            assert_never(revocation)

    path: Final = get_cli_token_file_path()
    match clear_cli_token(vault=vault):
        case SecretErased():
            click.echo("Logged out successfully. Authentication token cleared.")
        case CredentialNotCleared(detail=detail):
            click.echo(f"Your credential is still in {path}, which could not be removed: {detail}.")
            click.echo("Delete that file, or make the directory writable and run 'lite logout' again.")
        case SecretStranded():
            click.echo(STRANDED_CREDENTIAL_MESSAGE)
            click.echo("Unlock your keychain and run 'lite logout' again to clear it.")
        case KeyringNotInstalled():
            click.echo(UNCHECKED_KEYCHAIN_MESSAGE)
            click.echo(f"Install the keyring package with: {KEYRING_INSTALL_HINT}, then run 'lite logout' again.")
        case KeyringDisabled():
            click.echo(UNCHECKED_KEYCHAIN_MESSAGE)
            click.echo(f"Unset {DISABLE_KEYRING_ENV_VAR} and run 'lite logout' again to clear it.")
        case KeyringUnreachable():
            click.echo(UNCHECKED_KEYCHAIN_MESSAGE)
            click.echo("Unlock your keychain and run 'lite logout' again to clear it.")


@click.command(name="print-token")
@click.pass_context
def print_token(ctx: click.Context):
    """Print a valid API token for this proxy.

    Designed to be used as Claude Code's `apiKeyHelper`
    (https://docs.claude.com/en/docs/claude-code/settings): stdout must
    contain only the token, so all diagnostics go to stderr. The token
    expires after `LITELLM_CLI_JWT_EXPIRATION_HOURS` (default 24h); a
    `lite login --pkce` token renews itself here first, and once a token
    has expired for good, run the same `lite login` command again.
    """
    vault: Final = context_secret_vault(ctx)
    token_data: Final = load_token(vault=vault)
    if not token_data:
        click.echo("Not authenticated. Run 'lite login'.", err=True)
        sys.exit(1)

    # apiKeyHelper is invoked bare (no --base-url), so unless the caller
    # explicitly pointed us at a server, trust whichever one `lite login`
    # actually issued this token for -- that's the whole point of not
    # needing a wrapper command.
    ctx_obj: Final[CliContextObj] = ctx.obj
    issued_for_this_server: Final = token_data.get("base_url") == ctx_obj.get("base_url", "").rstrip("/")
    if ctx_obj.get("base_url_explicit") and not issued_for_this_server:
        click.echo("Not authenticated for this server. Run 'lite login'.", err=True)
        sys.exit(1)

    renews: Final = "refresh_token" in token_data
    if not is_cli_token_fresh(token_data) and not renews:
        click.echo("Token expired. Run 'lite login' again.", err=True)
        sys.exit(1)

    if token_data.get("key") is None:
        click.echo(keychain_unreadable_notice(vault), err=True)
        sys.exit(1)

    api_key: Final = (
        ctx_obj.get("api_key")
        if issued_for_this_server and ctx_obj.get("api_key_from_token_file")
        else fresh_api_key(
            token_data,
            _renewal_saver(vault),
            requests.Session(),
            reload=_renewal_reader(vault),
            warn=_warn,
        )
    )
    if not api_key:
        click.echo(f"Key expired. Run '{_login_command(renews)}' again.", err=True)
        sys.exit(1)

    click.echo(api_key)


@click.command(name="whoami")
@click.pass_context
def whoami(ctx: click.Context):
    """Show current authentication status"""
    vault: Final = context_secret_vault(ctx)
    token_data: Final = load_token(vault=vault)

    if not token_data:
        click.echo("Not authenticated. Run 'lite login' to authenticate.")
        return

    key_readable: Final = token_data.get("key") is not None
    click.echo("Authenticated" if key_readable else "Signed in, but the credential cannot be read")
    click.echo(f"User Email: {token_data.get('user_email') or 'Unknown'}")
    click.echo(f"User ID: {token_data.get('user_id') or 'Unknown'}")
    click.echo(f"User Role: {token_data.get('user_role') or 'Unknown'}")
    team_id: Final = token_data.get("team_id")
    if team_id:
        click.echo(f"Team ID: {team_id}")

    stamped: Final = token_data.get("timestamp")
    age_hours: Final = (time.time() - (stamped if isinstance(stamped, (int, float)) else 0.0)) / 3600
    click.echo(f"Token age: {age_hours:.1f} hours")

    if not key_readable:
        click.echo(keychain_unreadable_notice(vault))

    expires_at: Final = token_data.get("expires_at")
    if isinstance(expires_at, (int, float)):
        click.echo(_key_expiry_line(expires_at, renews="refresh_token" in token_data))
    elif age_hours > CLI_JWT_EXPIRATION_HOURS:
        click.echo(f"Warning: Token is more than {CLI_JWT_EXPIRATION_HOURS} hours old and may have expired.")


def _key_expiry_line(expires_at: float, renews: bool) -> str:
    remaining_hours: Final = (expires_at - time.time()) / 3600
    if remaining_hours <= 0:
        return f"Key expired. Run '{_login_command(renews)}' again"
    status: Final = f"Key expires in: {remaining_hours:.1f} hours"
    return f"{status}, renewed on next use" if renews else status


@click.group(name="auth")
def auth_group():
    """Manage CLI authentication (apiKeyHelper support, etc.)"""


auth_group.add_command(print_token)


# Export functions for use by other CLI commands
__all__ = ["auth_group", "login", "logout", "print_token", "prompt_team_selection", "whoami"]

# Export individual commands instead of grouping them
# login, logout, and whoami will be added as top-level commands
