import json
import os
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import requests
from rich.console import Console
from rich.table import Table


# Token storage utilities
def get_token_file_path() -> str:
    """Get the path to store the authentication token"""
    home_dir = Path.home()
    config_dir = home_dir / ".litellm"
    config_dir.mkdir(exist_ok=True)
    return str(config_dir / "token.json")


def save_token(token_data: Dict[str, Any]) -> None:
    """Save token data to file"""
    token_file = get_token_file_path()
    with open(token_file, "w") as f:
        json.dump(token_data, f, indent=2)
    # Set file permissions to be readable only by owner
    os.chmod(token_file, 0o600)


def load_token() -> Optional[Dict[str, Any]]:
    """Load token data from file"""
    token_file = get_token_file_path()
    if not os.path.exists(token_file):
        return None

    try:
        with open(token_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def clear_token() -> None:
    """Clear stored token"""
    token_file = get_token_file_path()
    if os.path.exists(token_file):
        os.remove(token_file)


def get_stored_api_key() -> Optional[str]:
    """Get the stored API key from token file"""
    # Use the SDK-level utility
    from litellm.litellm_core_utils.cli_token_utils import get_litellm_gateway_api_key

    return get_litellm_gateway_api_key()


# Team selection utilities
def display_teams_table(teams: List[Dict[str, Any]]) -> None:
    """Display teams in a formatted table"""
    console = Console()

    if not teams:
        console.print("❌ No teams found for your user.")
        return

    table = Table(title="Available Teams")
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

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
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


def display_interactive_team_selection(
    teams: List[Dict[str, Any]], selected_index: int = 0
) -> None:
    """Display teams with one highlighted for selection"""
    console = Console()

    # Clear the screen using Rich's method
    console.clear()

    console.print("🎯 Select a Team (Use ↑↓ arrows, Enter to select, 'q' to skip):\n")

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

        # Highlight the selected item
        if i == selected_index:
            console.print(f"➤ [bold cyan]{team_alias}[/bold cyan] ({team_id})")
            console.print(f"   Models: [yellow]{models_str}[/yellow]")
            console.print(f"   Budget: [blue]{budget_str}[/blue]\n")
        else:
            console.print(f"  [dim]{team_alias}[/dim] ({team_id})")
            console.print(f"   Models: [dim]{models_str}[/dim]")
            console.print(f"   Budget: [dim]{budget_str}[/dim]\n")


def prompt_team_selection(teams: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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
                click.echo(
                    f"✅ Selected team: {selected_team.get('team_alias', 'N/A')} ({selected_team.get('team_id')})"
                )
                return selected_team
            elif key == "quit" or key == "escape":
                # Clear screen
                console = Console()
                console.clear()
                click.echo("ℹ️ Team selection skipped.")
                return None
            elif key is None:
                # If we can't get key input, fall back to simple selection
                return prompt_team_selection_fallback(teams)

    except KeyboardInterrupt:
        console = Console()
        console.clear()
        click.echo("\n❌ Team selection cancelled.")
        return None
    except Exception:
        # If interactive mode fails, fall back to simple selection
        return prompt_team_selection_fallback(teams)


def prompt_team_selection_fallback(
    teams: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Fallback team selection for non-interactive environments"""
    if not teams:
        return None

    while True:
        try:
            choice = click.prompt(
                "\nSelect a team by entering the index number (or 'skip' to continue without a team)",
                type=str,
            ).strip()

            if choice.lower() == "skip":
                return None

            index = int(choice) - 1
            if 0 <= index < len(teams):
                selected_team = teams[index]
                click.echo(
                    f"\n✅ Selected team: {selected_team.get('team_alias', 'N/A')} ({selected_team.get('team_id')})"
                )
                return selected_team
            else:
                click.echo(
                    f"❌ Invalid selection. Please enter a number between 1 and {len(teams)}"
                )
        except ValueError:
            click.echo("❌ Invalid input. Please enter a number or 'skip'")
        except KeyboardInterrupt:
            click.echo("\n❌ Team selection cancelled.")
            return None


# Polling-based authentication - no local server needed


def _poll_for_authentication(base_url: str, key_id: str) -> Optional[dict]:
    """
    Poll the server for authentication completion and handle team selection.

    Returns:
        Dictionary with authentication data if successful, None otherwise
    """
    poll_url = f"{base_url}/sso/cli/poll/{key_id}"
    timeout = 300  # 5 minute timeout
    poll_interval = 2  # Poll every 2 seconds

    for attempt in range(timeout // poll_interval):
        try:
            response = requests.get(poll_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ready":
                    # Check if we need team selection first
                    if data.get("requires_team_selection"):
                        # Server returned teams list without JWT - need to select team.
                        # Newer servers may also return "team_details" containing
                        # objects with both team_id and team_alias. We prefer those
                        # for display, but continue to support the legacy list of
                        # team IDs for backwards compatibility.
                        teams = data.get("teams", [])
                        team_details = data.get("team_details")
                        user_id = data.get("user_id")

                        # Build a normalized list of team objects that always have
                        # "team_id" and optionally "team_alias".
                        normalized_teams: List[Dict[str, Any]] = []
                        if isinstance(team_details, list) and team_details:
                            for item in team_details:
                                if isinstance(item, dict):
                                    team_id = item.get("team_id") or item.get("id")
                                    if team_id is None:
                                        continue
                                    normalized_teams.append(
                                        {
                                            "team_id": team_id,
                                            "team_alias": item.get("team_alias"),
                                        }
                                    )
                        elif isinstance(teams, list):
                            for t in teams:
                                normalized_teams.append(
                                    {
                                        "team_id": str(t),
                                        "team_alias": None,
                                    }
                                )

                        if normalized_teams and len(normalized_teams) > 1:
                            # User has multiple teams - let them select
                            jwt_with_team = _handle_team_selection_during_polling(
                                base_url=base_url,
                                key_id=key_id,
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
                            else:
                                # Selection failed or was skipped - poll again without team_id
                                click.echo("⚠️ Team selection skipped, retrying...")
                                continue
                        else:
                            # Shouldn't happen, but fallback
                            click.echo("⚠️ No teams available, retrying...")
                            continue
                    else:
                        # JWT is ready (single team or team already selected)
                        api_key = data.get("key")
                        user_id = data.get("user_id")
                        teams = data.get("teams", [])
                        team_id = data.get("team_id")

                        # Show which team was assigned
                        if team_id and len(teams) == 1:
                            click.echo(f"\n✅ Automatically assigned to team: {team_id}")

                        if api_key:
                            return {
                                "api_key": api_key,
                                "user_id": user_id,
                                "teams": teams,
                                "team_id": team_id,
                            }
                elif data.get("status") == "pending":
                    # Still pending
                    if attempt % 10 == 0:  # Show progress every 20 seconds
                        click.echo("Still waiting for authentication...")
            else:
                click.echo(f"Polling error: HTTP {response.status_code}")

        except requests.RequestException as e:
            if attempt % 10 == 0:
                click.echo(f"Connection error (will retry): {e}")

        time.sleep(poll_interval)

    # Timeout reached
    return None


def _handle_team_selection_during_polling(
    base_url: str, key_id: str, teams: List[Dict[str, Any]]
) -> Optional[str]:
    """
    Handle team selection and re-poll with selected team_id.

    Args:
        teams: List of team IDs (strings)

    Returns:
        The JWT token with the selected team, or None if selection was skipped
    """
    if not teams:
        click.echo(
            "ℹ️ No teams found. You can create or join teams using the web interface."
        )
        return None

    click.echo("\n" + "=" * 60)
    click.echo("📋 Select a team for your CLI session...")

    team_id = _render_and_prompt_for_team_selection(teams)

    if not team_id:
        click.echo("ℹ️ No team selected.")
        return None

    click.echo(f"\n🔄 Generating JWT for team: {team_id}")

    # Re-poll with team_id to get JWT with correct team
    try:
        poll_url = f"{base_url}/sso/cli/poll/{key_id}?team_id={team_id}"
        response = requests.get(poll_url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ready":
                jwt_token = data.get("key")
                if jwt_token:
                    click.echo(f"✅ Successfully generated JWT for team: {team_id}")
                    return jwt_token

        click.echo(f"❌ Failed to get JWT with team. Status: {response.status_code}")
        return None

    except Exception as e:
        click.echo(f"❌ Error getting JWT with team: {e}")
        return None


def _render_and_prompt_for_team_selection(teams: List[Dict[str, Any]]) -> Optional[str]:
    """Render teams table and prompt user for a team selection.

    Returns the selected team_id as a string, or None if selection was
    cancelled or skipped without any teams available.
    """
    # Display teams as a simple list, but prefer showing aliases where
    # available while still keeping the underlying IDs intact.
    console = Console()
    table = Table(title="Available Teams")
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
            choice = click.prompt(
                "\nSelect a team by entering the index number (or 'skip' to use first team)",
                type=str,
            ).strip()

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
                click.echo(f"\n✅ Selected team: {team_alias} ({team_id})")
                return team_id

            click.echo(
                f"❌ Invalid selection. Please enter a number between 1 and {len(teams)}"
            )
        except ValueError:
            click.echo("❌ Invalid input. Please enter a number or 'skip'")
        except KeyboardInterrupt:
            click.echo("\n❌ Team selection cancelled.")
            return None


@click.command(name="login")
@click.pass_context
def login(ctx: click.Context):
    """Login to LiteLLM proxy using SSO authentication"""
    from litellm._uuid import uuid
    from litellm.constants import LITELLM_CLI_SOURCE_IDENTIFIER
    from litellm.proxy.client.cli.interface import show_commands

    base_url = ctx.obj["base_url"]

    # Check if we have an existing key to regenerate
    existing_key = get_stored_api_key()

    # Generate unique key ID for this login session
    key_id = f"sk-{str(uuid.uuid4())}"

    try:
        # Construct SSO login URL with CLI source and pre-generated key
        sso_url = f"{base_url}/sso/key/generate?source={LITELLM_CLI_SOURCE_IDENTIFIER}&key={key_id}"

        # If we have an existing key, include it as a parameter to the login endpoint
        # The server will encode it in the OAuth state parameter for the SSO flow
        if existing_key:
            sso_url += f"&existing_key={existing_key}"

        click.echo(f"Opening browser to: {sso_url}")
        click.echo("Please complete the SSO authentication in your browser...")
        click.echo(f"Session ID: {key_id}")

        # Open browser
        webbrowser.open(sso_url)

        # Poll for authentication completion
        click.echo("Waiting for authentication...")

        auth_result = _poll_for_authentication(base_url=base_url, key_id=key_id)

        if auth_result:
            api_key = auth_result["api_key"]
            user_id = auth_result["user_id"]

            # Save token data (simplified for CLI - we just need the key)
            save_token(
                {
                    "key": api_key,
                    "user_id": user_id or "cli-user",
                    "user_email": "unknown",
                    "user_role": "cli",
                    "auth_header_name": "Authorization",
                    "jwt_token": "",
                    "timestamp": time.time(),
                }
            )

            click.echo("\n✅ Login successful!")
            click.echo(f"JWT Token: {api_key[:20]}...")
            click.echo("You can now use the CLI without specifying --api-key")

            # Show available commands after successful login
            click.echo("\n" + "=" * 60)
            show_commands()
            return
        else:
            click.echo("❌ Authentication timed out. Please try again.")
            return

    except KeyboardInterrupt:
        click.echo("\n❌ Authentication cancelled by user.")
        return
    except Exception as e:
        click.echo(f"❌ Authentication failed: {e}")
        return


@click.command(name="logout")
def logout():
    """Logout and clear stored authentication"""
    clear_token()
    click.echo("✅ Logged out successfully. Authentication token cleared.")


@click.command(name="whoami")
def whoami():
    """Show current authentication status"""
    token_data = load_token()

    if not token_data:
        click.echo("❌ Not authenticated. Run 'litellm-proxy login' to authenticate.")
        return

    click.echo("✅ Authenticated")
    click.echo(f"User Email: {token_data.get('user_email', 'Unknown')}")
    click.echo(f"User ID: {token_data.get('user_id', 'Unknown')}")
    click.echo(f"User Role: {token_data.get('user_role', 'Unknown')}")

    # Check if token is still valid (basic timestamp check)
    timestamp = token_data.get("timestamp", 0)
    age_hours = (time.time() - timestamp) / 3600
    click.echo(f"Token age: {age_hours:.1f} hours")

    if age_hours > 24:
        click.echo("⚠️ Warning: Token is more than 24 hours old and may have expired.")


# Export functions for use by other CLI commands
__all__ = ["login", "logout", "whoami", "prompt_team_selection"]

# Export individual commands instead of grouping them
# login, logout, and whoami will be added as top-level commands
