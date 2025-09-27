"""Team management commands for LiteLLM CLI."""

from typing import Any, Dict, List, Optional

import click
import requests
from rich.console import Console
from rich.table import Table

from litellm.proxy.client import Client


@click.group()
def teams():
    """Manage teams and team assignments"""
    pass


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
    table.add_column("Role", style="red")
    
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
        
        # Try to determine role (this might vary based on API response structure)
        role = "Member"  # Default role
        if isinstance(team, dict) and 'members_with_roles' in team and team['members_with_roles']:
            # This would need to be implemented based on actual API response structure
            pass
        
        table.add_row(
            str(i + 1),
            team_alias,
            team_id,
            models_str,
            budget_str,
            role
        )
    
    console.print(table)


@teams.command()
@click.pass_context
def list(ctx: click.Context):
    """List teams that you belong to"""
    client = Client(ctx.obj["base_url"], ctx.obj["api_key"])
    
    try:
        # Use list() for simpler response structure (returns array directly)
        teams = client.teams.list()
        display_teams_table(teams)
    except requests.exceptions.HTTPError as e:
        click.echo(f"Error: HTTP {e.response.status_code}", err=True)
        error_body = e.response.json()
        click.echo(f"Details: {error_body.get('detail', 'Unknown error')}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


@teams.command()
@click.pass_context
def available(ctx: click.Context):
    """List teams that are available to join"""
    client = Client(ctx.obj["base_url"], ctx.obj["api_key"])
    
    try:
        teams = client.teams.get_available()
        if teams:
            console = Console()
            console.print("\n🎯 Available Teams to Join:")
            display_teams_table(teams)
        else:
            click.echo("ℹ️ No available teams to join.")
    except requests.exceptions.HTTPError as e:
        click.echo(f"Error: HTTP {e.response.status_code}", err=True)
        error_body = e.response.json()
        click.echo(f"Details: {error_body.get('detail', 'Unknown error')}", err=True)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


@teams.command()
@click.option("--team-id", type=str, help="Team ID to assign the key to")
@click.pass_context
def assign_key(ctx: click.Context, team_id: Optional[str]):
    """Assign your current CLI key to a team"""
    client = Client(ctx.obj["base_url"], ctx.obj["api_key"])
    api_key = ctx.obj["api_key"]
    
    if not api_key:
        click.echo("❌ No API key found. Please login first using 'litellm login'")
        raise click.Abort()
    
    try:
        # If no team_id provided, show teams and let user select
        if not team_id:
            teams = client.teams.list()
            
            if not teams:
                click.echo("❌ No teams found for your user.")
                return
            
            # Use interactive selection from auth module
            from .auth import prompt_team_selection
            selected_team = prompt_team_selection(teams)
            
            if selected_team:
                team_id = selected_team.get('team_id')
            else:
                click.echo("❌ Operation cancelled.")
                return
        
        # Update the key with the selected team
        if team_id:
            click.echo(f"\n🔄 Assigning your key to team: {team_id}")
            client.keys.update(key=api_key, team_id=team_id)
            click.echo(f"✅ Successfully assigned key to team: {team_id}")
            
            # Show team details if available
            teams = client.teams.list()
            for team in teams:
                if team.get('team_id') == team_id:
                    models = team.get('models', [])
                    if models:
                        click.echo(f"🎯 You can now access models: {', '.join(models)}")
                    else:
                        click.echo("🎯 You can now access all available models")
                    break
        
    except requests.exceptions.HTTPError as e:
        click.echo(f"Error: HTTP {e.response.status_code}", err=True)
        error_body = e.response.json()
        click.echo(f"Details: {error_body.get('detail', 'Unknown error')}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()
