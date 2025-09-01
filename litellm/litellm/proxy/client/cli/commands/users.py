import click
import rich
from ... import UsersManagementClient

@click.group()
def users():
    """Manage users on your LiteLLM proxy server"""
    pass

@users.command("list")
@click.pass_context
def list_users(ctx: click.Context):
    """List all users"""
    client = UsersManagementClient(base_url=ctx.obj["base_url"], api_key=ctx.obj["api_key"])
    users = client.list_users()
    if isinstance(users, dict) and "users" in users:
        users = users["users"]
    if not users:
        click.echo("No users found.")
        return
    from rich.table import Table
    from rich.console import Console
    table = Table(title="Users")
    table.add_column("User ID", style="cyan")
    table.add_column("Email", style="green")
    table.add_column("Role", style="magenta")
    table.add_column("Teams", style="yellow")
    for user in users:
        table.add_row(
            str(user.get("user_id", "")),
            str(user.get("user_email", "")),
            str(user.get("user_role", "")),
            ", ".join(user.get("teams", []) or [])
        )
    console = Console()
    console.print(table)

@users.command("get")
@click.option("--id", "user_id", help="ID of the user to retrieve")
@click.pass_context
def get_user(ctx: click.Context, user_id: str):
    """Get information about a specific user"""
    client = UsersManagementClient(base_url=ctx.obj["base_url"], api_key=ctx.obj["api_key"])
    result = client.get_user(user_id=user_id)
    rich.print_json(data=result)

@users.command("create")
@click.option("--email", required=True, help="User email")
@click.option("--role", default="internal_user", help="User role")
@click.option("--alias", default=None, help="User alias")
@click.option("--team", multiple=True, help="Team IDs (can specify multiple)")
@click.option("--max-budget", type=float, default=None, help="Max budget for user")
@click.pass_context
def create_user(ctx: click.Context, email, role, alias, team, max_budget):
    """Create a new user"""
    client = UsersManagementClient(base_url=ctx.obj["base_url"], api_key=ctx.obj["api_key"])
    user_data = {
        "user_email": email,
        "user_role": role,
    }
    if alias:
        user_data["user_alias"] = alias
    if team:
        user_data["teams"] = list(team)
    if max_budget is not None:
        user_data["max_budget"] = max_budget
    result = client.create_user(user_data)
    rich.print_json(data=result)

@users.command("delete")
@click.argument("user_ids", nargs=-1)
@click.pass_context
def delete_user(ctx: click.Context, user_ids):
    """Delete one or more users by user_id"""
    client = UsersManagementClient(base_url=ctx.obj["base_url"], api_key=ctx.obj["api_key"])
    result = client.delete_user(list(user_ids))
    rich.print_json(data=result) 