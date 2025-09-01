# stdlib imports
import os
import sys
from typing import TYPE_CHECKING

# third party imports
import click

from litellm._logging import verbose_logger

if TYPE_CHECKING:
    pass


def styled_prompt():
    """Create a styled blue box prompt for user input."""

    # Get terminal height to ensure we have enough space
    try:
        terminal_height = os.get_terminal_size().lines
        # Ensure we have at least 5 lines of space (for the box + some buffer)
        if terminal_height < 10:
            # If terminal is too small, just add some newlines to push content up
            click.echo("\n" * 3)
    except Exception as e:
        # Fallback if we can't get terminal size
        verbose_logger.debug(f"Error getting terminal size: {e}")
        click.echo("\n" * 3)
    
    # Unicode box drawing characters
    top_left = "┌"
    top_right = "┐"
    bottom_left = "└"
    bottom_right = "┘"
    horizontal = "─"
    vertical = "│"
    
    # Create the box with increased width
    width = 80
    top_line = top_left + horizontal * (width - 2) + top_right
    bottom_line = bottom_left + horizontal * (width - 2) + bottom_right
    
    # Create styled elements
    left_border = click.style(vertical, fg="blue", bold=True)
    right_border = click.style(vertical, fg="blue", bold=True)
    prompt_text = click.style("> ", fg="cyan", bold=True)
    
    # Display the complete box structure first to reserve space
    click.echo(click.style(top_line, fg="blue", bold=True))
    
    # Create empty space in the box for input
    empty_space = " " * (width - 4)
    click.echo(f"{left_border} {empty_space} {right_border}")
    
    # Display bottom border to complete the box
    click.echo(click.style(bottom_line, fg="blue", bold=True))
    
    # Now move cursor up to the input line and get input
    click.echo("\033[2A", nl=False)  # Move cursor up 2 lines
    click.echo(f"\r{left_border} {prompt_text}", nl=False)  # Position at start of input line
    
    try:
        # Get user input
        user_input = input().strip()
        
        # Move cursor down to after the box
        click.echo("\033[1B")  # Move cursor down 1 line
        click.echo("")  # Add some space after
        
    except (KeyboardInterrupt, EOFError):
        # Move cursor down and add space
        click.echo("\033[1B")
        click.echo("")
        raise
    
    return user_input


def show_commands():
    """Display available commands."""
    commands = [
        ("login", "Authenticate with the LiteLLM proxy server"),
        ("logout", "Clear stored authentication"),
        ("whoami", "Show current authentication status"),
        ("models", "Manage and view model configurations"),
        ("credentials", "Manage API credentials"),
        ("chat", "Interactive chat with models"),
        ("http", "Make HTTP requests to the proxy"),
        ("keys", "Manage API keys"),
        ("users", "Manage users"),
        ("version", "Show version information"),
        ("help", "Show this help message"),
        ("quit", "Exit the interactive session"),
    ]
    
    click.echo("Available commands:")
    for cmd, description in commands:
        click.echo(f"  {cmd:<20} {description}")
    click.echo()


def setup_shell(ctx: click.Context):
    """Set up the interactive shell with banner and initial info."""
    from litellm.proxy.common_utils.banner import show_banner
    
    show_banner()
    
    # Show server connection info
    base_url = ctx.obj.get("base_url")
    click.secho(f"Connected to LiteLLM server: {base_url}\n", fg="green")
    
    show_commands()


def handle_special_commands(user_input: str) -> bool:
    """Handle special commands like exit, help, clear. Returns True if command was handled."""
    if user_input.lower() in ["exit", "quit"]:
        click.echo("Goodbye!")
        return True
    elif user_input.lower() == "help":
        click.echo("")  # Add space before help
        show_commands()
        return True
    elif user_input.lower() == "clear":
        click.clear()
        from litellm.proxy.common_utils.banner import show_banner
        show_banner()
        show_commands()
        return True
    
    return False


def execute_command(user_input: str, ctx: click.Context):
    """Parse and execute a command."""
    # Parse command and arguments
    parts = user_input.split()
    command = parts[0]
    args = parts[1:] if len(parts) > 1 else []
    
    # Import cli here to avoid circular import
    from . import main
    cli = main.cli
    
    # Check if command exists
    if command not in cli.commands:
        click.echo(f"Unknown command: {command}")
        click.echo("Type 'help' to see available commands.")
        return
    
    # Execute the command
    try:
        # Create a new argument list for click to parse
        sys.argv = ["litellm-proxy"] + [command] + args
        
        # Get the command object and invoke it
        cmd = cli.commands[command]
        
        # Create a new context for the subcommand
        with ctx.scope():
            cmd.main(
                args,
                parent=ctx,
                standalone_mode=False
            )
            
    except click.ClickException as e:
        e.show()
    except click.Abort:
        click.echo("Command aborted.")
    except SystemExit:
        # Prevent the interactive shell from exiting on command errors
        pass
    except Exception as e:
        click.echo(f"Error executing command: {e}")


def interactive_shell(ctx: click.Context):
    """Run the interactive shell."""
    setup_shell(ctx)
    
    while True:
        try:
            # Add some space before the input box to ensure it's positioned well
            click.echo("\n")  # Extra spacing
            
            # Show styled prompt
            user_input = styled_prompt()
            
            if not user_input:
                continue
                
            # Handle special commands
            if handle_special_commands(user_input):
                if user_input.lower() in ["exit", "quit"]:
                    break
                continue
            
            # Execute regular commands
            execute_command(user_input, ctx)
                
        except (KeyboardInterrupt, EOFError):
            click.echo("\nGoodbye!")
            break
        except Exception as e:
            click.echo(f"Error: {e}") 