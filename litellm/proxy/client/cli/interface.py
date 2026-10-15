# stdlib imports
import functools
import os
import re
from typing import Final, TypeAlias

# third party imports
import click

from litellm._logging import verbose_logger

from .commands.auth import cli_context_obj, context_secret_vault

ShellScope: TypeAlias = tuple[str, ...]
SHELL_WORD: Final = re.compile(r"""(?P<prefix>[\w-]+=)?(?P<quoted>"[^"]*"|'[^']*')(?P<rest>\S*)|\S+""")
OPENING_QUOTE: Final = re.compile(r"""(?:[\w-]+=)?["']""")


def styled_prompt(scope: ShellScope = ()):
    """Create a styled blue box prompt for user input, labelled with the group the shell is in."""

    # Get terminal height to ensure we have enough space
    try:
        terminal_height: Final = os.get_terminal_size().lines
        # Ensure we have at least 5 lines of space (for the box + some buffer)
        if terminal_height < 10:
            # If terminal is too small, just add some newlines to push content up
            click.echo("\n" * 3)
    except Exception as e:
        # Fallback if we can't get terminal size
        verbose_logger.debug("Error getting terminal size: %s", e)
        click.echo("\n" * 3)

    # ASCII box drawing characters
    top_left: Final = "+"
    top_right: Final = "+"
    bottom_left: Final = "+"
    bottom_right: Final = "+"
    horizontal: Final = "-"
    vertical: Final = "|"

    # Create the box with increased width
    width: Final = 80
    top_line: Final = top_left + horizontal * (width - 2) + top_right
    bottom_line: Final = bottom_left + horizontal * (width - 2) + bottom_right

    # Create styled elements
    left_border: Final = click.style(vertical, fg="blue", bold=True)
    right_border: Final = click.style(vertical, fg="blue", bold=True)
    prompt_text: Final = click.style(" ".join((*scope, "> ")), fg="cyan", bold=True)

    # Display the complete box structure first to reserve space
    click.echo(click.style(top_line, fg="blue", bold=True))

    # Create empty space in the box for input
    empty_space: Final = " " * (width - 4)
    click.echo(f"{left_border} {empty_space} {right_border}")

    # Display bottom border to complete the box
    click.echo(click.style(bottom_line, fg="blue", bold=True))

    # Now move cursor up to the input line and get input
    click.echo("\033[2A", nl=False)  # Move cursor up 2 lines
    click.echo(f"\r{left_border} {prompt_text}", nl=False)  # Position at start of input line

    try:
        # Get user input
        user_input: Final = input().strip()

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
    """Display available commands, minus any the operator chose to hide."""
    from .commands.agents import agent_commands
    from .commands.config import hidden_command_names

    commands = [
        ("login", "Authenticate with the LiteLLM proxy server"),
        ("logout", "Clear stored authentication"),
        ("whoami", "Show current authentication status"),
        ("models", "Manage and view model configurations"),
        ("credentials", "Manage API credentials"),
        ("chat", "Interactive streaming chat with models"),
        ("http", "Make HTTP requests to the proxy"),
        ("keys", "Manage API keys"),
        ("teams", "Manage teams and team assignments"),
        ("users", "Manage users"),
    ]
    commands += [(c.name, c.get_short_help_str()) for c in agent_commands()]
    commands += [
        ("version", "Show version information"),
        ("help", "Show this help message"),
        ("quit", "Exit the interactive session"),
    ]

    hidden: Final = hidden_command_names()

    click.echo("Available commands:")
    for cmd, description in commands:
        if cmd not in hidden:
            click.echo(f"  {cmd:<20} {description}")
    click.echo()


def setup_shell(ctx: click.Context):
    """Set up the interactive shell with banner and initial info."""
    from litellm.proxy.common_utils.banner import show_banner

    show_banner()

    # Show server connection info
    base_url: Final = cli_context_obj(ctx).get("base_url")
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


def _subgroup(group: click.Group | None, name: str) -> click.Group | None:
    if group is None:
        return None
    child: Final = group.commands.get(name)
    return child if isinstance(child, click.Group) else None


def _group_at(group: click.Group, path: ShellScope) -> click.Group | None:
    """The group `path` names under `group`, or None when it names a plain command or nothing"""
    return functools.reduce(_subgroup, path, group)


def split_shell_line(line: str) -> ShellScope:
    """Split on spaces and unwrap a quote that opens a word or follows its leading `name=`, keeping the rest as typed"""
    words: Final = tuple(SHELL_WORD.finditer(line))
    if any(word["quoted"] is None and OPENING_QUOTE.match(word.group()) for word in words):
        raise ValueError("No closing quotation")
    return tuple(
        word.group() if word["quoted"] is None else f"{word['prefix'] or ''}{word['quoted'][1:-1]}{word['rest']}"
        for word in words
    )


def resolve_command_path(words: ShellScope, scope: ShellScope, cli: click.Group) -> ShellScope:
    """The full command path a line names: from the top after a leading `lite`, else inside the shell's group"""
    if words[0] == "lite":
        return words[1:]
    scope_group: Final = _group_at(cli, scope)
    if scope and scope_group is not None and (words[0] in scope_group.commands or words[0].startswith("-")):
        return (*scope, *words)
    return words


def scope_after(path: ShellScope, scope: ShellScope, cli: click.Group) -> ShellScope:
    """A group typed alone becomes the scope, a command run inside the scope keeps it, anything else leaves it"""
    group: Final = _group_at(cli, path)
    if path and group is not None and not group.invoke_without_command:
        return path
    return scope if path[: len(scope)] == scope else ()


def execute_command(path: ShellScope, ctx: click.Context, cli: click.Group) -> None:
    """Run a resolved command path as `lite` would, so its usage and errors name `lite <command>`."""
    try:
        if path[0].startswith("-"):
            cli.main(
                ["--base-url", ctx.obj["base_url"], *path],
                prog_name="lite",
                obj={"secret_vault": context_secret_vault(ctx)},
                standalone_mode=False,
            )
            return
        with ctx.scope():
            cli.commands[path[0]].main(list(path[1:]), prog_name=path[0], parent=ctx, standalone_mode=False)

    except click.ClickException as e:
        e.show()
    except click.Abort:
        click.echo("Command aborted.")
    except SystemExit:
        # Prevent the interactive shell from exiting on command errors
        pass
    except Exception as e:
        click.echo(f"Error executing command: {e}")


def run_shell_line(user_input: str, scope: ShellScope, ctx: click.Context, cli: click.Group) -> ShellScope:
    """Run one line typed at the prompt and return the scope the next prompt starts in."""
    try:
        words: Final = split_shell_line(user_input)
    except ValueError as e:
        click.echo(f"Could not read that command: {e}")
        return scope
    if not words:
        return scope

    path: Final = resolve_command_path(words, scope, cli)
    if not path:
        show_commands()
        return ()
    if path[0] not in cli.commands and not path[0].startswith("-"):
        click.echo(f"Unknown command: {path[0]}")
        click.echo("Type 'help' to see available commands.")
        return scope

    execute_command(path, ctx, cli)
    return scope_after(path, scope, cli)


def interactive_shell(ctx: click.Context, cli: click.Group):
    """Run the interactive shell."""
    setup_shell(ctx)
    scope: ShellScope = ()  # rebind-ok: a group typed alone stays the scope for the prompts after it

    while True:
        try:
            # Add some space before the input box to ensure it's positioned well
            click.echo("\n")  # Extra spacing

            # Show styled prompt
            user_input = styled_prompt(scope)

            if not user_input:
                continue

            # Handle special commands
            if handle_special_commands(user_input):
                if user_input.lower() in ["exit", "quit"]:
                    break
                scope = ()
                continue

            scope = run_shell_line(user_input, scope, ctx, cli)

        except (KeyboardInterrupt, EOFError):
            click.echo("\nGoodbye!")
            break
        except Exception as e:
            click.echo(f"Error: {e}")
