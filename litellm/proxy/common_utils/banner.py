from typing import Final

# LiteLLM ASCII banner
LITELLM_BANNER: Final = """   ██╗     ██╗████████╗███████╗██╗     ██╗     ███╗   ███╗
   ██║     ██║╚══██╔══╝██╔════╝██║     ██║     ████╗ ████║
   ██║     ██║   ██║   █████╗  ██║     ██║     ██╔████╔██║
   ██║     ██║   ██║   ██╔══╝  ██║     ██║     ██║╚██╔╝██║
   ███████╗██║   ██║   ███████╗███████╗███████╗██║ ╚═╝ ██║
   ╚══════╝╚═╝   ╚═╝   ╚══════╝╚══════╝╚══════╝╚═╝     ╚═╝"""


def show_banner():
    """Display the LiteLLM CLI banner."""
    import litellm
    from litellm._logging import print_startup_message

    if litellm.json_logs:
        print_startup_message(LITELLM_BANNER)
        return
    try:
        import click

        click.echo(f"\n{LITELLM_BANNER}\n")
    except ImportError:
        print("\n")  # noqa: T201
