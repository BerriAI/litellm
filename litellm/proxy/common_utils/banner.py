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
    try:
        import click

        click.echo(f"\n{LITELLM_BANNER}\n")
    except ImportError:
        print("\n")  # noqa: T201
    except UnicodeEncodeError:
        # Reaching here means the import succeeded and only the encode failed, so `click` is
        # bound and is the same channel the banner just went out on. Using it rather than
        # `print` keeps the line free of a T201 suppression, which the two ruff configs
        # disagree about: ruff.toml enables T20 and needs one, ruff-strict.toml does not
        # enable it and counts one as dead under RUF100.
        click.echo("\n   LiteLLM\n")
