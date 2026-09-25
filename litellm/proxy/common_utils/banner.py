import sys
from typing import Final

LITELLM_BANNER: Final = """   ██╗     ██╗████████╗███████╗██╗     ██╗     ███╗   ███╗
   ██║     ██║╚══██╔══╝██╔════╝██║     ██║     ████╗ ████║
   ██║     ██║   ██║   █████╗  ██║     ██║     ██╔████╔██║
   ██║     ██║   ██║   ██╔══╝  ██║     ██║     ██║╚██╔╝██║
   ███████╗██║   ██║   ███████╗███████╗███████╗██║ ╚═╝ ██║
   ╚══════╝╚═╝   ╚═╝   ╚══════╝╚══════╝╚══════╝╚═╝     ╚═╝"""

LITELLM_BANNER_ASCII: Final = r"""    _       _____  _______  ______  _       _       __  __
   | |     |_   _||__   __||  ____|| |     | |     |  \/  |
   | |       | |     | |   | |__   | |     | |     | \  / |
   | |       | |     | |   |  __|  | |     | |     | |\/| |
   | |____  _| |_    | |   | |____ | |____ | |____ | |  | |
   |______||_____|   |_|   |______||______||______||_|  |_|"""


def banner_for_encoding(encoding: str | None) -> str:
    """Return the banner variant that ``encoding`` can represent."""
    if encoding is None:
        return LITELLM_BANNER
    try:
        LITELLM_BANNER.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return LITELLM_BANNER_ASCII
    return LITELLM_BANNER


def _echo(text: str) -> None:
    try:
        import click

        click.echo(text)
    except ImportError:
        print(text)  # noqa: T201  # the banner is CLI output, and click is what we would print through


def show_banner() -> None:
    """Display the LiteLLM CLI banner. Never raises: the banner must not block startup."""
    encoding: Final[str | None] = getattr(sys.stdout, "encoding", None)
    try:
        _echo(f"\n{banner_for_encoding(encoding)}\n")
    except UnicodeEncodeError:
        pass
