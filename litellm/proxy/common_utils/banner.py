# ArcheOps ASCII banner
ARCHEOPS_BANNER = """   ___    ____   ____  _   _ _____  ___  ____   ____
  / _ \  / ___| |  _ \| | | | ____|/ _ \|  _ \ / ___|
 / /_\ \| |     | |_) | |_| |  _| | | | | |_) | |
/ /   \ \ |___  |  __/|  _  | |___| |_| |  __/| |___
/_/     \_\____| |_|   |_| |_|_____|\___/|_|    \____|"""


def show_banner():
    """Display the ArcheOps CLI banner."""
    try:
        import click

        click.echo(f"\n{ARCHEOPS_BANNER}\n")
    except ImportError:
        print("\n")  # noqa: T201
