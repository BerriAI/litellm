import shutil
import click

from ... import config
from ...cli.prisma import ensure_cached


@click.command('fetch', short_help='Download all required binaries.')
@click.option(
    '--force',
    is_flag=True,
    help='Download all binaries regardless of if they are already downloaded or not.',
)
def cli(force: bool) -> None:
    """Ensures all required binaries are available."""
    if force:
        shutil.rmtree(config.binary_cache_dir)

    directory = ensure_cached().cache_dir
    click.echo(
        f'Downloaded binaries to {click.style(str(directory), fg="green")}'
    )
