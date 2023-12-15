import click

from .utils import PathlibPath
from .._types import FuncType


schema: FuncType = click.option(
    '--schema',
    type=PathlibPath(exists=True, dir_okay=False, resolve_path=True),
    help='The location of the Prisma schema file.',
    required=False,
)

watch: FuncType = click.option(
    '--watch',
    is_flag=True,
    default=False,
    required=False,
    help='Watch the Prisma schema and rerun after a change',
)

skip_generate: FuncType = click.option(
    '--skip-generate',
    is_flag=True,
    default=False,
    help='Skip triggering generators (e.g. Prisma Client Python)',
)
