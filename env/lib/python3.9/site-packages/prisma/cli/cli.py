import os
import sys
import logging
import contextlib
from typing import List, Iterator, Optional, NoReturn

import click

from . import prisma
from .utils import error
from .custom import cli

from .. import _sync_http as http
from ..utils import DEBUG
from ..generator import Generator


__all__ = ('main', 'setup_logging')

log: logging.Logger = logging.getLogger(__name__)


# TODO: switch base cli to click as well to support autocomplete


def main(
    args: Optional[List[str]] = None,
    use_handler: bool = True,
    do_cleanup: bool = True,
) -> NoReturn:
    if args is None:
        args = sys.argv

    with setup_logging(use_handler), cleanup(do_cleanup):
        if len(args) > 1:
            if args[1] == 'py':
                cli.main(args[2:], prog_name='prisma py')
            else:
                sys.exit(prisma.run(args[1:]))
        else:
            if not os.environ.get('PRISMA_GENERATOR_INVOCATION'):
                error(
                    'This command is only intended to be invoked internally. '
                    'Please run the following instead:',
                    exit_=False,
                )
                click.echo('prisma <command>')
                click.echo('e.g.')
                click.echo('prisma generate')
                sys.exit(1)
            Generator.invoke()

    # mypy does not recognise sys.exit as a NoReturn for some reason
    raise SystemExit(0)


@contextlib.contextmanager
def setup_logging(use_handler: bool = True) -> Iterator[None]:
    handler = None
    logger = logging.getLogger()

    try:
        if DEBUG:
            logger.setLevel(logging.DEBUG)

            # the prisma CLI binary uses the DEBUG environment variable
            if os.environ.get('DEBUG') is None:
                os.environ['DEBUG'] = 'prisma:GeneratorProcess'
            else:
                log.debug('Not overriding the DEBUG environment variable.')
        else:
            logger.setLevel(logging.INFO)

        if use_handler:
            fmt = logging.Formatter(
                '[{levelname:<7}] {name}: {message}',
                style='{',
            )
            handler = logging.StreamHandler()
            handler.setFormatter(fmt)
            logger.addHandler(handler)

        yield
    finally:
        if use_handler and handler is not None:
            handler.close()
            logger.removeHandler(handler)


@contextlib.contextmanager
def cleanup(do_cleanup: bool = True) -> Iterator[None]:
    try:
        yield
    finally:
        if do_cleanup:
            http.client.close()


if __name__ == '__main__':
    main()
