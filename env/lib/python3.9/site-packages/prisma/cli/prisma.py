from __future__ import annotations

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Dict, NamedTuple

import click

from ._node import node, npm
from .. import config
from ..errors import PrismaError


log: logging.Logger = logging.getLogger(__name__)


def run(
    args: List[str],
    check: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> int:
    log.debug('Running prisma command with args: %s', args)

    default_env = {
        **os.environ,
        'PRISMA_HIDE_UPDATE_MESSAGE': 'true',
        'PRISMA_CLI_QUERY_ENGINE_TYPE': 'binary',
    }
    env = {**default_env, **env} if env is not None else default_env

    # TODO: ensure graceful termination
    entrypoint = ensure_cached().entrypoint
    process = node.run(
        str(entrypoint),
        *args,
        env=env,
        check=check,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    if args and args[0] in {'--help', '-h'}:
        click.echo(click.style('Python Commands\n', bold=True))
        click.echo(
            '  '
            + 'For Prisma Client Python commands run '
            + click.style('prisma py --help', bold=True)
        )

    return process.returncode


class CLICache(NamedTuple):
    cache_dir: Path
    entrypoint: Path


DEFAULT_PACKAGE_JSON: dict[str, Any] = {
    'name': 'prisma-binaries',
    'version': '1.0.0',
    'private': True,
    'description': 'Cache directory created by Prisma Client Python to store Prisma Engines',
    'main': 'node_modules/prisma/build/index.js',
    'author': 'RobertCraigie',
    'license': 'Apache-2.0',
}


def ensure_cached() -> CLICache:
    cache_dir = config.binary_cache_dir
    entrypoint = cache_dir / 'node_modules' / 'prisma' / 'build' / 'index.js'

    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)

    # We need to create a dummy `package.json` file so that `npm` doesn't try
    # and search for it elsewhere.
    #
    # If it finds a different `package.json` file then the `prisma` package
    # will be installed there instead of our cache directory.
    package = cache_dir / 'package.json'
    if not package.exists():
        package.write_text(json.dumps(DEFAULT_PACKAGE_JSON))

    if not entrypoint.exists():
        click.echo('Installing Prisma CLI')

        try:
            proc = npm.run(
                'install',
                f'prisma@{config.prisma_version}',
                cwd=config.binary_cache_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if proc.returncode != 0:
                click.echo(
                    f'An error ocurred while installing the Prisma CLI; npm install log: {proc.stdout.decode("utf-8")}'
                )
                proc.check_returncode()
        except Exception:
            # as we use the entrypoint existing to check whether or not we should run `npm install`
            # we need to make sure it doesn't exist if running `npm install` fails as it will otherwise
            # lead to a broken state, https://github.com/RobertCraigie/prisma-client-py/issues/705
            if entrypoint.exists():
                try:
                    entrypoint.unlink()
                except Exception:
                    pass
            raise

    if not entrypoint.exists():
        raise PrismaError(
            f'CLI installation appeared to complete but the expected entrypoint ({entrypoint}) could not be found.'
        )

    return CLICache(cache_dir=cache_dir, entrypoint=entrypoint)
