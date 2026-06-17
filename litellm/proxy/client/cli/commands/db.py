"""Database management commands for the LiteLLM proxy CLI."""

import os
import subprocess
import sys
import sysconfig

import click

# Import Prisma helpers from litellm-proxy-extras at module level.
# These are optional dependencies; the ImportError is handled at call time.
try:
    from litellm_proxy_extras.utils import (
        ProxyExtrasDBManager,
        _get_prisma_command,
        _get_prisma_env,
    )

    _PROXY_EXTRAS_AVAILABLE = True
except ImportError:
    _PROXY_EXTRAS_AVAILABLE = False
    ProxyExtrasDBManager = None  # type: ignore[assignment]
    _get_prisma_command = None  # type: ignore[assignment]
    _get_prisma_env = None  # type: ignore[assignment]


def _get_venv_scripts_dir() -> str:
    """Return the directory that holds console scripts for the running interpreter.

    The prisma engine shells out to the ``prisma-client-py`` console script via
    ``/bin/sh``, which only finds it when that directory is on PATH. In an
    *activated* venv it is; when ``litellm-proxy`` is invoked by absolute path
    without activation, it is not. We derive the directory from the current
    interpreter so the caller never has to activate the venv.

    ``sysconfig.get_path("scripts")`` is the authoritative answer; we fall back
    to ``dirname(sys.executable)`` if it is empty (defensive: should not happen
    on a normal CPython install).
    """
    scripts_dir = sysconfig.get_path("scripts")
    if not scripts_dir:
        scripts_dir = os.path.dirname(os.path.abspath(sys.executable))
    return scripts_dir


def _get_generate_env() -> dict:
    """Build the subprocess env for ``db generate``.

    Delegates to the shared ``_get_prisma_env()`` which now injects the venv
    scripts dir for all Prisma subprocesses (generate, db push, migrate deploy).
    Kept as a named wrapper for backward-compat with existing tests.
    """
    if _get_prisma_env:
        result = _get_prisma_env()
        if result is not None:
            return result
    return os.environ.copy()


@click.group()
def db() -> None:
    """Database management commands."""


@db.command(name="generate")
def db_generate() -> None:
    """Generate the Prisma client using the schema bundled with litellm-proxy-extras.

    Runs: prisma generate --schema <path_to_schema.prisma>

    I resolve the schema path from the installed litellm-proxy-extras package,
    so you never need to know internal site-packages paths. This fixes the gap
    where `migrate deploy` and `db push` already use the bundled schema but
    there was no equivalent for the generate step.
    """
    if not _PROXY_EXTRAS_AVAILABLE:
        click.echo(
            "Error: litellm-proxy-extras is not installed. "
            "Run: pip install 'litellm[proxy]'",
            err=True,
        )
        raise SystemExit(1)

    prisma_dir = ProxyExtrasDBManager._get_prisma_dir()
    schema_path = os.path.join(prisma_dir, "schema.prisma")

    if not os.path.exists(schema_path):
        click.echo(
            f"Error: schema.prisma not found at {schema_path}. "
            "Your litellm-proxy-extras installation may be incomplete.",
            err=True,
        )
        raise SystemExit(1)

    click.echo(f"Generating Prisma client from {schema_path} ...")
    try:
        subprocess.run(
            [_get_prisma_command(), "generate", "--schema", schema_path],
            check=True,
            env=_get_generate_env(),
        )
        click.echo("Prisma client generated successfully.")
    except subprocess.CalledProcessError as e:
        click.echo(
            f"Error: prisma generate failed (exit {e.returncode}).",
            err=True,
        )
        raise SystemExit(e.returncode)
