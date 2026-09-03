"""Database management commands for the LiteLLM proxy CLI."""

import os
import subprocess
import sys
import sysconfig
from typing import Callable, Optional

import click

# The Prisma schema ships inside the litellm package itself (litellm/proxy/
# schema.prisma); PrismaManager resolves that directory. This is the same
# schema `db push` operates on, so `db generate` stays consistent with it.
from litellm.proxy.db.prisma_client import PrismaManager

# The prisma *binary* discovery and PATH-injection helpers live in
# litellm-proxy-extras. These are optional dependencies; the ImportError is
# handled at call time so `litellm` works without the proxy extras installed.
try:
    from litellm_proxy_extras.utils import (
        _get_prisma_command as _imported_prisma_command,
    )
    from litellm_proxy_extras.utils import (
        _get_prisma_env as _imported_prisma_env,
    )

    _PROXY_EXTRAS_AVAILABLE = True
except ImportError:
    _imported_prisma_command = None
    _imported_prisma_env = None
    _PROXY_EXTRAS_AVAILABLE = False

# Bind to explicitly typed module-level names. litellm-proxy-extras ships no
# py.typed marker, so its exports are untyped (Any) at this import boundary;
# the annotations pin the signatures back down. These are the names the tests
# patch. The `any-ok` markers acknowledge the unavoidable untyped third-party
# boundary (the annotation is the concrete type the rest of the file relies on).
_get_prisma_command: Optional[Callable[[], str]] = (
    _imported_prisma_command  # any-ok: untyped optional import from litellm-proxy-extras
)
_get_prisma_env: Optional[Callable[[], dict[str, str]]] = (
    _imported_prisma_env  # any-ok: untyped optional import from litellm-proxy-extras
)


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


def _get_generate_env() -> dict[str, str]:
    """Build the subprocess env for ``db generate``.

    Delegates to the shared ``_get_prisma_env()`` which now injects the venv
    scripts dir for all Prisma subprocesses (generate, db push, migrate deploy).
    Kept as a named wrapper for backward-compat with existing tests.
    """
    if _get_prisma_env is not None:
        result = _get_prisma_env()
        if result is not None:
            return result
    return os.environ.copy()


@click.group()
def db() -> None:
    """Database management commands."""


@db.command(name="generate")
def db_generate() -> None:
    """Generate the Prisma client using litellm's bundled schema.

    Runs: prisma generate --schema <path_to_schema.prisma>

    The schema is resolved from the litellm package (litellm/proxy/schema.prisma)
    via PrismaManager, which is the same schema `db push` operates on. This keeps
    generate consistent with the existing push path and closes the gap where there
    was no out-of-the-box `generate` step after a plain pip install.
    """
    if not _PROXY_EXTRAS_AVAILABLE:
        click.echo(
            "Error: litellm-proxy-extras is not installed. Run: pip install 'litellm[proxy]'",
            err=True,
        )
        raise SystemExit(1)

    prisma_dir = PrismaManager._get_prisma_dir()
    schema_path = os.path.join(prisma_dir, "schema.prisma")

    if not os.path.exists(schema_path):
        click.echo(
            f"Error: schema.prisma not found at {schema_path}. "
            "Your litellm-proxy-extras installation may be incomplete.",
            err=True,
        )
        raise SystemExit(1)

    if _get_prisma_command is None:
        click.echo(
            "Error: litellm-proxy-extras is not installed. Run: pip install 'litellm[proxy]'",
            err=True,
        )
        raise SystemExit(1)

    click.echo(f"Generating Prisma client from {schema_path} ...")
    command: list[str] = [
        _get_prisma_command(),
        "generate",
        "--schema",
        schema_path,
    ]
    try:
        subprocess.run(
            command,
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
