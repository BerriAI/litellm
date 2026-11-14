"""Build the Prisma Python client from the schema LiteLLM ships, for installs that have none.

Shipped images bake the client at build time. A pip install has only the Prisma package, and nothing in
proxy startup generates the client, so the proxy cannot import it.
"""

import importlib
import importlib.util
import os
import subprocess
import sys
import sysconfig
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias

from litellm._logging import verbose_proxy_logger

PRISMA_PACKAGE: Final = "prisma"
SHIPPED_SCHEMA_PATH: Final = Path(__file__).resolve().parent.parent / "schema.prisma"


class PrismaRunner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        env: Mapping[str, str],
        stdout: int | None,
        stderr: int | None,
    ) -> "subprocess.CompletedProcess[str]": ...


@dataclass(frozen=True, slots=True)
class ClientPresent:
    pass


@dataclass(frozen=True, slots=True)
class ClientGenerated:
    pass


@dataclass(frozen=True, slots=True)
class ClientGenerationFailed:
    reason: str


PrismaClientStatus: TypeAlias = ClientPresent | ClientGenerated | ClientGenerationFailed


def generate_command(schema_path: Path) -> tuple[str, ...]:
    return ("prisma", "generate", "--schema", str(schema_path))


def manual_generate_hint(schema_path: Path = SHIPPED_SCHEMA_PATH) -> str:
    return " ".join(generate_command(schema_path))


def env_with_scripts_dir(env: Mapping[str, str], scripts_dir: str) -> Mapping[str, str]:
    """Put the interpreter's scripts dir first on PATH, so Prisma finds `prisma-client-py` in an unactivated venv."""
    path: Final = env.get("PATH")
    return MappingProxyType({**env, "PATH": os.pathsep.join((scripts_dir, path)) if path else scripts_dir})


def _client_is_generated(package_dir: Path) -> bool:
    return (package_dir / "client.py").is_file()


def _refresh_loaded_package(package_name: str) -> ClientGenerationFailed | None:
    """Re-run an already imported package's `__init__`, which only exposes the client when it was generated first."""
    loaded: Final = sys.modules.get(package_name)
    if loaded is None:
        return None
    importlib.invalidate_caches()
    try:
        importlib.reload(loaded)
    except ImportError as e:
        return ClientGenerationFailed(f"the client was generated but could not be loaded: {e}")
    return None


def ensure_prisma_client(
    *,
    package_dir: Path | None,
    schema_path: Path,
    run: PrismaRunner,
    timeout: float,
    env: Mapping[str, str],
    package_name: str = PRISMA_PACKAGE,
) -> PrismaClientStatus:
    if package_dir is None:
        return ClientGenerationFailed("the prisma package is not installed")
    if _client_is_generated(package_dir):
        return ClientPresent()
    try:
        run(generate_command(schema_path), timeout=timeout, env=env, stdout=None, stderr=None)
    except subprocess.TimeoutExpired:
        return ClientGenerationFailed(f"prisma generate timed out after {timeout}s")
    except subprocess.CalledProcessError as e:
        return ClientGenerationFailed(f"prisma generate exited {e.returncode}, see its output above")
    except OSError as e:
        return ClientGenerationFailed(f"could not run prisma generate: {e}")
    if not _client_is_generated(package_dir):
        return ClientGenerationFailed(f"prisma generate finished but wrote no client into {package_dir}")
    return _refresh_loaded_package(package_name) or ClientGenerated()


def _installed_package_dir(package_name: str) -> Path | None:
    spec: Final = importlib.util.find_spec(package_name)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def ensure_shipped_prisma_client() -> None:
    """Generate the client from LiteLLM's own schema if this install has none, and log the outcome."""
    from litellm_proxy_extras import prisma_toolchain  # pyright: ignore[reportUnknownVariableType]  # untyped in CI env

    status: Final = ensure_prisma_client(
        package_dir=_installed_package_dir(PRISMA_PACKAGE),
        schema_path=SHIPPED_SCHEMA_PATH,
        run=prisma_toolchain.run_prisma,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped in CI env
        timeout=prisma_toolchain.prisma_bootstrap_timeout(),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped in CI env
        env=env_with_scripts_dir(os.environ, sysconfig.get_path("scripts")),
    )
    match status:
        case ClientPresent():
            pass
        case ClientGenerated():
            verbose_proxy_logger.info("Generated the Prisma client from %s", SHIPPED_SCHEMA_PATH)
        case ClientGenerationFailed(reason=reason):
            verbose_proxy_logger.warning(
                "Could not generate the Prisma client (%s). Run '%s' to generate it.",
                reason,
                manual_generate_hint(),
            )
