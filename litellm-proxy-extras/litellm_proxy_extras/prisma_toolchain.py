"""Prepare the Node toolchain the Prisma CLI needs, separately from migrations.

The Prisma CLI is a Node program. The first invocation inside a fresh
container installs a private Node runtime and npm-installs the CLI itself,
which can take minutes on a cold or slow machine. Sharing one timeout between
that one-time bootstrap and the migration commands makes a slow bootstrap
indistinguishable from a slow migration, so the bootstrap gets killed long
before it can finish.

A killed bootstrap does not correct itself. The installer leaves its cache
directory behind, and Prisma decides whether to install by testing that
directory for existence alone, so every later attempt skips the install and
then fails on a Node binary that was never written. Deleting a cache directory
that exists without a Node binary is what turns a killed bootstrap back into a
recoverable one.

Both budgets are overridable so an operator can widen them without a release:
``LITELLM_PRISMA_BOOTSTRAP_TIMEOUT`` for the toolchain install and
``LITELLM_PRISMA_COMMAND_TIMEOUT`` for every individual Prisma command.
"""

import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from litellm_proxy_extras._logging import logger

try:
    from prisma import config as prisma_config
except ImportError:
    prisma_config = None

PRISMA_COMMAND_TIMEOUT_ENV_VAR = "LITELLM_PRISMA_COMMAND_TIMEOUT"
PRISMA_BOOTSTRAP_TIMEOUT_ENV_VAR = "LITELLM_PRISMA_BOOTSTRAP_TIMEOUT"
NODEENV_CACHE_DIR_ENV_VAR = "PRISMA_NODEENV_CACHE_DIR"

DEFAULT_PRISMA_COMMAND_TIMEOUT = 60.0
DEFAULT_PRISMA_BOOTSTRAP_TIMEOUT = 600.0

BOOTSTRAP_ARG = "--version"


@dataclass(frozen=True)
class ToolchainBootstrap:
    """Outcome of preparing the Prisma toolchain."""

    healed_incomplete_cache: bool
    ready: bool


def _timeout_from_env(env_var: str, default: float) -> float:
    raw = os.getenv(env_var)
    if raw is None:
        return default
    try:
        seconds = float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number, falling back to %ss", env_var, raw, default
        )
        return default
    if not math.isfinite(seconds) or seconds <= 0:
        logger.warning(
            "%s=%r is not a finite positive number, falling back to %ss",
            env_var,
            raw,
            default,
        )
        return default
    return seconds


def prisma_command_timeout() -> float:
    """Seconds any single Prisma command may run for."""
    return _timeout_from_env(
        PRISMA_COMMAND_TIMEOUT_ENV_VAR, DEFAULT_PRISMA_COMMAND_TIMEOUT
    )


def prisma_bootstrap_timeout() -> float:
    """Seconds the one-time Node toolchain install may run for."""
    return _timeout_from_env(
        PRISMA_BOOTSTRAP_TIMEOUT_ENV_VAR, DEFAULT_PRISMA_BOOTSTRAP_TIMEOUT
    )


def nodeenv_cache_dir() -> Optional[Path]:
    """Where Prisma installs its private Node runtime, or None if unknowable."""
    override = os.getenv(NODEENV_CACHE_DIR_ENV_VAR)
    if override:
        return Path(override).absolute()
    if prisma_config is not None:
        try:
            return Path(prisma_config.nodeenv_cache_dir).absolute()
        except (OSError, ValueError) as e:
            logger.warning("Could not read the Prisma nodeenv cache dir: %s", e)
    try:
        return Path.home() / ".cache" / "prisma-python" / "nodeenv"
    except RuntimeError:
        logger.warning(
            "No resolvable home directory, cannot locate the Prisma nodeenv cache"
        )
        return None


def node_binary_path(cache_dir: Path) -> Path:
    """Path the Node binary occupies once the toolchain is fully installed."""
    if os.name == "nt":
        return cache_dir / "Scripts" / "node.exe"
    return cache_dir / "bin" / "node"


def heal_incomplete_nodeenv_cache() -> bool:
    """Delete a nodeenv cache directory left without a Node binary.

    Returns True when a half-installed toolchain was removed, so the next
    Prisma invocation reinstalls it instead of failing on a missing binary.
    """
    cache_dir = nodeenv_cache_dir()
    if cache_dir is None:
        return False
    try:
        if not cache_dir.is_dir() or node_binary_path(cache_dir).exists():
            return False
    except OSError as e:
        logger.warning("Could not inspect the Node toolchain at %s: %s", cache_dir, e)
        return False
    logger.warning(
        "Node toolchain at %s has no %s, so a previous install was interrupted. "
        "Removing it so it can be reinstalled.",
        cache_dir,
        node_binary_path(cache_dir).name,
    )
    try:
        shutil.rmtree(cache_dir)
    except OSError as e:
        logger.warning("Could not remove %s: %s", cache_dir, e)
        return False
    return True


def ensure_prisma_toolchain(
    prisma_command: str, prisma_env: dict[str, str]
) -> ToolchainBootstrap:
    """Install whatever the Prisma CLI needs to run, under its own timeout.

    Never raises. A toolchain that cannot be prepared is reported so the
    caller can go on and let the real Prisma command produce the real error.
    """
    healed = heal_incomplete_nodeenv_cache()
    timeout = prisma_bootstrap_timeout()
    logger.info("Preparing the Prisma CLI toolchain (timeout %ss)", timeout)
    try:
        subprocess.run(
            [prisma_command, BOOTSTRAP_ARG],
            timeout=timeout,
            check=True,
            capture_output=True,
            text=True,
            env=prisma_env,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "Preparing the Prisma CLI toolchain timed out after %ss. Raise %s "
            "if this machine needs longer to install it.",
            timeout,
            PRISMA_BOOTSTRAP_TIMEOUT_ENV_VAR,
        )
        return ToolchainBootstrap(healed_incomplete_cache=healed, ready=False)
    except subprocess.CalledProcessError as e:
        logger.warning("Preparing the Prisma CLI toolchain failed: %s", e.stderr)
        return ToolchainBootstrap(healed_incomplete_cache=healed, ready=False)
    except OSError as e:
        logger.warning("Could not run the Prisma CLI: %s", e)
        return ToolchainBootstrap(healed_incomplete_cache=healed, ready=False)
    logger.info("Prisma CLI toolchain ready")
    return ToolchainBootstrap(healed_incomplete_cache=healed, ready=True)
