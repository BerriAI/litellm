"""Config loader for managed_agents.

Reads the `general_settings.managed_agents` yaml block at startup, validates
that each declared dockerfile path exists and is readable, hashes the
dockerfile + context contents, and builds an in-memory ``DOCKERFILE_REGISTRY``.

The template-create endpoint validates incoming ``dockerfile_id`` values
against this registry.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.proxy.managed_agents_endpoints.fargate.registry import (
    compute_dockerfile_hash,
)
from litellm.proxy.managed_agents_endpoints.types import ManagedAgentsConfig

_DOCKERFILE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

_BUILTIN_HARNESSES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "harnesses"
)


def _builtin_dockerfile_path(dockerfile_id: str) -> Optional[str]:
    """Return path to bundled harnesses/<id>/Dockerfile if it exists."""
    candidate = os.path.join(_BUILTIN_HARNESSES_DIR, dockerfile_id, "Dockerfile")
    return candidate if os.path.isfile(candidate) else None


@dataclass(frozen=True)
class DockerfileEntry:
    dockerfile_id: str
    path: str  # absolute path to Dockerfile
    context_dir: str  # absolute path to context dir (default: dockerfile dir)
    container_port: int
    content_hash: str  # sha256 of dockerfile + context
    build_platform: str  # e.g. "linux/amd64", "linux/arm64"


# Module-level state. Populated by ``initialize`` at proxy startup.
DOCKERFILE_REGISTRY: Dict[str, DockerfileEntry] = {}
MANAGED_AGENTS_CONFIG: Optional[ManagedAgentsConfig] = None


def _validate_dockerfile_id(dockerfile_id: str) -> None:
    """Reject ids with chars that aren't safe for ECR repo / task family names."""
    if not _DOCKERFILE_ID_RE.match(dockerfile_id):
        raise ValueError(
            f"dockerfile_id '{dockerfile_id}' invalid: only [a-zA-Z0-9_-] allowed"
        )


def _resolve_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def load_managed_agents_config(
    general_settings: dict,
) -> Optional[ManagedAgentsConfig]:
    """Parse the ``managed_agents`` block.

    Returns ``None`` if the block is absent or ``enabled`` is false. Raises
    ``ValueError`` on invalid shape (forbidden keys, wrong types, etc.).
    """
    raw = general_settings.get("managed_agents")
    if not raw:
        return None

    if not isinstance(raw, dict):
        raise ValueError(
            f"managed_agents config must be a mapping, got {type(raw).__name__}"
        )

    try:
        config = ManagedAgentsConfig(**raw)
    except ValidationError as e:
        raise ValueError(f"managed_agents config invalid: {e}") from e

    if not config.enabled:
        return None

    return config


def build_dockerfile_registry(
    config: ManagedAgentsConfig,
) -> Dict[str, DockerfileEntry]:
    """Validate each dockerfile path exists + is readable, compute hashes.

    Raises ``ValueError`` if a ``dockerfile_id`` contains disallowed chars and
    ``FileNotFoundError`` if any declared path is missing or unreadable.
    """
    registry: Dict[str, DockerfileEntry] = {}

    for dockerfile_id, dockerfile_cfg in config.dockerfiles.items():
        _validate_dockerfile_id(dockerfile_id)

        if dockerfile_cfg.path:
            abs_path = _resolve_path(dockerfile_cfg.path)
        else:
            builtin = _builtin_dockerfile_path(dockerfile_id)
            if builtin is None:
                raise FileNotFoundError(
                    f"managed_agents: dockerfile id '{dockerfile_id}' has no "
                    f"path set and no bundled harnesses/{dockerfile_id}/"
                    f"Dockerfile exists"
                )
            abs_path = builtin
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(
                f"managed_agents: dockerfile path for id '{dockerfile_id}' "
                f"does not exist: {abs_path}"
            )
        if not os.access(abs_path, os.R_OK):
            raise FileNotFoundError(
                f"managed_agents: dockerfile path for id '{dockerfile_id}' "
                f"is not readable: {abs_path}"
            )

        context_dir = os.path.dirname(abs_path)
        content_hash = compute_dockerfile_hash(abs_path, context_dir)

        entry = DockerfileEntry(
            dockerfile_id=dockerfile_id,
            path=abs_path,
            context_dir=context_dir,
            container_port=dockerfile_cfg.container_port,
            content_hash=content_hash,
            build_platform=dockerfile_cfg.build_platform,
        )
        registry[dockerfile_id] = entry

        verbose_proxy_logger.info(
            "managed_agents: loaded dockerfile id=%s path=%s hash=%s",
            dockerfile_id,
            abs_path,
            content_hash[:12],
        )

    return registry


def initialize(general_settings: dict) -> None:
    """Populate module-level ``DOCKERFILE_REGISTRY`` + ``MANAGED_AGENTS_CONFIG``.

    Called from ``proxy_server`` startup. No-op if managed_agents is absent or
    disabled. Refuses to start (raises) on any path missing — fail-fast at
    boot so bad config never makes it into a running proxy.
    """
    global DOCKERFILE_REGISTRY, MANAGED_AGENTS_CONFIG

    config = load_managed_agents_config(general_settings)
    if config is None:
        DOCKERFILE_REGISTRY = {}
        MANAGED_AGENTS_CONFIG = None
        return

    registry = build_dockerfile_registry(config)

    DOCKERFILE_REGISTRY = registry
    MANAGED_AGENTS_CONFIG = config

    verbose_proxy_logger.info(
        "managed_agents: registry initialized with %d dockerfile(s)",
        len(registry),
    )


def get_dockerfile(dockerfile_id: str) -> DockerfileEntry:
    """Lookup helper for endpoints. Raises ``KeyError`` if id not in registry."""
    if dockerfile_id not in DOCKERFILE_REGISTRY:
        raise KeyError(dockerfile_id)
    return DOCKERFILE_REGISTRY[dockerfile_id]


def list_dockerfiles() -> List[DockerfileEntry]:
    """Snapshot of registry, sorted by id."""
    return [DOCKERFILE_REGISTRY[k] for k in sorted(DOCKERFILE_REGISTRY.keys())]
