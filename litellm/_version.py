import importlib.metadata
from typing import Final


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except Exception:
        return None


_legacy_version: Final = _installed_version("litellm")
_core_version: Final = _installed_version("litellm-core")

if _legacy_version is not None and _core_version is not None:
    raise RuntimeError(
        "litellm and litellm-core are both installed and share the litellm namespace. "
        "Install them in separate environments."
    )

version = _legacy_version or _core_version or "unknown"
