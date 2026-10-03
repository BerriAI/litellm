import os
from importlib.metadata import PackageNotFoundError, version
from typing import Final

PROTOCOL_VERSION: Final = 4


def release_tag() -> str:
    if tag := os.environ.get("LITELLM_RELEASE_TAG", ""):
        return tag
    try:
        installed: Final = version("litellm")
    except PackageNotFoundError:
        return ""

    from packaging.version import Version

    parsed: Final = Version(installed)
    suffix: Final = f"-dev.{parsed.dev}" if parsed.dev is not None else f"-rc.{parsed.pre[1]}" if parsed.pre else ""
    return f"v{parsed.base_version}{suffix}"


def worker_image() -> str:
    override: Final = os.environ.get("LENS_WORKER_IMAGE", "")
    if override:
        return override
    tag: Final = release_tag()
    return f"ghcr.io/berriai/litellm-lens-worker:{tag}" if tag else "litellm-lens-worker:local"
