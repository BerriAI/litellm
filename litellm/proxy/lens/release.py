import os
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Final

PROTOCOL_VERSION: Final = 6


def release_tag() -> str:
    if "LITELLM_RELEASE_TAG" in os.environ:
        return os.environ["LITELLM_RELEASE_TAG"]
    try:
        installed: Final = distribution("litellm")
    except PackageNotFoundError:
        return ""
    if installed.read_text("direct_url.json") is not None:
        return ""
    if Path(str(installed.locate_file("litellm/proxy/lens/release.py"))).resolve() != Path(__file__).resolve():
        return ""

    from packaging.version import Version

    parsed: Final = Version(installed.version)
    suffix: Final = f"-dev.{parsed.dev}" if parsed.dev is not None else f"-rc.{parsed.pre[1]}" if parsed.pre else ""
    return f"v{parsed.base_version}{suffix}"


def worker_image() -> str:
    tag: Final = release_tag()
    if not tag:
        return ""
    override: Final = os.environ.get("LENS_WORKER_IMAGE", "")
    if override:
        return override
    package: Final = "litellm-lens-worker-dev" if tag.startswith("sha-") else "litellm-lens-worker"
    return f"ghcr.io/berriai/{package}:{tag}"
