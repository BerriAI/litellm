from collections.abc import Callable
from importlib.metadata import Distribution, PackageNotFoundError, distribution
from typing import Final


def get_distribution(lookup: Callable[[str], Distribution] | None = None) -> Distribution:
    find: Final = distribution if lookup is None else lookup
    try:
        return find("litellm-core")
    except PackageNotFoundError:
        return find("litellm")


def get_distribution_name(lookup: Callable[[str], Distribution] | None = None) -> str:
    try:
        return get_distribution(lookup).metadata["Name"] or "litellm"
    except PackageNotFoundError:
        return "litellm"


def get_version(lookup: Callable[[str], Distribution] | None = None) -> str:
    try:
        return get_distribution(lookup).version
    except PackageNotFoundError:
        return "unknown"


version: str = get_version()  # rebind-ok: existing importers assign fallback versions
