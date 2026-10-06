from collections.abc import Callable
from importlib.metadata import Distribution, PackageNotFoundError, distribution


def get_distribution(lookup: Callable[[str], Distribution] = distribution) -> Distribution:
    try:
        return lookup("litellm-core")
    except PackageNotFoundError:
        return lookup("litellm")


def get_distribution_name(lookup: Callable[[str], Distribution] = distribution) -> str:
    try:
        return get_distribution(lookup).metadata["Name"] or "litellm"
    except PackageNotFoundError:
        return "litellm"


def get_version(lookup: Callable[[str], Distribution] = distribution) -> str:
    try:
        return get_distribution(lookup).version
    except PackageNotFoundError:
        return "unknown"


version: str = get_version()  # rebind-ok: existing importers assign fallback versions
