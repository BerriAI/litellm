from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

from litellm._catalog._load import LoadedCatalog
from litellm.litellm_core_utils.offline_mode import is_offline_mode
from litellm.rust_bridge._native import (
    FetchedRegistry,
    ModelRegistry,
    RegistryNotModified,
    RegistryUnavailable,
    fetch_once,
    fetch_with_retry,
)


@dataclass(frozen=True, slots=True)
class Remote:
    url: str


@dataclass(frozen=True, slots=True)
class File:
    path: str


@dataclass(frozen=True, slots=True)
class Bundled:
    pass


Source: TypeAlias = Remote | File | Bundled


@dataclass(frozen=True, slots=True)
class CatalogUnavailable:
    reason: str
    retryable: bool = False


def fetch(source: Source, *, max_attempts: int = 1, timeout: float = 5.0) -> LoadedCatalog | CatalogUnavailable:
    if isinstance(source, Bundled):
        return LoadedCatalog(ModelRegistry.bundled())
    if isinstance(source, Remote) and is_offline_mode() and not source.url.startswith("file://"):
        return CatalogUnavailable("Model cost map fetching is disabled in offline mode")
    url: Final = Path(source.path).absolute().as_uri() if isinstance(source, File) else source.url
    try:
        outcome: Final = (
            fetch_once(url, if_none_match=None, timeout_seconds=timeout)
            if max_attempts == 1
            else fetch_with_retry(url, if_none_match=None, timeout_seconds=timeout, max_attempts=max_attempts)
        )
    except ValueError as error:
        return CatalogUnavailable(str(error))
    match outcome:
        case FetchedRegistry():
            return LoadedCatalog(outcome.registry)
        case RegistryUnavailable():
            return CatalogUnavailable(outcome.reason, outcome.retryable)
        case RegistryNotModified():
            return CatalogUnavailable("Model cost map returned not-modified without an installed validator")
