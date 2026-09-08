from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

from prometheus_client import REGISTRY, CollectorRegistry


def clear_prometheus_registry() -> None:
    for collector in tuple(REGISTRY._collector_to_names):  # pyright: ignore[reportPrivateUsage]  # prometheus_client has no public collector enumeration
        REGISTRY.unregister(collector)


@contextmanager
def isolated_prometheus_registry(registry: CollectorRegistry = REGISTRY) -> Iterator[None]:
    original: Final = tuple(registry._collector_to_names)  # pyright: ignore[reportPrivateUsage]  # prometheus_client has no public collector enumeration
    for collector in original:
        registry.unregister(collector)
    try:
        yield
    finally:
        for collector in tuple(registry._collector_to_names):  # pyright: ignore[reportPrivateUsage]  # remove collectors created inside the isolation scope
            registry.unregister(collector)
        for collector in original:
            registry.register(collector)
