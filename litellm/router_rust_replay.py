"""The Rust core's live routers, whose deployments a price data reload re-registers in
``litellm.model_cost`` until they retire (``_live_routers`` / ``_replay_live_router_model_cost``)."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Final

from litellm.utils import add_live_deployment_replay

if TYPE_CHECKING:
    from litellm.router_rust_registry import HostRegistry

_live_registries: Final[weakref.WeakSet[HostRegistry]] = weakref.WeakSet()


def track_live_registry(registry: HostRegistry) -> None:
    _live_registries.add(registry)


def untrack_live_registry(registry: HostRegistry) -> None:
    _live_registries.discard(registry)


def _replay_live_registries() -> None:
    for registry in tuple(_live_registries):
        registry.replay_model_cost_registrations()


add_live_deployment_replay(_replay_live_registries)
