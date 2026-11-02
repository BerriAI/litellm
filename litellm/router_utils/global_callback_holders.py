"""
The routers of this process that hold optional pre-call checks and routing callbacks, whatever core
serves them.

A router that retires hands the checks it vacates in its host's ``callbacks`` list to a surviving
router's same-keyed check (``routers_holding_global_callbacks``), and a router that drops a check at
runtime keeps it registered while another live router still has one of that type
(``live_check_owners``). The sets hold every router in the process, whatever core serves it, so
they live here, outside any one router core; both rules only ever consult the routers on the same
host (``on_host``), since another host's lists are not this host's.

A member of ``routers_holding_global_callbacks`` serves ``_global_callback_candidates()``,
``_core_adaptive_routers`` and ``_sync_adaptive_router_hooks()``; a member of ``live_check_owners``
serves ``optional_callbacks``.
"""

import threading
import weakref
from collections.abc import Iterable, Mapping, Sequence
from types import MethodType
from typing import TYPE_CHECKING, Final, Protocol, TypeVar

if TYPE_CHECKING:
    from litellm.integrations.custom_logger import CustomLogger


class GlobalCallbackHolder(Protocol):
    """A router whose callbacks a retiring router may hand its vacated checks to."""

    @property
    def _core_adaptive_routers(self) -> Mapping[str, object]: ...

    def _global_callback_candidates(self) -> "tuple[CustomLogger, ...]": ...

    def _sync_adaptive_router_hooks(self) -> None: ...


class CheckOwner(Protocol):
    """A live router with optional pre-call checks."""

    @property
    def optional_callbacks(self) -> "Sequence[object] | None": ...


_Router: Final = TypeVar("_Router")

routers_holding_global_callbacks: Final["weakref.WeakSet[GlobalCallbackHolder]"] = weakref.WeakSet()
routers_holding_global_callbacks_lock: Final = threading.Lock()
live_check_owners: Final["weakref.WeakSet[CheckOwner]"] = weakref.WeakSet()

OUTCOME_CALLBACK_LISTS: Final = (
    "success_callback",
    "_async_success_callback",
    "failure_callback",
    "_async_failure_callback",
)


def without_router_callbacks(
    callbacks: Iterable[object], owned: Iterable[object], method_owners: Iterable[object]
) -> list[object]:  # mutable-ok: a fresh callback list for the host
    """``callbacks`` without a router's own entries: the objects in ``owned`` and the bound methods of
    ``method_owners``. Everything other code registered stays."""
    owned_ids: Final = frozenset(map(id, owned))
    owner_ids: Final = frozenset(map(id, method_owners))
    return [
        callback
        for callback in callbacks
        if id(callback) not in owned_ids
        and not (isinstance(callback, MethodType) and id(callback.__self__) in owner_ids)
    ]


def on_host(routers: Iterable[_Router], host: object) -> tuple[_Router, ...]:
    """The routers among ``routers`` built on ``host``: a retiring router hands its vacated checks
    only to these, and a dropped check stays registered only for these."""
    return tuple(owner for owner in routers if getattr(owner, "_host", None) is host)


class _HeaderHookCallbacks(Protocol):
    """The host callback registration the adaptive response header hook goes through."""

    def get_custom_loggers_for_type(
        self, callback_type: "type[CustomLogger]"
    ) -> "list[CustomLogger]": ...  # mutable-ok: the manager's own return type

    def remove_callback_from_all_lists(self, obj: object, require_self: bool = False) -> None: ...

    def add_litellm_callback(self, callback: "CustomLogger") -> None: ...


def adaptive_routers_on_host(host: object) -> bool:
    """Whether a live router built on ``host`` (request-scoped ones included) has an adaptive router."""
    return any(getattr(owner, "_core_adaptive_routers", None) for owner in on_host(tuple(live_check_owners), host))


def sync_adaptive_response_header_hook(callbacks: _HeaderHookCallbacks, wanted: bool) -> None:
    """
    Exactly one ``AdaptiveRouterResponseHeaderHook`` on the host while ``wanted`` (some router on it has
    an adaptive router), none otherwise: the hook is stateless and shared by every router on the host,
    so it is added once, never per sync, and leaves with the last adaptive router.
    """
    from litellm.router_strategy.adaptive_router.hooks import AdaptiveRouterResponseHeaderHook

    present: Final = callbacks.get_custom_loggers_for_type(AdaptiveRouterResponseHeaderHook)
    for extra in present[1 if wanted else 0 :]:
        callbacks.remove_callback_from_all_lists(extra)
    if wanted and not present:
        callbacks.add_litellm_callback(AdaptiveRouterResponseHeaderHook())


__all__ = [
    "OUTCOME_CALLBACK_LISTS",
    "CheckOwner",
    "GlobalCallbackHolder",
    "adaptive_routers_on_host",
    "live_check_owners",
    "on_host",
    "routers_holding_global_callbacks",
    "routers_holding_global_callbacks_lock",
    "sync_adaptive_response_header_hook",
    "without_router_callbacks",
]
