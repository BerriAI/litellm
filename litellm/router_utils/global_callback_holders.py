"""
The routers of this process that hold optional pre-call checks and routing callbacks, whatever core
serves them.

A router that retires hands the checks it vacates in its host's ``callbacks`` list to a surviving
router's same-keyed check (``routers_holding_global_callbacks``), and a router that drops a check at
runtime keeps it registered while another live router still has one of that type
(``live_check_owners``). Both rules look at every router in the process, so the sets live here,
outside any one router core.

A member of ``routers_holding_global_callbacks`` serves ``_global_callback_candidates()``,
``_core_adaptive_routers`` and ``_sync_adaptive_router_hooks()``; a member of ``live_check_owners``
serves ``optional_callbacks``.
"""

import threading
import weakref
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, Protocol

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


routers_holding_global_callbacks: Final["weakref.WeakSet[GlobalCallbackHolder]"] = weakref.WeakSet()
routers_holding_global_callbacks_lock: Final = threading.Lock()
live_check_owners: Final["weakref.WeakSet[CheckOwner]"] = weakref.WeakSet()

__all__ = [
    "CheckOwner",
    "GlobalCallbackHolder",
    "live_check_owners",
    "routers_holding_global_callbacks",
    "routers_holding_global_callbacks_lock",
]
