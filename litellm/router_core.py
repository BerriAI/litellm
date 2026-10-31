"""
``RouterCore``: the interface ``litellm.Router`` delegates to, and ``make_core``, which picks one.

``PythonCore`` (``litellm/router_python_core.py``) is today's routing logic. The Rust core
(litellm-rust/crates/router-core) implements the same operations. The core is picked once, when
the router is built, from its ``RouterConfig``.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

from litellm._logging import verbose_router_logger
from litellm.router_config import RouterConfig

if TYPE_CHECKING:
    from litellm.router_api import (
        ApplyResult,
        ChangeSet,
        RouterHealth,
        RouterRegistry,
        RouterSettings,
        RoutingObservers,
    )
    from litellm.router_python_core import PythonCore
    from litellm.router_python_host import PythonHost
    from litellm.types.router import CustomRoutingStrategyBase


@runtime_checkable
class RouterCore(Protocol):
    async def call(self, op: str, request: Mapping[str, object]) -> object: ...

    def registry(self) -> "RouterRegistry": ...

    def settings(self) -> "RouterSettings": ...

    def health(self) -> "RouterHealth": ...

    async def select(
        self, model: str, request_kwargs: dict[str, object] | None = None
    ) -> object: ...  # mutable-ok: the selection cores take a dict

    def apply(self, changes: "ChangeSet") -> "ApplyResult": ...

    def observe(self) -> "RoutingObservers": ...

    def register_plugin(self, plugin: "CustomRoutingStrategyBase") -> None: ...

    def retire(self) -> None: ...


def make_core(config: RouterConfig, host: "PythonHost", core_class: "type[PythonCore] | None" = None) -> RouterCore:
    """
    The Rust core when the route catalog's router rule allows it and the Rust core serves this
    config, else the Python core. Decided once, for the router's life.
    """
    from litellm.router_python_core import PythonCore
    from litellm.rust_bridge.catalog import Route, RouteContext, decision
    from litellm.rust_bridge.configuration import Decision

    python_class: Final = core_class or PythonCore
    chosen: Final = decision(RouteContext(Route.ROUTER))
    if chosen is not Decision.PYTHON:
        from litellm.router_rust_core import RustCore, RustCoreDeclined
        from litellm.rust_bridge._native import RustBridgeDeclined

        try:
            return RustCore.build(config, host, python_class)
        except (RustBridgeDeclined, RustCoreDeclined) as declined:
            # A config the Rust core cannot take keeps the router on Python; an error building the
            # Python core itself is the router's own error and is raised once, as it is.
            if chosen is Decision.RUST_REQUIRED:
                raise
            verbose_router_logger.info("Router served by the Python core: %s", declined)
            if isinstance(declined, RustCoreDeclined):
                return declined.python
    return python_class(**config.arguments, _host=host)  # pyright: ignore[reportArgumentType]  # the arguments were bound against PythonCore's own signature
