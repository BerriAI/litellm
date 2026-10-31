"""
``RouterCore``: the interface ``litellm.Router`` delegates to, and ``make_core``, which picks one.

``PythonCore`` (``litellm/router_python_core.py``) is today's routing logic. The Rust core
(litellm-rust/crates/router-core) implements the same operations. The core is picked once, when
the router is built, from its ``RouterConfig``.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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
    from litellm.router_python_core import PythonCore

    return (core_class or PythonCore)(**config.arguments, _host=host)  # pyright: ignore[reportArgumentType]  # the arguments were bound against PythonCore's own signature
