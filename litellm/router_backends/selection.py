from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final, TypeAlias, cast

from typing_extensions import assert_never

from litellm._logging import verbose_router_logger
from litellm.router_backends.python_router import PythonRouter
from litellm.router_backends.rust_router import RustRouter, RustRouterDeclined, build_rust_router
from litellm.router_backends.rust_support import bind_arguments
from litellm.rust_bridge.catalog import Route, RouteContext, Rules, decision
from litellm.rust_bridge.configuration import Decision

_PYTHON_ROUTER_TYPE: TypeAlias = Callable[..., PythonRouter]
_PYTHON_ROUTER: Final = cast(_PYTHON_ROUTER_TYPE, PythonRouter)  # cast-ok: Router(...) forwards untyped arguments


class RustRouterUnsupportedError(ValueError):
    pass


def select_backend(
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    rules: Rules | None = None,
) -> PythonRouter | RustRouter:
    selected: Final = decision(RouteContext(Route.ROUTER), rules)
    match selected:
        case Decision.PYTHON:
            return _PYTHON_ROUTER(*args, **kwargs)
        case Decision.RUST_WITH_FALLBACK | Decision.RUST_REQUIRED:
            built: Final = build_rust_router(bind_arguments(args, kwargs))
            if isinstance(built, RustRouter):
                return built
            return _declined(built, selected, args, kwargs)
        case _:
            assert_never(selected)


def _declined(
    declined: RustRouterDeclined,
    selected: Decision,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> PythonRouter:
    if selected is Decision.RUST_REQUIRED:
        raise RustRouterUnsupportedError(f"the Rust router is required but cannot serve this config: {declined.reason}")
    verbose_router_logger.info("Rust router declined, using the Python router: %s", declined.reason)
    return _PYTHON_ROUTER(*args, **kwargs)
