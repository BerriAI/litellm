from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from pydantic import JsonValue, TypeAdapter

from litellm._v2.routing._types import (
    Candidate,
    CooldownPolicy,
    ModelConfig,
    RetryPolicy,
    RouterConfig,
    SelectionContext,
    Selector,
    Snapshot,
    Strategy,
)

if TYPE_CHECKING:
    from litellm.rust_bridge._native import RouterHandle

__all__ = (
    "Candidate",
    "CooldownPolicy",
    "ModelConfig",
    "RetryPolicy",
    "RouterConfig",
    "RouterHandle",
    "SelectionContext",
    "Selector",
    "Snapshot",
    "Strategy",
    "aclose",
    "close",
    "create",
    "reconfigure",
    "snapshot",
)

_SNAPSHOT: Final = TypeAdapter(Snapshot)


def __getattr__(name: str) -> object:
    if name == "RouterHandle":
        from litellm.rust_bridge._native import RouterHandle

        return RouterHandle
    raise AttributeError(name)


def create(
    *,
    model_list: Sequence[ModelConfig | Mapping[str, JsonValue]],
    strategy: Strategy = "weighted",
    retries: int = 0,
    timeout: float | None = None,
    fallbacks: Mapping[str, tuple[str, ...]] | None = None,
    cooldown: CooldownPolicy | None = None,
    selector: Selector | None = None,
) -> RouterHandle:
    from litellm.rust_bridge._native import routing_create

    if selector is not None:
        raise NotImplementedError("custom selection is not implemented in the API scaffold")
    config: Final = RouterConfig.model_validate(
        {
            "model_list": model_list,
            "strategy": strategy,
            "retry": RetryPolicy(retries=retries, fallbacks={} if fallbacks is None else fallbacks),
            "timeout": timeout,
            "cooldown": cooldown,
        }
    )
    return routing_create(config.model_dump(mode="json"))


def snapshot(router: RouterHandle) -> Snapshot:
    from litellm.rust_bridge._native import routing_snapshot

    return _SNAPSHOT.validate_python(routing_snapshot(router))


def reconfigure(router: RouterHandle, *, config: RouterConfig) -> None:
    from litellm.rust_bridge._native import routing_reconfigure

    routing_reconfigure(router, config.model_dump(mode="json"))


def close(router: RouterHandle) -> None:
    from litellm.rust_bridge._native import routing_close

    routing_close(router)


async def aclose(router: RouterHandle) -> None:
    close(router)
