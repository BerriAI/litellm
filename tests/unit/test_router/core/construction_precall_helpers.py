"""
Helpers for the construction / pre-call characterization tests.

``CallbackHost`` is a ``ScriptedHost`` whose callback manager also has
``remove_callback_from_list_by_object`` (litellm's ``LoggingCallbackManager`` has it; ``HostCallbacks``
does not), so the path that switches optional pre-call checks off at runtime can run.
"""

import copy
import gc
import uuid
from collections.abc import Generator, Mapping
from typing import Final

import litellm
from tests.unit.test_router.core.scripted_host import HostCallbacks, ScriptedHost, deployment


class RemovingHostCallbacks(HostCallbacks):
    def remove_callback_from_list_by_object(
        self,
        callback_list: object,
        obj: object,
        require_self: bool = True,
    ) -> None:
        """``LoggingCallbackManager.remove_callback_from_list_by_object``, on the host's lists."""
        if not isinstance(callback_list, list):
            return
        if require_self:
            doomed = [c for c in callback_list if getattr(c, "__self__", None) is obj]
        else:
            doomed = [c for c in callback_list if c == obj]
        for callback in doomed:
            callback_list.remove(callback)


class CallbackHost(ScriptedHost):
    __slots__ = ()

    @property
    def callbacks(self) -> RemovingHostCallbacks:  # pyright: ignore[reportIncompatibleVariableOverride]
        return RemovingHostCallbacks(self)


def callback_names(callbacks: list[object]) -> list[str]:  # mutable-ok: a fresh list
    """Each callback by name: a bound method by its function name, anything else by its class name."""
    names: Final[list[str]] = []
    for callback in callbacks:
        function = getattr(callback, "__func__", None)
        names.append(function.__name__ if function is not None else type(callback).__name__)
    return names


def host_callback_names(host: ScriptedHost) -> dict[str, list[str]]:  # mutable-ok: a fresh dict
    return {name: callback_names(callbacks) for name, callbacks in host.lists.items()}


def windowed(
    group: str, deployment_id: str, max_input_tokens: int, **params: object
) -> dict[str, object]:  # mutable-ok: a model_list entry
    """A deployment declaring ``max_input_tokens`` under a model name no other test uses: a deployment's
    model_info is written into the process-wide cost map under its id (and its schema fields, though not
    its token limits, under its model name)."""
    model_info: Final = {"max_input_tokens": max_input_tokens, **dict(params.pop("model_info", None) or {})}  # pyright: ignore[reportArgumentType]
    return deployment(
        group,
        deployment_id,
        model=f"openai/cw-{deployment_id}-{uuid.uuid4().hex[:8]}",
        model_info=model_info,
        **params,
    )


def collect_other_routers() -> None:
    """Drop unreachable routers from earlier tests: retire() and runtime pre-call-check changes look
    at every live router in the process, not only the ones on this host."""
    gc.collect()


def restoring_model_cost(*names: str) -> Generator[None, None, None]:
    """Undo what routers built meanwhile wrote into the process cost map: keys they added are removed
    and the named entries get their old value back."""
    added_before: Final = frozenset(litellm.model_cost)
    saved: Final = {name: copy.deepcopy(litellm.model_cost[name]) for name in names if name in litellm.model_cost}
    yield
    for key in frozenset(litellm.model_cost) - added_before:
        litellm.model_cost.pop(key, None)
    litellm.model_cost.update(saved)


__all__ = [
    "CallbackHost",
    "RemovingHostCallbacks",
    "callback_names",
    "collect_other_routers",
    "host_callback_names",
    "restoring_model_cost",
    "windowed",
]
