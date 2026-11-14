"""Shared helpers for ``test_gaps_l11100*.py``: strategy-router marker deployments and a host whose
callback registration can also remove a callback from every list (what the adaptive router's hook
sync asks of litellm's ``LoggingCallbackManager``)."""

from collections.abc import Callable, Mapping
from typing import Final

from litellm.router_core import RouterCore
from tests.unit.test_router.core.scripted_host import HostCallbacks, ScriptedHost

MakeCore = Callable[..., RouterCore]

TIERS: Final = {"SIMPLE": "small", "MEDIUM": "big", "COMPLEX": "big", "REASONING": "big"}


class RemovingHostCallbacks(HostCallbacks):
    """``HostCallbacks`` plus ``remove_callback_from_all_lists``, as litellm's callback manager has it."""

    def __init__(self, host: "RemovingHost") -> None:
        super().__init__(host)
        self._owner: Final = host

    def remove_callback_from_all_lists(self, obj: object, require_self: bool = False) -> None:
        del require_self
        for name, callbacks in self._owner.lists.items():
            self._owner.lists[name] = [callback for callback in callbacks if callback is not obj]


class RemovingHost(ScriptedHost):
    """A ``ScriptedHost`` whose callbacks can be removed from every list."""

    __slots__ = ()

    @property
    def callbacks(self) -> RemovingHostCallbacks:  # pyright: ignore[reportIncompatibleVariableOverride]  # same methods plus one
        return RemovingHostCallbacks(self)


def complexity_marker(
    name: str = "smart",
    deployment_id: str = "marker",
    *,
    config: Mapping[str, object] | None = None,
    **params: object,  # kwargs-ok: extra marker litellm_params
) -> dict[str, object]:  # mutable-ok: a model_list entry
    """An ``auto_router/complexity_router`` marker with the shipped tiers and default model ``big``."""
    return {
        "model_name": name,
        "litellm_params": {
            "model": "auto_router/complexity_router",
            "complexity_router_config": dict(config) if config is not None else {"tiers": dict(TIERS)},
            "complexity_router_default_model": "big",
            **params,
        },
        "model_info": {"id": deployment_id},
    }


def adaptive_marker(
    name: str, deployment_id: str, available_models: list[str], **params: object
) -> dict[str, object]:  # mutable-ok: a model_list entry  # kwargs-ok: extra marker litellm_params
    """An ``auto_router/adaptive_router`` marker over ``available_models``."""
    return {
        "model_name": name,
        "litellm_params": {
            "model": "auto_router/adaptive_router",
            "adaptive_router_config": {"available_models": available_models},
            **params,
        },
        "model_info": {"id": deployment_id},
    }


def marker(
    name: str, deployment_id: str, model: str, **params: object
) -> dict[str, object]:  # mutable-ok: a model_list entry  # kwargs-ok: marker litellm_params
    """Any strategy-router marker: ``model`` is its ``auto_router/...`` litellm model."""
    return {"model_name": name, "litellm_params": {"model": model, **params}, "model_info": {"id": deployment_id}}


def registered_tags(core: RouterCore, kind: str) -> dict[str, list[tuple[str, ...]]]:  # mutable-ok: a fresh dict
    """``{router name: [tags of each registered strategy]}`` for a pre-routing strategy registry."""
    registry: Final = core.registry().plugins(kind)  # pyright: ignore[reportCallIssue, reportArgumentType]  # one of the strategy kinds
    assert isinstance(registry, dict)
    return {name: [entry.tags for entry in entries] for name, entries in registry.items()}  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # TaggedPreRoutingStrategy lists


def model_ids(core: RouterCore) -> list[str]:  # mutable-ok: a fresh list
    """The ``model_info.id`` of every deployment on the model list, in order."""
    return [str(entry["model_info"]["id"]) for entry in core.registry().all()]  # pyright: ignore[reportAny]  # model_list rows


class ModelsEndpoint:
    """A stand-in for the ``AsyncHTTPHandler`` that model-info discovery reads ``/v1/models`` with:
    answers ``body`` (or raises ``error``) and records each request's url and headers."""

    def __init__(self, body: Mapping[str, object] | None = None, error: BaseException | None = None) -> None:
        self.body: Final = body or {"data": []}
        self.error: Final = error
        self.requests: Final[list[tuple[str, dict[str, str]]]] = []

    async def get(
        self, url: str, headers: dict[str, str] | None = None, **_: object
    ) -> object:  # kwargs-ok: httpx options
        import json

        import httpx

        self.requests.append((url, dict(headers or {})))
        if self.error is not None:
            raise self.error
        return httpx.Response(200, content=json.dumps(self.body).encode(), request=httpx.Request("GET", url))
