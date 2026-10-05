from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Final, cast  # noqa: TID251  # native Router dynamic attribute dispatch

from starlette.types import ASGIApp, Receive, Scope, Send

import litellm
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.chatgpt.authenticator import prevent_device_login
from litellm.proxy._types import UserAPIKeyAuth
from litellm.router import Router
from litellm.types.utils import CallTypes, CallTypesLiteral


@dataclass(frozen=True, slots=True)
class OfferingServingSnapshot:
    router: Router
    available_models: frozenset[str]
    unavailable_models: Mapping[str, str]
    allowed_deployments: frozenset[str]


_PINNED_SNAPSHOT: Final[ContextVar[tuple[int, OfferingServingSnapshot] | None]] = ContextVar(
    "model_offering_snapshot", default=None
)
_MODEL_MUTATIONS: Final = frozenset(("set_model_list", "add_deployment", "upsert_deployment", "delete_deployment"))
_VIEW_MEMBERS: Final = frozenset(("publish_snapshot", "pin_snapshot", "serving_snapshot", "latest_snapshot"))


class OfferingRouterView(Router):
    def __init__(self, snapshot: OfferingServingSnapshot) -> None:
        self._offering_snapshot = snapshot

    def __getattribute__(self, name: str) -> object:
        if name.startswith("_offering_") or name in _VIEW_MEMBERS:
            return cast(object, object.__getattribute__(self, name))  # cast-ok: native Router dispatch
        if name in _MODEL_MUTATIONS:
            return cast(  # cast-ok: native Router dispatch
                object, object.__getattribute__(self, "_offering_reject_mutation")
            )
        snapshot: Final = self.serving_snapshot()
        if name == "model_names":
            return snapshot.available_models | frozenset(snapshot.unavailable_models)
        return cast(object, getattr(snapshot.router, name))  # cast-ok: dynamic native Router compatibility boundary

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("_offering_"):
            object.__setattr__(self, name, value)
            return
        setattr(self.serving_snapshot().router, name, value)

    def _offering_reject_mutation(self, *args: object, **kwargs: object) -> None:
        raise litellm.BadRequestError(
            message="Model definitions are owned by the external offering configuration", model="", llm_provider=""
        )

    def serving_snapshot(self) -> OfferingServingSnapshot:
        pinned: Final = _PINNED_SNAPSHOT.get()
        return pinned[1] if pinned is not None and pinned[0] == id(self) else self._offering_snapshot

    def latest_snapshot(self) -> OfferingServingSnapshot:
        return self._offering_snapshot

    def publish_snapshot(self, snapshot: OfferingServingSnapshot) -> None:
        self._offering_snapshot = snapshot

    @contextmanager
    def pin_snapshot(self) -> Generator[None]:
        token: Final = _PINNED_SNAPSHOT.set((id(self), self._offering_snapshot))
        try:
            with prevent_device_login():
                yield
        finally:
            _PINNED_SNAPSHOT.reset(token)


class OfferingSnapshotMiddleware:
    def __init__(self, app: ASGIApp, router_getter: Callable[[], Router | None]) -> None:
        self.app = app
        self.router_getter = router_getter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        router: Final = self.router_getter()
        if scope["type"] != "http" or not isinstance(router, OfferingRouterView):
            await self.app(scope, receive, send)
            return
        with router.pin_snapshot():
            await self.app(scope, receive, send)


class OfferingAccessGuard(CustomLogger):
    def __init__(self, router: OfferingRouterView) -> None:
        self.router = router

    async def async_filter_listed_models(
        self, user_api_key_dict: UserAPIKeyAuth, model_names: Sequence[str]
    ) -> Sequence[str]:
        snapshot: Final = self.router.serving_snapshot()
        return tuple(name for name in model_names if name in snapshot.available_models)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: Mapping[str, object],
        call_type: CallTypesLiteral,
    ) -> Exception | None:
        if any(
            data.get(name) is not None
            for name in (
                "api_base",
                "base_url",
                "api_key",
                "user_config",
                "custom_llm_provider",
                "extra_headers",
                "headers",
                "provider_specific_header",
            )
        ):
            return litellm.BadRequestError(
                message="External offering mode does not accept caller-defined supplier connections",
                model="",
                llm_provider="",
            )
        model: Final = data.get("model")
        if not isinstance(model, str):
            return None
        snapshot: Final = self.router.serving_snapshot()
        if model in snapshot.available_models:
            return None
        reason: Final = snapshot.unavailable_models.get(model)
        if reason is not None:
            return litellm.ServiceUnavailableError(
                message=f"Model '{model}' is unavailable: {reason}", model=model, llm_provider=""
            )
        return litellm.NotFoundError(
            message=f"Model '{model}' is not an enabled gateway offering", model=model, llm_provider=""
        )

    async def async_pre_call_deployment_hook(self, kwargs: Mapping[str, object], call_type: CallTypes | None) -> None:
        model: Final = kwargs.get("model")
        if not isinstance(model, str) or call_type not in (
            CallTypes.acompletion,
            CallTypes.completion,
            CallTypes.aresponses,
            CallTypes.responses,
            CallTypes.anthropic_messages,
            CallTypes.aembedding,
            CallTypes.embedding,
        ):
            return
        if model not in self.router.serving_snapshot().allowed_deployments:
            raise litellm.NotFoundError(
                message="The requested deployment is not an enabled gateway offering", model=model, llm_provider=""
            )
