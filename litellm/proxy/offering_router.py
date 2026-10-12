from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, cast  # noqa: TID251  # native Router dynamic attribute dispatch

from pydantic import TypeAdapter
from starlette.types import ASGIApp, Receive, Scope, Send

import litellm
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.model_inventory import prevent_supplier_device_login
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.common_utils.model_listing_utils import TeamModelNameTranslator, alias_target, caller_alias_maps
from litellm.router import Router
from litellm.router_utils.common_utils import resolve_model_group_alias
from litellm.types.proxy.model_offerings import OpenRouterProviderPolicy
from litellm.types.utils import CallTypes, CallTypesLiteral


@dataclass(frozen=True, slots=True)
class OfferingServingSnapshot:
    router: Router
    available_models: frozenset[str]
    unavailable_models: Mapping[str, str]
    allowed_deployments: frozenset[str]
    allowed_deployment_ids: frozenset[str] = frozenset()
    provider_policies: Mapping[str, OpenRouterProviderPolicy] = field(default_factory=lambda: MappingProxyType({}))
    deployment_policies: Mapping[str, OpenRouterProviderPolicy] = field(default_factory=lambda: MappingProxyType({}))


_PINNED_SNAPSHOT: Final[ContextVar[tuple[int, OfferingServingSnapshot] | None]] = ContextVar(
    "model_offering_snapshot", default=None
)
_MODEL_MUTATIONS: Final = frozenset(("set_model_list", "add_deployment", "upsert_deployment", "delete_deployment"))
_VIEW_MEMBERS: Final = frozenset(("publish_snapshot", "pin_snapshot", "serving_snapshot", "latest_snapshot"))
_METADATA_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_POLICY_CALLS: Final = frozenset(("acompletion", "completion", "aresponses", "responses", "anthropic_messages"))
_ROUTING_OVERRIDES: Final = frozenset(
    (
        "provider",
        "models",
        "route",
        "fallbacks",
        "default_fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "model_list",
        "deployment_id",
        "specific_deployment",
        "model_id",
        "disable_fallbacks",
        "mock_response",
        "mock_tool_calls",
        "mock_timeout",
        "mock_provider_error",
    )
)


def _policy_override(data: Mapping[str, object]) -> bool:
    extra_body: Final = data.get("extra_body")
    return bool(_ROUTING_OVERRIDES.intersection(data)) or (
        isinstance(extra_body, Mapping)
        and bool((_ROUTING_OVERRIDES | frozenset(("model", "extra_body"))).intersection(extra_body))
    )


class OfferingRouterView(Router):
    """Request-pinned facade delegating native members to initialized Router instances.

    Router initialization belongs to the delegated instances. Reinitializing the
    facade would allocate unused clients and register duplicate global callbacks.
    """

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

    def _offering_reject_mutation(
        self,
        *args: object,
        **kwargs: object,  # kwargs-ok: blocks all native Router mutation signatures
    ) -> None:
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
            with prevent_supplier_device_login():
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
    def __init__(self, router: OfferingRouterView, general_settings: Mapping[str, object] | None = None) -> None:
        super().__init__()
        self.router = router
        self.general_settings: Mapping[str, object] = (
            general_settings if general_settings is not None else MappingProxyType({})
        )

    def _offering_name(self, name: str, auth: UserAPIKeyAuth) -> str:
        snapshot: Final = self.router.serving_snapshot()
        auth_metadata: Final = _METADATA_ADAPTER.validate_python(auth.model_dump())
        caller_target: Final = (
            alias_target(
                name,
                caller_alias_maps(
                    _METADATA_ADAPTER.validate_python(auth_metadata.get("aliases") or MappingProxyType({})),
                    _METADATA_ADAPTER.validate_python(auth_metadata.get("team_model_aliases") or MappingProxyType({})),
                    auth.team_id,
                    auth.team_id,
                ),
                snapshot.available_models | frozenset(snapshot.unavailable_models),
            )
            or name
        )
        target: Final = resolve_model_group_alias(snapshot.router.model_group_alias, caller_target) or caller_target
        rows: Final = TypeAdapter(tuple[Mapping[str, object], ...]).validate_python(
            snapshot.router.get_model_list() or ()
        )
        names: Final = tuple(name for name in (self._visible_model_name(row, auth) for row in rows) if name is not None)
        return TeamModelNameTranslator.resolve_public_name(target, names, snapshot.router, self.general_settings)

    @staticmethod
    def _visible_model_name(row: Mapping[str, object], auth: UserAPIKeyAuth) -> str | None:
        name: Final = row.get("model_name")
        if not isinstance(name, str):
            return None
        raw_model_info: Final = row.get("model_info")
        model_info: Final[Mapping[str, object]] = (
            _METADATA_ADAPTER.validate_python(raw_model_info)
            if isinstance(raw_model_info, Mapping)
            else MappingProxyType({})
        )
        return (
            name
            if auth.user_role == LitellmUserRoles.PROXY_ADMIN or model_info.get("team_id") in (None, auth.team_id)
            else None
        )

    async def async_filter_listed_models(
        self, user_api_key_dict: UserAPIKeyAuth, model_names: Sequence[str]
    ) -> Sequence[str]:
        snapshot: Final = self.router.serving_snapshot()
        return tuple(
            name for name in model_names if self._offering_name(name, user_api_key_dict) in snapshot.available_models
        )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: Mapping[str, object],
        call_type: CallTypesLiteral,
    ) -> dict[str, object] | Exception | None:  # mutable-ok: CustomLogger requires a fresh dict for request replacement
        if call_type in ("acreate_batch", "create_batch") or (
            call_type in ("acreate_file", "create_file") and data.get("purpose") == "batch"
        ):
            return litellm.BadRequestError(
                message="Native provider batches are unavailable in external offering mode because batch records cannot enforce offering selection",
                model="",
                llm_provider="",
            )
        extra_body: Final = data.get("extra_body")
        if isinstance(extra_body, Mapping) and "model" in extra_body:
            return litellm.BadRequestError(
                message="External offering mode does not accept supplier model overrides in extra_body",
                model="",
                llm_provider="",
            )
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
                "client",
            )
        ):
            return litellm.BadRequestError(
                message="External offering mode does not accept caller-defined supplier connections",
                model="",
                llm_provider="",
            )
        if any(name in data for name in ("specific_deployment", "deployment_id", "model_id", "model_list")):
            return litellm.BadRequestError(
                message="External offering mode does not accept caller-defined deployment selection",
                model="",
                llm_provider="",
            )
        model: Final = data.get("model")
        if not isinstance(model, str):
            return None
        snapshot: Final = self.router.serving_snapshot()
        offering: Final = self._offering_name(model, user_api_key_dict)
        if offering in snapshot.available_models:
            policy: Final = snapshot.provider_policies.get(offering)
            if policy is None:
                return None
            if call_type not in _POLICY_CALLS or _policy_override(data):
                return litellm.BadRequestError(
                    message="The offering provider policy does not accept caller routing overrides or this API",
                    model=model,
                    llm_provider="",
                )
            caller_body: Final = _METADATA_ADAPTER.validate_python(extra_body) if extra_body is not None else {}
            return {
                **data,
                "extra_body": {**caller_body, "provider": dict(policy.request_body())},
                "disable_fallbacks": True,
                "caching": False,
                "cache": {"no-cache": True, "no-store": True},
            }
        reason: Final = snapshot.unavailable_models.get(offering)
        if reason is not None:
            return litellm.ServiceUnavailableError(
                message=f"Model '{model}' is unavailable: {reason}", model=model, llm_provider=""
            )
        return litellm.NotFoundError(
            message=f"Model '{model}' is not an enabled gateway offering", model=model, llm_provider=""
        )

    async def async_pre_call_deployment_hook(
        self, kwargs: Mapping[str, object], call_type: CallTypes | None
    ) -> dict[str, object] | None:  # mutable-ok: CustomLogger requires a fresh dict for request replacement
        model: Final = kwargs.get("model")
        if not isinstance(model, str):
            return
        metadata: Final = _METADATA_ADAPTER.validate_python(
            kwargs.get("litellm_metadata") or kwargs.get("metadata") or {}
        )
        raw_model_info: Final = metadata.get("model_info")
        model_info: Final[Mapping[str, object]] = (
            _METADATA_ADAPTER.validate_python(raw_model_info)
            if isinstance(raw_model_info, Mapping)
            else MappingProxyType({})
        )
        deployment_id: Final = model_info.get("id")
        snapshot: Final = self.router.serving_snapshot()
        policy: Final = snapshot.deployment_policies.get(deployment_id) if isinstance(deployment_id, str) else None
        if policy is None and call_type not in (
            CallTypes.acompletion,
            CallTypes.completion,
            CallTypes.aresponses,
            CallTypes.responses,
            CallTypes.anthropic_messages,
            CallTypes.aembedding,
            CallTypes.embedding,
        ):
            return
        effective_model: Final = (
            model if model in snapshot.allowed_deployments else f"{kwargs.get('custom_llm_provider')}/{model}"
        )
        if effective_model not in snapshot.allowed_deployments or deployment_id not in snapshot.allowed_deployment_ids:
            raise litellm.NotFoundError(
                message="The requested deployment is not an enabled gateway offering", model=model, llm_provider=""
            )
        if policy is None:
            return
        extra_body: Final = kwargs.get("extra_body")
        if (
            call_type is None
            or call_type.value not in _POLICY_CALLS
            or any(kwargs.get(name) is not None for name in ("provider", "models", "route"))
            or not isinstance(extra_body, Mapping)
            or extra_body.get("provider") != policy.request_body()
            or any(name in extra_body for name in ("model", "models", "route", "extra_body"))
        ):
            raise litellm.BadRequestError(
                message="The request does not preserve the configured offering provider policy",
                model=model,
                llm_provider="",
            )
        return {**kwargs, "caching": False, "cache": {"no-cache": True, "no-store": True}}
