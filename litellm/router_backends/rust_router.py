from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias, TypeVar, cast
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict

import litellm
from litellm import constants
from litellm._logging import verbose_router_logger
from litellm.constants import RUNTIME_UPDATABLE_ROUTER_SETTINGS
from litellm.router_backends.python_router import PythonRouter
from litellm.router_backends.rust_call import (
    AttemptRouter,
    Kwargs,
    Operation,
    RoutedCall,
    as_mapping,
    as_sequence,
    metadata_key_for,
)
from litellm.router_backends.rust_support import unsupported_reason
from litellm.router_utils import clientside_credential_handler
from litellm.router_utils.cooldown_handlers import (
    _is_allowed_fails_set_on_router,  # pyright: ignore[reportPrivateUsage]  # the rule PythonRouter's cooldowns apply
)
from litellm.rust_bridge.bindings import NativeBinding
from litellm.types.router import Deployment, RetryPolicy


@dataclass(frozen=True, slots=True)
class RustRouterDeclined:
    reason: str


class RustRouterUnsupportedError(ValueError):
    pass


FallBack: TypeAlias = Callable[[PythonRouter], None]
_T = TypeVar("_T")


class NativeRouter(Protocol):
    def route(self, call: Mapping[str, object], driver: RoutedCall, asynchronous: bool) -> object: ...

    def replace(self, deployments: Sequence[Mapping[str, object]], settings: Mapping[str, object]) -> None: ...


NativeRouterType: TypeAlias = Callable[..., NativeRouter]


def _native_router_type(value: object) -> NativeRouterType | None:
    return cast(NativeRouterType, value) if isinstance(value, type) else None  # cast-ok: the extension's Router class


NATIVE_ROUTER: Final = NativeBinding("Router", validate=_native_router_type)

_REDIS_ARGUMENTS: Final = ("redis_url", "redis_host", "redis_port", "redis_password", "redis_db")
_UNSUPPORTED_REQUEST_KWARGS: Final = (
    "priority",
    "specific_deployment",
    "model_group_retry_policy",
    "include_fallback_errors",
    "_router_weights",
)
_MOCK_FAILURES: Final = (
    ("mock_testing_fallbacks", "fallbacks"),
    ("mock_testing_context_fallbacks", "context_window_fallbacks"),
    ("mock_testing_content_policy_fallbacks", "content_policy_fallbacks"),
    ("mock_testing_rate_limit_error", "rate_limit"),
)
_WEB_SEARCH_TOOLS: Final = frozenset({"web_search", "web_search_preview"})


class _ProjectedDeployment(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    model_name: str
    model: str
    weight: float | None = None
    cooldown_time: int | float | None = None
    num_retries: int | None = None
    supports_web_search: bool | None = None


class NormalizerView(AttemptRouter, Protocol):
    """The `PythonRouter` attributes the Rust backend projects or forwards, typed."""

    model_list: list[dict[str, object]]  # mutable-ok: PythonRouter's read surface
    model_names: set[str]  # mutable-ok: PythonRouter's read surface
    num_retries: int
    retry_after: float
    max_fallbacks: int
    fallbacks: object
    context_window_fallbacks: object
    content_policy_fallbacks: object
    retry_policy: object
    model_group_retry_policy: object
    allowed_fails: int | None
    cooldown_time: float
    disable_cooldowns: bool | None
    enable_pre_call_checks: bool
    fallback_access_check: object
    fallback_budget_check: object

    def get_model_names(self) -> list[str]: ...  # mutable-ok: PythonRouter's read surface

    def discard(self) -> None: ...

    def upsert_deployment(self, deployment: Deployment) -> Deployment | None: ...

    def add_deployment(self, deployment: Deployment) -> Deployment | None: ...

    def delete_deployment(self, id: str) -> Deployment | None: ...

    def update_settings(self, **kwargs: object) -> None: ...  # kwargs-ok: PythonRouter.update_settings' surface

    def get_settings(self) -> Mapping[str, object]: ...

    def _update_kwargs_before_fallbacks(
        self, model: str, kwargs: Kwargs, metadata_variable_name: str = "metadata"
    ) -> None: ...


def _raw(deployment: Deployment) -> Mapping[str, object]:
    return deployment.model_dump(exclude_none=True)


_PYTHON_ROUTER: Final = cast(  # cast-ok: Router(...) forwards untyped arguments
    Callable[..., PythonRouter], PythonRouter
)

_IS_CLIENTSIDE_CREDENTIAL: Final = cast(  # cast-ok: the helper's dict parameter is unparameterized
    Callable[[Kwargs], bool], clientside_credential_handler.is_clientside_credential
)
_NormalizerType: TypeAlias = Callable[..., NormalizerView]
_NORMALIZER: Final = cast(_NormalizerType, PythonRouter)  # cast-ok: forwards the caller's Router(...) arguments


def _first_present(sources: tuple[Mapping[str, object], ...], key: str) -> object:
    return next((source.get(key) for source in sources if source.get(key) is not None), None)


def _project(deployment: Mapping[str, object]) -> Mapping[str, object]:
    litellm_params: Final = as_mapping(deployment.get("litellm_params"))
    model_info: Final = as_mapping(deployment.get("model_info"))
    num_retries: Final = litellm_params.get("num_retries")
    return _ProjectedDeployment.model_validate(
        {
            "id": str(model_info.get("id")),
            "model_name": str(deployment.get("model_name")),
            "model": str(litellm_params.get("model")),
            "weight": litellm_params.get("weight"),
            "cooldown_time": _first_present((model_info, litellm_params), "cooldown_time"),
            "num_retries": int(num_retries)
            if isinstance(num_retries, (int, str)) and str(num_retries).isdigit()
            else None,
            "supports_web_search": model_info.get("supports_web_search"),
        }
    ).model_dump()


def _policy(policy: object) -> object:
    return policy.model_dump() if isinstance(policy, RetryPolicy) else policy


def _settings(router: NormalizerView) -> Mapping[str, object]:
    group_policies: Final = router.model_group_retry_policy
    return {
        "num_retries": router.num_retries,
        "retry_after": router.retry_after,
        "max_fallbacks": router.max_fallbacks,
        "fallbacks": router.fallbacks,
        "context_window_fallbacks": router.context_window_fallbacks,
        "content_policy_fallbacks": router.content_policy_fallbacks,
        "retry_policy": _policy(router.retry_policy),
        "model_group_retry_policy": (
            {str(group): _policy(policy) for group, policy in as_mapping(group_policies).items()}
            if group_policies is not None
            else None
        ),
        "allowed_fails": router.allowed_fails,
        "allowed_fails_set_on_router": _is_allowed_fails_set_on_router(router),  # pyright: ignore[reportArgumentType]  # reads only allowed_fails
        "cooldown_time": router.cooldown_time,
        "disable_cooldowns": bool(router.disable_cooldowns),
        "enable_pre_call_checks": router.enable_pre_call_checks,
        "fallback_checks": router.fallback_access_check is not None or router.fallback_budget_check is not None,
        "tunables": {
            "initial_retry_delay": constants.INITIAL_RETRY_DELAY,
            "max_retry_delay": constants.MAX_RETRY_DELAY,
            "jitter": constants.JITTER,
            "failure_threshold_percent": constants.DEFAULT_FAILURE_THRESHOLD_PERCENT,
            "failure_threshold_minimum_requests": constants.DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS,
            "single_deployment_traffic_failure_threshold": constants.SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD,
            "default_cooldown_time": constants.DEFAULT_COOLDOWN_TIME_SECONDS,
            "cooldown_redis_read_interval": constants.DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS,
        },
    }


def _providers() -> tuple[str, ...]:
    return tuple(
        str(provider.value) if isinstance(provider, Enum) else str(provider) for provider in litellm.provider_list
    )


def redis_url(arguments: Mapping[str, object]) -> str | None:
    """The URL `PythonRouter` would connect its router cache to, if any."""
    url: Final = arguments.get("redis_url")
    if isinstance(url, str):
        return url
    host: Final = arguments.get("redis_host")
    port: Final = arguments.get("redis_port")
    if host is None or port is None:
        return None
    password: Final = arguments.get("redis_password")
    credentials: Final = f":{quote(str(password), safe='')}@" if password else ""
    return f"redis://{credentials}{host}:{port}/{arguments.get('redis_db') or 0}"


class RustRouter:
    """The Rust-backed router. Members it does not emulate yet raise `NotImplementedError` naming them.

    Deployments are normalized by a `PythonRouter` built from the same arguments with its router
    callbacks removed (`discard`), which also builds each attempt's kwargs. Routing decisions,
    cooldowns and usage counters belong to the native router.
    """

    def __init__(
        self,
        arguments: Mapping[str, object],
        native: NativeRouterType,
        seed: int | None = None,
        required: bool = False,
    ) -> None:
        normalizer: Final = _NORMALIZER(
            **{name: value for name, value in arguments.items() if name not in _REDIS_ARGUMENTS}
        )
        normalizer.discard()
        object.__setattr__(self, "_normalizer", normalizer)
        object.__setattr__(self, "_arguments", MappingProxyType(dict(arguments)))
        object.__setattr__(self, "_updates", MappingProxyType({}))
        object.__setattr__(self, "_required", required)
        object.__setattr__(self, "_fall_back", None)
        object.__setattr__(
            self,
            "_native",
            native(
                [_project(deployment) for deployment in normalizer.model_list],
                _settings(normalizer),
                list(_providers()),
                redis_url(arguments),
                seed,
            ),
        )

    _normalizer: NormalizerView
    _native: NativeRouter
    _arguments: Mapping[str, object]
    _updates: Mapping[str, object]
    _required: bool
    _fall_back: FallBack | None

    def __getattr__(self, name: str) -> object:
        normalizer: Final[object] = self.__dict__.get("_normalizer")
        if normalizer is None or not hasattr(normalizer, name):
            raise AttributeError(name)
        raise NotImplementedError(f"the Rust router backend does not emulate Router.{name} yet")

    def __setattr__(self, name: str, value: object) -> None:
        raise NotImplementedError(f"the Rust router backend does not emulate setting Router.{name} yet")

    @property
    def model_list(self) -> list[dict[str, object]]:  # mutable-ok: PythonRouter's read surface
        return self._normalizer.model_list

    @property
    def model_names(self) -> set[str]:  # mutable-ok: PythonRouter's read surface
        return self._normalizer.model_names

    @property
    def total_calls(self) -> Mapping[str, int]:
        return self._normalizer.total_calls

    @property
    def success_calls(self) -> Mapping[str, int]:
        return self._normalizer.success_calls

    @property
    def fail_calls(self) -> Mapping[str, int]:
        return self._normalizer.fail_calls

    def get_model_names(self) -> list[str]:  # mutable-ok: PythonRouter's read surface
        return self._normalizer.get_model_names()

    def discard(self) -> None:
        return None

    def on_fall_back(self, fall_back: FallBack) -> None:
        """Where this instance hands over to a `PythonRouter` when a runtime change asks for something
        the Rust backend does not serve. The hand-over is permanent; in-memory state is not carried."""
        object.__setattr__(self, "_fall_back", fall_back)

    def upsert_deployment(self, deployment: Deployment) -> Deployment | None:
        return self._change(lambda: self._normalizer.upsert_deployment(deployment), deployments=(deployment,))

    def add_deployment(self, deployment: Deployment) -> Deployment | None:
        return self._change(lambda: self._normalizer.add_deployment(deployment), deployments=(deployment,))

    def delete_deployment(self, id: str) -> Deployment | None:
        return self._change(lambda: self._normalizer.delete_deployment(id))

    def update_settings(self, **kwargs: object) -> None:  # kwargs-ok: PythonRouter.update_settings' surface
        updatable: Final = {name: value for name, value in kwargs.items() if name in RUNTIME_UPDATABLE_ROUTER_SETTINGS}
        self._change(lambda: self._normalizer.update_settings(**kwargs), updates=updatable)

    @property
    def retry_policy(self) -> object:
        return self._normalizer.retry_policy

    def get_settings(self) -> Mapping[str, object]:
        return self._normalizer.get_settings()

    def _change(
        self,
        apply: Callable[[], _T],
        deployments: Sequence[Deployment] = (),
        updates: Mapping[str, object] = MappingProxyType({}),
    ) -> _T:
        """Applies a runtime change on the Python side, which normalizes deployments and resolves
        settings, then swaps the native snapshot. A change the Rust backend cannot serve falls back
        to `PythonRouter` for good, or raises when Rust is required."""
        merged: Final = MappingProxyType({**self._updates, **updates})
        prospective: Final = {
            **self._arguments,
            **merged,
            "model_list": [*self._normalizer.model_list, *(_raw(deployment) for deployment in deployments)],
        }
        reason: Final = unsupported_reason(prospective)
        if reason is not None:
            return self._hand_over(reason, apply, merged)
        result: Final = apply()
        object.__setattr__(self, "_updates", merged)
        self._native.replace(
            [_project(deployment) for deployment in self._normalizer.model_list], _settings(self._normalizer)
        )
        return result

    def _hand_over(self, reason: str, apply: Callable[[], _T], updates: Mapping[str, object]) -> _T:
        if self._required or self._fall_back is None:
            raise RustRouterUnsupportedError(f"the Rust router cannot serve this change: {reason}")
        result: Final = apply()
        python: Final = _PYTHON_ROUTER(**{**self._arguments, "model_list": copy.deepcopy(self._normalizer.model_list)})
        if updates:
            settings_view: Final = cast(NormalizerView, python)  # cast-ok: PythonRouter's untyped update_settings
            settings_view.update_settings(**updates)
        verbose_router_logger.info("Rust router handing over to the Python router for good: %s", reason)
        self._fall_back(python)
        return result

    async def acompletion(
        self,
        model: str,
        messages: Sequence[Mapping[str, object]],
        **kwargs: object,  # kwargs-ok: forwards litellm.acompletion's kwargs
    ) -> object:
        call: Final = self._call("completion", model, {**kwargs, "messages": messages})
        return await self._route(call, True)

    def completion(
        self,
        model: str,
        messages: Sequence[Mapping[str, object]],
        **kwargs: object,  # kwargs-ok: forwards litellm.completion's kwargs
    ) -> object:
        call: Final = self._call("completion", model, {**kwargs, "messages": messages})
        return self._native.route(call.spec, call.driver, False)

    async def aresponses(
        self,
        custom_llm_provider: str | None = None,
        client: object = None,
        **kwargs: object,  # kwargs-ok: forwards litellm.aresponses' kwargs
    ) -> object:
        """`factory_function`'s wrapper drops its own `custom_llm_provider` and `client`."""
        del custom_llm_provider, client
        return await self._route(self._call("responses", _model(kwargs), kwargs), True)

    async def aanthropic_messages(
        self,
        custom_llm_provider: str | None = None,
        client: object = None,
        **kwargs: object,  # kwargs-ok: forwards litellm.anthropic_messages' kwargs
    ) -> object:
        """`factory_function`'s wrapper drops its own `custom_llm_provider` and `client`."""
        del custom_llm_provider, client
        return await self._route(self._call("anthropic_messages", _model(kwargs), kwargs), True)

    async def _route(self, call: _Call, asynchronous: bool) -> object:
        native_route: Final = self._native.route(call.spec, call.driver, asynchronous)
        routed: Final = cast(Awaitable[object], native_route)  # cast-ok: an asynchronous route is a coroutine
        return await routed

    def _call(self, operation: Operation, model: str, kwargs: Kwargs) -> _Call:
        unsupported: Final = [name for name in _UNSUPPORTED_REQUEST_KWARGS if name in kwargs]
        if unsupported:
            raise NotImplementedError(f"the Rust router backend does not support request option {unsupported[0]!r} yet")
        clientside: Final = _IS_CLIENTSIDE_CREDENTIAL(kwargs)
        if clientside:
            raise NotImplementedError("the Rust router backend does not support client-side credentials yet")
        kwargs["model"] = model  # rebind-ok: PythonRouter stamps the caller's kwargs the same way
        if operation == "completion":
            kwargs.setdefault("stream", False)
        self._normalizer._update_kwargs_before_fallbacks(  # pyright: ignore[reportPrivateUsage]  # PythonRouter's entry stamps
            model=model, kwargs=kwargs, metadata_variable_name=metadata_key_for(operation)
        )
        overrides: Final = {
            name: kwargs[name]
            for name in ("fallbacks", "context_window_fallbacks", "content_policy_fallbacks")
            if name in kwargs
        }
        mock: Final = next((failure for flag, failure in _MOCK_FAILURES if _flag(kwargs.get(flag))), None)
        tools: Final = (as_mapping(tool) for tool in as_sequence(kwargs.get("tools")))
        spec: Final = {
            "operation": operation,
            "model": model,
            "num_retries": kwargs.get("num_retries"),
            "disable_fallbacks": kwargs.get("disable_fallbacks") is True,
            **overrides,
            "shared_logging": kwargs.get("litellm_logging_obj") is not None,
            "web_search": any(tool.get("type") in _WEB_SEARCH_TOOLS for tool in tools),
            "mock": mock,
        }
        driver: Final = self._driver(operation, kwargs, overrides, spec)
        driver.start(model)
        return _Call(spec, driver)

    def _driver(
        self, operation: Operation, kwargs: Kwargs, overrides: Mapping[str, object], spec: Mapping[str, object]
    ) -> RoutedCall:
        async def resume(point: Mapping[str, object], hop_kwargs: Kwargs) -> object:
            """A stream that failed after its content reached the caller falls back as its own call."""
            model: Final = str(point["model_group"])
            self._normalizer._update_kwargs_before_fallbacks(  # pyright: ignore[reportPrivateUsage]  # as the stream wrapper re-enters the chain
                model=model, kwargs=hop_kwargs, metadata_variable_name=metadata_key_for(operation)
            )
            driver: Final = self._driver(operation, hop_kwargs, overrides, spec)
            return await self._route(_Call({**spec, "model": model, "mock": None, "resume": point}, driver), True)

        return RoutedCall(
            self._normalizer,
            kwargs,
            operation,
            lambda name: overrides.get(name, _attribute(self._normalizer, name)),
            resume,
        )


@dataclass(frozen=True, slots=True)
class _Call:
    spec: Mapping[str, object]
    driver: RoutedCall


def _model(kwargs: Kwargs) -> str:
    model: Final = kwargs.pop("model", None)
    if not isinstance(model, str):
        raise TypeError("the router needs a model group to route to")
    return model


def _attribute(owner: object, name: str) -> object:
    return cast(object, getattr(owner, name))  # cast-ok: getattr is typed Any


def _flag(value: object) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def build_rust_router(
    arguments: Mapping[str, object], native: NativeRouterType | None = None, required: bool = False
) -> RustRouter | RustRouterDeclined:
    reason: Final = unsupported_reason(arguments)
    if reason is not None:
        return RustRouterDeclined(reason)
    native_type: Final = native if native is not None else NATIVE_ROUTER.load()
    if native_type is None:
        return RustRouterDeclined("the native router is not available in this build")
    return RustRouter(arguments, native_type, required=required)
