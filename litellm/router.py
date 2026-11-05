"""
litellm.Router: load balancing, retries, fallbacks and cooldowns across deployments.

``Router`` holds no routing logic. It builds its configuration (``litellm/router_config.py``), builds
its ``RouterCore`` from it (the Rust core, ``litellm/router_core.py``) and delegates every member to it:
the endpoint methods (acompletion, aembedding, aresponses, aanthropic_messages, ...) and the entry
points (registry, settings, health, select, apply, observe, register_plugin, retire). The members
outside code used before the entry points live in ``DeprecatedRouterMembers`` until they are removed.
"""

import functools
import inspect
import weakref
from collections.abc import Callable, Mapping, Sequence
from types import FunctionType, MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal

from litellm.router_config import RouterConfig, build_router_config
from litellm.router_core import RouterCore, make_core
from litellm.router_deprecated import DeprecatedRouterMembers
from litellm.router_python_host import PythonHost, python_host
from litellm.router_rust.steps.prerouting import as_output_cap, stamped_client_ceiling
from litellm.router_rust.surface.registry import model_info_is_active_for_environment
from litellm.types.litellm_params import RoutingStrategyName
from litellm.types.router import (
    AlertingConfig,
    AllowedFailsPolicy,
    AssistantsTypedDict,
    AutoRouterCapabilityLimit,
    CustomRoutingStrategyBase,
    Deployment,
    DeploymentTypedDict,
    FallbackAccessCheck,
    FallbackBudgetCheck,
    GuardrailTypedDict,
    LiteLLM_Params,
    ModelGroupInfo,
    OptionalPreCallChecks,
    RetryPolicy,
    RouterGeneralSettings,
    RouterModelGroupAliasItem,
    RoutingGroup,
    RoutingPlugin,
    SearchToolTypedDict,
)
from litellm.types.utils import GenericBudgetConfigType

if TYPE_CHECKING:
    from litellm.router_rust.core import RustCore


# ``Router(...)``'s parameters, their defaults and its docstring: ``router_arguments`` binds a call
# against them and ``Router.__init__`` reports them. The body is never run.
def _router_init(
    self,
    model_list: list[DeploymentTypedDict] | list[dict[str, Any]] | None = None,
    ## ASSISTANTS API ##
    assistants_config: AssistantsTypedDict | None = None,
    ## SEARCH API ##
    search_tools: list[SearchToolTypedDict] | None = None,
    ## GUARDRAIL API ##
    guardrail_list: list[GuardrailTypedDict] | None = None,
    ## CACHING ##
    redis_url: str | None = None,
    redis_host: str | None = None,
    redis_port: int | None = None,
    redis_password: str | None = None,
    redis_db: int | None = None,
    cache_responses: bool | None = False,
    cache_kwargs: dict = {},  # additional kwargs to pass to RedisCache (see caching.py)
    caching_groups: list[tuple] | None = None,  # if you want to cache across model groups
    client_ttl: int = 3600,  # ttl for cached clients - will re-initialize after this time in seconds
    ## SCHEDULER ##
    polling_interval: float | None = None,
    default_priority: int | None = None,
    ## RELIABILITY ##
    num_retries: int | None = None,
    max_fallbacks: int | None = None,  # max fallbacks to try before exiting the call. Defaults to 5.
    timeout: float | None = None,
    stream_timeout: float | None = None,
    default_litellm_params: dict | None = None,  # default params for Router.chat.completion.create
    default_max_parallel_requests: int | None = None,
    set_verbose: bool = False,
    debug_level: Literal["DEBUG", "INFO"] = "INFO",
    default_fallbacks: list[str] | None = None,  # generic fallbacks, works across all deployments
    fallbacks: list = [],
    context_window_fallbacks: list = [],
    content_policy_fallbacks: list = [],
    model_group_alias: dict[str, str | RouterModelGroupAliasItem] | None = {},
    enable_pre_call_checks: bool = False,
    enable_tag_filtering: bool = False,
    tag_filtering_match_any: bool = True,
    tag_routing_prefix: str = "",
    plugins: list[RoutingPlugin] | None = None,
    retry_after: int = 0,  # min time to wait before retrying a failed request
    retry_policy: RetryPolicy | dict | None = None,  # set custom retries for different exceptions
    model_group_retry_policy: dict[str, RetryPolicy] = {},  # set custom retry policies based on model group
    allowed_fails: int | None = None,  # Number of times a deployment can failbefore being added to cooldown
    allowed_fails_policy: AllowedFailsPolicy | None = None,  # set custom allowed fails policy
    cooldown_time: float | None = None,  # (seconds) time to cooldown a deployment after failure
    disable_cooldowns: bool | None = None,
    fallback_on_unsupported_params: bool = True,
    routing_strategy: RoutingStrategyName = "simple-shuffle",
    optional_pre_call_checks: OptionalPreCallChecks | None = None,
    routing_strategy_args: dict = {},  # just for latency-based
    routing_groups: list[RoutingGroup | dict] | None = None,
    provider_budget_config: GenericBudgetConfigType | None = None,
    alerting_config: AlertingConfig | None = None,
    router_general_settings: RouterGeneralSettings | None = RouterGeneralSettings(),
    deployment_affinity_ttl_seconds: int = 3600,
    model_group_affinity_config: dict[str, list[str]] | None = None,
    ignore_invalid_deployments: bool = False,
    enable_health_check_routing: bool = False,
    health_check_staleness_threshold: int | None = None,
    health_check_ignore_transient_errors: bool = False,
    background_health_check_model_groups: Sequence[str] | None = None,
    enable_weighted_failover: bool = False,
    enable_mid_stream_fallback_continuation: bool = False,
    fallback_access_check: FallbackAccessCheck | None = None,
    fallback_budget_check: FallbackBudgetCheck | None = None,
    auto_router_capability_limit: AutoRouterCapabilityLimit | None = None,
    session_affinity_repin_on_failover: bool = False,
    _host: PythonHost = python_host,
) -> None:
    """
    Initialize the Router class with the given parameters for caching, reliability, and routing strategy.

    Args:
        model_list (Optional[list]): List of models to be used. Defaults to None.
        redis_url (Optional[str]): URL of the Redis server. Defaults to None.
        redis_host (Optional[str]): Hostname of the Redis server. Defaults to None.
        redis_port (Optional[int]): Port of the Redis server. Defaults to None.
        redis_password (Optional[str]): Password of the Redis server. Defaults to None.
        cache_responses (Optional[bool]): Flag to enable caching of responses. Defaults to False.
        cache_kwargs (dict): Additional kwargs to pass to RedisCache. Defaults to {}.
        caching_groups (Optional[List[tuple]]): List of model groups for caching across model groups. Defaults to None.
        client_ttl (int): Time-to-live for cached clients in seconds. Defaults to 3600.
        polling_interval: (Optional[float]): frequency of polling queue. Only for '.scheduler_acompletion()'. Default is 3ms.
        default_priority: (Optional[int]): the priority given to every chat or text completion request that sets no 'priority' of its own; '.acompletion()' and '.atext_completion()' queue it through the scheduler. Default is None.
        num_retries (Optional[int]): Number of retries for failed requests. Defaults to 2.
        timeout (Optional[float]): Timeout for requests. Defaults to None.
        default_litellm_params (dict): Default parameters for Router.chat.completion.create. Defaults to {}.
        set_verbose (bool): Flag to set verbose mode. Defaults to False.
        debug_level (Literal["DEBUG", "INFO"]): Debug level for logging. Defaults to "INFO".
        fallbacks (List): List of fallback options. Defaults to [].
        context_window_fallbacks (List): List of context window fallback options. Defaults to [].
        enable_pre_call_checks (boolean): Filter out deployments which are outside context window limits for a given prompt
        model_group_alias (Optional[dict]): Alias for model groups. Defaults to {}.
        retry_after (int): Minimum time to wait before retrying a failed request. Defaults to 0.
        allowed_fails (Optional[int]): Number of allowed fails before adding to cooldown. Defaults to None.
        cooldown_time (float): Time to cooldown a deployment after failure in seconds. Defaults to 1.
        routing_strategy (Literal["simple-shuffle", "least-busy", "usage-based-routing", "latency-based-routing", "cost-based-routing"]): Routing strategy used for the implicit "default" group (any model not claimed by an entry in `routing_groups`). Defaults to "simple-shuffle".
        routing_strategy_args (dict): Additional args for the default group's routing strategy (e.g. latency window). Defaults to {}.
        routing_groups (Optional[List[RoutingGroup]]): Named subsets of `model_name`s with a group routing strategy.
            Priority groups apply only to group calls and may overlap. Other groups supply their members' default
            strategy, with at most one such group per model. Unclaimed models use the top-level strategy.
        alerting_config (AlertingConfig): Slack alerting configuration. Defaults to None.
        provider_budget_config (ProviderBudgetConfig): Provider budget configuration. Use this to set llm_provider budget limits. example $100/day to OpenAI, $100/day to Azure, etc. Defaults to None.
        deployment_affinity_ttl_seconds (int): TTL for user-key -> deployment affinity mapping. Defaults to 3600.
        session_affinity_repin_on_failover (bool): After a successful replacement response, move a session pin
            whose deployment is in cooldown or removed. Async routes only. Defaults to False. Applied at startup.
        ignore_invalid_deployments (bool): Ignores invalid deployments, and continues with other deployments. Default is to raise an error.
        enable_weighted_failover (bool): When True and the routing strategy is "simple-shuffle", a retryable failure on one deployment causes the request to re-pick (weighted) across the other deployments in the same model group before any cross-group fallback runs. Bounded by `max_fallbacks`. Async-only: currently honored by `router.acompletion()` and other async entrypoints. The sync `router.completion()` path falls back to the regular fallback flow. Defaults to False.
        enable_mid_stream_fallback_continuation (bool): When True, a chat-completions stream that breaks after plain assistant text has been delivered continues on a fallback deployment via assistant prefill instead of surfacing the error. Only deployments whose model supports assistant prefill are eligible, so the partial text is continued, not regenerated; if none is, the original error is surfaced. Streams that emitted tool calls, thinking blocks, audio/images, or constrained (JSON / forced tool_choice) output are never continued. Async-only. Defaults to False.
        fallback_access_check (Optional[FallbackAccessCheck]): Awaited before each cross-model-group fallback attempt on the async path; a fallback target it rejects is skipped. Defaults to None (every configured fallback is attempted).
        fallback_budget_check (Optional[FallbackBudgetCheck]): Awaited before each cross-model-group fallback attempt on the async path; a fallback target it rejects as over budget is skipped. Defaults to None (budget is not re-checked on fallback).
    Returns:
        Router: An instance of the litellm.Router class.

    Example Usage:
    ```python
    from litellm import Router
    model_list = [
    {
        "model_name": "azure-gpt-3.5-turbo", # model alias
        "litellm_params": { # params for litellm completion/embedding call
            "model": "azure/<your-deployment-name-1>",
            "api_key": <your-api-key>,
            "api_version": <your-api-version>,
            "api_base": <your-api-base>
        },
    },
    {
        "model_name": "azure-gpt-3.5-turbo", # model alias
        "litellm_params": { # params for litellm completion/embedding call
            "model": "azure/<your-deployment-name-2>",
            "api_key": <your-api-key>,
            "api_version": <your-api-version>,
            "api_base": <your-api-base>
        },
    },
    {
        "model_name": "openai-gpt-3.5-turbo", # model alias
        "litellm_params": { # params for litellm completion/embedding call
            "model": "gpt-3.5-turbo",
            "api_key": <your-api-key>,
        },
    ]

    router = Router(model_list=model_list, fallbacks=[{"azure-gpt-3.5-turbo": ["openai-gpt-3.5-turbo"]}])
    ```
    """


_CORE_SIGNATURE: Final = inspect.signature(_router_init)


def router_arguments(
    *args: object,
    **kwargs: object,  # kwargs-ok: Router's constructor arguments, as passed
) -> Mapping[str, object]:
    """The constructor arguments as passed, by name (positional ones bound to their parameter)."""
    try:
        bound: Final = _CORE_SIGNATURE.bind_partial(None, *args, **kwargs).arguments
    except TypeError as error:
        raise TypeError(f"Router.__init__() {error}") from None
    return MappingProxyType({name: value for name, value in bound.items() if name != "self"})


def _class_descriptor(cls: type, name: str) -> bool:
    return any(hasattr(vars(base).get(name), "__set__") for base in cls.__mro__ if name in vars(base))


if TYPE_CHECKING:
    from typing import overload

    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    from litellm.types.llms.openai import AllMessageValues, OpenAIFileObject
    from litellm.types.utils import EmbeddingResponse, LiteLLMBatch, ModelResponse

    class Router(RustCore, DeprecatedRouterMembers):
        """What type checkers see: the constructor's parameters and the endpoints' return types. The
        endpoints are served by the core at run time (litellm/router_rust/invoke/endpoints.py)."""

        __init__ = _router_init  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]  # Router(...)'s parameters

        @classmethod
        def from_config(cls, config: RouterConfig, host: PythonHost = python_host) -> "Router": ...

        @overload
        async def acompletion(
            self, model: str, messages: list[AllMessageValues], stream: Literal[True], **kwargs: object
        ) -> CustomStreamWrapper: ...
        @overload
        async def acompletion(
            self, model: str, messages: list[AllMessageValues], stream: Literal[False] = False, **kwargs: object
        ) -> ModelResponse: ...
        @overload
        async def acompletion(
            self, model: str, messages: list[AllMessageValues], stream: bool = False, **kwargs: object
        ) -> CustomStreamWrapper | ModelResponse: ...
        async def acompletion(  # pyright: ignore[reportIncompatibleMethodOverride]  # the endpoint's typed signature
            self, model: str, messages: list[AllMessageValues], stream: bool = False, **kwargs: object
        ) -> CustomStreamWrapper | ModelResponse: ...
        def completion(  # pyright: ignore[reportIncompatibleMethodOverride]  # see acompletion
            self, model: str, messages: list[dict[str, str]], **kwargs: object
        ) -> ModelResponse | CustomStreamWrapper: ...
        def embedding(  # pyright: ignore[reportIncompatibleMethodOverride]  # see acompletion
            self, model: str, input: str | list[object], is_async: bool | None = False, **kwargs: object
        ) -> EmbeddingResponse: ...
        async def aembedding(  # pyright: ignore[reportIncompatibleMethodOverride]  # see acompletion
            self, model: str, input: str | list[object], is_async: bool | None = True, **kwargs: object
        ) -> EmbeddingResponse: ...
        async def acreate_file(self, model: str, **kwargs: object) -> OpenAIFileObject: ...  # pyright: ignore[reportIncompatibleMethodOverride]  # see acompletion
        async def acreate_batch(self, model: str, **kwargs: object) -> LiteLLMBatch: ...  # pyright: ignore[reportIncompatibleMethodOverride]  # see acompletion
        async def aretrieve_batch(self, model: str | None = None, **kwargs: object) -> LiteLLMBatch: ...  # pyright: ignore[reportIncompatibleMethodOverride]  # see acompletion
        def __getattr__(self, name: str) -> Any: ...  # the endpoints served by name

else:
    _MISSING: Final = object()
    _replaced_core_members: Final[  # mutable-ok: originals of patched core members, restored on delete
        dict[tuple[type, str], list[object]]
    ] = {}

    def _core_class() -> type:
        from litellm.router_rust.core import RustCore

        return RustCore

    class _RouterType(type):
        """
        Class-level reads of core members (``Router._helper``) go to the core's class, and so do
        class-level patches, which put the original back when the patch is removed. ``dir`` lists
        the core's members and endpoints too, so ``Mock(spec=Router)`` accepts them.
        """

        def __getattr__(cls, name: str) -> object:
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            core: Final = _core_class()
            try:
                return getattr(core, name)
            except AttributeError:
                if not _is_endpoint(name):
                    raise
                return _endpoint_function(name)

        def __setattr__(cls, name: str, value: object) -> None:
            core: Final = _core_class()
            if any(name in vars(base) for base in cls.__mro__) or not (hasattr(core, name) or _is_endpoint(name)):
                super().__setattr__(name, value)
                return
            _replaced_core_members.setdefault((core, name), []).append(vars(core).get(name, _MISSING))
            setattr(core, name, value)

        def __delattr__(cls, name: str) -> None:
            core: Final = _core_class()
            replaced: Final = _replaced_core_members.get((core, name))
            if not replaced or any(name in vars(base) for base in cls.__mro__):
                super().__delattr__(name)
                return
            original: Final = replaced.pop()
            if original is _MISSING:
                delattr(core, name)
                return
            setattr(core, name, original)

        def __dir__(cls) -> list[str]:  # mutable-ok: dir() returns a list
            from litellm.router_rust.invoke.endpoints import endpoint_names

            return sorted({*type.__dir__(cls), *dir(_core_class()), *endpoint_names()})

    def _is_endpoint(name: str) -> bool:
        from litellm.router_rust.invoke.endpoints import endpoint_names

        return name in endpoint_names()

    @functools.cache
    def _endpoint_function(name: str) -> Callable[..., object]:
        """An endpoint the core serves by name, read off the class (``Router.aretrieve_batch``): a
        function that calls the instance's endpoint, so a class-level patch of it replaces it."""

        def endpoint(
            self: object,
            *args: object,
            **kwargs: object,  # kwargs-ok: the endpoint's own arguments
        ) -> object:
            served: Final = _core_class().__getattr__(self, name)  # pyright: ignore[reportAny]  # the core's endpoint lookup
            return served(*args, **kwargs)  # pyright: ignore[reportAny]  # see above

        endpoint.__name__ = endpoint.__qualname__ = name
        return endpoint

    def _patchable_method(core: object, name: str) -> bool:
        """A method of the core: an instance-level replacement of it is also kept on the facade, so
        ``mock.patch.object(router, name)`` finds it there and nested patches unwind in order. A core
        that serves its members from ``__getattr__`` is asked for the bound method."""
        static: Final = inspect.getattr_static(type(core), name, None)
        if static is not None:
            return isinstance(static, FunctionType | staticmethod | classmethod)
        return hasattr(type(core), "__getattr__") and inspect.ismethod(getattr(core, name, None))

    class Router(DeprecatedRouterMembers, metaclass=_RouterType):
        def __init_subclass__(cls, **kwargs: object) -> None:  # kwargs-ok: class keywords, refused with the class
            raise TypeError(
                "litellm.Router cannot be subclassed. Use the router's entry points instead: "
                "register_plugin() for custom routing, apply() for settings, observe() for outcomes. "
                "See https://docs.litellm.ai/docs/routing_plugins."
            )

        def __init__(self, *args: object, **kwargs: object) -> None:  # kwargs-ok: Router's constructor arguments
            self._attach(make_core(build_router_config(router_arguments(*args, **kwargs)), python_host))

        @classmethod
        def from_config(cls, config: RouterConfig, host: PythonHost = python_host) -> "Router":
            router: Final = cls.__new__(cls)
            router._attach(make_core(config, host))
            return router

        def _attach(self, core: RouterCore) -> None:
            pending: Final = self.__dict__.pop("_pending", {})
            self.__dict__["_core"] = core
            core.__dict__["_core_owner"] = weakref.ref(self)  # rebind-ok: the facade records itself on the core it owns
            for name, value in pending.items():
                setattr(core, name, value)

        @property
        def _core(self) -> RouterCore:
            core: Final = self.__dict__.get("_core")
            if core is None:
                raise AttributeError("_core")
            return core

        def __getattribute__(self, name: str) -> object:
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                pass
            facade: Final = object.__getattribute__(self, "__dict__")
            core: Final = facade.get("_core")
            if core is None:
                pending: Final = facade.get("_pending", {})
                if name in pending:
                    return pending[name]
                raise AttributeError(name)
            try:
                return getattr(core, name)
            except AttributeError as error:
                if error.obj is not core or error.name != name:
                    raise
                # What reading an attribute the router lacks raised before: it names the Router.
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'", name=name, obj=self
                ) from None

        def __setattr__(self, name: str, value: object) -> None:
            if _class_descriptor(type(self), name):
                object.__setattr__(self, name, value)
                return
            if "_core" not in self.__dict__:
                self.__dict__.setdefault("_pending", {})[name] = value
                return
            core: Final = self._core
            mirrored: Final = name in self.__dict__ or _patchable_method(core, name)
            setattr(core, name, value)
            if mirrored:
                self.__dict__[name] = value

        def __delattr__(self, name: str) -> None:
            if _class_descriptor(type(self), name):
                object.__delattr__(self, name)
                return
            self.__dict__.pop(name, None)
            delattr(self._core, name)

        def __dir__(self) -> list[str]:  # mutable-ok: dir() returns a list
            core: Final = self.__dict__.get("_core")
            return sorted({*super().__dir__(), *(dir(core) if core is not None else ())})

    Router.__init__.__signature__ = _CORE_SIGNATURE.replace(  # pyright: ignore[reportFunctionMemberAccess]  # inspect reads it
        parameters=[parameter for parameter in _CORE_SIGNATURE.parameters.values() if parameter.name != "_host"]
    )
    Router.__init__.__doc__ = _router_init.__doc__


__all__ = [
    "AllowedFailsPolicy",
    "AssistantsTypedDict",
    "CustomRoutingStrategyBase",
    "Deployment",
    "LiteLLM_Params",
    "ModelGroupInfo",
    "RetryPolicy",
    "Router",
    "as_output_cap",
    "model_info_is_active_for_environment",
    "router_arguments",
    "stamped_client_ceiling",
]
