"""
litellm.Router: load balancing, retries, fallbacks and cooldowns across deployments.

``Router`` holds no routing logic. It builds its configuration (``litellm/router_config.py``), picks
one ``RouterCore`` at construction (``litellm/router_core.py``) and delegates every member to it:
the endpoint methods (acompletion, aembedding, aresponses, aanthropic_messages, ...) and the entry
points (registry, settings, health, select, apply, observe, register_plugin, retire). The members
outside code used before the entry points live in ``DeprecatedRouterMembers`` until they are removed.
"""

import inspect
import weakref
from collections.abc import Mapping
from types import CellType, FunctionType, MappingProxyType
from typing import TYPE_CHECKING, Final

from litellm import router_python_core
from litellm.router_config import RouterConfig, build_router_config
from litellm.router_core import RouterCore, make_core
from litellm.router_deprecated import DeprecatedRouterMembers
from litellm.router_python_core import (
    PythonCore,
    as_output_cap,
    model_info_is_active_for_environment,
    stamped_client_ceiling,
)
from litellm.router_python_host import PythonHost, python_host
from litellm.types.router import (
    AllowedFailsPolicy,
    AssistantsTypedDict,
    CustomRoutingStrategyBase,
    Deployment,
    LiteLLM_Params,
    ModelGroupInfo,
    RetryPolicy,
)

_CORE_SIGNATURE: Final = inspect.signature(PythonCore.__init__)


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

    class Router(PythonCore, DeprecatedRouterMembers):
        @classmethod
        def from_config(cls, config: RouterConfig, host: PythonHost = python_host) -> "Router": ...

else:
    _MISSING: Final = object()
    _replaced_core_members: Final[  # mutable-ok: originals of patched core members, restored on delete
        dict[tuple[type, str], list[object]]
    ] = {}
    _FACADE_ONLY: Final = frozenset(
        {
            "__module__",
            "__qualname__",
            "__doc__",
            "__dict__",
            "__weakref__",
            "__init__",
            "__init_subclass__",
            "__annotations__",
            "__firstlineno__",
            "__static_attributes__",
            "__classcell__",
            "_core_class",
        }
    )

    def _with_class_cell(value: object, cell: "CellType") -> object:
        """``value`` with its ``__class__`` cell (what zero-argument ``super()`` reads) replaced by ``cell``."""
        if isinstance(value, staticmethod | classmethod):
            return type(value)(_function_with_class_cell(value.__func__, cell))
        if isinstance(value, property):
            return property(
                *(
                    None if f is None else _function_with_class_cell(f, cell)
                    for f in (value.fget, value.fset, value.fdel)
                ),
                value.__doc__,
            )
        return _function_with_class_cell(value, cell)

    def _function_with_class_cell(value: object, cell: "CellType") -> object:
        """A plain function with its ``__class__`` cell replaced by ``cell``; anything else unchanged."""
        if not isinstance(value, FunctionType) or "__class__" not in value.__code__.co_freevars:
            return value
        closure: Final = tuple(
            cell if name == "__class__" else original
            for name, original in zip(value.__code__.co_freevars, value.__closure__ or (), strict=True)
        )
        rebound: Final = FunctionType(value.__code__, value.__globals__, value.__name__, value.__defaults__, closure)
        rebound.__kwdefaults__ = value.__kwdefaults__
        rebound.__qualname__ = value.__qualname__
        rebound.__dict__.update(value.__dict__)
        return rebound

    class _Absent:
        """A facade-only member, hidden from the core class: reading it raises the usual AttributeError."""

        def __set_name__(self, owner: type, name: str) -> None:
            self._name = name

        def __get__(self, instance: object, owner: type | None = None) -> object:
            if instance is None:
                raise AttributeError(
                    f"type object {getattr(owner, '__name__', owner)!r} has no attribute {self._name!r}"
                )
            raise AttributeError(f"{type(instance).__name__!r} object has no attribute {self._name!r}")

    def _core_class_for(facade: type, parent: type) -> type:
        """
        A core class carrying a Router subclass's own members, so the core runs its overrides. It
        subclasses the facade too, so an override's ``super(Sub, self)`` finds the core it runs on.
        """
        cell: Final = CellType()
        members: Final = {
            name: _with_class_cell(value, cell) for name, value in vars(facade).items() if name not in _FACADE_ONLY
        }
        namespace: Final = {
            # The facade's own plumbing sits in the core class's MRO ahead of the parent core: put
            # the parent core's versions back, and hide the facade-only members.
            "__init__": parent.__init__,
            "__getattribute__": object.__getattribute__,
            "__setattr__": parent.__setattr__,
            "__delattr__": parent.__delattr__,
            "__dir__": parent.__dir__,
            **{name: _Absent() for name in ("_attach", "_core", "from_config", "_core_class")},
            **members,
            "__module__": facade.__module__,
        }
        core_class: Final = _CoreType(f"{facade.__name__}Core", (facade, parent), namespace)
        cell.cell_contents = core_class
        for name in members:
            type.__delattr__(facade, name)
        return core_class

    class _RouterType(type):
        """
        Class-level reads of core members (``Router._helper(...)``) go to the router's core class, and
        so do class-level patches, which put the original back when the patch is removed.
        ``isinstance`` and ``dir`` see the core class too.
        """

        def __getattr__(cls, name: str) -> object:
            if name == "_core_class":
                raise AttributeError(name)
            return getattr(cls._core_class, name)

        def __setattr__(cls, name: str, value: object) -> None:
            core_class: Final = vars(cls).get("_core_class") or getattr(cls, "_core_class", None)
            if core_class is None or any(name in vars(base) for base in cls.__mro__) or not hasattr(core_class, name):
                super().__setattr__(name, value)
                return
            _replaced_core_members.setdefault((core_class, name), []).append(vars(core_class).get(name, _MISSING))
            setattr(core_class, name, value)

        def __delattr__(cls, name: str) -> None:
            core_class: Final = getattr(cls, "_core_class", None)
            replaced: Final = _replaced_core_members.get((core_class, name)) if core_class is not None else None
            if not replaced or any(name in vars(base) for base in cls.__mro__):
                super().__delattr__(name)
                return
            original: Final = replaced.pop()
            if original is _MISSING:
                delattr(core_class, name)
                return
            setattr(core_class, name, original)

        def __instancecheck__(cls, instance: object) -> bool:
            return type.__instancecheck__(cls, instance) or isinstance(instance, cls._core_class)

        def __dir__(cls) -> list[str]:  # mutable-ok: dir() returns a list
            return sorted({*type.__dir__(cls), *dir(cls._core_class)})

    class _CoreType(_RouterType):
        """The metaclass of a Router subclass's core class: a plain class, none of the facade's redirects."""

        def __getattr__(cls, name: str) -> object:
            raise AttributeError(f"type object {cls.__name__!r} has no attribute {name!r}")

        __setattr__ = type.__setattr__
        __delattr__ = type.__delattr__
        __instancecheck__ = type.__instancecheck__
        __dir__ = type.__dir__

    def _is_core(instance: object) -> bool:
        return isinstance(type(instance), _CoreType)

    def _patchable_method(core: object, name: str) -> bool:
        """A method of the core: an instance-level replacement of it is also kept on the facade, so
        ``mock.patch.object(router, name)`` finds it there and nested patches unwind in order. A core
        that serves its members from another core (``__getattr__``) is asked for the bound method."""
        static: Final = inspect.getattr_static(type(core), name, None)
        if static is not None:
            return isinstance(static, FunctionType | staticmethod | classmethod)
        return hasattr(type(core), "__getattr__") and inspect.ismethod(getattr(core, name, None))

    class Router(DeprecatedRouterMembers, metaclass=_RouterType):
        _core_class: type[PythonCore] = PythonCore

        def __init_subclass__(cls, **kwargs: object) -> None:  # kwargs-ok: class keywords, passed on to super()
            if isinstance(cls, _CoreType):
                # A core class built below: the parent core's hooks wire it, as for any core class.
                super().__init_subclass__(**kwargs)
                return
            # DeprecatedRouterMembers wires a subclass's overrides of old members; here that runs
            # on the core class built from it, which is where the router's own calls land.
            super(DeprecatedRouterMembers, cls).__init_subclass__(**kwargs)
            type.__setattr__(cls, "_core_class", _core_class_for(cls, cls._core_class))

        def __init__(self, *args: object, **kwargs: object) -> None:  # kwargs-ok: Router's constructor arguments
            self._attach(
                make_core(build_router_config(router_arguments(*args, **kwargs)), python_host, type(self)._core_class)
            )

        @classmethod
        def from_config(cls, config: RouterConfig, host: PythonHost = python_host) -> "Router":
            router: Final = cls.__new__(cls)
            router._attach(make_core(config, host, cls._core_class))
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
                if _is_core(self):
                    raise
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
            if _is_core(self):
                super().__setattr__(name, value)
                return
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
            if _is_core(self):
                super().__delattr__(name)
                return
            if _class_descriptor(type(self), name):
                object.__delattr__(self, name)
                return
            self.__dict__.pop(name, None)
            delattr(self._core, name)

        def __dir__(self) -> list[str]:  # mutable-ok: dir() returns a list
            if _is_core(self):
                return super().__dir__()
            return sorted({*super().__dir__(), *dir(self.__dict__.get("_core", type(self)._core_class))})

            # Always track deployment successes for cooldown logic, regardless of TPM/RPM limits
            increment_deployment_successes_for_current_minute(
                litellm_router_instance=self,
                deployment_id=str(deployment_id),
            )

            total_tokens: Final[float] = standard_logging_object.get("total_tokens", 0)
            counted_tokens: Final = get_counted_usage_tokens(litellm_params)
            deployment_name: Final = metadata.get("deployment", None)
            return await self._increment_deployment_usage(
                deployment_id=str(deployment_id),
                deployment_name=deployment_name if isinstance(deployment_name, str) else None,
                model_group=model_group,
                total_tokens=total_tokens if counted_tokens is None else max(0, total_tokens - counted_tokens),
                rpm_increment=1 if counted_tokens is None else 0,
                parent_otel_span=get_parent_otel_span_from_kwargs(kwargs),
            )

        except Exception as e:
            verbose_router_logger.debug(
                "litellm.router.Router::deployment_callback_on_success(): Exception occured - %s", e
            )

    async def _repin_session_after_success(self, litellm_params: Mapping[str, object], response: object) -> None:
        if not self.session_affinity_repin_on_failover:
            return
        try:
            from litellm.router_utils.pre_call_checks.deployment_affinity_check import completed_affinity_response

            if not completed_affinity_response(response):
                return
            callback: Final = next(
                (logger for logger in self._optional_loggers() if isinstance(logger, DeploymentAffinityCheck)),
                None,
            )
            if callback is None:
                return
            model_group: Final = callback.get_repin_model_group(litellm_params)
            if model_group is None or self._is_priority_routing_group(model_group):
                return
            registered_ids: Final = self.get_candidate_model_ids_for_route(
                model=model_group, team_id=get_request_team_id(litellm_params)
            )
            cooldown_ids: Final = frozenset(
                model_id
                for model_id, _ in await self.cooldown_cache.async_get_active_cooldowns(
                    model_ids=list(registered_ids), parent_otel_span=None
                )
            )
            await callback.async_repin_session_after_success(litellm_params, model_group, registered_ids - cooldown_ids)
        except Exception as error:  # noqa: BLE001  # Optional pin updates must not interrupt success accounting
            verbose_router_logger.debug("Session re-pinning failed: %s", error)

    async def increment_deployment_usage_for_response(
        self,
        response: object,
        request_kwargs: dict[str, object],
    ) -> None:
        if response is None:
            return
        try:
            deployment_metadata: Final = find_deployment_metadata(request_kwargs)
            model_group: Final = request_kwargs.get("model")
            if deployment_metadata is None or not isinstance(model_group, str):
                return
            model_info: Final = deployment_metadata["model_info"]
            deployment_id: Final = model_info.get("id") if isinstance(model_info, dict) else None
            if deployment_id is None:
                return
            total_tokens: Final = response_total_token_count(response)
            deployment_name: Final = deployment_metadata.get("deployment")
            deployment_metadata[ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY] = total_tokens
            try:
                await self._increment_deployment_usage(
                    deployment_id=str(deployment_id),
                    deployment_name=deployment_name if isinstance(deployment_name, str) else None,
                    model_group=model_group,
                    total_tokens=total_tokens,
                    rpm_increment=1,
                    parent_otel_span=get_parent_otel_span_from_kwargs(request_kwargs),
                )
            except Exception:
                deployment_metadata.pop(ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY, None)
                raise
        except Exception as e:
            verbose_router_logger.debug(
                "litellm.router.Router::increment_deployment_usage_for_response(): Exception occured - %s", e
            )

    async def _increment_deployment_usage(
        self,
        *,
        deployment_id: str,
        deployment_name: str | None,
        model_group: str,
        total_tokens: float,
        rpm_increment: int,
        parent_otel_span: Span | None,
    ) -> str | None:
        from litellm.types.caching import RedisPipelineIncrementOperation

        if total_tokens <= 0 and rpm_increment <= 0:
            return None
        if not self._cached_deployment_tracks_usage(deployment_id, model_group):
            return None

        current_minute: Final = get_utc_datetime().strftime("%H-%M")  # use the same timezone regardless of system clock
        tpm_key: Final = RouterCacheEnum.TPM.value.format(
            id=deployment_id, current_minute=current_minute, model=deployment_name
        )
        rpm_key: Final = RouterCacheEnum.RPM.value.format(
            id=deployment_id, current_minute=current_minute, model=deployment_name
        )
        pipeline_operations: Final[list[RedisPipelineIncrementOperation]] = [
            RedisPipelineIncrementOperation(key=key, increment_value=increment_value, ttl=RoutingArgs.ttl.value)
            for key, increment_value in ((tpm_key, total_tokens), (rpm_key, rpm_increment))
        ]
        post_increment_values: Final = await self.cache.async_increment_cache_pipeline(
            increment_list=pipeline_operations,
            parent_otel_span=parent_otel_span,
        )
        if post_increment_values is not None and self.cache.redis_cache is not None:
            for operation, value in zip(pipeline_operations, post_increment_values):
                await self.cache.async_set_cache(
                    operation["key"], int(value), local_only=True, ttl=RoutingArgs.ttl.value
                )
        return tpm_key

    def _deployment_tracks_usage(self, deployment_id: str, model_group: str) -> bool:
        deployment_info: Final = self.get_deployment(model_id=deployment_id)
        if deployment_info is None:
            return False
        deployment_model_info: Final = self.get_router_model_info(
            deployment=deployment_info,
            received_model_name=model_group,
        )
        configured_limits: Final = (
            deployment_info.get("tpm", None),
            deployment_info.get("rpm", None),
            deployment_info.litellm_params.tpm,
            deployment_info.litellm_params.rpm,
            deployment_model_info.get("tpm", None),
            deployment_model_info.get("rpm", None),
        )
        ## Nothing to track only when neither tpm/rpm nor itpm/otpm limits are
        ## set. IO deployments still record TPM/RPM usage here so TPM-aware
        ## routing strategies see their real load in mixed model groups; their
        ## itpm/otpm enforcement runs separately in ModelRateLimitingCheck.
        return any(limit is not None for limit in configured_limits) or deployment_has_io_token_limits(
            deployment_info.model_dump()
        )

    def sync_deployment_callback_on_success(
        self,
        kwargs,  # kwargs to completion
        completion_response,  # response from completion
        start_time,
        end_time,  # start/end time
    ) -> str | None:
        """
        Tracks the number of successes for a deployment in the current minute (using in-memory cache)

        Returns:
        - key: str - The key used to increment the cache
        - None: if no key is found
        """
        if is_batch_retrieve_call_type(kwargs.get("call_type")):
            return None
        id = None
        if kwargs["litellm_params"].get("metadata") is None:
            pass
        else:
            model_group: Final = kwargs["litellm_params"]["metadata"].get("model_group", None)
            model_info: Final = kwargs["litellm_params"].get("model_info", {}) or {}
            id = model_info.get("id", None)
            if model_group is None or id is None:
                return None
            elif isinstance(id, int):
                id = str(id)

        if id is not None:
            key: Final = increment_deployment_successes_for_current_minute(
                litellm_router_instance=self,
                deployment_id=id,
            )
            return key

        return None

    def deployment_callback_on_failure(
        self,
        kwargs,  # kwargs to completion
        completion_response,  # response from completion
        start_time,
        end_time,  # start/end time
    ) -> bool:
        """
        2 jobs:
        - Tracks the number of failures for a deployment in the current minute (using in-memory cache)
        - Puts the deployment in cooldown if it exceeds the allowed fails / minute

        Returns:
        - True if the deployment should be put in cooldown
        - False if the deployment should not be put in cooldown
        """
        verbose_router_logger.debug("Router: Entering 'deployment_callback_on_failure'")
        try:
            exception: Final = kwargs.get("exception", None)

            if is_advisor_orchestration_failure(exception):
                verbose_router_logger.debug(
                    "Router: Exiting 'deployment_callback_on_failure' without cooldown. "
                    "Failure originated from advisor orchestration, not the selected deployment."
                )
                return False

            # Cache litellm_params to avoid repeated dict lookups
            litellm_params: Final = kwargs.get("litellm_params", {})
            _model_info: Final = litellm_params.get("model_info", {})

            if is_background_response_poll_not_found(exception, litellm_params):
                verbose_router_logger.debug(
                    "Router: Exiting 'deployment_callback_on_failure' without cooldown. "
                    "Provider 404 came from the background response poll, not the deployment's health."
                )
                return False

            if is_video_lookup_not_found(exception, kwargs):
                verbose_router_logger.debug(
                    "Router: Exiting 'deployment_callback_on_failure' without cooldown. "
                    "Provider 404 says the video is not on this deployment's account, not that the deployment is down."
                )
                return False

            exception_status: Final = getattr(exception, "status_code", "")

            if is_caller_timeout_408(kwargs, exception_status):
                verbose_router_logger.debug(
                    "Router: Exiting 'deployment_callback_on_failure' without cooldown. "
                    "A timeout the caller set caused this 408, not the deployment's health."
                )
                return False

            caller_failure_model_info: Final = (
                _MODEL_INFO_ADAPTER.validate_python(_model_info) if isinstance(_model_info, dict) else None
            )
            caller_failure_model_id: Final = (
                caller_failure_model_info.get("id") if caller_failure_model_info is not None else None
            )
            caller_failure_deployment: Final = (
                self.get_deployment(model_id=caller_failure_model_id)
                if isinstance(caller_failure_model_id, str)
                else None
            )
            if is_caller_scoped_auth_failure(caller_failure_deployment, exception_status):
                verbose_router_logger.debug(
                    "Router: Exiting 'deployment_callback_on_failure' without cooldown. "
                    "Caller-scoped OAuth authentication failed, not the deployment."
                )
                return False

            exception_headers: Final = litellm.litellm_core_utils.exception_mapping_utils._get_response_headers(
                original_exception=exception
            )

            # Determine cooldown time with priority: deployment config > response header > router default
            deployment_cooldown: Final = _first_present(
                _model_info if isinstance(_model_info, dict) else None, litellm_params, key="cooldown_time"
            )

            header_cooldown = None
            if exception_headers is not None:
                header_cooldown = litellm.utils._get_retry_after_from_exception_header(
                    response_headers=exception_headers
                )
            ##############################################
            # Logic to determine cooldown time
            # 1. Check if a cooldown time is set in the deployment config
            # 2. Check if a cooldown time is set in the response header
            # 3. If no cooldown time is set, use the router default cooldown time
            ##############################################
            if deployment_cooldown is not None and deployment_cooldown >= 0:
                _time_to_cooldown = deployment_cooldown
            elif header_cooldown is not None and header_cooldown >= 0:
                _time_to_cooldown = header_cooldown
            else:
                _time_to_cooldown = self.cooldown_time

            if isinstance(_model_info, dict):
                deployment_id: Final[str | None] = _model_info.get("id")
                if deployment_id is None:
                    return False
                increment_deployment_failures_for_current_minute(
                    litellm_router_instance=self,
                    deployment_id=deployment_id,
                )
                result: Final = set_cooldown_deployments(
                    litellm_router_instance=self,
                    exception_status=exception_status,
                    original_exception=exception,
                    deployment=deployment_id,
                    time_to_cooldown=_time_to_cooldown,
                    requested_model_group=(
                        get_litellm_metadata_from_kwargs(
                            cast("dict[str, object]", kwargs)  # cast-ok: untyped kwargs dict
                        )
                        or {}
                    ).get("model_group"),
                    request_kwargs=cast("dict[str, object]", kwargs),  # cast-ok: kwargs is the untyped request dict
                )  # setting deployment_id in cooldown deployments

                return result
            else:
                verbose_router_logger.debug(
                    "Router: Exiting 'deployment_callback_on_failure' without cooldown. No model_info found."
                )
                return False

        except Exception as e:
            raise e

    async def async_deployment_callback_on_failure(
        self, kwargs, completion_response: object | None, start_time, end_time
    ):
        """
        Update RPM usage for a deployment
        """
        if is_batch_retrieve_call_type(kwargs.get("call_type")):
            return
        deployment_name: Final = kwargs["litellm_params"]["metadata"].get(
            "deployment", None
        )  # handles wildcard routes - by giving the original name sent to `litellm.completion`
        model_group: Final = kwargs["litellm_params"]["metadata"].get("model_group", None)
        model_info: Final = kwargs["litellm_params"].get("model_info", {}) or {}
        id = model_info.get("id", None)
        if model_group is None or id is None:
            return
        elif isinstance(id, int):
            id = str(id)
        parent_otel_span: Final = get_parent_otel_span_from_kwargs(kwargs)

        dt: Final = get_utc_datetime()
        current_minute: Final = dt.strftime("%H-%M")  # use the same timezone regardless of system clock

        ## RPM
        rpm_key: Final = RouterCacheEnum.RPM.value.format(id=id, current_minute=current_minute, model=deployment_name)
        with service_target(ROUTER_USAGE_TARGET):
            await self.cache.async_increment_cache(
                key=rpm_key,
                value=1,
                parent_otel_span=parent_otel_span,
                ttl=RoutingArgs.ttl.value,
            )

    def _get_metadata_variable_name_from_kwargs(self, kwargs: dict) -> Literal["metadata", "litellm_metadata"]:
        """
        Helper to return what the "metadata" field should be called in the request data

        - New endpoints return `litellm_metadata`
        - Old endpoints return `metadata`

        Context:
        - LiteLLM used `metadata` as an internal field for storing metadata
        - OpenAI then started using this field for their metadata
        - LiteLLM is now moving to using `litellm_metadata` for our metadata
        """
        return get_metadata_variable_name_from_kwargs(kwargs)

    def log_retry(self, kwargs: dict, e: Exception) -> dict:
        """
        When a retry or fallback happens, record which model group, deployment and attempt just failed and why,
        and count it toward the request-wide num_retries_per_request cap
        """
        from litellm.litellm_core_utils.get_litellm_params import get_base_model_from_litellm_call_metadata
        from litellm.types.router import RetryAttemptRecord

        _metadata_var: Final = "litellm_metadata" if "litellm_metadata" in kwargs else "metadata"
        request_metadata: Final = cast("Mapping[str, object]", kwargs[_metadata_var])  # cast-ok: untyped kwargs
        model_group: Final = kwargs.get("model")
        model_info: Final = request_metadata.get("model_info")
        deployment_id: Final = model_info.get("id") if isinstance(model_info, Mapping) else None
        base_model: Final = get_base_model_from_litellm_call_metadata(
            metadata=request_metadata
        ) or self._deployment_params_base_model(deployment_id if isinstance(deployment_id, str) else None)
        deployment_model: Final = request_metadata.get("deployment")
        attempted_retries: Final = request_metadata.get("attempted_retries")
        attempt_record: Final[RetryAttemptRecord] = {
            "model_group": model_group if isinstance(model_group, str) else None,
            "deployment_id": deployment_id if isinstance(deployment_id, str) else None,
            "model": deployment_model if isinstance(deployment_model, str) else None,
            "base_model": base_model if isinstance(base_model, str) else None,
            "exception_type": type(e).__name__,
            "exception_string": str(e),
            "attempted_retries": attempted_retries if type(attempted_retries) is int else None,
        }
        earlier_breadcrumbs: Final = request_metadata.get("previous_models")
        kept_breadcrumbs: Final[tuple[object, ...]] = (
            tuple(earlier_breadcrumbs)[-(RETRY_BREADCRUMB_LIMIT - 1) :]
            if isinstance(earlier_breadcrumbs, (list, tuple))
            else ()
        )
        breadcrumbs: Final = (*kept_breadcrumbs, attempt_record)
        earlier: Final = request_metadata.get("request_retry_count")
        request_retry_count: Final = (earlier if type(earlier) is int and 0 <= earlier else 0) + 1
        decision: Final = request_metadata.get("routing_decision")
        router_name: Final = decision.get("router_model_name") if isinstance(decision, Mapping) else None
        if isinstance(router_name, str):
            phase_event(
                "litellm.routing.retry",
                {
                    "litellm.routing.router_model_name": router_name,
                    "litellm.retry.count": request_retry_count,
                    "error.type": type(e).__name__,
                    **({"litellm.deployment.model_group": model_group} if isinstance(model_group, str) else {}),
                    **({"litellm.deployment.id": deployment_id} if isinstance(deployment_id, str) else {}),
                },
            )
        kwargs[_metadata_var]["previous_models"] = breadcrumbs  # rebind-ok: the logging object already holds this dict
        kwargs[_metadata_var]["request_retry_count"] = request_retry_count  # rebind-ok: same dict, read by the cap
        return kwargs

    def _deployment_params_base_model(self, deployment_id: str | None) -> str | None:
        """The ``base_model`` a deployment declares under ``litellm_params`` (the Azure named-deployment
        shape), the same key the success path's ``get_base_model_from_metadata`` reads."""
        deployment: Final = self.get_deployment(model_id=deployment_id) if deployment_id is not None else None
        if deployment is None:
            return None
        extra_params: Final = cast(  # cast-ok: pydantic extras are untyped, base_model is isinstance-checked below
            "Mapping[str, object]", deployment.litellm_params.model_extra or {}
        )
        base_model: Final = extra_params.get("base_model")
        return base_model if isinstance(base_model, str) else None

    def _update_usage(self, deployment_id: str, parent_otel_span: Span | None) -> int:
        """
        Update deployment rpm for that minute

        Returns:
        - int: request count
        """
        rpm_key: Final = deployment_id

        request_count = self.cache.get_cache(key=rpm_key, parent_otel_span=parent_otel_span, local_only=True)
        if request_count is None:
            request_count = 1
            self.cache.set_cache(key=rpm_key, value=request_count, local_only=True, ttl=60)  # only store for 60s
        else:
            request_count += 1
            self.cache.set_cache(key=rpm_key, value=request_count, local_only=True)  # don't change existing ttl

        return request_count

    def _has_default_fallbacks(self) -> bool:
        if self.fallbacks is None:
            return False
        for fallback in self.fallbacks:
            if isinstance(fallback, dict):
                if "*" in fallback:
                    return True
        return False

    def _has_content_policy_fallback(self, model_group: str, kwargs: Mapping[str, Any]) -> bool:
        """
        Whether a content-policy fallback would resolve for this request, keyed the same way
        async_function_with_fallbacks_common_utils resolves it: the tier a pre-routing hook
        selected wins over the requested group. Raising without this returning True would turn
        a deliverable response into an error the fallback chain cannot recover from.
        """
        content_policy_fallbacks: Final = kwargs.get("content_policy_fallbacks", self.content_policy_fallbacks)
        if content_policy_fallbacks is not None:
            return has_unattempted_fallback_target(
                self._get_fallback_model_group_for_lookup_groups(
                    fallbacks=content_policy_fallbacks,
                    lookup_groups=fallback_lookup_groups(kwargs, model_group),
                ),
                kwargs,
            )
        if self._has_default_fallbacks():
            return True
        verbose_router_logger.debug(
            "No content-policy fallback available. Returning original response. model=%s, content_policy_fallbacks=%s",
            model_group,
            content_policy_fallbacks,
        )
        return False

    def _refusal_fallback_available(self, model_group: str, kwargs: Mapping[str, Any]) -> bool:
        """
        Whether a safeguard refusal can actually be recovered by the dispatcher. A configured
        content-policy list is authoritative; with none configured at all, the dispatcher falls
        through to the generic fallbacks lookup, so the gate mirrors that reachability and arms
        on a resolving generic chain (tier first, then the requested group, then "*").
        """
        if fallbacks_disabled_for_request(kwargs):
            return False
        content_policy_fallbacks: Final = kwargs.get("content_policy_fallbacks", self.content_policy_fallbacks)
        if content_policy_fallbacks is not None:
            return self._has_content_policy_fallback(model_group, kwargs)
        if self._has_default_fallbacks():
            return True
        fallbacks: Final = kwargs.get("fallbacks", self.fallbacks)
        if fallbacks is None:
            return False
        resolved, _ = get_fallback_model_group_for_lookup_groups(
            fallbacks=fallbacks,
            lookup_groups=fallback_lookup_groups(kwargs, model_group),
        )
        return has_unattempted_fallback_target(resolved, kwargs)

    def _anthropic_messages_order_levels(self, model_group: str, kwargs: Mapping[str, Any]) -> tuple[int, ...]:
        """
        The distinct deployment order levels the fallback dispatcher would see for this request,
        computed the same way: the tier a pre-routing hook selected wins over the requested group.
        """
        request_team_id: Final[str | None] = (kwargs.get("metadata", {}) or {}).get("user_api_key_team_id")
        order_model_group: Final = get_pre_routing_selection(kwargs) or model_group
        all_deployments: Final = self.get_model_list(model_name=order_model_group, team_id=request_team_id) or ()
        return tuple(
            sorted(
                {
                    litellm.utils.get_deployment_order(d)
                    for d in all_deployments
                    if litellm.utils.get_deployment_order(d) is not None
                }
            )
        )

    def _anthropic_messages_stream_can_fall_back(self, model_group: str, kwargs: Mapping[str, Any]) -> bool:
        """
        Whether async_function_with_fallbacks_common_utils could still route a
        MidStreamFallbackError somewhere for this request (order levels, weighted
        failover, content-policy or generic fallbacks), which is the only case where
        holding lifecycle frames back from the client buys a clean retry. Errs toward
        True whenever a dispatcher path might reach a fallback.
        """
        if fallbacks_disabled_for_request(kwargs):
            return False
        if self.enable_weighted_failover:
            return True
        order_levels: Final = self._anthropic_messages_order_levels(model_group, kwargs)
        if len(order_levels) > 1:
            current_target: Final = kwargs.get("_target_order")
            skip_up_to: Final = current_target if current_target is not None else order_levels[0]
            if any(o > skip_up_to for o in order_levels):
                return True
        content_policy_fallbacks: Final = kwargs.get("content_policy_fallbacks", self.content_policy_fallbacks)
        if content_policy_fallbacks is not None and self._has_content_policy_fallback(model_group, kwargs):
            return True
        fallbacks: Final = kwargs.get("fallbacks", self.fallbacks)
        if not fallbacks:
            return False
        if check_non_standard_fallback_format(fallbacks=fallbacks):
            return True
        resolved, _ = get_fallback_model_group_for_lookup_groups(
            fallbacks=fallbacks,
            lookup_groups=fallback_lookup_groups(kwargs, model_group),
        )
        return has_unattempted_fallback_target(resolved, kwargs)

    def _should_raise_content_policy_error(self, model: str, response: ModelResponse, kwargs: dict) -> bool:
        """
        Determines if a content policy error should be raised.

        Only raised if a fallback is available.

        Else, original response is returned.
        """
        if response.choices and len(response.choices) > 0:
            if response.choices[0].finish_reason != "content_filter":
                return False

        return self._has_content_policy_fallback(model, kwargs)

    def _should_raise_anthropic_refusal_error(
        self, model: str, original_generic_function: Callable, response: object, kwargs: Mapping[str, object]
    ) -> bool:
        """
        The /v1/messages twin of _should_raise_content_policy_error: an Anthropic safeguard
        refusal (stop_reason "refusal" carrying stop_details) re-enters the fallback chain only
        when a content-policy fallback is configured; a plain refusal without stop_details, or
        any response with nothing configured, is returned to the client unchanged.
        """
        from litellm.llms.anthropic.messages.native.utils import (
            get_safeguard_refusal_stop_details,
        )

        if getattr(original_generic_function, "__name__", "") != "anthropic_messages":
            return False
        if get_safeguard_refusal_stop_details(response) is None:
            return False
        return self._refusal_fallback_available(model, kwargs)

    def _get_healthy_deployments(self, model: str, parent_otel_span: Span | None):
        _all_deployments: list = []
        try:
            _, _all_deployments = self._common_checks_available_deployment(
                model=model,
            )
            if isinstance(_all_deployments, dict):
                return []
        except Exception:
            pass

        unhealthy_deployments: Final = get_cooldown_deployments(
            litellm_router_instance=self, parent_otel_span=parent_otel_span
        )
        unhealthy_set: Final = set(unhealthy_deployments)
        healthy_deployments: list = [d for d in _all_deployments if d["model_info"]["id"] not in unhealthy_set]
        healthy_deployments = self._filter_blocked_deployments(healthy_deployments)

        return healthy_deployments, _all_deployments

    async def _async_get_healthy_deployments(
        self, model: str, parent_otel_span: Span | None, request_kwargs: Mapping[str, object] | None = None
    ) -> tuple[list[dict], list[dict]]:
        """
        Returns Tuple of:
        - Tuple[List[Dict], List[Dict]]:
            1. healthy_deployments: list of healthy deployments the caller of ``request_kwargs`` may use
            2. all_deployments: list of all deployments that caller may use
        """
        _all_deployments: list = []
        try:
            _, _all_deployments = self._common_checks_available_deployment(
                model=model,
                request_kwargs=request_kwargs,
            )
            if isinstance(_all_deployments, dict):
                return [], _all_deployments
        except Exception:
            pass

        unhealthy_deployments: Final = await async_get_cooldown_deployments(
            litellm_router_instance=self, parent_otel_span=parent_otel_span
        )
        # Convert to set for O(1) lookup instead of O(n)
        unhealthy_deployments_set: Final = set(unhealthy_deployments)
        healthy_deployments: list = [
            d for d in _all_deployments if d["model_info"]["id"] not in unhealthy_deployments_set
        ]
        healthy_deployments = self._filter_blocked_deployments(healthy_deployments)
        return healthy_deployments, _all_deployments

    def routing_strategy_pre_call_checks(self, deployment: dict):
        """
        Mimics 'async_routing_strategy_pre_call_checks'

        Ensures consistent update rpm implementation for 'usage-based-routing-v2'

        Returns:
        - None

        Raises:
        - Rate Limit Exception - If the deployment is over it's tpm/rpm limits
        """
        for _callback in self._routing_callbacks():
            if isinstance(_callback, CustomLogger):
                _callback.pre_call_check(deployment)

    async def async_routing_strategy_pre_call_checks(
        self,
        deployment: dict,
        parent_otel_span: Span | None,
        logging_obj: LiteLLMLogging | None = None,
    ):
        """
        For usage-based-routing-v2, enables running rpm checks before the call is made, inside the semaphore.

        -> makes the calls concurrency-safe, when rpm limits are set for a deployment

        Returns:
        - None

        Raises:
        - Rate Limit Exception - If the deployment is over it's tpm/rpm limits
        """
        for _callback in self._routing_callbacks():
            if isinstance(_callback, CustomLogger):
                try:
                    await _callback.async_pre_call_check(deployment, parent_otel_span)
                except litellm.RateLimitError as e:
                    ## LOG FAILURE EVENT
                    if logging_obj is not None:
                        asyncio.create_task(
                            logging_obj.dispatch_failure_handlers(
                                exception=e,
                                traceback_exception=traceback.format_exc(),
                                prefer_async_handlers=True,
                            )
                        )
                    set_cooldown_deployments(
                        litellm_router_instance=self,
                        exception_status=e.status_code,
                        original_exception=e,
                        deployment=deployment["model_info"]["id"],
                        time_to_cooldown=self.cooldown_time,
                    )
                    raise e
                except Exception as e:
                    ## LOG FAILURE EVENT
                    if logging_obj is not None:
                        asyncio.create_task(
                            logging_obj.dispatch_failure_handlers(
                                exception=e,
                                traceback_exception=traceback.format_exc(),
                                prefer_async_handlers=True,
                            )
                        )
                    raise e

    def _deployment_as_configured_now(self, deployment: Mapping[str, object]) -> Mapping[str, object]:
        model_info: Final = deployment.get("model_info")
        model_id: Final = (
            _MODEL_INFO_ADAPTER.validate_python(model_info).get("id") if isinstance(model_info, dict) else None
        )
        current: Final = self.get_deployment(model_id) if isinstance(model_id, str) else None
        return deployment if current is None else current.model_dump(exclude_none=True)

    async def async_admit_deployment_turn(
        self, model_group: str, opened_on: Mapping[str, object], request_kwargs: Mapping[str, object]
    ) -> None:
        deployment: Final = self._deployment_as_configured_now(opened_on)
        budget_limiter: Final = self.get_router_deployment_budget_limiter()
        if budget_limiter is not None:
            try:
                await budget_limiter.async_filter_deployments(  # pyright: ignore[reportUnknownMemberType]  # legacy signature takes untyped deployment dicts
                    model=model_group,
                    healthy_deployments=[dict(deployment)],
                    messages=None,
                    request_kwargs=dict(request_kwargs),
                )
            except ValueError as e:
                raise litellm.RateLimitError(
                    message=str(e),
                    llm_provider="",
                    model=model_group,
                    category=litellm.RateLimitErrorCategory.LITELLM_RATE_LIMIT,
                ) from e
        await self._admit_turn_through_routing_strategy(model_group, deployment, request_kwargs)
        await self.async_routing_strategy_pre_call_checks(  # pyright: ignore[reportUnknownMemberType]  # legacy signature takes an untyped deployment dict
            deployment=dict(without_io_token_limits(deployment)), parent_otel_span=None
        )

    async def _admit_turn_through_routing_strategy(
        self, model_group: str, deployment: Mapping[str, object], request_kwargs: Mapping[str, object]
    ) -> None:
        routing_kwargs: Final = {key: value for key, value in request_kwargs.items() if key != "litellm_logging_obj"}
        strategy, selector = self._get_routing_context(  # pyright: ignore[reportUnknownMemberType]  # legacy signature takes an untyped request dict
            model=model_group, request_kwargs=routing_kwargs
        )
        if strategy == "simple-shuffle" or selector is None:
            return
        selected: Final = await self._select_deployment_async(  # pyright: ignore[reportUnknownMemberType]  # legacy signature takes untyped deployment dicts
            strategy=strategy,
            selector=selector,
            model=model_group,
            healthy_deployments=[dict(deployment)],
            messages=None,
            input=None,
            request_kwargs=routing_kwargs,
        )
        if selected is None:
            no_deployment: Final = await async_raise_no_deployment_exception(
                litellm_router_instance=self, model=model_group, parent_otel_span=None
            )
            raise litellm.RateLimitError(
                message=str(no_deployment),
                llm_provider="",
                model=model_group,
                category=litellm.RateLimitErrorCategory.LITELLM_RATE_LIMIT,
            ) from no_deployment
        await self._async_override_selector_pre_call_check(  # pyright: ignore[reportUnknownMemberType]  # legacy signature takes an untyped deployment dict
            strategy, selector, dict(deployment), None
        )

    @contextlib.asynccontextmanager
    async def _deployment_slot(
        self, deployment: dict, kwargs: Mapping[str, object], parent_otel_span: Span | None
    ) -> AsyncGenerator[None, None]:
        """Holds the deployment's max_parallel_requests slot, if it has one, around the provider call. Routing
        strategy pre-call checks run inside the slot so their rpm accounting stays concurrency-safe."""
        max_parallel_requests_limit: Final = self._get_client(
            deployment=deployment,
            kwargs=kwargs,
            client_type="max_parallel_requests",
        )
        async with contextlib.AsyncExitStack() as slot:
            if isinstance(max_parallel_requests_limit, MaxParallelRequestsLimit):
                slot.enter_context(max_parallel_requests_limit)
            await self.async_routing_strategy_pre_call_checks(deployment=deployment, parent_otel_span=parent_otel_span)
            yield

    async def async_callback_filter_deployments(
        self,
        model: str,
        healthy_deployments: list[dict],
        messages: list[AllMessageValues] | None,
        parent_otel_span: Span | None,
        request_kwargs: dict | None = None,
        logging_obj: LiteLLMLogging | None = None,
    ):
        """
        For usage-based-routing-v2, enables running rpm checks before the call is made, inside the semaphore.

        -> makes the calls concurrency-safe, when rpm limits are set for a deployment

        Returns:
        - None

        Raises:
        - Rate Limit Exception - If the deployment is over it's tpm/rpm limits
        """
        returned_healthy_deployments = healthy_deployments
        for _callback in self._routing_callbacks():
            if isinstance(_callback, CustomLogger):
                try:
                    returned_healthy_deployments = await _callback.async_filter_deployments(
                        model=model,
                        healthy_deployments=returned_healthy_deployments,
                        messages=messages,
                        request_kwargs=request_kwargs,
                        parent_otel_span=parent_otel_span,
                    )
                except Exception as e:
                    ## LOG FAILURE EVENT
                    if logging_obj is not None:
                        asyncio.create_task(
                            logging_obj.dispatch_failure_handlers(
                                exception=e,
                                traceback_exception=traceback.format_exc(),
                                prefer_async_handlers=True,
                            )
                        )
                    raise e
        return returned_healthy_deployments

    @staticmethod
    def _json_default_stable_id(value: object) -> str:
        """json.dumps default= for generate_model_id: plain str() on an arbitrary
        object (e.g. a RoutingPlugin instance) falls back to object.__repr__'s
        `<module.Class object at 0x...>`, so the hash -- and deployment id -- would
        change every restart. Use the class name instead, stable across restarts."""
        return f"{type(value).__module__}.{type(value).__qualname__}"

    @staticmethod
    def generate_model_id(model_group: str, litellm_params: dict) -> str:  # mutable-ok: hashed read-only
        """
        Helper function to consistently generate the same id for a deployment

        - create a string from all the litellm params
        - hash
        - use hash as id
        """
        # Optimized: Use list and join instead of string concatenation in loop
        # This avoids creating many temporary string objects (O(n) vs O(n²) complexity)
        parts: Final = [model_group]
        for k, v in litellm_params.items():
            if isinstance(k, str):
                parts.append(k)
            elif isinstance(k, dict):
                parts.append(json.dumps(k, default=Router._json_default_stable_id))
            else:
                parts.append(str(k))

            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, dict):
                parts.append(json.dumps(v, default=Router._json_default_stable_id))
            else:
                parts.append(str(v))

        concat_str: Final = "".join(parts)
        hash_object: Final = hashlib.sha256(concat_str.encode())

        return hash_object.hexdigest()

    @staticmethod
    def _inherit_builtin_cache_pricing(model_info: dict, backend_model: str, custom_llm_provider: str | None) -> None:
        """Fill missing cache pricing on a custom-priced deployment entry from
        the backend model's built-in cost map entry, so a deployment that
        only spells out ``input_cost_per_token``/``output_cost_per_token``
        does not silently bill cache_read/cache_creation at 0.

        User-specified cache fields always win; only ``None``/missing entries
        are inherited, and a user-set cache_read price also covers explicit
        cache hits. No-op when the backend model has no canonical entry.
        """
        cache_fields: Final = (
            "cache_creation_input_token_cost",
            "cache_creation_input_token_cost_above_1hr",
            "cache_creation_input_token_cost_above_200k_tokens",
            "cache_read_input_token_cost",
            "cache_read_input_token_cost_above_200k_tokens",
        )
        inherited_fields: Final = (
            cache_fields
            if model_info.get("cache_read_input_token_cost") is not None
            else (*cache_fields, "explicit_cache_read_input_token_cost")
        )
        missing_fields: Final = tuple(f for f in inherited_fields if model_info.get(f) is None)
        if not missing_fields:
            return
        try:
            backend_info: Final = litellm.get_model_info(model=backend_model, custom_llm_provider=custom_llm_provider)
        except Exception:
            return
        for field in missing_fields:
            backend_value = backend_info.get(field)
            if backend_value is not None:
                model_info[field] = backend_value

    @staticmethod
    def _cost_map_backend_model(deployment: Deployment) -> str:
        model_info_base_model: Final = deployment.model_info.base_model
        if isinstance(model_info_base_model, str) and model_info_base_model:
            return model_info_base_model
        params_base_model: Final = deployment.litellm_params.get("base_model")
        if isinstance(params_base_model, str) and params_base_model:
            return params_base_model
        params: Final = _LITELLM_PARAMS_ADAPTER.validate_python(deployment.litellm_params.model_dump(exclude_none=True))
        invoked_model: Final = params.get("model_id")
        if (
            isinstance(invoked_model, str)
            and invoked_model
            and (
                deployment.litellm_params.custom_llm_provider == "bedrock"
                or deployment.litellm_params.model.startswith("bedrock/")
            )
        ):
            profile_model: Final = invoked_model if invoked_model.startswith("bedrock/") else f"bedrock/{invoked_model}"
            profile_info: Final = lookup_model_info_for_registration(model=profile_model, custom_llm_provider="bedrock")
            if not isinstance(profile_info, ModelInfoLookupMiss):
                return profile_model
        return deployment.litellm_params.model

    @staticmethod
    def _inherit_builtin_service_tier_pricing(
        model_info: dict,  # mutable-ok: deployment cost-map entry filled in place
        backend_model: str,
        custom_llm_provider: str | None,
    ) -> None:
        """Inherit missing tier rates so a standalone entry does not fall back to custom standard rates."""
        if ptu_terms(model_info) is not None and is_ptu_cost_attribution_enabled():
            return
        if all(model_info.get(field) is None for field in ("input_cost_per_token", "output_cost_per_token")):
            return
        try:
            backend_info: Final = litellm.get_model_info(model=backend_model, custom_llm_provider=custom_llm_provider)
        except Exception:  # noqa: BLE001  # get_model_info raises plain Exception for an unmapped backend model
            return
        backend_entry: Final = litellm.model_cost.get(backend_info.get("key") or "")
        if not isinstance(backend_entry, dict):
            return
        for field, backend_value in backend_entry.items():
            if not field.endswith(SERVICE_TIER_COST_KEY_SUFFIXES):
                continue
            if model_info.get(field) is not None or backend_value is None:
                continue
            model_info[field] = copy.deepcopy(backend_value)

    @staticmethod
    def _inherit_builtin_base_rates_for_off_peak(
        model_info: dict,  # mutable-ok: cost-map entry filled in place
        backend_model: str,
        custom_llm_provider: str | None,
    ) -> None:
        """Fill missing pricing fields on a deployment entry that only sets
        ``off_peak_pricing``, from the backend model's built-in cost map entry.

        Cost lookup selects the deployment-scoped entry over the shared backend
        entry only when the deployment entry carries a base pricing field, and
        ``off_peak_pricing`` is deliberately kept off the shared entry, so a
        deployment spelling out only its off-peak schedule would otherwise
        never receive the discount. The backend model's entire canonical cost
        map entry is copied, field by field, so threshold, tiered,
        service-tier, cache, character, and per-second rates as well as
        companion billing fields like ``web_search_billing_unit`` and the
        regional uplift multipliers all carry over, and peak-hour billing
        through the deployment entry matches the shared backend entry exactly.
        The raw ``litellm.model_cost`` entry is the copy source rather than
        ``get_model_info``'s view of it, since that view synthesizes zero flat
        token rates for backends without one and storing those would mark a
        tiered-only backend explicitly priced free. Values are deep-copied to
        keep the builtin entry isolated. User-specified fields always win;
        no-op when any base pricing field is already set or the backend model
        has no canonical entry.
        """
        if not model_info.get("off_peak_pricing"):
            return
        if any(
            model_info.get(field) is not None
            for field in ("input_cost_per_token", "input_cost_per_second", "cost_per_second", "tiered_pricing")
        ):
            return
        try:
            backend_info: Final = litellm.get_model_info(model=backend_model, custom_llm_provider=custom_llm_provider)
        except Exception:  # noqa: BLE001  # get_model_info raises plain Exception for an unmapped backend model
            return
        backend_entry: Final = litellm.model_cost.get(backend_info.get("key") or "")
        if not isinstance(backend_entry, dict):
            return
        for field, backend_value in backend_entry.items():
            if model_info.get(field) is not None or backend_value is None:
                continue
            model_info[field] = copy.deepcopy(backend_value)

    @staticmethod
    def _inherit_builtin_tiered_output_rate(
        model_info: dict, backend_model: str, custom_llm_provider: str | None
    ) -> None:
        """Fill a missing entry-level output rate on a deployment entry whose tier
        table omits one, from the backend model's built-in cost map entry.

        A deployment's custom pricing is registered as its own standalone
        ``litellm.model_cost`` entry holding only the supplied fields, and the
        tiered-cost output fallback reads that same entry, so a tier table that
        spells out only input-side rates would bill every completion at 0.

        A user-specified ``output_cost_per_token`` always wins. No-op without a
        tier table, when every tier declares its own output rate, or when the
        backend model has no canonical entry or no flat output rate:
        ``get_model_info`` synthesizes a zero for tiered-only backends, and
        storing that zero would mark the deployment as explicitly priced free.
        """
        tiers: Final = model_info.get("tiered_pricing")
        if not isinstance(tiers, list) or not tiers:
            return
        if model_info.get("output_cost_per_token") is not None:
            return
        if all(isinstance(tier, dict) and "output_cost_per_token" in tier for tier in tiers):
            return
        try:
            backend_info: Final = litellm.get_model_info(model=backend_model, custom_llm_provider=custom_llm_provider)
        except Exception:  # noqa: BLE001  # get_model_info raises plain Exception for an unmapped backend model
            return
        backend_rate: Final = backend_info.get("output_cost_per_token")
        if backend_rate:
            model_info["output_cost_per_token"] = backend_rate

    def _create_deployment(
        self,
        deployment_info: dict,
        _model_name: str,
        _litellm_params: dict,
        _model_info: dict,
        *,
        declared_id: str | None = None,
        duplicate_ids: frozenset[str] = frozenset(),
    ) -> Deployment | None:
        """
        Create a deployment object and add it to the model list

        If the deployment is not active for the current environment, it is ignored

        Returns:
        - Deployment: The deployment object
        - None: If the deployment is not active for the current environment (if 'supported_environments' is set in litellm_params)
        """
        try:
            config_sourced: Final = _model_info.get("db_model") is not True
            identity_error: Final = (
                ptu_identity_error(
                    declared_id=declared_id,
                    taken=declared_id in duplicate_ids,
                    current_id=_model_info.get("id"),
                    model_name=_model_name,
                )
                if config_sourced and ptu_terms(_model_info) is not None
                else None
            )
            ptu_error: Final = (
                (ptu_config_error(_model_info, model_name=_model_name) or identity_error) if config_sourced else None
            )
            declares_split: Final = declares_ptu_shares(_model_info)
            if ptu_error is not None and (declares_split or is_ptu_cost_attribution_enabled()):
                raise ValueError(ptu_error)
            access_windows_error: Final = access_windows_config_error(_model_info, model_name=_model_name)
            if access_windows_error is not None:
                raise ValueError(access_windows_error)
            capacity_warning: Final = (
                ptu_capacity_warning(
                    _model_name,
                    MappingProxyType(
                        {  # pyright: ignore[reportUnknownArgumentType]  # router deployment dicts are untyped
                            "model_info": _model_info,
                            "litellm_params": _litellm_params,
                        }
                    ),
                )
                if is_ptu_cost_attribution_enabled()
                else None
            )
            if capacity_warning is not None:
                verbose_router_logger.warning(capacity_warning)
            zeroed_pricing: Final = zeroed_ptu_pricing(_model_info, _litellm_params) if config_sourced else None
            merged_params: Final[Mapping[str, object]] = (
                _litellm_params if zeroed_pricing is None else MappingProxyType({**_litellm_params, **zeroed_pricing})
            )
            litellm_params: Final[LiteLLM_Params] = LiteLLM_Params.model_validate(dict(**merged_params))
            declared_params: Final = _LITELLM_PARAMS_ADAPTER.validate_python(_litellm_params)
            warn_on_provider_credential_mismatch(model_name=_model_name, litellm_params=declared_params)
            warn_on_ignored_model_capability_flags(model_name=_model_name, litellm_params=declared_params)
            deployment = without_model_capability_flags(
                Deployment(
                    **deployment_info,
                    model_name=_model_name,
                    litellm_params=litellm_params,
                    model_info=_model_info,
                )
            )
            Router._register_deployment_pricing(deployment)

            ## Check if LLM Deployment is allowed for this deployment
            if self.deployment_is_active_for_environment(deployment=deployment) is not True:
                verbose_router_logger.warning(
                    "Ignoring deployment %s as it is not active for environment %s",
                    deployment.model_name,
                    deployment.model_info["supported_environments"],
                )
                return None

            # Validate tag_regex patterns BEFORE adding the deployment so we never
            # have partially-initialised router state if a pattern is invalid.
            _tag_regex: Final = deployment.litellm_params.get("tag_regex") or []
            for pattern in _tag_regex:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(
                        f"Invalid regex in tag_regex for model '{deployment.model_name}': {pattern!r} — {exc}"
                    ) from exc

            deployment = self._add_deployment(deployment=with_admin_api_key_secret_reference(deployment))

            model: Final = deployment.to_json(exclude_none=True)

            self._add_model_to_list_and_index_map(model=model, model_id=deployment.model_info.id)
            return deployment
        except Exception as e:
            if self.ignore_invalid_deployments:
                if isinstance(e, litellm.BadRequestError):
                    self._provider_unresolved_deployments = (
                        *self._provider_unresolved_deployments,
                        partial(
                            self._create_deployment,
                            deployment_info=deployment_info,
                            _model_name=_model_name,
                            _litellm_params=_litellm_params,
                            _model_info=_model_info,
                            declared_id=declared_id,
                            duplicate_ids=duplicate_ids,
                        ),
                    )
                verbose_router_logger.exception(
                    "Error creating deployment: %s, ignoring and continuing with other deployments.", e
                )
                return None
            else:
                raise e

    def _is_auto_router_deployment(self, litellm_params: LiteLLM_Params) -> bool:
        """
        Check if the deployment is an auto-router deployment (semantic router).

        Returns True if the litellm_params model starts with "auto_router/"
        but NOT "auto_router/complexity_router" or "auto_router/adaptive_router"
        (which use the complexity-router and adaptive-router strategies).
        """
        return classify_strategy_router_model(litellm_params.model) == "semantic"

    @staticmethod
    def _deployment_tags(deployment: Deployment) -> tuple[str, ...]:
        """Deployment tags used to disambiguate strategy registries keyed by model_name."""
        return tuple(deployment.litellm_params.tags or ())

    def init_auto_router_deployment(self, deployment: Deployment):
        """
        Initialize the auto-router deployment.

        This will initialize the auto-router and add it to the auto-routers dictionary.
        """
        from litellm.router_strategy.auto_router.auto_router import AutoRouter

        auto_router_config_path: Final[str | None] = deployment.litellm_params.auto_router_config_path
        auto_router_config: Final[str | None] = deployment.litellm_params.auto_router_config
        if auto_router_config_path is None and auto_router_config is None:
            raise ValueError(
                "auto_router_config_path or auto_router_config is required for auto-router deployments. Please set it in the litellm_params"
            )

        default_model: Final[str | None] = deployment.litellm_params.auto_router_default_model
        if default_model is None:
            raise ValueError(
                "auto_router_default_model is required for auto-router deployments. Please set it in the litellm_params"
            )

        embedding_model: Final[str | None] = deployment.litellm_params.auto_router_embedding_model
        if embedding_model is None:
            raise ValueError(
                "auto_router_embedding_model is required for auto-router deployments. Please set it in the litellm_params"
            )

        max_input_chars: Final[int | None] = deployment.litellm_params.auto_router_max_input_chars

        autor_router: Final[AutoRouter] = AutoRouter(
            model_name=deployment.model_name,
            auto_router_config_path=auto_router_config_path,
            auto_router_config=auto_router_config,
            default_model=default_model,
            embedding_model=embedding_model,
            litellm_router_instance=self,
            max_input_chars=(max_input_chars if max_input_chars is not None else DEFAULT_AUTO_ROUTER_MAX_INPUT_CHARS),
        )
        self._register_pre_routing_strategy(
            registry=self.auto_routers,
            deployment=deployment,
            strategy=autor_router,
            strategy_label="Auto-router",
        )

    def _is_complexity_router_deployment(self, litellm_params: LiteLLM_Params) -> bool:
        """
        Check if the deployment is a complexity-router deployment.

        Returns True if the litellm_params model starts with "auto_router/complexity_router"
        """
        return classify_strategy_router_model(litellm_params.model) == "complexity"

    def config_deployments(self) -> Iterator[Mapping[str, object]]:
        """The model_list rows that came from config.yaml rather than the DB (``model_info.db_model`` unset)."""
        for deployment in self.model_list:
            if not isinstance(deployment, Mapping):
                continue
            model_info = deployment.get("model_info")
            if not (isinstance(model_info, Mapping) and model_info.get("db_model")):
                yield deployment

    def auto_router_capability_violation(self, capability: GatedAutoRouterCapability) -> str | None:
        """
        Why one more router claiming ``capability`` cannot join this router, or None when it can.

        Judged against every deployment currently on the model_list; an upsert pops the row being
        edited first, so an edit of an existing gated router keeps its own slot. The limit is
        resolved on every call through ``auto_router_capability_limit``; unset means unlimited,
        which is the SDK default, and the proxy injects a resolver backed by its license.
        """
        limit: Final = self.auto_router_capability_limit() if self.auto_router_capability_limit is not None else None
        others: Final = count_capability_routers(
            (deployment for deployment in self.model_list if isinstance(deployment, Mapping)),
            capability=capability,
        )
        return capability_limit_violation(capability=capability, held=others + 1, limit=limit)

    def init_complexity_router_deployment(self, deployment: Deployment):
        """
        Initialize the complexity-router deployment.

        This will initialize the complexity-router and add it to the complexity-routers dictionary.
        """
        # Import here to avoid circular imports — ComplexityRouter is a CustomLogger
        # subclass that imports litellm internals which depend on router.py.
        # This matches the AutoRouter pattern in init_auto_router_deployment above.
        from litellm.router_strategy.complexity_router.complexity_router import (
            ComplexityRouter,
        )
        from litellm.router_strategy.complexity_router.config import (
            ComplexityRouterConfig,
        )

        complexity_router_config: Final[dict | None] = deployment.litellm_params.complexity_router_config
        for capability in claimed_capabilities(complexity_router_config):
            if (limit_violation := self.auto_router_capability_violation(capability)) is not None:
                raise ValueError(limit_violation)

        default_model: Final = (
            ComplexityRouterConfig.model_validate(complexity_router_config).resolve_default_model(
                deployment.litellm_params.complexity_router_default_model
            )
            if complexity_router_config
            else deployment.litellm_params.complexity_router_default_model
        )

        if default_model is None:
            raise ValueError(
                "complexity_router_default_model is required for complexity-router deployments, "
                "or configure tiers in complexity_router_config. Please set it in the litellm_params"
            )

        complexity_router: Final[ComplexityRouter] = ComplexityRouter(
            model_name=deployment.model_name,
            default_model=default_model,
            litellm_router_instance=self,
            complexity_router_config=complexity_router_config,
        )
        self._register_pre_routing_strategy(
            registry=self.complexity_routers,
            deployment=deployment,
            strategy=complexity_router,
            strategy_label="Complexity-router",
        )
        if complexity_router._uses_deployment_pin:
            self._ensure_deployment_affinity_callback()

    def _is_adaptive_router_deployment(self, litellm_params: LiteLLM_Params) -> bool:
        """True when this deployment opts in via the `auto_router/adaptive_router` model prefix."""
        return classify_strategy_router_model(litellm_params.model) == "adaptive"

    def _deployment_participates_in_adaptive_routing(self, litellm_params: LiteLLM_Params) -> bool:
        """True when this deployment owns an `adaptive_routers` entry once finalized:
        a dedicated adaptive router, or a complexity router whose config enables the
        adaptive companion. Mirrors the two arms of
        `_finalize_adaptive_router_if_configured`, which is the registry's only writer."""
        if self._is_adaptive_router_deployment(litellm_params=litellm_params):
            return True
        if not self._is_complexity_router_deployment(litellm_params=litellm_params):
            return False
        config: Final = litellm_params.complexity_router_config
        if not config:
            return False
        adaptive_flag: Final[object] = config.get("adaptive")
        return bool(adaptive_flag)

    @staticmethod
    def _has_registered_strategy(
        registry: dict[str, list[TaggedPreRoutingStrategy[_PreRoutingStrategyT]]],
        model_name: str,
        tags: tuple[str, ...],
    ) -> bool:
        """True when a strategy for this (model_name, tags) pair is already registered."""
        return any(existing.tags == tags for existing in registry.get(model_name, []))

    def _register_pre_routing_strategy(
        self,
        registry: dict[str, list[TaggedPreRoutingStrategy[_PreRoutingStrategyT]]],
        deployment: Deployment,
        strategy: _PreRoutingStrategyT,
        strategy_label: str,
    ) -> None:
        """
        Register `strategy` under `deployment.model_name`, scoped by its tags.
        Reusing a `model_name` is allowed when tags differ; a repeat of the same
        (model_name, tags) pair is a misconfiguration and is rejected.
        """
        tags: Final = self._deployment_tags(deployment)
        if self._has_registered_strategy(registry, deployment.model_name, tags):
            raise ValueError(
                f"{strategy_label} deployment {deployment.model_name} with tags {list(tags)} already exists. "
                "Please use a different model name or set different tags."
            )
        registry[deployment.model_name] = [
            *registry.get(deployment.model_name, []),
            TaggedPreRoutingStrategy(
                tags=tags,
                strategy=strategy,
                definition_fingerprint=self._routing_definition_fingerprint(deployment),
                deployment=deployment,
            ),
        ]

    @staticmethod
    def _routing_definition_fingerprint(deployment: Deployment) -> str | None:
        """Identify configured routing fields, excluding external files and live strategy state."""
        try:
            definition: Final = json.dumps(
                deployment.litellm_params.model_dump(
                    include=STRATEGY_ROUTER_PARAM_FIELDS | {"model", "tags"}, exclude_none=True, warnings=False
                ),
                sort_keys=True,
                default=Router._json_default_stable_id,
            )
        except (TypeError, ValueError):
            return None
        return Router.generate_model_id(deployment.model_name, {"routing_definition": definition})

    @staticmethod
    def _unregister_pre_routing_strategy(
        registry: dict[str, list[TaggedPreRoutingStrategy[_PreRoutingStrategyT]]],
        model_name: str,
        tags: tuple[str, ...],
    ) -> bool:
        """Drop the strategy registered for this exact (model_name, tags) pair, leaving
        strategies registered under the same name with different tags in place. Returns
        whether anything was actually dropped."""
        existing: Final = registry.get(model_name, [])
        remaining: Final = [entry for entry in existing if entry.tags != tags]
        if len(remaining) == len(existing):
            return False
        if remaining:
            registry[model_name] = remaining
        else:
            registry.pop(model_name, None)
        return True

    def _unregister_pre_routing_strategy_for_deployment(self, deployment: Deployment) -> None:
        """
        Release the pre-routing strategy a deployment holds, so removing it from the
        model_list also frees its (model_name, tags) slot.

        Without this, re-adding the deployment (an edit arriving via upsert_deployment,
        or a router recreated under a name that was deleted earlier) hits the
        "already exists" guard in `_register_pre_routing_strategy`, which
        `ignore_invalid_deployments` swallows - the deployment then silently never
        makes it back into the model_list.

        Released from every registry rather than the first match, because registration is
        one-to-many: a complexity router configured with `adaptive` is also registered in
        `adaptive_routers` under the same (model_name, tags) by the deferred finalize pass.
        Guarded on the auto_router/ prefix so removing a *regular* deployment can't evict a
        router that merely shares its model_name.
        """
        if not deployment.litellm_params.model.startswith("auto_router/"):
            return
        model_name: Final = deployment.model_name
        tags: Final = self._deployment_tags(deployment)
        for registry in (self.auto_routers, self.complexity_routers, self.quality_routers):
            self._unregister_pre_routing_strategy(registry, model_name, tags)
        if self._unregister_pre_routing_strategy(self.adaptive_routers, model_name, tags):
            self._sync_adaptive_router_hooks()

    def _finalize_adaptive_router_if_configured(self) -> None:
        """Locate every adaptive-router deployment in the finalized model_list and
        build an AdaptiveRouter for each. Safe no-op when none are configured.
        Idempotent: skips any deployment whose (model_name, tags) pair is already
        initialized, so hot-reloads don't rebuild routers that would lose state."""
        for entry in self.model_list or []:
            lp = entry.get("litellm_params") if isinstance(entry, dict) else entry.litellm_params
            lp_model = (lp.get("model") if isinstance(lp, dict) else lp.model) if lp else None
            if not (lp_model and lp_model.startswith("auto_router/adaptive_router")):
                continue
            model_name = entry.get("model_name") if isinstance(entry, dict) else entry.model_name
            if not model_name or not lp:
                continue
            deployment = Deployment(
                model_name=model_name,
                litellm_params=(lp if not isinstance(lp, dict) else LiteLLM_Params.model_validate(lp)),
                model_info=(entry.get("model_info") if isinstance(entry, dict) else entry.model_info),
            )
            if self._has_registered_strategy(self.adaptive_routers, model_name, self._deployment_tags(deployment)):
                continue
            self.init_adaptive_router_deployment(deployment=deployment)

        for model_name, tagged_complexity_routers in self.complexity_routers.items():
            for tagged in tagged_complexity_routers:
                complexity_router = tagged.strategy
                if not complexity_router.config.adaptive:
                    continue
                if self._has_registered_strategy(self.adaptive_routers, model_name, tagged.tags):
                    continue
                adaptive_router: AdaptiveRouter | None = complexity_router._ensure_adaptive_router()
                if adaptive_router is not None:
                    self.adaptive_routers[model_name] = [
                        *self.adaptive_routers.get(model_name, []),
                        TaggedPreRoutingStrategy(
                            tags=tagged.tags,
                            strategy=adaptive_router,
                            definition_fingerprint=tagged.definition_fingerprint,
                            deployment=tagged.deployment,
                        ),
                    ]

        self._sync_adaptive_router_hooks()

    def _sync_adaptive_router_hooks(self) -> None:
        """Rebuild the AdaptiveRouterPostCallHook set so it is exactly one hook per
        currently registered adaptive router. Run at every point the adaptive registry
        changes, otherwise a released router keeps recording turns through its hook."""
        from litellm.router_strategy.adaptive_router.hooks import (
            AdaptiveRouterPostCallHook,
            AdaptiveRouterResponseHeaderHook,
        )

        for callback in litellm.logging_callback_manager.get_custom_loggers_for_type(AdaptiveRouterPostCallHook):
            litellm.logging_callback_manager.remove_callback_from_all_lists(callback)
        for tagged_adaptive_routers in self.adaptive_routers.values():
            for tagged in tagged_adaptive_routers:
                litellm.logging_callback_manager.add_litellm_callback(
                    AdaptiveRouterPostCallHook(adaptive_router=tagged.strategy)
                )
        if self.adaptive_routers:
            litellm.logging_callback_manager.add_litellm_callback(AdaptiveRouterResponseHeaderHook())

    def init_adaptive_router_deployment(self, deployment: Deployment) -> None:
        """
        Build an AdaptiveRouter instance for this deployment and register its
        post-call hook. Multiple adaptive routers can coexist on a single Router,
        keyed by `deployment.model_name`.

        `model_to_prefs` and `model_to_cost` are derived from the OTHER models
        already registered in `self.model_list` whose `model_name` appears in
        `available_models`. Models not yet registered fall back to defaults.
        """
        # Local import: AdaptiveRouter -> hooks -> classifier all import litellm
        # internals which transitively import this module. (AGENTS.md exception clause.)
        from litellm.router_strategy.adaptive_router.adaptive_router import (
            AdaptiveRouter,
        )
        from litellm.router_strategy.adaptive_router.hooks import (
            AdaptiveRouterPostCallHook,
        )
        from litellm.types.router import (
            AdaptiveRouterConfig,
            AdaptiveRouterPreferences,
        )

        raw_config: Final = deployment.litellm_params.adaptive_router_config
        if raw_config is None:
            raise ValueError("adaptive_router_config is required for adaptive-router deployments.")

        config: Final = AdaptiveRouterConfig(**raw_config)

        model_to_prefs: Final[dict[str, AdaptiveRouterPreferences]] = {}
        model_to_cost: Final[dict[str, float]] = {}
        # O(k) via the name→indices map: only touch deployments whose name
        # is listed in `available_models`, instead of scanning model_list.
        for name in config.available_models:
            deployments = self.deployments_named(name)
            if not deployments:
                continue
            d = deployments[0]
            mi_dict: dict[str, Any] = d.get("model_info") or {}
            prefs_raw = mi_dict.get("adaptive_router_preferences")
            if prefs_raw is not None:
                model_to_prefs[name] = AdaptiveRouterPreferences(**prefs_raw)

            # model_info is the conventional pricing location elsewhere in LiteLLM; litellm_params wins if set.
            cost = d.get("litellm_params", {}).get("input_cost_per_token")
            if cost is None:
                cost = mi_dict.get("input_cost_per_token")
            if cost is not None:
                model_to_cost[name] = float(cost)

        adaptive_router: Final = AdaptiveRouter(
            router_name=deployment.model_name,
            config=config,
            model_to_prefs=model_to_prefs,
            model_to_cost=model_to_cost,
            litellm_router_instance=self,
        )
        self._register_pre_routing_strategy(
            registry=self.adaptive_routers,
            deployment=deployment,
            strategy=adaptive_router,
            strategy_label="Adaptive-router",
        )
        litellm.logging_callback_manager.add_litellm_callback(
            AdaptiveRouterPostCallHook(adaptive_router=adaptive_router)
        )
        verbose_router_logger.info(
            "AdaptiveRouter[%s] initialized with %d models",
            deployment.model_name,
            len(config.available_models),
        )

    def _is_quality_router_deployment(self, litellm_params: LiteLLM_Params) -> bool:
        """
        Check if the deployment is a quality-router deployment.

        Returns True if the litellm_params model starts with "auto_router/quality_router".
        """
        return classify_strategy_router_model(litellm_params.model) == "quality"

    def init_quality_router_deployment(self, deployment: Deployment):
        """
        Initialize the quality-router deployment.

        Resolves the default model from either `quality_router_default_model` or
        `quality_router_config["default_model"]`, then instantiates the
        QualityRouter and stores it in `self.quality_routers`.
        """
        # Import here to mirror the AutoRouter / ComplexityRouter init pattern
        # and avoid circular imports.
        from litellm.router_strategy.quality_router.quality_router import (
            QualityRouter,
        )

        quality_router_config: Final[dict | None] = deployment.litellm_params.quality_router_config

        default_model: str | None = deployment.litellm_params.quality_router_default_model
        if default_model is None and quality_router_config:
            default_model = quality_router_config.get("default_model")

        if default_model is None:
            raise ValueError(
                "quality_router_default_model is required for quality-router deployments, "
                "or set default_model in quality_router_config. Please configure it in the litellm_params"
            )

        quality_router: Final[QualityRouter] = QualityRouter(
            model_name=deployment.model_name,
            default_model=default_model,
            litellm_router_instance=self,
            quality_router_config=quality_router_config,
        )
        self._register_pre_routing_strategy(
            registry=self.quality_routers,
            deployment=deployment,
            strategy=quality_router,
            strategy_label="Quality-router",
        )

    def deployment_is_active_for_environment(self, deployment: Deployment) -> bool:
        """
        Function to check if a llm deployment is active for a given environment. Allows using the same config.yaml across multople environments

        Requires `LITELLM_ENVIRONMENT` to be set in .env. Valid values for environment:
            - development
            - staging
            - production

        Raises:
        - ValueError: If LITELLM_ENVIRONMENT is not set in .env or not one of the valid values
        - ValueError: If supported_environments is not set in model_info or not one of the valid values
        """
        return model_info_is_active_for_environment(model_info=deployment.model_info)

    def set_model_list(self, model_list: list):
        original_model_list: Final = copy.deepcopy(model_list)
        self._discovered_model_info_cache.flush_cache()
        with self._deployment_index_lock:
            self.model_list = []
            self.model_id_to_deployment_index_map = {}  # Reset the index
            self.model_name_to_deployment_indices = {}  # Reset the model_name index
            self.team_model_to_deployment_indices = {}  # Reset the team_model index
        self.team_pattern_routers = {}
        self.team_public_model_names = frozenset()
        # Reset per-strategy router registries so hot-reload doesn't leave
        # stale routers pointing at the old model_list.
        self.quality_routers = {}
        self.complexity_routers = {}
        self.auto_routers = {}
        self._provider_unresolved_deployments = ()
        self._invalidate_model_group_info_cache()
        self._invalidate_access_groups_cache()
        # we add api_base/api_key each model so load balancing between azure/gpt on api_base1 and api_base2 works

        declared_ids: Final = tuple(
            str(entry["model_info"]["id"])
            for entry in original_model_list
            if isinstance(entry.get("model_info"), dict) and entry["model_info"].get("id") is not None
        )
        duplicate_ids: Final = frozenset(model_id for model_id in declared_ids if declared_ids.count(model_id) > 1)

        ptu_declared: Final = tuple(
            str(entry.get("model_name"))
            for entry in original_model_list
            if isinstance(entry.get("model_info"), dict)
            and entry["model_info"].get("db_model") is not True
            and declares_ptu(entry["model_info"])
        )
        if ptu_declared and not is_ptu_cost_attribution_enabled():
            verbose_router_logger.warning(
                "PTU fields are set on config.yaml deployment(s) %s, but PTU cost attribution is disabled, so no "
                "flat cost accrues and this traffic is billed per token. Set %s=True to enable it",
                ", ".join(ptu_declared),
                PTU_COST_ATTRIBUTION_ENV_VAR,
            )

        for model in original_model_list:
            _model_name = model.pop("model_name")
            _litellm_params = model.pop("litellm_params")
            ## check if litellm params in os.environ
            if isinstance(_litellm_params, dict):
                for k, v in _litellm_params.items():
                    if isinstance(v, str) and v.startswith("os.environ/") and not holds_secret_pointer(k):
                        _litellm_params[k] = get_secret(v)

            _model_info: dict = model.pop("model_info", {})

            declared_id = None if _model_info.get("id") is None else str(_model_info["id"])

            # check if model info has id
            if "id" not in _model_info:
                _id = self.generate_model_id(_model_name, _litellm_params)
                _model_info["id"] = _id

            if _litellm_params.get("organization", None) is not None and isinstance(
                _litellm_params["organization"], list
            ):  # Addresses https://github.com/BerriAI/litellm/issues/3949
                for org in _litellm_params["organization"]:
                    _litellm_params["organization"] = org
                    self._create_deployment(
                        deployment_info=model,
                        _model_name=_model_name,
                        _litellm_params=_litellm_params,
                        _model_info=_model_info,
                        declared_id=declared_id,
                        duplicate_ids=duplicate_ids,
                    )
            else:
                self._create_deployment(
                    deployment_info=model,
                    _model_name=_model_name,
                    _litellm_params=_litellm_params,
                    _model_info=_model_info,
                    declared_id=declared_id,
                    duplicate_ids=duplicate_ids,
                )

        verbose_router_logger.debug("\nInitialized Model List %s", self.get_model_names())
        self.model_names = {m["model_name"] for m in model_list}

        # Note: model_name_to_deployment_indices is already built incrementally
        # by _create_deployment -> _add_model_to_list_and_index_map

        # Deferred: build the AdaptiveRouter strategy now that all underlying
        # deployments have been registered.
        self._finalize_adaptive_router_if_configured()

    def _add_deployment(self, deployment: Deployment) -> Deployment:
        import os

        #### VALIDATE MODEL ########
        # Check if this is a prompt management model before validating as LLM provider
        litellm_model: Final = deployment.litellm_params.model
        if isinstance(deployment.litellm_params.drop_params, str):
            verbose_router_logger.warning(
                "model=%s drop_params=%r is not a flag value, treating it as unset",
                deployment.model_name,
                deployment.litellm_params.drop_params,
            )
        is_prompt_management_model = False

        if "/" in litellm_model:
            split_litellm_model: Final = litellm_model.split("/")[0]
            if split_litellm_model in litellm._known_custom_logger_compatible_callbacks:
                is_prompt_management_model = True

        if is_prompt_management_model:
            # For prompt management models, skip LLM provider validation
            # The actual model will be resolved at runtime from the prompt file
            _model = litellm_model
            custom_llm_provider = None
            dynamic_api_key = None
            api_base = None
        elif declared_authenticating_provider(litellm_model, deployment.litellm_params.custom_llm_provider) is not None:
            _model, custom_llm_provider = (  # rebind-ok: each branch resolves the deployment provider
                self._model_group_llm_provider(deployment.litellm_params)
            )
        else:
            # check if model provider in supported providers
            (
                _model,
                custom_llm_provider,
                dynamic_api_key,
                api_base,
            ) = litellm.get_llm_provider(
                model=deployment.litellm_params.model,
                custom_llm_provider=deployment.litellm_params.get("custom_llm_provider", None),
                api_base=deployment.litellm_params.api_base,
                litellm_params=deployment.litellm_params,
            )
            # done reading model["litellm_params"]
            # Check if provider is supported: either in enum or JSON-configured
            if (
                custom_llm_provider not in litellm.provider_list
                and not JSONProviderRegistry.exists(custom_llm_provider)
                and not is_registered_custom_provider(custom_llm_provider)
            ):
                raise Exception(f"Unsupported provider - {custom_llm_provider}")

        #### DEPLOYMENT NAMES INIT ########
        self.deployment_names.append(deployment.litellm_params.model)
        ############ Users can either pass tpm/rpm as a litellm_param or a router param ###########
        # for get_available_deployment, we use the litellm_param["rpm"]
        # in this snippet we also set rpm to be a litellm_param
        if deployment.litellm_params.rpm is None and getattr(deployment, "rpm", None) is not None:
            deployment.litellm_params.rpm = getattr(deployment, "rpm")

        if deployment.litellm_params.tpm is None and getattr(deployment, "tpm", None) is not None:
            deployment.litellm_params.tpm = getattr(deployment, "tpm")

        # Check if user is trying to use model_name == "*"
        # this is a catch all model for their specific api key
        # if deployment.model_name == "*":
        #     if deployment.litellm_params.model == "*":
        #         # user wants to pass through all requests to litellm.acompletion for unknown deployments
        #         self.router_general_settings.pass_through_all_models = True
        #     else:
        #         self.default_deployment = deployment.to_json(exclude_none=True)
        # Check if user is using provider specific wildcard routing
        # example model_name = "databricks/*" or model_name = "anthropic/*"
        if "*" in deployment.model_name:
            # store this as a regex pattern - all deployments matching this pattern will be sent to this deployment
            # Store deployment.model_name as a regex pattern
            self.pattern_router.add_pattern(deployment.model_name, deployment.to_json(exclude_none=True))
            if deployment.model_info.id:
                self.provider_default_deployment_ids.append(deployment.model_info.id)

        _team_id: Final = deployment.model_info.get("team_id")
        _team_public_model_name: Final[str | None] = deployment.model_info.get("team_public_model_name")
        if _team_id is not None and _team_public_model_name is not None and "*" in _team_public_model_name:
            if _team_id not in self.team_pattern_routers:
                self.team_pattern_routers[_team_id] = PatternMatchRouter()
            self.team_pattern_routers[_team_id].add_pattern(
                _team_public_model_name, deployment.to_json(exclude_none=True)
            )

        # Azure GPT-Vision Enhancements, users can pass os.environ/
        data_sources: Final = deployment.litellm_params.get("dataSources", []) or []

        for data_source in data_sources:
            params = data_source.get("parameters", {})
            for param_key in ["endpoint", "key"]:
                # if endpoint or key set for Azure GPT Vision Enhancements, check if it's an env var
                if param_key in params and params[param_key].startswith("os.environ/"):
                    env_name = params[param_key].replace("os.environ/", "")
                    params[param_key] = os.environ.get(env_name, "")

        # # init OpenAI, Azure clients
        # InitalizeOpenAISDKClient.set_client(
        #     litellm_router_instance=self, model=deployment.to_json(exclude_none=True)
        # )

        if custom_llm_provider is not None:
            self._initialize_deployment_for_pass_through(
                deployment=deployment,
                custom_llm_provider=custom_llm_provider,
            )

        #########################################################
        # Check if this is an auto-router deployment
        #########################################################
        if self._is_auto_router_deployment(litellm_params=deployment.litellm_params):
            self.init_auto_router_deployment(deployment=deployment)

        #########################################################
        # Check if this is a complexity-router deployment
        #########################################################
        if self._is_complexity_router_deployment(litellm_params=deployment.litellm_params):
            self.init_complexity_router_deployment(deployment=deployment)

        # NOTE: adaptive-router deployments are deferred to the end of
        # set_model_list() because their init needs visibility into the OTHER
        # deployments listed in `available_models` (which may not yet have
        # been processed when this one is created).
        #########################################################
        # Check if this is a quality-router deployment
        #########################################################
        if self._is_quality_router_deployment(litellm_params=deployment.litellm_params):
            self.init_quality_router_deployment(deployment=deployment)

        return deployment

    def _initialize_deployment_for_pass_through(self, deployment: Deployment, custom_llm_provider: str):
        """
        Optional: Register vertex credentials for pass-through endpoints if `deployment.litellm_params.use_in_pass_through` is True

        Other providers need no registration here: PassthroughEndpointRouter.get_credentials resolves their credentials per-request from the live router deployments
        """
        if deployment.litellm_params.use_in_pass_through is not True:
            return
        if custom_llm_provider != "vertex_ai":
            return
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            passthrough_endpoint_router,
        )

        credential_name: Final = deployment.litellm_params.litellm_credential_name
        credential_values: Final = (
            CredentialAccessor.get_credential_values(credential_name) if credential_name is not None else {}
        )
        from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES

        vertex_project: Final = (
            cast("str | None", credential_values.get("vertex_project"))  # cast-ok: vertex credential values are str
            or deployment.litellm_params.vertex_project
        )
        vertex_location: Final = (
            cast("str | None", credential_values.get("vertex_location"))  # cast-ok: vertex credential values are str
            or deployment.litellm_params.vertex_location
        )
        vertex_credentials: Final = (
            cast(  # cast-ok: vertex_credentials holds the typed credential union
                "VERTEX_CREDENTIALS_TYPES | None", credential_values.get("vertex_credentials")
            )
            or deployment.litellm_params.vertex_credentials
        )

        if vertex_project is None or vertex_location is None:
            raise ValueError(
                "vertex_project, and vertex_location must be set in litellm_params for pass-through endpoints."
            )
        passthrough_endpoint_router.add_vertex_credentials(
            project_id=vertex_project,
            location=vertex_location,
            vertex_credentials=vertex_credentials,
        )

    def add_deployment(self, deployment: Deployment) -> Deployment | None:
        """
        Parameters:
        - deployment: Deployment - the deployment to be added to the Router

        Returns:
        - The added deployment
        - OR None (if deployment already exists)
        """
        # check if deployment already exists

        _deployment_model_id: Final = deployment.model_info.id
        if _deployment_model_id and self.has_model_id(_deployment_model_id):
            return None

        declared_params: Final = deployment.litellm_params.model_dump(exclude_none=True)
        warn_on_provider_credential_mismatch(model_name=deployment.model_name, litellm_params=declared_params)
        warn_on_ignored_model_capability_flags(model_name=deployment.model_name, litellm_params=declared_params)
        served: Final = with_admin_api_key_secret_reference(without_model_capability_flags(deployment))

        # add to model list
        _deployment: Final = served.to_json(exclude_none=True)
        # initialize client
        self._add_deployment(deployment=served)

        Router._register_deployment_pricing(deployment=served)

        # add to model names
        self._add_model_to_list_and_index_map(model=_deployment, model_id=served.model_info.id)
        self.model_names.add(served.model_name)
        self._sync_deployment_budget_config(deployment=served)
        return served

    def _update_deployment_indices_after_removal(self, model_id: str, removal_idx: int) -> None:
        """
        Helper method to update deployment indices after a deployment has been removed from model_list.

        Parameters:
        - model_id: str - the id of the deployment that was removed
        - removal_idx: int - the index where the deployment was removed from model_list
        """
        self._discovered_model_info_cache.delete_cache(model_id)
        # Update indices for all models after the removed one
        for deployment_id, idx in self.model_id_to_deployment_index_map.items():
            if idx > removal_idx:
                self.model_id_to_deployment_index_map[deployment_id] = idx - 1
        # Remove the deleted model from index
        if model_id in self.model_id_to_deployment_index_map:
            del self.model_id_to_deployment_index_map[model_id]

        # Update model_name_to_deployment_indices
        for model_name, indices in list(self.model_name_to_deployment_indices.items()):
            # Build new list without mutating the original
            updated_indices = []
            for idx in indices:
                if idx == removal_idx:
                    # Skip the removed index
                    continue
                elif idx > removal_idx:
                    # Decrement indices after removal
                    updated_indices.append(idx - 1)
                else:
                    # Keep indices before removal unchanged
                    updated_indices.append(idx)

            # Update or remove the entry
            if len(updated_indices) > 0:
                self.model_name_to_deployment_indices[model_name] = updated_indices
            else:
                del self.model_name_to_deployment_indices[model_name]
                self.model_names.discard(model_name)

        # Update team_model_to_deployment_indices
        for key, indices in list(self.team_model_to_deployment_indices.items()):
            # Build new list without mutating the original
            updated_indices = []
            for idx in indices:
                if idx == removal_idx:
                    # Skip the removed index
                    continue
                elif idx > removal_idx:
                    # Decrement indices after removal
                    updated_indices.append(idx - 1)
                else:
                    # Keep indices before removal unchanged
                    updated_indices.append(idx)

            # Update or remove the entry
            if len(updated_indices) > 0:
                self.team_model_to_deployment_indices[key] = updated_indices
            else:
                del self.team_model_to_deployment_indices[key]
        self.team_public_model_names = frozenset(
            public_model_name for _, public_model_name in self.team_model_to_deployment_indices
        )

        self.pattern_router.remove_deployment(model_id)
        for team_id in list(self.team_pattern_routers.keys()):
            team_pattern_router = self.team_pattern_routers[team_id]
            team_pattern_router.remove_deployment(model_id)
            if not team_pattern_router.patterns:
                del self.team_pattern_routers[team_id]

    def _update_team_model_index(self, model: dict, idx: int) -> None:
        """
        Helper to update team_model_to_deployment_indices for a single deployment.

        Parameters:
        - model: dict - the deployment to index
        - idx: int - the index in model_list
        """
        team_id: Final = (model.get("model_info") or {}).get("team_id")
        team_public_model_name: Final = (model.get("model_info") or {}).get("team_public_model_name")
        if team_id and team_public_model_name:
            key: Final = (team_id, team_public_model_name)
            self.team_public_model_names = self.team_public_model_names | frozenset({team_public_model_name})
            if key not in self.team_model_to_deployment_indices:
                self.team_model_to_deployment_indices[key] = []
            if idx not in self.team_model_to_deployment_indices[key]:
                self.team_model_to_deployment_indices[key].append(idx)

    def _add_model_to_list_and_index_map(self, model: dict, model_id: str | None = None) -> None:
        """
        Helper method to add a model to the model_list and update both indices.

        Parameters:
        - model: dict - the model to add to the list
        - model_id: Optional[str] - the model ID to use for indexing. If None, will try to get from model["model_info"]["id"]
        """
        with self._deployment_index_lock:
            idx: Final = len(self.model_list)
            self.model_list.append(model)
            self._invalidate_model_group_info_cache()
            self._invalidate_access_groups_cache()

            # Update model_id index for O(1) lookup
            if model_id is not None:
                self.model_id_to_deployment_index_map[model_id] = idx
            elif model.get("model_info", {}).get("id") is not None:
                self.model_id_to_deployment_index_map[model["model_info"]["id"]] = idx

            # Update model_name index for O(1) lookup
            model_name: Final = model.get("model_name")
            if model_name:
                if model_name not in self.model_name_to_deployment_indices:
                    self.model_name_to_deployment_indices[model_name] = []
                self.model_name_to_deployment_indices[model_name].append(idx)

            # Update team_model index for O(1) team-scoped lookup
            self._update_team_model_index(model, idx)

    def upsert_deployment(self, deployment: Deployment) -> Deployment | None:
        """
        Add or update deployment
        Parameters:
        - deployment: Deployment - the deployment to be added to the Router

        Returns:
        - The added/updated deployment
        """
        _deployment_model_id: Final = deployment.model_info.id or ""
        _deployment_on_router: Final[Deployment | None] = self.get_deployment(model_id=_deployment_model_id)
        served: Final = without_model_capability_flags(deployment)
        try:
            if _deployment_on_router is not None:
                # deployment with this model_id exists on the router
                if (
                    deployment.model_name == _deployment_on_router.model_name
                    and served.litellm_params == _deployment_on_router.litellm_params
                    and deployment.model_info == _deployment_on_router.model_info
                ):
                    # No need to update
                    return None

                # if there is a new litellm param -> then update the deployment
                # remove the previous deployment
                with self._deployment_index_lock:
                    removal_idx: Final = self.model_id_to_deployment_index_map.get(_deployment_model_id)
                    if removal_idx is not None:
                        self.model_list.pop(removal_idx)
                        self._invalidate_model_group_info_cache()
                        self._invalidate_access_groups_cache()
                        self._update_deployment_indices_after_removal(
                            model_id=_deployment_model_id, removal_idx=removal_idx
                        )

                # Free the outgoing deployment's pre-routing strategy slot (keyed by the
                # OLD model_name/tags) before the re-add below re-registers it.
                self._unregister_pre_routing_strategy_for_deployment(deployment=_deployment_on_router)

            # if the model_id is not in router
            self.add_deployment(deployment=deployment)
            # add_deployment() builds every strategy EXCEPT the adaptive one, which
            # set_model_list() defers until the whole model_list is visible. Re-run that
            # deferred pass so an adaptive router whose slot was just released above is
            # rebuilt rather than left unregistered.
            if self._deployment_participates_in_adaptive_routing(litellm_params=deployment.litellm_params) or (
                _deployment_on_router is not None
                and self._deployment_participates_in_adaptive_routing(
                    litellm_params=_deployment_on_router.litellm_params
                )
            ):
                self._finalize_adaptive_router_if_configured()
            return deployment
        except Exception as e:
            if self.ignore_invalid_deployments:
                verbose_router_logger.warning(
                    "Error upserting deployment %s (id=%s): %s. Dropping it and continuing with other deployments.",
                    deployment.model_name,
                    deployment.model_info.id,
                    e,
                )
                self._restore_deployment_after_failed_upsert(
                    previous_deployment=_deployment_on_router, model_id=_deployment_model_id
                )
                return None
            else:
                raise e

    def _restore_deployment_after_failed_upsert(self, previous_deployment: Deployment | None, model_id: str) -> None:
        """Put a deployment back the way it was before a failed upsert popped it.

        A rollback re-admits state that was already serving, so it does not go through the
        capability ceiling a newcomer gets: with the ceiling tightened since the deployment first
        registered, judging the rollback would drop a serving router over an unrelated failed edit.
        """
        if previous_deployment is None or self.has_model_id(model_id):
            return
        limit_resolver: Final = self.auto_router_capability_limit
        self.auto_router_capability_limit = None
        try:
            self.add_deployment(deployment=previous_deployment)
            verbose_router_logger.info(
                "Restored deployment %s (id=%s); it keeps serving its previous configuration.",
                previous_deployment.model_name,
                model_id,
            )
        except Exception as restore_error:  # noqa: BLE001  # best-effort restore: a second failure must not abort the reload
            verbose_router_logger.warning(
                "Could not restore previously served deployment %s (id=%s) after the failed upsert: %s",
                previous_deployment.model_name,
                model_id,
                restore_error,
            )
        finally:
            self.auto_router_capability_limit = limit_resolver

    @staticmethod
    def _backend_cost_map_keys(model: str, custom_llm_provider: str | None) -> tuple[str, ...]:
        """The ``litellm.model_cost`` keys a deployment's shared backend info is registered under."""
        backend_key: Final = model if custom_llm_provider is None else f"{custom_llm_provider}/{model}"
        if "responses/" in backend_key:
            return (backend_key, backend_key.replace("responses/", ""))
        return (backend_key,)

    @staticmethod
    def _deployment_model_cost_payload(deployment: Deployment) -> dict[str, object]:  # mutable-ok: cost-map entry
        """The ``model_info`` a deployment contributes to ``litellm.model_cost``.

        Custom pricing lives on ``litellm_params`` rather than ``model_info``, and
        the built-in cache-pricing inheritance is derived rather than stored, so
        both are folded back in here. That keeps this reproducible from a
        deployment alone, which is what lets a refresh rebuild the same entries.
        """
        model_info: Final = dict(
            _MODEL_INFO_ADAPTER.validate_python(deployment.model_info.model_dump(exclude_none=True))
        )
        for field in CustomPricingLiteLLMParams.model_fields:
            field_value = deployment.litellm_params.get(field)
            if field_value is not None:
                model_info[field] = field_value
        backend_model: Final = Router._cost_map_backend_model(deployment)
        pricing_provider: Final = (
            deployment.model_info.pricing_provider or deployment.litellm_params.custom_llm_provider
        )
        params: Final = _LITELLM_PARAMS_ADAPTER.validate_python(deployment.litellm_params.model_dump(exclude_none=True))
        if (
            params.get("model_id")
            and backend_model != deployment.litellm_params.model
            and (
                deployment.litellm_params.custom_llm_provider == "bedrock"
                or deployment.litellm_params.model.startswith("bedrock/")
            )
        ):
            profile_info: Final = lookup_model_info_for_registration(model=backend_model, custom_llm_provider="bedrock")
            if not isinstance(profile_info, ModelInfoLookupMiss):
                explicit_info: Final = _MODEL_INFO_ADAPTER.validate_python(model_info)
                priced_info: Final = model_info_with_pricing_overrides(profile_info, explicit_info)
                model_info.update(
                    {key: value for key, value in priced_info.items() if is_server_derived_pricing_key(key)}
                )
        Router._inherit_builtin_base_rates_for_off_peak(
            model_info=model_info,
            backend_model=backend_model,
            custom_llm_provider=pricing_provider,
        )
        if model_info.get("input_cost_per_token") is not None:
            Router._inherit_builtin_cache_pricing(
                model_info=model_info,
                backend_model=backend_model,
                custom_llm_provider=pricing_provider,
            )
        Router._inherit_builtin_service_tier_pricing(
            model_info=model_info,
            backend_model=backend_model,
            custom_llm_provider=pricing_provider,
        )
        Router._inherit_builtin_tiered_output_rate(
            model_info=model_info,
            backend_model=backend_model,
            custom_llm_provider=pricing_provider,
        )
        return model_info

    @staticmethod
    def _register_deployment_pricing(deployment: Deployment) -> None:
        """Register a deployment's custom/inherited pricing in ``litellm.model_cost``.

        Takes only a ``Deployment``, so it registers pricing for a deployment that
        is never added to ``self.model_list`` (a per-request client-side-credential
        deployment) just as readily as one that is.
        """
        Router._register_deployment_in_model_cost(
            model_id=deployment.model_info.id,
            model_info=Router._deployment_model_cost_payload(deployment),
            model=deployment.litellm_params.model,
            custom_llm_provider=deployment.litellm_params.custom_llm_provider,
            pricing_provider=deployment.model_info.pricing_provider,
        )

    @staticmethod
    def _register_deployment_in_model_cost(
        *,
        model_id: str | None,
        model_info: dict,  # mutable-ok: cost-map entry
        model: str,
        custom_llm_provider: str | None,
        pricing_provider: str | None = None,
    ) -> None:
        """Write a deployment's metadata into ``litellm.model_cost``.

        Runs when a deployment is added and again after a price data reload, so
        the entries a refresh rebuilds are the ones a fresh boot would produce.
        Nothing is recorded for replay: a refresh walks the live routers instead,
        so a deleted, repointed or never-added deployment, and a discarded router,
        drop out of the rebuild on their own.

        A strategy-router alias is never the deployment actually called or
        billed, so custom pricing configured on it must not become a cost-map
        price: an explicit zero would let ``_is_cost_explicitly_configured``
        treat the alias as a genuinely free model and waive budget checks for
        requests that route to (and bill as) a real deployment.
        """
        if classify_strategy_router_model(model) is not None:
            model_info = {k: v for k, v in model_info.items() if k not in CustomPricingLiteLLMParams.model_fields}

        if model_id is not None:
            litellm.register_model(
                model_cost={model_id: model_info},
                persist_across_reloads=False,
                warning_display_name=model,
                custom_llm_provider=pricing_provider or custom_llm_provider,
                add_to_provider_model_sets=False,
            )

        ## OLD MODEL REGISTRATION ## Kept to prevent breaking changes
        backend_keys: Final = Router._backend_cost_map_keys(model=model, custom_llm_provider=custom_llm_provider)

        # For the shared backend key, keep only cost-map schema fields
        # (minus custom pricing) so that one deployment's pricing overrides
        # or custom metadata (id, access_via_team_ids, arbitrary keys)
        # don't pollute another deployment sharing the same backend model
        # name. Each deployment's full model_info is already stored under
        # its unique model_id above.
        shared_model_info: Final = shared_backend_model_info(model_info)
        litellm.register_model(
            model_cost={_key: keep_registered_mode(_key, shared_model_info) for _key in backend_keys},
            persist_across_reloads=False,
        )

    def replay_model_cost_registrations(self) -> None:
        """Re-assert this router's deployments onto a freshly fetched catalog.

        Reads ``model_list`` at call time, so only deployments the router still
        serves are restored, plus any config deployment the fresh catalog now resolves.
        """
        provider_unresolved: Final = self._provider_unresolved_deployments
        self._provider_unresolved_deployments = ()
        for create_deployment in provider_unresolved:
            create_deployment()
        for entry in tuple(self.model_list):
            try:
                deployment = entry if isinstance(entry, Deployment) else Deployment(**entry)
            except Exception:  # noqa: BLE001  # a malformed entry must not abort the rest of the rebuild
                verbose_router_logger.exception(
                    "Router: could not rebuild cost-map entry for a deployment during a price data reload"
                )
                continue
            Router._register_deployment_in_model_cost(
                model_id=deployment.model_info.id,
                model_info=Router._deployment_model_cost_payload(deployment),
                model=deployment.litellm_params.model,
                custom_llm_provider=deployment.litellm_params.custom_llm_provider,
                pricing_provider=deployment.model_info.pricing_provider,
            )
        self._invalidate_model_group_info_cache()

    _replay_model_cost_registrations = replay_model_cost_registrations

    def delete_deployment(self, id: str) -> Deployment | None:
        """
        Parameters:
        - id: str - the id of the deployment to be deleted

        Returns:
        - The deleted deployment
        - OR None (if deleted deployment not found)
        """
        try:
            with self._deployment_index_lock:
                deployment_idx: Final = self.model_id_to_deployment_index_map.get(id)
                if deployment_idx is None:
                    return None
                # Pop the item from the list first
                item: Final = self.model_list.pop(deployment_idx)
                self._invalidate_model_group_info_cache()
                self._invalidate_access_groups_cache()
                self._update_deployment_indices_after_removal(model_id=id, removal_idx=deployment_idx)
            _budget_limiter: Final = self.get_router_deployment_budget_limiter()
            if _budget_limiter is not None:
                _budget_limiter.unregister_deployment_budget(model_id=id)
            try:
                self._unregister_pre_routing_strategy_for_deployment(
                    deployment=item if isinstance(item, Deployment) else Deployment(**item)
                )
            except Exception:
                verbose_router_logger.exception(
                    "delete_deployment: could not release pre-routing strategies for model_id=%s; "
                    "the deployment is out of the model_list and its indices are repaired",
                    id,
                )
        except Exception:
            return None
        return item

    def get_router_deployment_budget_limiter(
        self,
    ) -> RouterBudgetLimiting | None:
        """
        Return the router's deployment-budget callback.

        Uses exact-type matching so proxy subclasses (e.g. virtual-key model budgets)
        registered on litellm.callbacks are not mistaken for router deployment budgets.
        """
        if self.router_budget_logger is not None:
            return self.router_budget_logger

        if self.optional_callbacks:
            for _cb in self.optional_callbacks:
                if type(_cb) is RouterBudgetLimiting:
                    self.router_budget_logger = _cb
                    return _cb
        return None

    _get_router_deployment_budget_limiter = get_router_deployment_budget_limiter

    def _deployment_has_budget_limits(self, deployment: Deployment) -> bool:
        return (
            deployment.litellm_params.get("max_budget") is not None
            and deployment.litellm_params.get("budget_duration") is not None
            and deployment.model_info.id is not None
        )

    def _sync_deployment_budget_config(self, deployment: Deployment) -> None:
        model_id: Final = deployment.model_info.id
        if model_id is None:
            return

        _budget_limiter = self.get_router_deployment_budget_limiter()

        if not self._deployment_has_budget_limits(deployment=deployment):
            if _budget_limiter is not None:
                _budget_limiter.unregister_deployment_budget(model_id=model_id)
            return

        if _budget_limiter is None:
            self.add_optional_pre_call_checks(optional_pre_call_checks=["router_budget_limiting"])
            _budget_limiter = self.get_router_deployment_budget_limiter()

        if _budget_limiter is not None:
            _budget_limiter.register_deployment_budget(deployment=deployment.to_json(exclude_none=True))

    def deployments_named(self, model_name: str) -> tuple[DeploymentTypedDict, ...] | None:
        """The deployments indexed under `model_name`, read under the index lock, or None when the name is not indexed."""
        with self._deployment_index_lock:
            indices: Final = self.model_name_to_deployment_indices.get(model_name)
            return None if indices is None else tuple(self.model_list[idx] for idx in indices)

    def _team_deployments_named(self, team_id: str, public_model_name: str) -> tuple[DeploymentTypedDict, ...] | None:
        """The deployments `team_id` serves under `public_model_name`, or None when that pair is not indexed."""
        with self._deployment_index_lock:
            indices: Final = self.team_model_to_deployment_indices.get((team_id, public_model_name))
            return None if indices is None else tuple(self.model_list[idx] for idx in indices)

    def get_deployment(self, model_id: str) -> Deployment | None:
        """
        Returns -> Deployment or None

        Raise Exception -> if model found in invalid format
        """
        # Use O(1) lookup via model_id_to_deployment_index_map only
        with self._deployment_index_lock:
            idx: Final = self.model_id_to_deployment_index_map.get(model_id)
            model: Final = None if idx is None else self.model_list[idx]
        if model is not None:
            if isinstance(model, dict):
                return Deployment(**model)
            elif isinstance(model, Deployment):
                return model
            else:
                raise Exception(f"Model invalid format - {type(model)}")

        return None

    def get_deployment_credentials(self, model_id: str) -> dict | None:
        """
        Returns -> dict of credentials for a given model id.

        Returns None if the deployment is paused via `LiteLLM_ProxyModelTable.blocked`,
        so file/batch/passthrough callers that resolve credentials directly cannot keep
        using a paused deployment.
        """
        deployment: Final = self.get_deployment(model_id=model_id)
        if deployment is None or self.is_deployment_blocked(deployment):
            return None
        return CredentialLiteLLMParams.model_validate(
            deployment.litellm_params.model_dump(exclude_none=True)
        ).model_dump(exclude_none=True)

    def get_deployment_by_model_group_name(self, model_group_name: str) -> Deployment | None:
        """
        Returns -> Deployment or None

        Raise Exception -> if model found in invalid format

        Optimized with O(1) index lookup instead of O(n) linear scan.
        """
        # O(1) lookup in model_name index
        with self._deployment_index_lock:
            indices: Final = self.model_name_to_deployment_indices.get(model_group_name)
            # Return first deployment for this model_name
            model: Final = self.model_list[indices[0]] if indices else None
        if model is not None:
            if isinstance(model, dict):
                return Deployment(**model)
            elif isinstance(model, Deployment):
                return model
            else:
                raise Exception(f"Model Name invalid - {type(model)}")
        return None

    @staticmethod
    def deployment_usable_by_team(model: Mapping[str, object] | Deployment, team_id: str | None) -> bool:
        """
        A team-scoped deployment (``model_info.team_id`` set) is only usable by
        callers from that same team; deployments without a team owner are shared.
        """
        model_info: Final = model.get("model_info") if isinstance(model, dict) else model.model_info
        owner_team_id: Final = cast(Mapping[str, object], model_info).get("team_id") if model_info is not None else None
        return team_may_use_deployment(owner_team_id, team_id)

    _deployment_usable_by_team = deployment_usable_by_team

    def _get_model_group_deployment_usable_by_team(
        self, model_group_name: str, team_id: str | None
    ) -> Deployment | None:
        """
        Like ``get_deployment_by_model_group_name``, but skips deployments owned
        by other teams so a shared model name never resolves another team's
        credentials.
        """
        with self._deployment_index_lock:
            indices: Final = self.model_name_to_deployment_indices.get(model_group_name) or ()
            usable: Final = (
                self.model_list[idx] for idx in indices if self.deployment_usable_by_team(self.model_list[idx], team_id)
            )
            first_usable: Final = next(usable, None)
        if first_usable is None:
            return None
        return Deployment(**first_usable) if isinstance(first_usable, dict) else first_usable

    async def arefresh_model_info(self, *, client: AsyncHTTPHandler | None = None) -> None:
        """Refresh token limits advertised by configured OpenAI-compatible deployments."""
        deployments: Final = iter(tuple(self.model_list))

        async def refresh_worker() -> None:
            for raw_deployment in deployments:
                try:
                    await self._arefresh_deployment_model_info(raw_deployment, client=client)
                except Exception:  # noqa: BLE001  # one invalid deployment must not prevent refreshing the others
                    verbose_router_logger.debug("Could not refresh deployment model info")

        await asyncio.gather(*(refresh_worker() for _ in range(MODEL_INFO_REFRESH_CONCURRENCY)))
        self._invalidate_model_group_info_cache()

    async def _arefresh_deployment_model_info(
        self, raw_deployment: Mapping[str, object], *, client: AsyncHTTPHandler | None
    ) -> None:
        deployment: Final = Deployment.model_validate(raw_deployment)
        params: Final = LiteLLM_Params.model_validate(
            MappingProxyType(
                {
                    **deployment.litellm_params.model_dump(exclude_none=True),
                    **(
                        self.get_deployment_credentials_with_provider(deployment.model_info.id or "")
                        or MappingProxyType({})
                    ),
                }
            )
        )
        model, provider, dynamic_api_key, api_base = litellm.get_llm_provider(model=params.model, litellm_params=params)
        if provider not in MODEL_INFO_DISCOVERY_PROVIDERS:
            return
        if api_base is None or "*" in model or params.get("use_clientside_credentials"):
            return
        api_key: Final = resolve_api_key_secret_reference(
            params.api_key or dynamic_api_key,
            model=model,
            llm_provider=provider,
            upstream=MappingProxyType({"api_base": params.api_base}),
        )
        headers: Final = TypeAdapter(Mapping[str, str]).validate_python(
            params.get("extra_headers") or params.get("headers") or MappingProxyType({})
        )
        auth_headers: Final = (
            MappingProxyType({"authorization": f"Bearer {api_key}"}) if api_key else MappingProxyType({})
        )
        limits: Final = await get_openai_compatible_model_info(
            model=model,
            api_base=api_base,
            headers=MappingProxyType(
                {
                    **auth_headers,
                    **MappingProxyType({key.lower(): value for key, value in headers.items()}),
                }
            ),
            client=client or get_async_httpx_client(llm_provider=LlmProviders.OPENAI),
            cache=self.cache.in_memory_cache,
        )
        model_id: Final = deployment.model_info.id
        if not limits or model_id is None or self.get_model_info(model_id) is not raw_deployment:
            return
        self._discovered_model_info_cache.max_size_in_memory = max(len(self.model_list), 1)
        self._discovered_model_info_cache.delete_cache(model_id)
        self._discovered_model_info_cache.set_cache(
            model_id, DiscoveredDeploymentModelInfo(deployment=raw_deployment, limits=limits)
        )
        self._invalidate_model_group_info_cache()

    def get_discovered_model_info(self, model_id: str | None) -> Mapping[str, int]:
        cached: Final[object] = self._discovered_model_info_cache.get_cache(model_id)
        if (
            model_id is not None
            and isinstance(cached, DiscoveredDeploymentModelInfo)
            and cached.deployment is self.get_model_info(model_id)
        ):
            configured: Final = TypeAdapter(Mapping[str, object]).validate_python(cached.deployment["model_info"])
            return MappingProxyType({key: value for key, value in cached.limits.items() if configured.get(key) is None})
        return MappingProxyType({})

    def get_model_listing_info(self, model_name: str) -> DeploymentModelListingInfo | None:
        """
        Return what the concrete deployments behind model_name contribute to its
        /v1/models entry: the cost-map keys for their underlying models, plus the widest
        configured or discovered token limits. Resolved via O(1) index
        lookup.

        Returns None for wildcard-expanded or unknown names, where the listed name is the
        real model name and no deployment-specific information exists, and treats a
        malformed configured limit as absent rather than failing the listing.

        The whole group is read rather than just its first deployment, so a group that
        mixes models does not advertise a window that depends on config order; the widest
        one is reported, which is what get_model_group_info already shows the Admin UI.
        Keys are deduplicated, so the ordinary group of interchangeable deployments of one
        model still costs the caller a single cost-map lookup. Unlike get_model_group_info,
        this never triggers pattern matching or deep copies, so it is safe to call per
        listed model on the /v1/models hot path.
        """
        deployments: Final = self.deployments_named(model_name)
        if not deployments:
            return None

        model_infos: Final = tuple(
            MappingProxyType(
                {
                    **self.get_discovered_model_info((deployment.get("model_info") or MappingProxyType({})).get("id")),
                    **MappingProxyType(
                        {
                            k: v
                            for k, v in (deployment.get("model_info") or MappingProxyType({})).items()
                            if v is not None
                        }
                    ),
                }
            )
            for deployment in deployments
        )
        raw_deployments: Final[tuple[object, ...]] = deployments
        pricing_keys: Final = (deployment_pricing_models(deployment) for deployment in raw_deployments)
        cost_map_keys: Final = tuple(dict.fromkeys(keys[0] for keys in pricing_keys if keys))
        return DeploymentModelListingInfo(
            cost_map_keys=cost_map_keys,
            max_input_tokens=self._widest_configured_limit(model_infos, "max_input_tokens"),
            max_output_tokens=self._widest_configured_limit(model_infos, "max_output_tokens"),
        )

    @staticmethod
    def _widest_configured_limit(model_infos: Sequence[Mapping[str, object]], field: str) -> int | None:
        """The largest usable value of ``field`` across a group's configured model_info blocks."""
        limits: Final = tuple(
            limit
            for limit in (coerce_token_limit(model_info.get(field)) for model_info in model_infos)
            if limit is not None
        )
        return max(limits) if limits else None

    def get_configured_token_limits(self, model_name: str) -> "tuple[int | None, int | None]":
        """
        Return (max_input_tokens, max_output_tokens) configured or discovered for a concrete
        deployment of model_name, via O(1) index lookup.

        Returns (None, None) for wildcard-expanded or unknown names, and treats a
        malformed configured value as absent rather than failing the caller.

        Deliberately reads one deployment rather than aggregating the group the way
        get_model_listing_info does: its caller truncates an embedding input to this
        value, so the widest window in a mixed group would be the wrong answer there.
        """
        deployment: Final = self.get_deployment_by_model_group_name(model_group_name=model_name)
        if deployment is None:
            return (None, None)

        model_info: Final = MappingProxyType(
            {
                **self.get_discovered_model_info(deployment.model_info.id),
                **deployment.model_info.model_dump(exclude_none=True),
            }
        )
        return (
            coerce_token_limit(model_info.get("max_input_tokens")),
            coerce_token_limit(model_info.get("max_output_tokens")),
        )

    def get_configured_mode(self, model_name: str) -> "str | None":
        """Return the mode explicitly configured for a concrete deployment."""
        deployment: Final = self.get_deployment_by_model_group_name(model_group_name=model_name)
        if deployment is None:
            return None

        mode: Final = deployment.model_info.get("mode")
        if isinstance(mode, str) and mode.strip():
            return mode
        return None

    def get_configured_display_name(self, model_name: str) -> "str | None":
        """
        Return the display_name explicitly configured in a concrete deployment's
        model_info for model_name, via O(1) index lookup.

        Returns None for wildcard-expanded or unknown names, and treats a
        non-string or empty configured value as absent rather than failing the
        listing. Like get_configured_token_limits, this never triggers pattern
        matching or deep copies, so it is safe to call per listed model on the
        /v1/models hot path.
        """
        deployment: Final = self.get_deployment_by_model_group_name(model_group_name=model_name)
        if deployment is None:
            return None

        display_name: Final = deployment.model_info.get("display_name")
        if isinstance(display_name, str) and display_name.strip():
            return display_name
        return None

    def routable_model_group(self, model_name: str) -> str:
        """
        The model group a request to model_name routes to: its target when
        model_name is a `model_group_alias`, else model_name itself.
        """
        target: Final = self.get_model_from_alias(model_name)
        return target if target is not None else model_name

    def _routable_deployments(self, model_name: str, team_id: str | None) -> tuple[DeploymentTypedDict, ...]:
        """
        The deployments a request from team_id to model_name can route to, in
        model_list order and via the same O(1) index lookup as
        get_configured_display_name, selected the way routing selects them: a
        `model_group_alias` reads its target's deployments, another team's
        deployment of the name is left out, a caller with no team reads the
        deployments no team owns (every deployment of the name when a team owns
        each one, which is what an admin's request routes to), and one an admin
        paused via `LiteLLM_ProxyModelTable.blocked` is left out.

        Returns an empty tuple for wildcard-expanded or unknown names.
        """
        named: Final = self.get_all_deployments(model_name=self.routable_model_group(model_name), team_id=team_id)
        usable: Final = tuple(deployment for deployment in named if self.deployment_usable_by_team(deployment, team_id))
        return tuple(
            deployment
            for deployment in (usable or named)
            if _configured_model_info(deployment).get("blocked") is not True
        )

    def get_configured_service_tiers(self, model_name: str, team_id: str | None = None) -> tuple[object, ...]:
        """
        Return the service_tiers value each deployment a request from team_id
        to model_name can route to (see _routable_deployments) configures in
        its model_info, unvalidated and in model_list order, None for a
        deployment that sets none; the caller validates the shape it expects
        and decides how the deployments combine.

        Returns an empty tuple for wildcard-expanded or unknown names.
        """
        return tuple(
            _configured_model_info(deployment).get("service_tiers")
            for deployment in self._routable_deployments(model_name, team_id)
        )

    def get_routable_upstream_model(self, model_name: str, team_id: str | None = None) -> str | None:
        """
        Return the `litellm_params.model` of the first deployment a request
        from team_id to model_name can route to (see _routable_deployments).

        Returns None for wildcard-expanded or unknown names, and when the team
        can route to no deployment of the name.
        """
        deployment: Final = next(iter(self._routable_deployments(model_name, team_id)), None)
        return deployment["litellm_params"].get("model") if deployment is not None else None

    def get_credential_deployment(self, model_id: str, team_id: str | None = None) -> Deployment | None:
        """
        The deployment a passthrough endpoint (files, batches, etc.) resolves for a
        model id or model name: by deployment id first, then by model_name, then by
        the team's exact public model name, then by wildcard pattern (team wildcards
        before global ones, so a global "openai/*" never shadows the team's own
        entry). Name and wildcard lookups never resolve another team's deployment.

        Returns None when nothing matches or the match is paused via
        `LiteLLM_ProxyModelTable.blocked`, so callers cannot bypass an admin pause
        by resolving the deployment directly.
        """
        deployment: Final = (
            self.get_deployment(model_id=model_id)
            or self._get_model_group_deployment_usable_by_team(model_group_name=model_id, team_id=team_id)
            or self._get_team_public_name_deployment(model_id=model_id, team_id=team_id)
            or self._get_wildcard_deployment_usable_by_team(model_id=model_id, team_id=team_id)
        )
        if deployment is None or self.is_deployment_blocked(deployment):
            return None
        return deployment

    def _get_team_public_name_deployment(self, model_id: str, team_id: str | None) -> Deployment | None:
        if team_id is None:
            return None
        with self._deployment_index_lock:
            team_indices: Final = self.team_model_to_deployment_indices.get((team_id, model_id))
            team_model: Final = self.model_list[team_indices[0]] if team_indices else None
        if team_model is None:
            return None
        return Deployment(**team_model) if isinstance(team_model, dict) else team_model

    def _get_wildcard_deployment_usable_by_team(self, model_id: str, team_id: str | None) -> Deployment | None:
        team_pattern_router: Final = self.team_pattern_routers.get(team_id) if team_id is not None else None
        team_wildcard_models: Final = team_pattern_router.route(model_id) if team_pattern_router else None
        global_wildcard_models: Final = tuple(
            wildcard_model
            for wildcard_model in (self.pattern_router.route(model_id) or ())
            if self.deployment_usable_by_team(wildcard_model, team_id)
        )
        potential_wildcard_models: Final = team_wildcard_models or global_wildcard_models
        if not potential_wildcard_models:
            return None
        wildcard_deployment: Final = potential_wildcard_models[0]
        if isinstance(wildcard_deployment, dict):
            return Deployment(**wildcard_deployment)
        if isinstance(wildcard_deployment, Deployment):
            return wildcard_deployment
        return None

    def get_deployment_credentials_with_provider(
        self, model_id: str, team_id: str | None = None
    ) -> dict[str, Any] | None:
        """
        Get API credentials and provider info from a model name in model_list.
        Useful for passthrough endpoints (files, batches, etc.) that need credentials.

        Resolves the deployment with `get_credential_deployment` (by deployment id,
        then model_name, team public model name, and wildcard pattern).

        Args:
            model_id: Model ID or model name from model_list (e.g., "gpt-4o-litellm")
            team_id: Optional team id of the caller. When set, team-scoped
                deployments (indexed by team public model name, including team
                wildcard models like "openai/*") are also considered. Name and
                wildcard lookups never resolve a deployment owned by a
                different team, so shared model names can't leak another
                team's credentials.

        Returns:
            Dictionary containing api_key, api_base, custom_llm_provider, etc.
            Returns None if model not found, or if the resolved deployment is
            paused via `LiteLLM_ProxyModelTable.blocked` (so passthrough callers
            cannot bypass an admin pause by resolving credentials directly).

        Example:
            credentials = router.get_deployment_credentials_with_provider("gpt-4o-litellm")
            # Returns: {"api_key": "sk-...", "custom_llm_provider": "openai", "model": "gpt-4o", ...}
        """
        deployment: Final = self.get_credential_deployment(model_id=model_id, team_id=team_id)
        if deployment is None:
            return None

        # Get basic credentials
        credentials: Final = CredentialLiteLLMParams.model_validate(
            deployment.litellm_params.model_dump(exclude_none=True)
        ).model_dump(exclude_none=True)

        # Resolve litellm_credential_name to actual credentials
        if deployment.litellm_params.litellm_credential_name is not None:
            credential_values: Final = CredentialAccessor.get_credential_values(
                deployment.litellm_params.litellm_credential_name
            )
            if not credential_values:
                verbose_router_logger.warning(
                    "Credential '%s' not found in credential_list", deployment.litellm_params.litellm_credential_name
                )
            credentials.update(credential_values)
            # Remove the credential name since we've resolved it
            credentials.pop("litellm_credential_name", None)

        credentials["model"] = deployment.litellm_params.model

        # Add custom_llm_provider
        if deployment.litellm_params.custom_llm_provider:
            credentials["custom_llm_provider"] = deployment.litellm_params.custom_llm_provider
        elif "/" in deployment.litellm_params.model:
            # Extract provider from "provider/model" format
            credentials["custom_llm_provider"] = deployment.litellm_params.model.split("/")[0]
        else:
            credentials["custom_llm_provider"] = "openai"  # default

        return credentials

    @overload
    def get_router_model_info(
        self,
        deployment: Union[dict, "Deployment"],
        received_model_name: str,
        id: None = None,
    ) -> ModelMapInfo:
        pass

    @overload
    def get_router_model_info(self, deployment: None, received_model_name: str, id: str) -> ModelMapInfo:
        pass

    def get_router_model_info(
        self,
        deployment: Union[dict, "Deployment"] | None,
        received_model_name: str,
        id: str | None = None,
    ) -> ModelMapInfo:
        """
        For a given model id, return the model info (max tokens, input cost, output cost, etc.).

        Augment litellm info with additional params set in `model_info`.

        For azure models, ignore the `model:`. Only set max tokens, cost values if base_model is set.
        For every other provider, a base_model naming the same model as `model:` can only widen its context window.

        Returns
        - ModelInfo - If found -> typed dict with max tokens, input cost, etc.

        Raises:
        - ValueError -> If model is not mapped yet
        """
        if id is not None:
            _deployment: Final = self.get_deployment(model_id=id)
            if _deployment is not None:
                deployment = _deployment

        if deployment is None:
            raise ValueError("Deployment not found")

        ## GET BASE MODEL
        base_model = (deployment.get("model_info") or {}).get("base_model", None)
        if base_model is None:
            base_model = (deployment.get("litellm_params") or {}).get("base_model", None)

        model = base_model

        ## GET PROVIDER - reuse LiteLLM_Params if already constructed
        litellm_params_data: Final = deployment.get("litellm_params")
        litellm_params: LiteLLM_Params
        if isinstance(litellm_params_data, LiteLLM_Params):
            litellm_params = litellm_params_data
        elif isinstance(litellm_params_data, dict) and "model" in litellm_params_data:
            litellm_params = LiteLLM_Params.model_validate(litellm_params_data)
        else:
            raise ValueError(
                f"Deployment missing valid litellm_params. "
                f"Got: {type(litellm_params_data).__name__}, "
                f"deployment_id: {(deployment.get('model_info') or {}).get('id', 'unknown')}"
            )
        _model, custom_llm_provider, _, _ = litellm.get_llm_provider(
            model=litellm_params.model,
            litellm_params=litellm_params,
        )

        ## SET MODEL TO 'model=' - if base_model is None + not azure
        if custom_llm_provider == "azure" and base_model is None:
            # Router init auto-registers every deployment name into
            # litellm.model_cost as a zeroed stub, so membership alone can't
            # tell a resolvable name apart; require usable limits/costs.
            _azure_fallback_key = _model if _model.startswith("azure/") else f"azure/{_model}"
            _fallback_entry = litellm.model_cost.get(_azure_fallback_key)
            _fallback_resolves = _fallback_entry is not None and (
                (_fallback_entry.get("max_input_tokens") or 0) > 0
                or (_fallback_entry.get("input_cost_per_token") or 0) > 0
            )
            if _fallback_resolves:
                verbose_router_logger.debug(
                    "Azure deployment '%s' has no base_model set; using '%s' from the model cost map for max tokens, cost tracking, etc.",
                    _model,
                    _azure_fallback_key,
                )
            else:
                verbose_router_logger.error(
                    "Could not identify azure model '%s'. Set azure 'base_model' for accurate max tokens, cost tracking, etc.- https://docs.litellm.ai/docs/proxy/cost_tracking#spend-tracking-for-azure-openai-models",
                    _model,
                )
        elif custom_llm_provider != "azure":
            model = _model

            if "*" in model:  # only call pattern_router for wildcard models
                potential_models: Final = self.pattern_router.route(received_model_name)
                if potential_models is not None:
                    for potential_model in potential_models:
                        try:
                            if (potential_model.get("model_info") or {}).get("id") == (
                                deployment.get("model_info") or {}
                            ).get("id"):
                                model = (potential_model.get("litellm_params") or {}).get("model")
                                break
                        except Exception:
                            pass

        ## GET LITELLM MODEL INFO - raises exception, if model is not mapped
        if model is None:
            # Handle case where base_model is None (e.g., Azure models without base_model set)
            # Use the original model from litellm_params
            model = _model

        if not model.startswith(f"{custom_llm_provider}/"):
            model_info_name = f"{custom_llm_provider}/{model}"
        else:
            model_info_name = model

        model_info: Final = litellm.get_model_info(model=model_info_name)
        if model_info is None:
            return model_info

        ## CHECK USER SET MODEL INFO
        raw_user_model_info: Final = deployment.get("model_info")
        user_model_info: Final = _MODEL_INFO_ADAPTER.validate_python(
            raw_user_model_info.model_dump(exclude_none=True)
            if isinstance(raw_user_model_info, BaseModel)
            else raw_user_model_info or {}
        )

        # get_model_info() hands back an lru_cache'd dict, so merge into a copy; unset
        # values are skipped or Deployment's None pricing defaults would erase the map's
        merged_model_info: Final[ModelMapInfo] = {
            **copy.deepcopy(model_info),
            **_base_model_widened_window(
                model_info,
                _declared_base_model(user_model_info) or _declared_base_model(litellm_params.model_extra or {}),
                custom_llm_provider,
            ),
            **self.get_discovered_model_info((deployment.get("model_info") or {}).get("id")),
            **MappingProxyType({key: value for key, value in user_model_info.items() if value is not None}),
        }

        return merged_model_info

    def get_model_info(self, id: str) -> dict | None:
        """
        For a given model id, return the model info

        Returns
        - dict: the model in list with 'model_name', 'litellm_params', Optional['model_info']
        - None: could not find deployment in list

        Optimized with O(1) index lookup instead of O(n) linear scan.
        """
        # O(1) lookup via model_id_to_deployment_index_map
        with self._deployment_index_lock:
            idx: Final = self.model_id_to_deployment_index_map.get(id)
            return None if idx is None else self.model_list[idx]

    def get_model_group(self, id: str) -> list | None:
        """
        Return list of all models in the same model group as that model id
        """

        model_info: Final = self.get_model_info(id=id)
        if model_info is None:
            return None

        model_name: Final = model_info["model_name"]
        return self.get_model_list(model_name=model_name)

    def get_deployment_model_info(self, model_id: str, model_name: str) -> ModelInfo | None:
        """
        For a given model id, return the model info

        1. Check if model_id is in model info
        2. If not, check if litellm model name is in model info
        3. If not, return None
        """
        from litellm.utils import cost_map_omits_token_price, update_dictionary

        model_info: ModelInfo | None = None
        custom_model_info: dict | None = None
        litellm_model_name_model_info: ModelInfo | None = None
        base_model_key: str | None = None

        try:
            custom_model_info = (
                {
                    **copy.deepcopy(litellm.model_cost.get(model_id) or MappingProxyType({})),
                    **self.get_discovered_model_info(model_id),
                }
                if model_id in litellm.model_cost
                else None
            )
        except Exception:
            pass

        try:
            litellm_model_name_model_info = litellm.get_model_info(model=model_name)
        except Exception:
            pass

        ## check for base model
        try:
            if custom_model_info is not None:
                base_model: Final = custom_model_info.get("base_model", None)
                if base_model is not None:
                    ## update litellm model info with base model info
                    base_model_info: Final = copy.deepcopy(litellm.get_model_info(model=base_model))
                    if base_model_info is not None:
                        base_model_key = base_model_info.get("key")
                        # Base model provides defaults, custom model info overrides
                        custom_model_info = update_dictionary(
                            cast(dict, base_model_info),
                            custom_model_info,
                        )
        except Exception:
            pass

        # Three mutually exclusive scenarios for the model's metadata:
        if custom_model_info is not None and litellm_model_name_model_info is not None:
            # (1) It has both custom model_info set and exists in the built-in map
            # merge with custom overriding built-in
            model_info = cast(
                ModelInfo,
                update_dictionary(
                    copy.deepcopy(cast(dict, litellm_model_name_model_info)),
                    custom_model_info,
                ),
            )
        elif litellm_model_name_model_info is not None:
            # (2) Built-in only — no custom pricing to merge
            model_info = copy.deepcopy(litellm_model_name_model_info)
        elif custom_model_info is not None:
            # (3) Custom only — model not in built-in cost map yet
            # custom_model_info already includes base_model defaults at this point, if applicable
            model_info = cast(ModelInfo, custom_model_info)

        if model_info is None:
            return None
        builtin_key: Final = (
            litellm_model_name_model_info.get("key") if litellm_model_name_model_info is not None else None
        )
        if cost_map_omits_token_price(model_id, builtin_key, base_model_key):
            return cast(  # cast-ok: TypedDict spread with overridden keys loses its type
                ModelInfo, {**model_info, "input_cost_per_token": None, "output_cost_per_token": None}
            )
        return model_info

    @staticmethod
    def _model_group_llm_provider(litellm_params: LiteLLM_Params) -> tuple[str, str]:
        try:
            return resolve_model_and_provider_for_metadata(
                model=litellm_params.model,
                custom_llm_provider=litellm_params.custom_llm_provider,
            )
        except litellm.exceptions.BadRequestError as e:
            verbose_router_logger.error("litellm.router.py::get_model_group_info() - %s", e)
            return "", ""

    def _set_model_group_info(
        self, model_group: str, user_facing_model_group_name: str, team_id: str | None = None
    ) -> ModelGroupInfo | None:
        """
        For a given model group name, return the combined model info

        Returns:
        - ModelGroupInfo if able to construct a model group
        - None if error constructing model group info
        """
        model_group_info: ModelGroupInfo | None = None

        total_tpm: int | None = None
        total_rpm: int | None = None
        total_itpm: int | None = None
        total_otpm: int | None = None
        configurable_clientside_auth_params: CONFIGURABLE_CLIENTSIDE_AUTH_PARAMS = None
        reasoning_efforts_initialized = False
        reasoning_efforts_unknown = False
        model_list: Final = self.get_model_list_of_routed_group(model_group)
        model_group_description: Final[str | None] = next(
            (d for d in (_configured_model_info(m).get("description") for m in model_list) if isinstance(d, str)),
            None,
        )
        for model in model_list:
            is_match = False
            if (
                "model_name" in model
                and model["model_name"] == model_group
                or "model_name" in model
                and self.pattern_router.route(model_group) is not None
            ):  # exact match
                is_match = True

            if not is_match:
                continue
            # model in model group found #
            litellm_params = LiteLLM_Params(**model["litellm_params"])
            # get configurable clientside auth params
            configurable_clientside_auth_params = litellm_params.configurable_clientside_auth_params

            # Cache nested dict access to avoid repeated temporary dict allocations
            model_litellm_params = model.get("litellm_params", {})
            model_info_dict = model.get("model_info", {})

            # get model tpm
            _deployment_tpm: int | None = None
            if _deployment_tpm is None:
                _deployment_tpm = model.get("tpm", None)
            if _deployment_tpm is None:
                _deployment_tpm = model_litellm_params.get("tpm", None)
            if _deployment_tpm is None:
                _deployment_tpm = model_info_dict.get("tpm", None)

            # get model rpm
            _deployment_rpm: int | None = None
            if _deployment_rpm is None:
                _deployment_rpm = model.get("rpm", None)
            if _deployment_rpm is None:
                _deployment_rpm = model_litellm_params.get("rpm", None)
            if _deployment_rpm is None:
                _deployment_rpm = model_info_dict.get("rpm", None)

            _deployment_itpm: int | None = model.get("itpm")
            if _deployment_itpm is None:
                _deployment_itpm = model_litellm_params.get("itpm", None)
            if _deployment_itpm is None:
                _deployment_itpm = model_info_dict.get("itpm", None)

            _deployment_otpm: int | None = model.get("otpm")
            if _deployment_otpm is None:
                _deployment_otpm = model_litellm_params.get("otpm", None)
            if _deployment_otpm is None:
                _deployment_otpm = model_info_dict.get("otpm", None)

            # get model info
            try:
                model_id = model_info_dict.get("id", None)
                if model_id is not None:
                    model_info = self.get_deployment_model_info(model_id=model_id, model_name=litellm_params.model)
                else:
                    model_info = None
            except Exception:
                model_info = None

            deployment_is_mapped = deployment_is_catalog_mapped(model_info, model_info_dict)

            # get llm provider
            litellm_model, llm_provider = self._model_group_llm_provider(litellm_params)

            if model_info is None:
                supported_openai_params = litellm.get_supported_openai_params(
                    model=litellm_model, custom_llm_provider=llm_provider
                )
                if supported_openai_params is None:
                    supported_openai_params = []

                # Get mode from database model_info if available, otherwise default to "chat"
                db_model_info = model.get("model_info", {})
                mode = db_model_info.get("mode", "chat")
                input_cost_per_token = _cost_value_as_float(db_model_info.get("input_cost_per_token"))
                output_cost_per_token = _cost_value_as_float(db_model_info.get("output_cost_per_token"))

                model_info = ModelMapInfo(
                    key=model_group,
                    max_tokens=None,
                    max_input_tokens=None,
                    max_output_tokens=None,
                    input_cost_per_token=input_cost_per_token,
                    output_cost_per_token=output_cost_per_token,
                    litellm_provider=llm_provider,
                    mode=mode,
                    supported_openai_params=supported_openai_params,
                    supports_system_messages=None,
                )

            if model_group_info is None:
                model_group_info = ModelGroupInfo(
                    **{
                        "model_group": user_facing_model_group_name,
                        "providers": [llm_provider],
                        "description": model_group_description,
                        **model_info,
                        "supports_fast_mode": True,
                        "supported_reasoning_efforts": None,
                        "supported_endpoints": None,
                    }
                )
            else:
                # if max_input_tokens > curr
                # if max_output_tokens > curr
                # if input_cost_per_token > curr
                # if output_cost_per_token > curr
                # supports_parallel_function_calling == True
                # supports_vision == True
                # supports_function_calling == True
                if llm_provider not in model_group_info.providers:
                    model_group_info.providers.append(llm_provider)
                if (
                    model_info.get("max_input_tokens", None) is not None
                    and model_info["max_input_tokens"] is not None
                    and (
                        model_group_info.max_input_tokens is None
                        or model_info["max_input_tokens"] > model_group_info.max_input_tokens
                    )
                ):
                    model_group_info.max_input_tokens = model_info["max_input_tokens"]
                if (
                    model_info.get("max_output_tokens", None) is not None
                    and model_info["max_output_tokens"] is not None
                    and (
                        model_group_info.max_output_tokens is None
                        or model_info["max_output_tokens"] > model_group_info.max_output_tokens
                    )
                ):
                    model_group_info.max_output_tokens = model_info["max_output_tokens"]
                _input_cost_per_token = _cost_value_as_float(model_info.get("input_cost_per_token"))
                if _input_cost_per_token is not None and (
                    model_group_info.input_cost_per_token is None
                    or _input_cost_per_token > (model_group_info.input_cost_per_token or 0.0)
                ):
                    model_group_info.input_cost_per_token = _input_cost_per_token
                _output_cost_per_token = _cost_value_as_float(model_info.get("output_cost_per_token"))
                if _output_cost_per_token is not None and (
                    model_group_info.output_cost_per_token is None
                    or _output_cost_per_token > (model_group_info.output_cost_per_token or 0.0)
                ):
                    model_group_info.output_cost_per_token = _output_cost_per_token
                if (
                    model_info.get("supports_parallel_function_calling", None) is not None
                    and model_info["supports_parallel_function_calling"] is True
                ):
                    model_group_info.supports_parallel_function_calling = True
                if model_info.get("supports_vision", None) is not None and model_info["supports_vision"] is True:
                    model_group_info.supports_vision = True
                if (
                    model_info.get("supports_function_calling", None) is not None
                    and model_info["supports_function_calling"] is True
                ):
                    model_group_info.supports_function_calling = True
                if (
                    model_info.get("supports_web_search", None) is not None
                    and model_info["supports_web_search"] is True
                ):
                    model_group_info.supports_web_search = True
                if (
                    model_info.get("supports_url_context", None) is not None
                    and model_info["supports_url_context"] is True
                ):
                    model_group_info.supports_url_context = True

                if model_info.get("supports_reasoning", None) is not None and model_info["supports_reasoning"] is True:
                    model_group_info.supports_reasoning = True
                if (
                    model_info.get("supported_openai_params", None) is not None
                    and model_info["supported_openai_params"] is not None
                ):
                    model_group_info.supported_openai_params = model_info["supported_openai_params"]
                if model_info.get("tpm", None) is not None and _deployment_tpm is None:
                    _deployment_tpm = model_info.get("tpm")
                if model_info.get("rpm", None) is not None and _deployment_rpm is None:
                    _deployment_rpm = model_info.get("rpm")

            model_group_info.supported_endpoints = (
                tuple(
                    dict.fromkeys(
                        (
                            *(model_group_info.supported_endpoints or ()),
                            *(
                                deployment_supported_endpoints(
                                    model_info.get("supported_endpoints"), model, llm_provider
                                )
                                if team_may_use_deployment(_configured_model_info(model).get("team_id"), team_id)
                                else ()
                            ),
                        )
                    )
                )
                or None
            )
            model_group_info.supports_fast_mode = model_group_info.supports_fast_mode and (
                AnthropicModelInfo.supports_fast_mode(litellm_model, llm_provider)
            )
            deployment_reasoning_efforts = resolve_supported_reasoning_efforts(
                model_info, deployment_is_mapped=deployment_is_mapped
            )
            if deployment_reasoning_efforts is None:
                reasoning_efforts_unknown = True
                model_group_info.supported_reasoning_efforts = None
            elif not reasoning_efforts_initialized:
                reasoning_efforts_initialized = True
                if not reasoning_efforts_unknown:
                    model_group_info.supported_reasoning_efforts = deployment_reasoning_efforts
            elif not reasoning_efforts_unknown:
                model_group_info.supported_reasoning_efforts = intersect_supported_reasoning_efforts(
                    model_group_info.supported_reasoning_efforts,
                    deployment_reasoning_efforts,
                )

            if _deployment_tpm is not None:
                if total_tpm is None:
                    total_tpm = 0
                total_tpm += _deployment_tpm

            if _deployment_rpm is not None:
                if total_rpm is None:
                    total_rpm = 0
                total_rpm += _deployment_rpm

            if _deployment_itpm is not None:
                if total_itpm is None:
                    total_itpm = 0
                total_itpm += _deployment_itpm

            if _deployment_otpm is not None:
                if total_otpm is None:
                    total_otpm = 0
                total_otpm += _deployment_otpm
        if model_group_info is not None:
            ## UPDATE WITH TOTAL TPM/RPM FOR MODEL GROUP
            if total_tpm is not None:
                model_group_info.tpm = total_tpm

            if total_rpm is not None:
                model_group_info.rpm = total_rpm

            if total_itpm is not None:
                model_group_info.itpm = total_itpm

            if total_otpm is not None:
                model_group_info.otpm = total_otpm

            ## UPDATE WITH CONFIGURABLE CLIENTSIDE AUTH PARAMS FOR MODEL GROUP
            if configurable_clientside_auth_params is not None:
                model_group_info.configurable_clientside_auth_params = configurable_clientside_auth_params

        return model_group_info

    def get_model_group_info(
        self, model_group: str, *, include_hidden: bool = False, team_id: str | None = None
    ) -> ModelGroupInfo | None:
        """
        For a given model group name, return the combined model info

        Returns:
        - ModelGroupInfo if able to construct a model group
        - None if error constructing model group info or hidden model group (unless include_hidden)
        """
        ## Check if model group alias
        if model_group in self.model_group_alias:
            item: Final = self.model_group_alias[model_group]
            if isinstance(item, str):
                _router_model_group = item
            elif isinstance(item, dict):
                if item["hidden"] is True and not include_hidden:
                    return None
                else:
                    _router_model_group = item["model"]
            else:
                return None

            return self._set_model_group_info(
                model_group=_router_model_group,
                user_facing_model_group_name=model_group,
                team_id=team_id,
            )

        ## Check if actual model
        return self._set_model_group_info(
            model_group=model_group, user_facing_model_group_name=model_group, team_id=team_id
        )

    async def get_model_group_usage(self, model_group: str) -> tuple[int | None, int | None]:
        """
        Returns current tpm/rpm usage for model group

        Parameters:
        - model_group: str - the received model name from the user (can be a wildcard route).

        Returns:
        - usage: Tuple[tpm, rpm]
        """
        dt: Final = get_utc_datetime()
        current_minute: Final = dt.strftime("%H-%M")  # use the same timezone regardless of system clock
        tpm_keys: Final[list[str]] = []
        rpm_keys: Final[list[str]] = []

        model_list: Final = self.get_model_list(model_name=model_group)
        if model_list is None:  # no matching deployments
            return None, None

        for model in model_list:
            id: str | None = model.get("model_info", {}).get("id")
            litellm_model: str | None = model["litellm_params"].get(
                "model"
            )  # USE THE MODEL SENT TO litellm.completion() - consistent with how global_router cache is written.
            if id is None or litellm_model is None:
                continue
            tpm_keys.append(
                RouterCacheEnum.TPM.value.format(
                    id=id,
                    model=litellm_model,
                    current_minute=current_minute,
                )
            )
            rpm_keys.append(
                RouterCacheEnum.RPM.value.format(
                    id=id,
                    model=litellm_model,
                    current_minute=current_minute,
                )
            )
        combined_tpm_rpm_keys: Final = tpm_keys + rpm_keys

        combined_tpm_rpm_values: Final = await self.cache.async_batch_get_cache(keys=combined_tpm_rpm_keys)
        if combined_tpm_rpm_values is None:
            return None, None

        tpm_usage_list: Final[list | None] = combined_tpm_rpm_values[: len(tpm_keys)]
        rpm_usage_list: Final[list | None] = combined_tpm_rpm_values[len(tpm_keys) :]

        ## TPM
        tpm_usage: int | None = None
        if tpm_usage_list is not None:
            for t in tpm_usage_list:
                if isinstance(t, int):
                    if tpm_usage is None:
                        tpm_usage = 0
                    tpm_usage += t
        ## RPM
        rpm_usage: int | None = None
        if rpm_usage_list is not None:
            for t in rpm_usage_list:
                if isinstance(t, int):
                    if rpm_usage is None:
                        rpm_usage = 0
                    rpm_usage += t
        return tpm_usage, rpm_usage

    async def get_model_group_io_token_usage(self, model_group: str) -> tuple[int | None, int | None]:
        """
        Returns current ITPM/OTPM usage for a model group (sum across deployments).
        """
        dt: Final = get_utc_datetime()
        current_minute: Final = dt.strftime("%H-%M")
        itpm_keys: Final[list[str]] = []
        otpm_keys: Final[list[str]] = []

        model_list: Final = self.get_model_list(model_name=model_group)
        if model_list is None:
            return None, None

        for model in model_list:
            model_id: str | None = model.get("model_info", {}).get("id")
            litellm_model: str | None = model["litellm_params"].get("model")
            if model_id is None or litellm_model is None:
                continue
            itpm_keys.append(
                RouterCacheEnum.ITPM.value.format(
                    id=model_id,
                    model=litellm_model,
                    current_minute=current_minute,
                )
            )
            otpm_keys.append(
                RouterCacheEnum.OTPM.value.format(
                    id=model_id,
                    model=litellm_model,
                    current_minute=current_minute,
                )
            )

        combined_values: Final = await self.cache.async_batch_get_cache(keys=itpm_keys + otpm_keys)
        if combined_values is None:
            return None, None

        itpm_values: Final = combined_values[: len(itpm_keys)]
        otpm_values: Final = combined_values[len(itpm_keys) :]

        total_itpm: int | None = None
        for value in itpm_values:
            if isinstance(value, int):
                total_itpm = (total_itpm or 0) + value

        total_otpm: int | None = None
        for value in otpm_values:
            if isinstance(value, int):
                total_otpm = (total_otpm or 0) + value

        return total_itpm, total_otpm

    @lru_cache(maxsize=DEFAULT_MAX_LRU_CACHE_SIZE)
    def _cached_get_model_group_info(self, model_group: str) -> ModelGroupInfo | None:
        """
        Cached version of get_model_group_info, uses @lru_cache wrapper

        This is a speed optimization, since set_response_headers makes a call to get_model_group_info on every request
        """
        return self.get_model_group_info(model_group)

    def cached_model_group_info(self, model_group: str) -> ModelGroupInfo | None:
        return self._cached_get_model_group_info(model_group)

    async def get_remaining_model_group_usage(self, model_group: str) -> dict[str, int]:
        model_group_info: Final = self._cached_get_model_group_info(model_group)

        returned_dict: Final[dict[str, int]] = {}

        # ITPM/OTPM groups emit input/output token headers, but they may also set
        # tpm/rpm, so build both sets rather than returning early - clients and
        # prometheus gauges that read the standard headers still get data.
        if model_group_info is not None and (model_group_info.itpm is not None or model_group_info.otpm is not None):
            current_itpm, current_otpm = await self.get_model_group_io_token_usage(model_group)
            returned_dict.update(
                build_io_token_rate_limit_headers(
                    itpm_limit=model_group_info.itpm,
                    otpm_limit=model_group_info.otpm,
                    current_itpm=current_itpm,
                    current_otpm=current_otpm,
                )
            )

        tpm_limit: Final = model_group_info.tpm if model_group_info is not None else None
        rpm_limit: Final = model_group_info.rpm if model_group_info is not None else None

        if tpm_limit is None and rpm_limit is None:
            return returned_dict

        current_tpm, current_rpm = await self.get_model_group_usage(model_group)

        if tpm_limit is not None:
            returned_dict["x-ratelimit-remaining-tokens"] = tpm_limit - (current_tpm or 0)
            returned_dict["x-ratelimit-limit-tokens"] = tpm_limit
        if rpm_limit is not None:
            returned_dict["x-ratelimit-remaining-requests"] = rpm_limit - (current_rpm or 0)
            returned_dict["x-ratelimit-limit-requests"] = rpm_limit

        return returned_dict

    async def set_response_headers(
        self,
        response: object,
        model_group: str | None = None,
        request_kwargs: dict[str, object] | None = None,
    ) -> Any:
        """
        Add the most accurate rate limit headers for a given model response.

        ## TODO: add model group rate limit headers
        # - if healthy_deployments > 1, return model group rate limit headers
        # - else return the model's rate limit headers
        """
        response = prepare_response_for_header_attachment(response)
        if response is None:
            return response

        additional_headers: Final = ensure_response_additional_headers(response)
        apply_response_model_id(
            response, find_deployment_metadata(request_kwargs) if request_kwargs is not None else None
        )
        additional_headers["x-litellm-model-group"] = model_group
        apply_quality_router_decision_headers(additional_headers, request_kwargs)
        additional_headers.update(complexity_router_decision_headers(request_kwargs))

        if model_group is not None:
            remaining_usage: Final = await self.get_remaining_model_group_usage(model_group)
            apply_remaining_usage_headers(additional_headers, remaining_usage)
        return response

    def _build_model_name_index(self, model_list: list) -> None:
        """
        Build model_name -> deployment indices mapping for O(1) lookups.

        This index allows us to find all deployments for a given model_name in O(1) time
        instead of O(n) linear scan through the entire model_list.
        """
        self.model_name_to_deployment_indices.clear()
        self.team_model_to_deployment_indices.clear()
        self.team_public_model_names = frozenset()

        for idx, model in enumerate(model_list):
            model_name = model.get("model_name")
            if model_name:
                if model_name not in self.model_name_to_deployment_indices:
                    self.model_name_to_deployment_indices[model_name] = []
                self.model_name_to_deployment_indices[model_name].append(idx)

            self._update_team_model_index(model, idx)

    def _build_model_id_to_deployment_index_map(self, model_list: list):
        """
        Build model index from model list to enable O(1) lookups immediately.
        This is called during initialization to avoid the race condition where
        requests arrive before model_id_to_deployment_index_map is populated.
        """
        # First populate the model_list
        self.model_list = []
        self._invalidate_model_group_info_cache()
        self._invalidate_access_groups_cache()
        for _, model in enumerate(model_list):
            # Extract model_info from the model dict
            model_info = model.get("model_info", {})
            model_id = model_info.get("id")

            # If no ID exists, generate one using the same logic as set_model_list
            if model_id is None:
                model_name = model.get("model_name", "")
                litellm_params = model.get("litellm_params", {})
                model_id = self.generate_model_id(model_name, litellm_params)
                # Update the model_info in the original list
                if "model_info" not in model:
                    model["model_info"] = {}
                model["model_info"]["id"] = model_id

            self._add_model_to_list_and_index_map(model=model, model_id=model_id)

    def get_model_ids(self, model_name: str | None = None, exclude_team_models: bool = False) -> list[str]:
        """
        if 'model_name' is none, returns all.

        Returns list of model id's.

        Optimized with O(1) or O(k) index lookup when model_name provided,
        instead of O(n) linear scan.
        """
        ids: Final = []

        if model_name is not None:
            # O(1) lookup in model_name index, then O(k) iteration where k = deployments for this model_name
            for model in self.deployments_named(model_name) or ():
                if "model_info" in model and "id" in model["model_info"]:
                    if exclude_team_models and model["model_info"].get("team_id"):
                        continue
                    ids.append(model["model_info"]["id"])
        else:
            # When model_name is None, return all model IDs
            # Use the index map keys for O(n) where n = total deployments
            with self._deployment_index_lock:
                id_indices: Final = tuple(self.model_id_to_deployment_index_map.items())
                models: Final = tuple(self.model_list)
            for model_id, idx in id_indices:
                model = models[idx]
                if "model_info" in model and "id" in model["model_info"]:
                    if exclude_team_models and model["model_info"].get("team_id"):
                        continue
                    ids.append(model_id)

        return ids

    def get_candidate_model_ids_for_route(self, model: str, team_id: str | None = None) -> frozenset[str]:
        """
        Deployment ids that could serve ``model`` for ``team_id``, following the same
        precedence ``_common_checks_available_deployment`` uses to build a candidate pool:
        ``model_group_alias``, then a routing group, then the first matching early-resolve
        path for a name that is not a ``model_name`` (team route, wildcard pattern via
        ``get_deployments_by_pattern``, team pattern router, default deployment), then the
        ``model_name`` and team indexes. Delegating to the router's own resolvers keeps this
        aligned with how a route actually resolves rather than re-deriving it, and unlike
        ``_common_checks_available_deployment`` it is read-only: it does not apply request
        fallbacks and (with ``include_team_models`` left off) does not raise. Lets a pre-call
        check tell a genuine cross-group route from same-group unavailability without leaking
        deployment ids into request kwargs bound for the provider.
        """
        resolved: Final = self.get_model_from_alias(model=model) or model
        routing_group_members: Final = self._get_routing_group_deployments(model=resolved, team_id=team_id)
        if routing_group_members is not None:
            return self._deployment_ids(routing_group_members)
        early: Final = self._try_early_resolve_deployments_for_model_not_in_names(
            model=resolved, request_team_id=team_id
        )
        if early is not None:
            early_deployments: Final = early[1]
            return self._deployment_ids(
                (early_deployments,) if isinstance(early_deployments, Mapping) else early_deployments
            )
        return self._deployment_ids(self.get_all_deployments(model_name=resolved, team_id=team_id))

    @staticmethod
    def _deployment_ids(deployments: Sequence[Mapping[str, object]]) -> frozenset[str]:
        return frozenset(
            str(model_info["id"])
            for deployment in deployments
            for model_info in (deployment.get("model_info"),)
            if isinstance(model_info, Mapping) and model_info.get("id") is not None
        )

    def _region_routable_deployments(
        self,
        deployments: Sequence[DeploymentTypedDict],
        request_kwargs: Mapping[str, object] | None,
        has_request_input: bool,
    ) -> tuple[Mapping[str, Any], ...]:
        """
        `deployments` without the ones `_pre_call_checks` keeps this request away from for being outside its allowed
        region, typed for the tag routing filter
        """
        allowed_model_region: Final = None if request_kwargs is None else request_kwargs.get("allowed_model_region")
        if allowed_model_region is None or not self.enable_pre_call_checks or not has_request_input:
            return tuple(deployments)
        if not isinstance(allowed_model_region, str):
            return ()
        return tuple(
            deployment
            for deployment in deployments
            if is_region_allowed(
                litellm_params=LiteLLM_Params.model_validate(deployment["litellm_params"]),
                allowed_model_region=allowed_model_region,
            )
        )

    async def _async_routable_cooled_deployment_ids(
        self,
        model: str,
        request_deployments: Sequence[DeploymentTypedDict],
        cooldown_deployments: Iterable[str],
        request_kwargs: Mapping[str, object],
        has_request_input: bool,
    ) -> frozenset[str]:
        """
        Ids of the cooling deployments in `request_deployments` that region and tag routing would send this request
        to, each tag routing pick counted along with the picks it falls back to once that one is unavailable, judged
        without recording a tag match on the request
        """
        cooled_ids: Final = frozenset(cooldown_deployments)
        metadata_variable_name: Final = get_metadata_variable_name_from_kwargs(request_kwargs)
        metadata: Final = request_kwargs.get(metadata_variable_name)
        if metadata_variable_name in request_kwargs and not isinstance(metadata, Mapping):
            return frozenset()
        tag_request: Final[Mapping[str, Any]] = (
            {**request_kwargs, metadata_variable_name: {**metadata}}
            if isinstance(metadata, Mapping)
            else request_kwargs
        )
        routable_ids = frozenset[str]()  # rebind-ok: gains each pass's tag routing picks
        remaining = tuple(  # rebind-ok: each pass drops what tag routing just picked
            deployment
            for deployment in self._region_routable_deployments(request_deployments, request_kwargs, has_request_input)
            if self._deployment_ids((deployment,)) <= cooled_ids
        )
        while remaining:
            try:
                picked = await get_deployments_for_tag(
                    llm_router_instance=self,
                    model=model,
                    healthy_deployments=remaining,
                    request_kwargs=tag_request,
                    metadata_variable_name=metadata_variable_name,
                    request_deployments=request_deployments,
                )
            except ValueError:  # tag routing sends this request to none of the rest
                break
            picked_identities = frozenset(id(deployment) for deployment in picked)
            rest = tuple(deployment for deployment in remaining if id(deployment) not in picked_identities)
            if len(rest) == len(remaining):
                break
            routable_ids |= self._deployment_ids(
                tuple(deployment for deployment in remaining if id(deployment) in picked_identities)
            )
            remaining = rest
        return routable_ids

    def has_model_id(self, candidate_id: str) -> bool:
        """
        O(1) membership check for a deployment ID without allocating large lists.

        Note: Call sites may pass a variable named `model` when it actually
        contains a deployment ID. This helper expects the deployment ID string.

        Uses the existing `model_id_to_deployment_index_map` which is kept
        in sync by `_build_model_id_to_deployment_index_map` and model-list
        mutation helpers.
        """
        return candidate_id in self.model_id_to_deployment_index_map

    def resolve_model_name_from_model_id(
        self, model_id: str | None, custom_llm_provider: str | None = None
    ) -> str | None:
        """
        Resolve model_name from model_id.

        This method attempts to find the correct model_name to use with the router
        so that litellm_params can be automatically injected from the model config.

        Strategy:
        1. First, check if model_id directly matches a model_name or deployment ID
        2. If not, search through router's model_list to find a match by litellm_params.model
        3. If not, keep model_id when the router routes that name itself (an alias, a routing group, or a wildcard
           whose deployment serves `custom_llm_provider`), else name it `custom_llm_provider/model_id` when only that
           form routes to such a wildcard
        4. Return the model_name if found, None otherwise

        Args:
            model_id: The model_id extracted from decoded video_id
                     (could be model_name or litellm_params.model value)
            custom_llm_provider: The provider extracted from the same id

        Returns:
            model_name if found, None otherwise. If None, the request will fall through
            to normal flow using environment variables.
        """
        if not model_id:
            return None

        # Strategy 1: Check if model_id directly matches a model_name or deployment ID
        if model_id in self.model_names:
            return model_id
        if self.has_model_id(model_id):
            deployment = self.get_deployment(model_id=model_id)
            if deployment is not None and deployment.model_name:
                return deployment.model_name
            return model_id

        # Strategy 2: Search through router's model_list to find by litellm_params.model
        all_models: Final = self.get_model_list(model_name=None)
        if not all_models:
            return None

        for deployment in all_models:
            litellm_params = deployment.get("litellm_params", {})
            actual_model = litellm_params.get("model")

            # Match by exact match or by checking if actual_model ends with /model_id or :model_id
            # e.g., model_id="veo-2.0-generate-001" matches actual_model="vertex_ai/veo-2.0-generate-001"
            matches = (
                actual_model == model_id
                or (actual_model and actual_model.endswith(f"/{model_id}"))
                or (actual_model and actual_model.endswith(f":{model_id}"))
            )

            if matches:
                model_name = deployment.get("model_name")
                if model_name:
                    return model_name

        if self.is_recognized_model(model_id):
            return model_id
        if custom_llm_provider is None:
            return None
        if self.pattern_router.routes_to_provider(model_id, custom_llm_provider):
            return model_id
        provider_prefixed: Final = f"{custom_llm_provider}/{model_id}"
        if not model_id.startswith(f"{custom_llm_provider}/") and self.pattern_router.routes_to_provider(
            provider_prefixed, custom_llm_provider
        ):
            return provider_prefixed
        return None

    def resolve_served_model_name(self, model_name: str, served_model: str) -> str:
        """
        `served_model` when `model_name` routes to it (the alias target, the routing group member, or the name
        itself), so a request can return to the group that served an earlier one, else `model_name`.
        """
        target: Final = self.get_model_from_alias(model_name) or model_name
        routing_group: Final = self.get_routing_group(target)
        members: Final = routing_group.models if routing_group is not None else ()
        return served_model if served_model == target or served_model in members else model_name

    def map_team_model(
        self, team_model_name: str | None, team_id: str, custom_llm_provider: str | None = None
    ) -> str | None:
        """
        Resolve explicit team models or select a deployment for model-less calls.

        Model-less calls with no provider use the first team deployment. With a provider, they
        use the first team deployment of that provider, pinned by ID when its public name also
        covers another provider.
        """
        models: Final = self.get_model_list(model_name=team_model_name, team_id=team_id)
        if not models:
            return None
        if team_model_name is None and custom_llm_provider is not None:

            def team_deployment_public_name(deployment: DeploymentTypedDict) -> str | None:
                model_info: Final = _configured_model_info(deployment)
                if model_info.get("team_id") != team_id:
                    return None
                team_public_model_name: Final = model_info.get("team_public_model_name")
                return (
                    team_public_model_name
                    if isinstance(team_public_model_name, str) and team_public_model_name
                    else deployment["model_name"]
                )

            matching_deployment: Final = next(
                (
                    model
                    for model in models
                    if team_deployment_public_name(model) is not None
                    and provider_for_generic_call(model["litellm_params"]) == custom_llm_provider
                ),
                None,
            )
            if matching_deployment is None:
                return None
            public_name: Final = team_deployment_public_name(matching_deployment)
            if public_name is None:
                return None
            same_public_name_deployments: Final = tuple(
                candidate for candidate in models if team_deployment_public_name(candidate) == public_name
            )
            if all(
                provider_for_generic_call(candidate["litellm_params"]) == custom_llm_provider
                for candidate in same_public_name_deployments
            ):
                return public_name
            deployment_id: Final = _configured_model_info(matching_deployment).get("id")
            return deployment_id if isinstance(deployment_id, str) else None
        for model in models:
            if model.get("model_info", {}).get("team_id") == team_id:
                if team_model_name is None:
                    # No model was specified (e.g. vector store endpoints).
                    # Return the deployment's public model name so the router
                    # can route to it and inject the BYOK API key.
                    return model.get("model_info", {}).get("team_public_model_name") or model.get("model_name")
                return team_model_name

        # No team-scoped deployment found; wildcard/pattern routes are
        # handled downstream by the pattern_router in _common_checks_available_deployment.
        return None

    def should_include_deployment(
        self, model_name: str, model: DeploymentTypedDict | dict, team_id: str | None = None
    ) -> bool:
        """
        Get the team-specific model name if team_id matches the deployment.
        """
        if (
            team_id is not None
            and (model.get("model_info") or {}).get("team_id") == team_id
            and model_name == (model.get("model_info") or {}).get("team_public_model_name")
        ):
            return True
        elif model_name is not None and model["model_name"] == model_name:
            # Fallback: check by internal model_name for non-team deployments
            # or deployments that haven't been migrated to team_public_model_name yet
            model_team_id: Final = (model.get("model_info") or {}).get("team_id")
            if (
                team_id is None  # requester has no team constraint
                or model_team_id is None  # global deployment - accessible to all teams
                or model_team_id == team_id  # deployment belongs to requester's team
            ):
                return True
        # No match: deployment is for a different team or doesn't match the requested model
        return False

    def get_all_deployments(
        self,
        model_name: str,
        model_alias: str | None = None,
        team_id: str | None = None,
    ) -> list[DeploymentTypedDict]:
        """
        Return all deployments of a model name

        Used for accurate 'get_model_list'.

        if team_id specified, only return team-specific models

        Optimized with O(1) index lookup instead of O(n) linear scan.

        Note: when team_id is provided, O(1) lookup in
        `team_model_to_deployment_indices` only applies when `model_name` is the
        team public model name. If a caller passes an internal deployment model
        name (for example, `model_name_<team_id>_<uuid>`), this method falls back
        to the standard model-name index / scan path.
        """
        returned_models: Final[list[DeploymentTypedDict]] = []

        # O(1) lookup in team_model index when team_id is provided
        if team_id is not None:
            team_models: Final = self._team_deployments_named(team_id, model_name)
            if team_models is not None:
                # O(k) where k = team deployments for this model_name (typically 1-10)
                for model in team_models:
                    if not self.should_include_deployment(model_name=model_name, model=model, team_id=team_id):
                        continue
                    if model_alias is not None:
                        alias_model = model.copy()
                        alias_model["model_name"] = model_alias
                        returned_models.append(alias_model)
                    else:
                        returned_models.append(model)
                if returned_models:
                    return returned_models

        # O(1) lookup in model_name index
        named_models: Final = self.deployments_named(model_name)
        if named_models is not None:
            # O(k) where k = deployments for this model_name (typically 1-10)
            for model in named_models:
                if self.should_include_deployment(model_name=model_name, model=model, team_id=team_id):
                    if model_alias is not None:
                        # Optimized: Use shallow copy since we only modify top-level model_name
                        # This is much faster than deepcopy for nested dict structures
                        alias_model = model.copy()
                        alias_model["model_name"] = model_alias
                        returned_models.append(alias_model)
                    else:
                        returned_models.append(model)
        elif team_id is not None:
            # Fallback: if team_id is provided and model_name not in index,
            # check if model_name matches any team_public_model_name
            # O(n) scan but only when team_id lookup fails
            for _idx, model in enumerate(self.model_list):
                if self.should_include_deployment(model_name=model_name, model=model, team_id=team_id):
                    if model_alias is not None:
                        # Optimized: Use shallow copy since we only modify top-level model_name
                        alias_model = model.copy()
                        alias_model["model_name"] = model_alias
                        returned_models.append(alias_model)
                    else:
                        returned_models.append(model)

        return returned_models

    _get_all_deployments = get_all_deployments

    def get_model_names(self, team_id: str | None = None) -> list[str]:
        """
        Returns all possible model names for the router, including models defined via model_group_alias.

        If a team_id is provided, only deployments configured with that team_id (i.e. team‐specific models)
        will yield their team public name.
        """
        deployments: Final = self.get_model_list() or []
        model_names: Final = []

        for deployment in deployments:
            model_info = deployment.get("model_info")
            if self._is_team_specific_model(model_info):
                team_model_name = self._get_team_specific_model(deployment=deployment, team_id=team_id)
                if team_model_name:
                    model_names.append(team_model_name)
            else:
                model_names.append(deployment.get("model_name", ""))

        return model_names

    def get_fully_blocked_model_names(self) -> set[str]:
        """
        Returns the set of model_names where every backing deployment has `blocked=True`.

        Used by `/v1/models` to hide paused models from client listings while still
        surfacing them on admin endpoints (e.g. `/model/info`). A model with at least
        one non-blocked deployment is still serviceable and remains visible.
        """
        deployments: Final = self.get_model_list() or []
        blocked_by_name: Final[dict[str, bool]] = {}
        for deployment in deployments:
            name = deployment.get("model_name") or ""
            if not name:
                continue
            is_blocked = (deployment.get("model_info") or {}).get("blocked") is True
            if name in blocked_by_name:
                blocked_by_name[name] = blocked_by_name[name] and is_blocked
            else:
                blocked_by_name[name] = is_blocked
        return {name for name, fully_blocked in blocked_by_name.items() if fully_blocked}

    @staticmethod
    def are_all_deployments_blocked(
        deployments: list[DeploymentTypedDict],
    ) -> bool:
        return len(deployments) > 0 and all(
            (deployment.get("model_info") or {}).get("blocked") is True for deployment in deployments
        )

    _are_all_deployments_blocked = are_all_deployments_blocked

    def _is_model_fully_blocked(self, model: str) -> bool:
        deployments: Final = self.get_model_list(model_name=model) or []
        return self.are_all_deployments_blocked(deployments=deployments)

    async def async_get_fully_unhealthy_model_names(self) -> set[str]:
        """
        Returns the set of model names where every backing deployment is currently
        marked unhealthy by background health checks (and the health state is not stale).

        Used by `/v1/models?healthy_only=true` to hide models that cannot serve any
        request. A model with at least one healthy (or unknown-health) deployment
        remains visible. Returns an empty set when no health state is available, so
        callers fail open to the unfiltered listing.

        Notes:
            - Mirrors `_async_filter_health_check_unhealthy_deployments`: when
              `allowed_fails_policy` is set, cooldown is the sole routing exclusion
              mechanism, so nothing is hidden here either.
            - Team-specific public model names (`team_public_model_name`) are
              aggregated alongside `model_name`, so team aliases of fully-unhealthy
              deployments are hidden too (unlike `get_fully_blocked_model_names`,
              which matches `model_name` only).
            - Wildcard routes (e.g. `openai/*`) are matched by their literal
              deployment name only; models expanded from a wildcard route are not
              hidden (fail open).
            - Intentionally diverges from the routing-time safety net (which
              bypasses the health filter when every candidate is unhealthy and
              still attempts the request): hiding here is presentation-only —
              it answers "should this model be advertised?", not "should a
              request for it still be attempted?". A hidden model can still be
              called directly.
        """
        if self.allowed_fails_policy is not None:
            return set()
        unhealthy_ids: Final = await self.health_state_cache.async_get_unhealthy_deployment_ids()
        if not unhealthy_ids:
            return set()
        deployments: Final = self.get_model_list() or []
        unhealthy_by_name: Final[dict[str, bool]] = {}
        for deployment in deployments:
            model_info = deployment.get("model_info") or {}
            names = [deployment.get("model_name") or ""]
            team_public_model_name = model_info.get("team_public_model_name")
            if team_public_model_name:
                names.append(team_public_model_name)
            is_unhealthy = model_info.get("id") in unhealthy_ids
            for name in names:
                if not name:
                    continue
                if name in unhealthy_by_name:
                    unhealthy_by_name[name] = unhealthy_by_name[name] and is_unhealthy
                else:
                    unhealthy_by_name[name] = is_unhealthy
        return {name for name, fully_unhealthy in unhealthy_by_name.items() if fully_unhealthy}

    def _get_team_specific_model(self, deployment: DeploymentTypedDict, team_id: str | None = None) -> str | None:
        """
        Get the team-specific model name if team_id matches the deployment.

        Args:
            deployment: DeploymentTypedDict - The model deployment
            team_id: Optional[str] - If passed, will return router models set with a `team_id` matching the passed `team_id`.

        Returns:
            str: The `team_public_model_name` if team_id matches
            None: If team_id doesn't match or no team info exists
        """
        model_info: Final[dict | None] = deployment.get("model_info") or {}
        if model_info is None:
            return None
        if team_id == model_info.get("team_id"):
            return model_info.get("team_public_model_name")
        return None

    def _is_team_specific_model(self, model_info: dict | None) -> bool:
        """
        Check if model info contains team-specific configuration.

        Args:
            model_info: Model information dictionary

        Returns:
            bool: True if model has team-specific configuration
        """
        return bool(model_info and model_info.get("team_id"))

    def get_model_list_from_model_alias(self, model_name: str | None = None) -> list[DeploymentTypedDict]:
        """
        Helper function to get model list from model alias.

        Used by `.get_model_list` to get model list from model alias.
        """
        returned_models: Final[list[DeploymentTypedDict]] = []

        if model_name is not None:
            # Fast path: direct dict lookup avoids scanning all aliases for non-alias model names.
            if model_name not in self.model_group_alias:
                return returned_models
            alias_items = [(model_name, self.model_group_alias[model_name])]
        else:
            alias_items = list(self.model_group_alias.items())

        for model_alias, model_value in alias_items:
            if isinstance(model_value, str):
                _router_model_name: str = model_value
            elif isinstance(model_value, dict):
                _model_value = RouterModelGroupAliasItem(**model_value)
                if _model_value["hidden"] is True and model_name is None:
                    continue
                _router_model_name = _model_value["model"]
            else:
                continue

            if (alias_group := self.get_routing_group(_router_model_name)) is not None:
                returned_models.extend(
                    {**row, "model_name": model_alias} for row in self._materialize_routing_group_rows((alias_group,))
                )
            else:
                returned_models.extend(self.get_all_deployments(model_name=_router_model_name, model_alias=model_alias))

        return returned_models

    def get_model_list_from_routing_groups(self, model_name: str | None = None) -> Sequence[DeploymentTypedDict]:
        """
        Callable routing groups materialized as model-list rows, mirroring
        `get_model_list_from_model_alias`: each member deployment is emitted
        under the group's name (via `get_all_deployments`' `model_alias`
        rewrite), which is what surfaces groups in `get_model_names`,
        `/v1/models` discovery, `get_model_group_usage`, and the
        blocked/unhealthy hiding that all read `get_model_list`.
        """
        if model_name is not None:
            group: Final = self.get_routing_group(model_name)
            return self._materialize_routing_group_rows((group,)) if group is not None else ()
        cached: Final = self._routing_group_rows
        if cached is not None:
            return cached
        rows: Final = self._materialize_routing_group_rows(
            tuple(
                callable_group
                for name in self.routing_groups
                if (callable_group := self.get_routing_group(name)) is not None
            )
        )
        self._routing_group_rows = rows
        return rows

    def _materialize_routing_group_rows(self, groups: tuple[RoutingGroup, ...]) -> tuple[DeploymentTypedDict, ...]:
        return tuple(
            self._as_routing_group_row(apply_routing_group_priority(group, member, deployment))
            for group in groups
            for member in group.models
            for deployment in self.get_all_deployments(model_name=member, model_alias=group.group_name)
        )

    @staticmethod
    def _as_routing_group_row(deployment: DeploymentTypedDict) -> DeploymentTypedDict:
        """
        A member deployment re-emitted under its group's name must not carry
        the member's `access_groups`: access groups grant member names, never
        the group, so inheriting them here would let a key holding a member's
        access group list and call the whole group.
        """
        model_info: Final = {k: v for k, v in (deployment.get("model_info") or {}).items() if k != "access_groups"}
        return {**deployment, "model_info": model_info}

    TIER_PARAMS_NEVER_DROPPED: Final = frozenset(all_litellm_params) | frozenset(
        {
            "additional_drop_params",
            "drop_params",
            "messages",
            "model",
            "extra_headers",
            "max_tokens",
            "max_completion_tokens",
        }
    Router.__init__.__signature__ = _CORE_SIGNATURE.replace(  # pyright: ignore[reportFunctionMemberAccess]  # inspect reads it
        parameters=[parameter for parameter in _CORE_SIGNATURE.parameters.values() if parameter.name != "_host"]
    )
    Router.__init__.__doc__ = PythonCore.__init__.__doc__


def __getattr__(name: str) -> object:
    """Module-level helpers that lived in this module before the core moved out."""
    return getattr(router_python_core, name)


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
