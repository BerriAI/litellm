"""
The deployment outcome callbacks ``RustCore`` serves as members: ``deployment_callback_on_failure``,
``deployment_callback_on_success`` and ``sync_deployment_callback_on_success``. router-core counts
the outcome and decides the cooldown; this module reads litellm's logging kwargs, drops the
failures the Python router does not hold against the deployment, and drives the Rust call. The
stream wrappers report a stream's outcome through the same members when the stream ends.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Final, Literal

from litellm.litellm_core_utils.core_helpers import as_str_mapping

if TYPE_CHECKING:
    from litellm.router_rust.steps.callbacks import OptionalChecks
    from litellm.router_rust.surface.registry import HostRegistry
    from litellm.router_utils.cooldown_cache import CooldownCache
    from litellm.rust_bridge._native import (
        RouterCall,
        RouterCoreHandle,
        _FailureReportSteps,  # pyright: ignore[reportPrivateUsage]  # the stub's step shapes
        _SuccessReportSteps,  # pyright: ignore[reportPrivateUsage]  # see above
    )
    from litellm.types.router import Deployment


def _cast_status(exception_status: object) -> int | None:
    if isinstance(exception_status, int) and not isinstance(exception_status, bool):
        return exception_status
    if isinstance(exception_status, str):
        try:
            return int(exception_status)
        except ValueError:
            return 500
    return None


def failure_status(exception_status: object) -> int | None:
    """``cast_exception_status_to_int``: a number as given, a numeric string read as one, any
    other string 500. A number no status can hold reads as no status."""
    status: Final = _cast_status(exception_status)
    return status if status is not None and 0 <= status <= 65535 else None


def _logged_deployment_id(kwargs: Mapping[str, object]) -> object:
    """``litellm_params.model_info.id`` of a logged call."""
    litellm_params: Final = as_str_mapping(kwargs.get("litellm_params")) or {}
    return (as_str_mapping(litellm_params.get("model_info")) or {}).get("id")


def first_cooldown(*sources: Mapping[str, object] | None) -> int | float | None:
    """``_first_present(*sources, key="cooldown_time")``, the number as it was written."""
    for source in sources:
        value = source.get("cooldown_time") if source is not None else None
        if isinstance(value, int | float) and not isinstance(value, bool):
            return value
    return None


def _configured_cooldown(model_info: object, litellm_params: Mapping[str, object]) -> float | None:
    """``first_cooldown`` of a logged call, as the float router-core takes."""
    seconds: Final = first_cooldown(as_str_mapping(model_info), litellm_params)
    return None if seconds is None else float(seconds)


def python_cooldown_time(seconds: float, *sources: float | None) -> float:
    """``seconds`` as ``deployment_callback_on_failure`` hands them on. Router-core's duration is
    always a float; Python's is the number that won its precedence, an int or a float as it was
    written. ``sources`` run in that precedence, and only the type is read from them."""
    return next((source for source in sources if source is not None and source == seconds), seconds)


def _fallback_cooldown(deployment: "Deployment | None") -> float | None:
    """``first_cooldown`` of the deployment a failed hop tried, as ``_core_get_model_info`` lists it."""
    if deployment is None:
        return None
    listed: Final = deployment.model_dump()
    return first_cooldown(as_str_mapping(listed.get("model_info")), as_str_mapping(listed.get("litellm_params")))


def _deployment_cooldown(deployment: Mapping[str, object]) -> float | None:
    """``_configured_cooldown`` of a deployment as listed."""
    return _configured_cooldown(deployment.get("model_info"), as_str_mapping(deployment.get("litellm_params")) or {})


def logged_call(op: str, kwargs: Mapping[str, object], started: datetime | None = None) -> Mapping[str, object]:
    """The logging kwargs ``deployment_callback_on_failure`` would read for a call made with
    ``kwargs``: the call's logging object's (a proxy request carries one), else what the request says.
    Read before the failure logger stamps the call's ``end_time``, so it is stamped here."""
    ended: Final = datetime.now()  # noqa: DTZ005  # naive to match the logging pipeline's api_call_start_time
    details: Final = as_str_mapping(getattr(kwargs.get("litellm_logging_obj"), "model_call_details", None))
    call_type: Final = op.rsplit(".", 1)[-1]
    if details is not None:
        return {"call_type": call_type, **details, "end_time": ended}
    fields: Final = ("metadata", "litellm_metadata", "client_side_timeout", "timeout")
    return {
        "call_type": call_type,
        "litellm_params": {name: kwargs.get(name) for name in fields},
        "start_time": started,
        "end_time": ended,
    }


def _authenticates_per_caller(litellm_params: object) -> bool:
    """``github_copilot_per_user_credential_name`` in ``_should_run_cooldown_logic``. Per-user params
    with no credential raise there, which ends Python's failure callback short of any cooldown."""
    from litellm.exceptions import BadRequestError
    from litellm.llms.github_copilot.per_user_auth import github_copilot_per_user_credential_name

    try:
        return github_copilot_per_user_credential_name(litellm_params) is not None
    except BadRequestError:
        return True


def failure_already_logged(kwargs: Mapping[str, object], event: Literal["async_failure", "sync_failure"]) -> bool:
    """``should_run_logging``: a logging object logs one failure a request. The attempts of a request
    that shares one (``litellm_logging_obj``) after the first logged failure run no failure callback;
    without one, every litellm call logs on its own."""
    from litellm.litellm_core_utils.litellm_logging import Logging

    logging_obj: Final = kwargs.get("litellm_logging_obj")
    return isinstance(logging_obj, Logging) and logging_obj.model_call_details.get(f"has_logged_{event}") is True


def _success_target(kwargs: Mapping[str, object]) -> tuple[str, Mapping[str, object]] | None:
    """The deployment id and metadata a success callback counts under, or None."""
    params: Final = as_str_mapping(kwargs.get("litellm_params")) or {}
    metadata: Final = as_str_mapping(params.get("metadata"))
    if metadata is None:
        return None
    model_info: Final = as_str_mapping(params.get("model_info")) or {}
    deployment_id: Final = model_info.get("id")
    if metadata.get("model_group") is None or deployment_id is None:
        return None
    return str(deployment_id), metadata


class RustHealthCallbacks:
    """Outcome callbacks read into the Rust core's health state."""

    _handle: "RouterCoreHandle"
    _cooldown_given: object

    def _drive_sync_steps(self, call: "RouterCall[_FailureReportSteps | _SuccessReportSteps]") -> Mapping[str, object]:
        raise NotImplementedError

    async def _drive_cache_steps(
        self, call: "RouterCall[_FailureReportSteps | _SuccessReportSteps]"
    ) -> Mapping[str, object]:
        raise NotImplementedError

    def _sync_allowed_fails(self) -> None:
        raise NotImplementedError

    def _core_get_allowed_fails_from_policy(self, exception: Exception) -> int | None:
        """The router ``allowed_fails_policy``'s allowance for this exception, decided by router-core."""
        from litellm.router_rust.core import attempt_failure

        return self._handle.allowed_fails_for(dict(attempt_failure(exception)))

    def _deployment_of(self, deployment_id: object) -> "Deployment | None":
        # RustCore sets `_hosted` as an attribute in its constructor.
        hosted: Final[HostRegistry] = self.__dict__["_hosted"]  # pyright: ignore[reportAny]  # RustCore's own attribute
        return hosted._core_get_deployment(model_id=deployment_id) if isinstance(deployment_id, str) else None  # pyright: ignore[reportPrivateUsage]  # the host registry's id lookup

    def deployment_tracks_usage(self, deployment_id: object, model_group: object) -> bool:
        """``_cached_deployment_tracks_usage``. A deployment whose model is not mapped raises there,
        which ends Python's usage write, so it reads as not tracked."""
        # RustCore sets `_hosted` as an attribute in its constructor.
        hosted: Final[HostRegistry] = self.__dict__["_hosted"]  # pyright: ignore[reportAny]  # RustCore's own attribute
        if not isinstance(deployment_id, str) or not isinstance(model_group, str):
            return False
        try:
            return hosted.cached_deployment_tracks_usage(deployment_id, model_group)
        except Exception:  # noqa: BLE001  # as `deployment_callback_on_success` swallows it
            return False

    def cooldown_event(self, deployment_id: str, exception_status: object, cooldown_time: float | None) -> None:
        """The end of ``set_cooldown_deployments``, once the cooldown is written: the cooldown event
        Prometheus reads, as a task of the running loop. Outside one this raises ``RuntimeError``."""
        import asyncio

        from litellm.router_api import RoutingContext
        from litellm.router_utils import cooldown_handlers

        _ = asyncio.get_running_loop()
        _ = asyncio.create_task(  # fire and forget, as set_cooldown_deployments schedules it
            cooldown_handlers.router_cooldown_event_callback(
                ctx=RoutingContext(self),  # pyright: ignore[reportArgumentType]  # the core's RoutingContext
                deployment_id=deployment_id,
                exception_status=exception_status,  # pyright: ignore[reportArgumentType]  # the status as raised
                cooldown_time=cooldown_time,
            )
        )

    def cooldown_event_on_own_loop(
        self, deployment_id: str, exception_status: object, cooldown_time: float | None
    ) -> None:
        """``cooldown_event`` for a sync walk ``function_with_fallbacks`` would run on its private
        loop: ``set_cooldown_deployments`` finds that loop however the caller runs, and the event's
        task keeps what it raises to itself."""
        import asyncio

        from litellm.litellm_core_utils import asyncify

        async def fire() -> None:
            self.cooldown_event(deployment_id, exception_status, cooldown_time)
            await asyncio.sleep(0)

        asyncify.run_async_function(fire)  # pyright: ignore[reportUnknownMemberType]  # the untyped sync bridge

    def python_cooldown(self, seconds: float, configured: float | None = None, error: object = None) -> float:
        """The ``time_to_cooldown`` Python hands ``set_cooldown_deployments``, for router-core's
        ``seconds``. A failure callback's precedence is the ``configured`` number, then the ``retry-after``
        header of ``error``, then the router's own; a path passing neither hands on the router's own."""
        from litellm.router_rust.core import retry_after_seconds

        retry_after: Final = retry_after_seconds(error) if isinstance(error, BaseException) else None
        default: Final = self._cooldown_given
        return python_cooldown_time(
            seconds,
            configured,
            None if retry_after is None else int(retry_after),
            default if isinstance(default, int | float) else None,
        )

    def cooled_down(
        self,
        done: Mapping[str, object],
        deployment_id: object,
        exception_status: object,
        *,
        own_loop: bool = False,
        configured: float | None = None,
        error: object = None,
    ) -> bool:
        """Whether a failure report cooled the deployment down, with its cooldown event, which
        carries ``python_cooldown`` of ``configured`` and ``error``."""
        cooled: Final = bool(done.get("cooled"))
        if cooled and isinstance(deployment_id, str):
            seconds: Final = done.get("cooldown_time")
            event: Final = self.cooldown_event_on_own_loop if own_loop else self.cooldown_event
            event(
                deployment_id,
                exception_status,
                self.python_cooldown(seconds, configured, error) if isinstance(seconds, float) else None,
            )
        return cooled

    def per_user_deployment(self, deployment_id: object) -> bool:
        """The failed deployment authenticates per caller (``_should_run_cooldown_logic``), which is
        all ``set_cooldown_deployments`` reads when it is called without the request's kwargs."""
        deployment: Final = self._deployment_of(deployment_id)
        return deployment is not None and _authenticates_per_caller(deployment.litellm_params)

    def ran_as_request_deployment(self, logged_id: object, deployment_id: object) -> bool:
        """Whether a call against ``deployment_id`` ran under credentials the request supplied
        (``_handle_clientside_credential``): as ``logged_id``, a request-scoped deployment the router
        does not hold, so its failure is that id's, never the shared deployment's."""
        return (
            logged_id is not None
            and deployment_id is not None
            and str(logged_id) != str(deployment_id)
            and self._deployment_of(logged_id) is None
        )

    def failure_exits(
        self, exception: object, logged: Mapping[str, object], deployment_id: object
    ) -> Mapping[str, bool]:
        """What the Python router does not hold against the deployment, for every failure it logs.
        ``outside_deployment``: ``deployment_callback_on_failure``'s early exits, counting nothing.
        ``caller_scoped``: ``_should_run_cooldown_logic``'s per-caller exits, counted but not cooled;
        its exits on the exception's class are router-core's (``FailureKind``)."""
        from litellm.router_utils.cooldown_handlers import (
            _is_per_user_provider_request,  # pyright: ignore[reportPrivateUsage]  # the cooldown check's own rule
            is_background_response_poll_not_found,
            is_caller_scoped_auth_failure,
            is_caller_timeout_408,
            is_failure_outside_selected_deployment,
            is_response_lookup_not_found,
            is_video_lookup_not_found,
        )

        status: Final[object] = getattr(exception, "status_code", "")
        litellm_params: Final = as_str_mapping(logged.get("litellm_params")) or {}
        outside: Final = (
            is_failure_outside_selected_deployment(exception if isinstance(exception, BaseException) else None)
            or (isinstance(exception, Exception) and is_background_response_poll_not_found(exception, litellm_params))
            or is_video_lookup_not_found(exception, logged)
            or is_response_lookup_not_found(exception, logged)
            or is_caller_timeout_408(logged, status)  # pyright: ignore[reportArgumentType]  # the status as raised
            or is_caller_scoped_auth_failure(self._deployment_of(deployment_id), status)  # pyright: ignore[reportArgumentType]  # see above
        )
        return {
            "outside_deployment": outside,
            "caller_scoped": _is_per_user_provider_request(logged) or self.per_user_deployment(deployment_id),
        }

    def blocked_failure_exempt(self, exception: BaseException, kwargs: Mapping[str, object], generic: bool) -> bool:
        """The exits of ``_trigger_cooldown_for_failed_deployment`` for a failure the request's shared
        logging object does not log: the selection tag is not one of them; a stream stall and a
        generic call's 404 are."""
        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.router_utils.cooldown_handlers import (
            cast_exception_status_to_int,
            is_advisor_orchestration_failure,
            is_caller_timeout_408,
            is_stream_stall,
        )
        from litellm.router_utils.fallback_event_handlers import (
            _REQUEST_SCOPED_STATUS_CODES,  # pyright: ignore[reportPrivateUsage]  # the fallback path's own rule
        )

        exception_status: Final[object] = getattr(exception, "status_code", "")
        logging_obj: Final = kwargs.get("litellm_logging_obj")
        details: Final[Mapping[str, object]] = (
            logging_obj.model_call_details if isinstance(logging_obj, Logging) else {}
        )
        return (
            (isinstance(exception, Exception) and is_advisor_orchestration_failure(exception))
            or is_stream_stall(exception)
            or (generic and cast_exception_status_to_int(exception_status) in _REQUEST_SCOPED_STATUS_CODES)  # pyright: ignore[reportArgumentType]  # the status as raised
            or is_caller_timeout_408(details, exception_status, ended=datetime.now())  # pyright: ignore[reportArgumentType]  # noqa: DTZ005  # see logged_call
        )

    def trigger_fallback_cooldown(
        self, kwargs: Mapping[str, object], exception: object, generic: bool, *, own_loop: bool = False
    ) -> None:
        """The failed fallback hop's ``_trigger_cooldown_for_failed_deployment``, which runs only when
        the request's logging object blocked the hop's failure callbacks. ``generic``: the call's kwargs
        carry ``original_generic_function``. ``own_loop``: see ``cooldown_event_on_own_loop``."""
        from litellm._logging import verbose_router_logger
        from litellm.router_utils.fallback_event_handlers import (
            _failure_callbacks_blocked_logging_object,  # pyright: ignore[reportPrivateUsage]  # the fallback path's own rule
        )

        blocked: Final = _failure_callbacks_blocked_logging_object(kwargs)
        if blocked is None or not isinstance(exception, Exception):
            return
        try:
            call: Final = self._fallback_failure_call(kwargs, exception, blocked.model_call_details, generic)
            if call is not None:
                _ = self.cooled_down(
                    self._drive_sync_steps(call),
                    getattr(exception, "failed_deployment_id", None),
                    getattr(exception, "status_code", ""),
                    own_loop=own_loop,
                    configured=_fallback_cooldown(
                        self._deployment_of(getattr(exception, "failed_deployment_id", None))
                    ),
                    error=exception,
                )
        except Exception as e:  # noqa: BLE001  # best effort, as the Python helper: it never breaks the fallback response
            verbose_router_logger.debug("Error triggering cooldown for fallback deployment: %s", e)

    def _fallback_failure_call(
        self,
        kwargs: Mapping[str, object],
        exception: Exception,
        model_call_details: Mapping[str, object],
        generic: bool,
    ) -> "RouterCall[_FailureReportSteps] | None":
        """The Rust call recording a failed hop, or None past the exits of
        ``_trigger_cooldown_for_failed_deployment``: another list than ``failure_exits``, checked per
        failed hop. The selection tag is not on it; a stream stall and a generic call's 404 are."""
        from litellm.router_rust.core import attempt_failure
        from litellm.router_utils.cooldown_handlers import (
            _is_per_user_provider_request,  # pyright: ignore[reportPrivateUsage]  # the cooldown check's own rule
            cast_exception_status_to_int,
            is_advisor_orchestration_failure,
            is_caller_scoped_auth_failure,
            is_caller_timeout_408,
            is_stream_stall,
        )
        from litellm.router_utils.fallback_event_handlers import (
            _REQUEST_SCOPED_STATUS_CODES,  # pyright: ignore[reportPrivateUsage]  # the fallback path's own rule
        )

        self._sync_allowed_fails()
        exception_status: Final[object] = getattr(exception, "status_code", "")
        if (
            is_advisor_orchestration_failure(exception)
            or is_stream_stall(exception)
            or (generic and cast_exception_status_to_int(exception_status) in _REQUEST_SCOPED_STATUS_CODES)  # pyright: ignore[reportArgumentType]  # the status as raised
            or is_caller_timeout_408(model_call_details, exception_status, ended=datetime.now())  # pyright: ignore[reportArgumentType]  # noqa: DTZ005  # see logged_call
        ):
            return None
        deployment_id: Final[object] = getattr(exception, "failed_deployment_id", None)
        if not isinstance(deployment_id, str):
            return None
        deployment: Final = self._deployment_of(deployment_id)
        if is_caller_scoped_auth_failure(deployment, exception_status):  # pyright: ignore[reportArgumentType]  # see above
            return None
        return self._handle.report_failure(
            deployment_id,
            {
                **attempt_failure(exception),
                "status": failure_status(exception_status),
                "caller_scoped": _is_per_user_provider_request(kwargs) or self.per_user_deployment(deployment_id),
            },
            deployment_cooldown=True,
            count_only=exception_status == "",
            cooldown_time=None if deployment is None else _deployment_cooldown(deployment.model_dump()),
        )

    def deployment_callback_on_failure(
        self,
        kwargs: Mapping[str, object],
        completion_response: object,
        start_time: object,
        end_time: object,
    ) -> bool:
        """Count the failure and cool the deployment down when the rules say so."""
        call: Final = self._failure_call(kwargs)
        litellm_params: Final = as_str_mapping(kwargs.get("litellm_params")) or {}
        return call is not None and self.cooled_down(
            self._drive_sync_steps(call),
            _logged_deployment_id(kwargs),
            getattr(kwargs.get("exception"), "status_code", ""),
            configured=first_cooldown(as_str_mapping(litellm_params.get("model_info")), litellm_params),
            error=kwargs.get("exception"),
        )

    async def report_stream_failure(
        self, kwargs: Mapping[str, object], callbacks: Mapping[str, bool] | None = None
    ) -> None:
        """A stream's failure after its attempt returned it, as an async call's failure callbacks
        record it: counted, cooled down by the rules, and one more request this minute
        (``async_deployment_callback_on_failure``). No cooldown event: the stream logs its failure
        through ``dispatch_failure_handlers``, whose sync callbacks run in an executor thread, where
        ``set_cooldown_deployments`` has no loop to schedule one on."""
        call: Final = self._failure_call(kwargs, counts_request=True, callbacks=callbacks)
        if call is not None:
            _ = await self._drive_cache_steps(call)

    def report_stream_failure_sync(
        self, kwargs: Mapping[str, object], callbacks: Mapping[str, bool] | None = None
    ) -> None:
        """``report_stream_failure`` for a sync stream: a sync call's failure callback counts no request.
        No cooldown event: the stream logs its failure in a ``threading.Thread`` of its own, which
        has no running loop whatever the consumer's thread has."""
        call: Final = self._failure_call(kwargs, callbacks=callbacks)
        if call is not None:
            _ = self._drive_sync_steps(call)

    def _failure_call(
        self,
        kwargs: Mapping[str, object],
        counts_request: bool = False,
        callbacks: Mapping[str, bool] | None = None,
    ) -> "RouterCall[_FailureReportSteps] | None":
        """The Rust call recording a logged failure, or None when it records nothing: one the Python
        router does not hold against the deployment is still a request of the minute
        (``async_deployment_callback_on_failure`` makes none of the failure callback's exits)."""
        from litellm.litellm_core_utils import core_helpers
        from litellm.router_rust.core import attempt_failure

        self._sync_allowed_fails()
        exception: Final = kwargs.get("exception")
        litellm_params: Final = as_str_mapping(kwargs.get("litellm_params")) or {}
        model_info: Final = litellm_params.get("model_info", {})
        info: Final = as_str_mapping(model_info)
        deployment_id: Final = info.get("id") if info is not None else None
        if deployment_id is None:
            return None
        exits: Final = self.failure_exits(exception, kwargs, deployment_id)
        if exits["outside_deployment"] and not counts_request:
            return None
        exception_status: Final[object] = getattr(exception, "status_code", "")
        described: Final[Mapping[str, object]] = (
            attempt_failure(exception) if isinstance(exception, BaseException) else {"ok": False, "kind": "APIError"}
        )
        failure: Final = {**described, **exits, "status": failure_status(exception_status)}
        metadata: Final[object] = core_helpers.get_litellm_metadata_from_kwargs(dict(kwargs))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # an untyped helper's dict
        group: Final = (as_str_mapping(metadata) or {}).get("model_group")  # pyright: ignore[reportUnknownArgumentType]  # read as a mapping
        usage_model: Final = (as_str_mapping(metadata) or {}).get("deployment")  # pyright: ignore[reportUnknownArgumentType]  # see above
        return self._handle.report_failure(
            str(deployment_id),
            failure,
            deployment_cooldown=True,
            count_only=exception_status == "",
            requested_group=group if isinstance(group, str) else None,
            cooldown_time=_configured_cooldown(model_info, litellm_params),
            counts_request=counts_request,
            usage_model=usage_model if isinstance(usage_model, str) else None,
            callbacks=callbacks,
        )

    def sync_deployment_callback_on_success(
        self,
        kwargs: Mapping[str, object],
        completion_response: object,
        start_time: object,
        end_time: object,
    ) -> str | None:
        """Count a success this minute; returns the counter's key."""
        return self.report_stream_success_sync(kwargs)

    def report_stream_success_sync(
        self, kwargs: Mapping[str, object], callbacks: Mapping[str, bool] | None = None
    ) -> str | None:
        """``sync_deployment_callback_on_success``, for a sync stream that ended."""
        from litellm.litellm_core_utils.batch_utils import is_batch_retrieve_call_type

        if is_batch_retrieve_call_type(kwargs.get("call_type")):
            return None
        target: Final = _success_target(kwargs)
        if target is None:
            return None
        key: Final = self._drive_sync_steps(self._handle.count_success(target[0], callbacks)).get("key")
        return key if isinstance(key, str) else None

    async def repin_after_success(self, kwargs: Mapping[str, object], completion_response: object) -> None:
        """The session re-pin ``deployment_callback_on_success`` runs for a success logged against a
        deployment the router serves (``_repin_session_after_success``). The Rust core runs it from the
        attempt itself, or from the stream wrapper when a stream ends."""
        from litellm.litellm_core_utils.batch_utils import is_batch_retrieve_call_type

        call_type: Final = kwargs.get("call_type")
        if call_type == "_arealtime" or is_batch_retrieve_call_type(call_type):
            return
        target: Final = _success_target(kwargs)
        # RustCore sets these as attributes in its constructor.
        hosted: Final[HostRegistry] = self.__dict__["_hosted"]  # pyright: ignore[reportAny]  # RustCore's own attribute
        if target is None or hosted._core_get_deployment(model_id=target[0]) is None:  # pyright: ignore[reportPrivateUsage]  # the host registry's id lookup
            return
        checks: Final[OptionalChecks] = self.__dict__["_checks"]  # pyright: ignore[reportAny]  # see above
        cooldowns: Final[CooldownCache] = self.__dict__["_cooldowns"]  # pyright: ignore[reportAny]  # see above
        await checks.repin_session_after_success(
            as_str_mapping(kwargs.get("litellm_params")) or {}, completion_response, cooldowns
        )

    async def deployment_callback_on_success(
        self,
        kwargs: Mapping[str, object],
        completion_response: object,
        start_time: object,
        end_time: object,
    ) -> str | None:
        """Count a success and the call's tokens and request against a tracked deployment; returns
        the counter's key, as the sync callback does."""
        return await self.report_stream_success(kwargs, completion_response)

    async def report_stream_success(
        self,
        kwargs: Mapping[str, object],
        completion_response: object,
        callbacks: Mapping[str, bool] | None = None,
    ) -> str | None:
        """``deployment_callback_on_success``, for a stream that ended. The re-pin is the host's half
        of that callback, so it reads ``callbacks`` here."""
        from litellm.litellm_core_utils.batch_utils import is_batch_retrieve_call_type
        from litellm.router_utils.router_callbacks.track_deployment_metrics import get_counted_usage_tokens

        call_type: Final = kwargs.get("call_type")
        if call_type == "_arealtime" or is_batch_retrieve_call_type(call_type):
            return
        logged: Final = as_str_mapping(kwargs.get("standard_logging_object"))
        if logged is None:
            return
        target: Final = _success_target(kwargs)
        if target is None:
            return
        if (callbacks or {}).get("success", True):
            await self.repin_after_success(kwargs, completion_response)
        tokens: Final[object] = logged.get("total_tokens", 0)
        total: Final = float(tokens) if isinstance(tokens, int | float) else 0.0
        counted: Final = get_counted_usage_tokens(as_str_mapping(kwargs.get("litellm_params")) or {})
        usage_model: Final = target[1].get("deployment")
        finished: Final = await self._drive_cache_steps(
            self._handle.report_success(
                target[0],
                total if counted is None else max(0.0, total - counted),
                1.0 if counted is None else 0.0,
                usage_model if isinstance(usage_model, str) else None,
                self.deployment_tracks_usage(target[0], target[1].get("model_group")),
                callbacks,
            )
        )
        key: Final = finished.get("key")
        return key if isinstance(key, str) else None

    async def count_realtime_session_success(self, deployment_id: str) -> None:
        """``increment_deployment_successes_for_current_minute`` for a realtime session that ended
        cleanly: a success alone, as ``deployment_callback_on_success`` skips ``_arealtime``. A
        counter that cannot be written does not fail the served session."""
        from litellm._logging import verbose_router_logger

        try:
            await self._drive_cache_steps(self._handle.count_success(deployment_id))
        except Exception as e:  # noqa: BLE001  # as the Rust call loop treats a success it could not record
            verbose_router_logger.debug("count_realtime_session_success: not recorded. Got - %s", e)


__all__ = [
    "RustHealthCallbacks",
    "failure_status",
    "logged_call",
]
