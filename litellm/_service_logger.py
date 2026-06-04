import asyncio
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional, Union

import litellm
from litellm._logging import verbose_logger

from .integrations.custom_logger import CustomLogger
from .integrations.datadog.datadog import DataDogLogger
from .integrations.opentelemetry import OpenTelemetry
from .integrations.prometheus_services import PrometheusServicesLogger
from .types.services import ServiceLoggerPayload, ServiceTypes

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.proxy._types import UserAPIKeyAuth

    Span = Union[_Span, Any]
    OTELClass = OpenTelemetry
else:
    Span = Any
    OTELClass = Any
    UserAPIKeyAuth = Any


def _get_otel_v2_class() -> Optional[type]:
    """Return the ``OpenTelemetryV2`` class, or ``None`` if the OTel SDK is absent.

    Imported lazily: ``litellm.integrations.otel.logger`` imports the OpenTelemetry
    SDK at module scope, so importing it eagerly would break installs without the
    SDK. The V2 logger only exists when ``LITELLM_OTEL_V2`` is enabled (which
    requires the SDK), so a failed import simply means "no V2 logger in play".
    """
    try:
        from litellm.integrations.otel.logger import OpenTelemetryV2

        return OpenTelemetryV2
    except Exception:
        return None


class ServiceLogging(CustomLogger):
    """
    Separate class used for monitoring health of litellm-adjacent services (redis/postgres).
    """

    def __init__(self, mock_testing: bool = False) -> None:
        self.mock_testing = mock_testing
        self.mock_testing_sync_success_hook = 0
        self.mock_testing_async_success_hook = 0
        self.mock_testing_sync_failure_hook = 0
        self.mock_testing_async_failure_hook = 0
        if "prometheus_system" in litellm.service_callback:
            self.prometheusServicesLogger = PrometheusServicesLogger()

    def _resolve_otel_service_logger(self, callback: Any) -> Optional[Any]:
        """Resolve the OTel logger (legacy or V2) to emit a service span on.

        Returns the logger instance whose ``async_service_*_hook`` should fire for
        this ``callback``, or ``None`` when ``callback`` is not an OTel callback.

        The V2 ``OpenTelemetryV2`` logger is a plain ``CustomLogger`` and is NOT a
        subclass of the legacy ``OpenTelemetry``, so the legacy ``isinstance``
        check alone misses it — which is why redis/postgres service spans never
        showed up under ``LITELLM_OTEL_V2``. Match both the legacy and V2 types,
        whether the callback is the logger instance itself or the ``"otel"`` string
        (which routes to the proxy's registered ``open_telemetry_logger``).
        """
        otel_v2_cls = _get_otel_v2_class()

        def _is_otel_logger(obj: Any) -> bool:
            if isinstance(obj, OpenTelemetry):
                return True
            return otel_v2_cls is not None and isinstance(obj, otel_v2_cls)

        if _is_otel_logger(callback):
            return callback
        if callback == "otel":
            from litellm.proxy.proxy_server import open_telemetry_logger

            if open_telemetry_logger is not None and _is_otel_logger(
                open_telemetry_logger
            ):
                return open_telemetry_logger
        return None

    def service_success_hook(
        self,
        service: ServiceTypes,
        duration: float,
        call_type: str,
        parent_otel_span: Optional[Span] = None,
        start_time: Optional[Union[datetime, float]] = None,
        end_time: Optional[Union[float, datetime]] = None,
    ):
        """
        Handles both sync and async monitoring by checking for existing event loop.
        """

        if self.mock_testing:
            self.mock_testing_sync_success_hook += 1

        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            # Check if the loop is running
            if loop.is_running():
                # If we're in a running loop, create a task
                loop.create_task(
                    self.async_service_success_hook(
                        service=service,
                        duration=duration,
                        call_type=call_type,
                        parent_otel_span=parent_otel_span,
                        start_time=start_time,
                        end_time=end_time,
                    )
                )
            else:
                # Loop exists but not running, we can use run_until_complete
                loop.run_until_complete(
                    self.async_service_success_hook(
                        service=service,
                        duration=duration,
                        call_type=call_type,
                        parent_otel_span=parent_otel_span,
                        start_time=start_time,
                        end_time=end_time,
                    )
                )
        except RuntimeError:
            # No event loop exists, create a new one and run
            asyncio.run(
                self.async_service_success_hook(
                    service=service,
                    duration=duration,
                    call_type=call_type,
                    parent_otel_span=parent_otel_span,
                    start_time=start_time,
                    end_time=end_time,
                )
            )

    def service_failure_hook(
        self, service: ServiceTypes, duration: float, error: Exception, call_type: str
    ):
        """
        [TODO] Not implemented for sync calls yet. V0 is focused on async monitoring (used by proxy).
        """
        if self.mock_testing:
            self.mock_testing_sync_failure_hook += 1

    async def async_service_success_hook(
        self,
        service: ServiceTypes,
        call_type: str,
        duration: float,
        parent_otel_span: Optional[Span] = None,
        start_time: Optional[Union[datetime, float]] = None,
        end_time: Optional[Union[datetime, float]] = None,
        event_metadata: Optional[dict] = None,
    ):
        """
        - For counting if the redis, postgres call is successful
        """
        if self.mock_testing:
            self.mock_testing_async_success_hook += 1

        payload = ServiceLoggerPayload(
            is_error=False,
            error=None,
            service=service,
            duration=duration,
            call_type=call_type,
            event_metadata=event_metadata,
        )

        # OTel loggers already fired this event. ``service_callback`` can hold more
        # than one reference that resolves to the *same* logger — the ``"otel"``
        # string AND the registered instance both map to ``open_telemetry_logger``
        # (the V2 logger self-registers its instance even when the string is
        # present, unlike V1). Without this guard each such reference emits its own
        # span, so a single DB call shows up as duplicate ``postgres ...`` spans.
        emitted_otel_logger_ids: set = set()
        for callback in litellm.service_callback:
            if callback == "prometheus_system":
                await self.init_prometheus_services_logger_if_none()
                await self.prometheusServicesLogger.async_service_success_hook(
                    payload=payload
                )
            elif callback == "datadog" or isinstance(callback, DataDogLogger):
                await self.init_datadog_logger_if_none()
                await self.dd_logger.async_service_success_hook(
                    payload=payload,
                    parent_otel_span=parent_otel_span,
                    start_time=start_time,
                    end_time=end_time,
                    event_metadata=event_metadata,
                )
            else:
                _otel_logger_to_use = self._resolve_otel_service_logger(callback)
                # No ``parent_otel_span is not None`` gate: a background service
                # call (no request on the stack) has no parent, and dropping it
                # here is what hid those calls from traces entirely. The OTel
                # logger decides what to do with a missing parent — legacy V1
                # no-ops, V2 emits a root span (and skips metrics-only pings).
                if (
                    _otel_logger_to_use is not None
                    and id(_otel_logger_to_use) not in emitted_otel_logger_ids
                ):
                    emitted_otel_logger_ids.add(id(_otel_logger_to_use))
                    await _otel_logger_to_use.async_service_success_hook(
                        payload=payload,
                        parent_otel_span=parent_otel_span,
                        start_time=start_time,
                        end_time=end_time,
                        event_metadata=event_metadata,
                    )

    async def init_prometheus_services_logger_if_none(self):
        """
        initializes prometheusServicesLogger if it is None or no attribute exists on ServiceLogging Object

        """
        if not hasattr(self, "prometheusServicesLogger"):
            self.prometheusServicesLogger = PrometheusServicesLogger()
        elif self.prometheusServicesLogger is None:
            self.prometheusServicesLogger = self.prometheusServicesLogger()
        return

    async def init_datadog_logger_if_none(self):
        """
        initializes dd_logger if it is None or no attribute exists on ServiceLogging Object

        """
        from litellm.integrations.datadog.datadog import DataDogLogger

        if not hasattr(self, "dd_logger"):
            self.dd_logger: DataDogLogger = DataDogLogger()

        return

    async def init_otel_logger_if_none(self):
        """
        initializes otel_logger if it is None or no attribute exists on ServiceLogging Object

        """
        from litellm.proxy.proxy_server import open_telemetry_logger

        if not hasattr(self, "otel_logger"):
            if open_telemetry_logger is not None and isinstance(
                open_telemetry_logger, OpenTelemetry
            ):
                self.otel_logger: OpenTelemetry = open_telemetry_logger
            else:
                verbose_logger.warning(
                    "ServiceLogger: open_telemetry_logger is None or not an instance of OpenTelemetry"
                )
        return

    async def async_service_failure_hook(
        self,
        service: ServiceTypes,
        duration: float,
        error: Union[str, Exception],
        call_type: str,
        parent_otel_span: Optional[Span] = None,
        start_time: Optional[Union[datetime, float]] = None,
        end_time: Optional[Union[float, datetime]] = None,
        event_metadata: Optional[dict] = None,
    ):
        """
        - For counting if the redis, postgres call is unsuccessful
        """
        if self.mock_testing:
            self.mock_testing_async_failure_hook += 1

        error_message = ""
        if isinstance(error, Exception):
            error_message = str(error)
        elif isinstance(error, str):
            error_message = error

        payload = ServiceLoggerPayload(
            is_error=True,
            error=error_message,
            service=service,
            duration=duration,
            call_type=call_type,
            event_metadata=event_metadata,
        )

        # Dedupe OTel loggers per event — see ``async_service_success_hook`` for why
        # the same logger can be referenced twice in ``service_callback``.
        emitted_otel_logger_ids: set = set()
        for callback in litellm.service_callback:
            if callback == "prometheus_system":
                await self.init_prometheus_services_logger_if_none()
                await self.prometheusServicesLogger.async_service_failure_hook(
                    payload=payload,
                    error=error,
                )
            elif callback == "datadog" or isinstance(callback, DataDogLogger):
                await self.init_datadog_logger_if_none()
                await self.dd_logger.async_service_failure_hook(
                    payload=payload,
                    error=error_message,
                    parent_otel_span=parent_otel_span,
                    start_time=start_time,
                    end_time=end_time,
                    event_metadata=event_metadata,
                )
            else:
                _otel_logger_to_use = self._resolve_otel_service_logger(callback)

                if not isinstance(error, str):
                    error = str(error)

                # See the success hook: no parent gate, so background failures
                # are traced too. V1 no-ops without a parent; V2 emits a root.
                if (
                    _otel_logger_to_use is not None
                    and id(_otel_logger_to_use) not in emitted_otel_logger_ids
                ):
                    emitted_otel_logger_ids.add(id(_otel_logger_to_use))
                    await _otel_logger_to_use.async_service_failure_hook(
                        payload=payload,
                        error=error,
                        parent_otel_span=parent_otel_span,
                        start_time=start_time,
                        end_time=end_time,
                        event_metadata=event_metadata,
                    )

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: Optional[str] = None,
    ):
        """
        Hook to track failed litellm-service calls
        """
        return await super().async_post_call_failure_hook(
            request_data,
            original_exception,
            user_api_key_dict,
        )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Hook to track latency for litellm proxy llm api calls
        """
        try:
            _duration = end_time - start_time
            if isinstance(_duration, timedelta):
                _duration = _duration.total_seconds()
            elif isinstance(_duration, float):
                pass
            else:
                raise Exception(
                    "Duration={} is not a float or timedelta object. type={}".format(
                        _duration, type(_duration)
                    )
                )  # invalid _duration value
            # Batch polling callbacks (check_batch_cost) don't include call_type in kwargs.
            # Use .get() to avoid KeyError.
            await self.async_service_success_hook(
                service=ServiceTypes.LITELLM,
                duration=_duration,
                call_type=kwargs.get("call_type", "unknown"),
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as e:
            raise e
