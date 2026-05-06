import asyncio
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

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


_DEPENDENCY_COMPONENT_BY_SERVICE: Dict[ServiceTypes, Literal["db", "redis"]] = {
    ServiceTypes.DB: "db",
    ServiceTypes.BATCH_WRITE_TO_DB: "db",
    ServiceTypes.REDIS: "redis",
}


def _record_dependency_status(
    service: ServiceTypes,
    is_error: bool,
    error_message: Optional[str] = None,
) -> None:
    """
    Mirror DB/Redis service-call outcomes into the proxy connection-status
    tracker so the liveness probe can report a real "down" state without
    running a separate poller. Lazy-imported to avoid pulling the proxy
    package into _service_logger's import graph at module load.
    """
    component = _DEPENDENCY_COMPONENT_BY_SERVICE.get(service)
    if component is None:
        return
    try:
        from litellm.proxy.health_check_utils.connection_status import (
            connection_status_tracker,
        )
    except Exception:
        return

    try:
        if is_error:
            connection_status_tracker.mark_down(component, error_message)
        else:
            connection_status_tracker.mark_up(component)
    except Exception:
        # Tracker writes must never break a real service call path.
        pass


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

        _record_dependency_status(service, is_error=False)

        payload = ServiceLoggerPayload(
            is_error=False,
            error=None,
            service=service,
            duration=duration,
            call_type=call_type,
            event_metadata=event_metadata,
        )

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
            elif callback == "otel" or isinstance(callback, OpenTelemetry):
                _otel_logger_to_use: Optional[OpenTelemetry] = None
                if isinstance(callback, OpenTelemetry):
                    _otel_logger_to_use = callback
                else:
                    from litellm.proxy.proxy_server import open_telemetry_logger

                    if open_telemetry_logger is not None and isinstance(
                        open_telemetry_logger, OpenTelemetry
                    ):
                        _otel_logger_to_use = open_telemetry_logger

                if _otel_logger_to_use is not None and parent_otel_span is not None:
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

        _record_dependency_status(service, is_error=True, error_message=error_message)

        payload = ServiceLoggerPayload(
            is_error=True,
            error=error_message,
            service=service,
            duration=duration,
            call_type=call_type,
            event_metadata=event_metadata,
        )

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
            elif callback == "otel" or isinstance(callback, OpenTelemetry):
                _otel_logger_to_use: Optional[OpenTelemetry] = None
                if isinstance(callback, OpenTelemetry):
                    _otel_logger_to_use = callback
                else:
                    from litellm.proxy.proxy_server import open_telemetry_logger

                    if open_telemetry_logger is not None and isinstance(
                        open_telemetry_logger, OpenTelemetry
                    ):
                        _otel_logger_to_use = open_telemetry_logger

                if not isinstance(error, str):
                    error = str(error)

                if _otel_logger_to_use is not None and parent_otel_span is not None:
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
            )
        except Exception as e:
            raise e
