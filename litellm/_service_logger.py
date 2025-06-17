import asyncio
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.proxy._types import UserAPIKeyAuth

from .integrations.custom_logger import CustomLogger
from .integrations.datadog.datadog import DataDogLogger
from .integrations.opentelemetry import OpenTelemetry
from .integrations.prometheus_services import PrometheusServicesLogger
from .types.services import ServiceLoggerPayload, ServiceTypes

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = Union[_Span, Any]
    OTELClass = OpenTelemetry
else:
    Span = Any
    OTELClass = Any


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
                from litellm.proxy.proxy_server import open_telemetry_logger

                await self.init_otel_logger_if_none()

                if (
                    parent_otel_span is not None
                    and open_telemetry_logger is not None
                    and isinstance(open_telemetry_logger, OpenTelemetry)
                ):
                    await self.otel_logger.async_service_success_hook(
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
                from litellm.proxy.proxy_server import open_telemetry_logger

                await self.init_otel_logger_if_none()

                if not isinstance(error, str):
                    error = str(error)

                if (
                    parent_otel_span is not None
                    and open_telemetry_logger is not None
                    and isinstance(open_telemetry_logger, OpenTelemetry)
                ):
                    await self.otel_logger.async_service_success_hook(
                        payload=payload,
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
            await self.async_service_success_hook(
                service=ServiceTypes.LITELLM,
                duration=_duration,
                call_type=kwargs["call_type"],
            )
        except Exception as e:
            raise e
