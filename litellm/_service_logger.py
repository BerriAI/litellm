import litellm, traceback
from litellm.proxy._types import UserAPIKeyAuth
from .types.services import ServiceTypes, ServiceLoggerPayload
from .integrations.prometheus_services import PrometheusServicesLogger
from .integrations.custom_logger import CustomLogger
from datetime import timedelta
from typing import Union


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
        self, service: ServiceTypes, duration: float, call_type: str
    ):
        """
        [TODO] Not implemented for sync calls yet. V0 is focused on async monitoring (used by proxy).
        """
        if self.mock_testing:
            self.mock_testing_sync_success_hook += 1

    def service_failure_hook(
        self, service: ServiceTypes, duration: float, error: Exception, call_type: str
    ):
        """
        [TODO] Not implemented for sync calls yet. V0 is focused on async monitoring (used by proxy).
        """
        if self.mock_testing:
            self.mock_testing_sync_failure_hook += 1

    async def async_service_success_hook(
        self, service: ServiceTypes, duration: float, call_type: str
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
        )
        for callback in litellm.service_callback:
            if callback == "prometheus_system":
                await self.prometheusServicesLogger.async_service_success_hook(
                    payload=payload
                )

    async def async_service_failure_hook(
        self,
        service: ServiceTypes,
        duration: float,
        error: Union[str, Exception],
        call_type: str,
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
        )
        for callback in litellm.service_callback:
            if callback == "prometheus_system":
                if self.prometheusServicesLogger is None:
                    self.prometheusServicesLogger = self.prometheusServicesLogger()
                await self.prometheusServicesLogger.async_service_failure_hook(
                    payload=payload
                )

    async def async_post_call_failure_hook(
        self, original_exception: Exception, user_api_key_dict: UserAPIKeyAuth
    ):
        """
        Hook to track failed litellm-service calls
        """
        return await super().async_post_call_failure_hook(
            original_exception, user_api_key_dict
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
