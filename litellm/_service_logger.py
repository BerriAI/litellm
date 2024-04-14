import litellm
from .types.services import ServiceTypes, ServiceLoggerPayload
from .integrations.prometheus_services import PrometheusServicesLogger


class ServiceLogging:
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

    def service_success_hook(self, service: ServiceTypes, duration: float):
        """
        [TODO] Not implemented for sync calls yet. V0 is focused on async monitoring (used by proxy).
        """
        if self.mock_testing:
            self.mock_testing_sync_success_hook += 1

    def service_failure_hook(
        self, service: ServiceTypes, duration: float, error: Exception
    ):
        """
        [TODO] Not implemented for sync calls yet. V0 is focused on async monitoring (used by proxy).
        """
        if self.mock_testing:
            self.mock_testing_sync_failure_hook += 1

    async def async_service_success_hook(self, service: ServiceTypes, duration: float):
        """
        - For counting if the redis, postgres call is successful
        """
        if self.mock_testing:
            self.mock_testing_async_success_hook += 1

        payload = ServiceLoggerPayload(
            is_error=False, error=None, service=service, duration=duration
        )
        for callback in litellm.service_callback:
            if callback == "prometheus_system":
                await self.prometheusServicesLogger.async_service_success_hook(
                    payload=payload
                )

    async def async_service_failure_hook(
        self, service: ServiceTypes, duration: float, error: Exception
    ):
        """
        - For counting if the redis, postgres call is unsuccessful
        """
        if self.mock_testing:
            self.mock_testing_async_failure_hook += 1

        payload = ServiceLoggerPayload(
            is_error=True, error=str(error), service=service, duration=duration
        )
        for callback in litellm.service_callback:
            if callback == "prometheus_system":
                if self.prometheusServicesLogger is None:
                    self.prometheusServicesLogger = self.prometheusServicesLogger()
                await self.prometheusServicesLogger.async_service_failure_hook(
                    payload=payload
                )
