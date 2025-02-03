"""
Base class for health check integrations
"""

from abc import ABC, abstractmethod

from litellm.types.integrations.base_health_check import IntegrationHealthCheckStatus


class HealthCheckIntegration(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    async def async_health_check(self) -> IntegrationHealthCheckStatus:
        """
        Check if the service is healthy
        """
        pass
