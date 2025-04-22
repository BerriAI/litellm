"""
Base class for Additional Logging Utils for CustomLoggers 

- Health Check for the logging util
- Get Request / Response Payload for the logging util
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from litellm.types.integrations.base_health_check import IntegrationHealthCheckStatus


class AdditionalLoggingUtils(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    async def async_health_check(self) -> IntegrationHealthCheckStatus:
        """
        Check if the service is healthy
        """
        pass

    @abstractmethod
    async def get_request_response_payload(
        self,
        request_id: str,
        start_time_utc: Optional[datetime],
        end_time_utc: Optional[datetime],
    ) -> Optional[dict]:
        """
        Get the request and response payload for a given `request_id`
        """
        return None
