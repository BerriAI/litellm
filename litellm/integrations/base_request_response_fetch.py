"""
Base class for getting request / response payload from custom loggers
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional


class BaseRequestResponseFetchFromCustomLogger(ABC):
    def __init__(self):
        super().__init__()

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
