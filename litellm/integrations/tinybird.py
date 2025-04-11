import os
from typing import Optional
import subprocess
import sys

from litellm.integrations.custom_logger import CustomLogger


class TinybirdLogger(CustomLogger):
    def __init__(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None, datasource_name: Optional[str] = None
    ) -> None:
        super().__init__()

        try:
            from tb.litellm.handler import TinybirdLitellmAsyncHandler, TinybirdLitellmSyncHandler
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tinybird-python-sdk"])
            from tb.litellm.handler import TinybirdLitellmAsyncHandler, TinybirdLitellmSyncHandler

        self.validate_environment(api_key=api_key, api_base=api_base)
        self.api_base = api_base or os.getenv("TINYBIRD_API_HOST")
        self.api_key: str = api_key or os.getenv("TINYBIRD_TOKEN") or ""
        self.datasource_name = datasource_name or os.getenv("TINYBIRD_DATASOURCE_NAME") or "litellm"
        self.async_handler = TinybirdLitellmAsyncHandler(
            api_url=self.api_base, 
            tinybird_token=self.api_key, 
            datasource_name=self.datasource_name
        )
        self.sync_handler = TinybirdLitellmSyncHandler(
            api_url=self.api_base, 
            tinybird_token=self.api_key, 
            datasource_name=self.datasource_name
        )

    def validate_environment(self, api_key: Optional[str], api_base: Optional[str]):
        """
        Expects
        TINYBIRD_TOKEN
        TINYBIRD_API_HOST
        in the environment
        """
        missing_keys = []
        if api_key is None and os.getenv("TINYBIRD_TOKEN", None) is None:
            missing_keys.append("TINYBIRD_TOKEN")
        if api_base is None and os.getenv("TINYBIRD_API_HOST", None) is None:
            missing_keys.append("TINYBIRD_API_HOST")

        if len(missing_keys) > 0:
            raise Exception("Missing keys={} in environment.".format(missing_keys))

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        return self.sync_handler.log_success_event(kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        return await self.async_handler.async_log_success_event(kwargs, response_obj, start_time, end_time)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        return self.sync_handler.log_failure_event(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        return await self.async_handler.async_log_failure_event(kwargs, response_obj, start_time, end_time)
