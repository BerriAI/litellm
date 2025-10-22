#### What this does ####
#    On success, logs events to Rubrik
import os
import json
import httpx
from typing import Optional
from litellm.integrations.custom_logger import CustomLogger
from litellm._logging import verbose_logger

from litellm.types.utils import StandardLoggingPayload

class RubrikLogger(CustomLogger):
    def __init__(self):
        # Instance variables
        verbose_logger.debug("initializing rubrik logger")
        self.key = os.getenv("RUBRIK_API_KEY")
        self.webhook_url = os.getenv("RUBRIK_WEBHOOK_URL")
        if self.webhook_url is None:
            raise ValueError("environment variable RUBRIK_WEBHOOK_URL not set")

        if self.webhook_url.endswith("/"):
            self.webhook_url = self.webhook_url[:-1]

        # Cache the httpx client
        self.client: Optional[httpx.AsyncClient] = None

    def log_success(self, model, messages, response_obj, start_time, end_time, print_verbose, kwargs=None):
        print_verbose("RubrikLogger: Logging Success")

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ):
        standard_logging_payload: StandardLoggingPayload = kwargs["standard_logging_object"]

        try:
            # Initialize client if not already done
            if self.client is None:
                self.client = httpx.AsyncClient()

            # Convert the payload to JSON
            payload_json = json.dumps(standard_logging_payload, default=str)

            verbose_logger.debug(f"RubrikLogger: Sending payload to {self.webhook_url}")
            verbose_logger.debug(f"RubrikLogger: Payload = {payload_json}")
            # Send POST request to the webhook
            webhook_endpoint = f"{self.webhook_url}/litellm"
            headers = {"Content-Type": "application/json"}
            if self.key:
                headers["Authorization"] = f"Bearer {self.key}"

            response = await self.client.post(
                webhook_endpoint,
                content=payload_json,
                headers=headers,
                timeout=10.0
            )
            response.raise_for_status()
            verbose_logger.debug(f"Successfully sent payload to {webhook_endpoint}")
        except Exception as e:
            verbose_logger.exception(f"Error sending payload to Rubrik webhook")