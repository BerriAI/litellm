"""
BETA

This is the PubSub logger for GCS PubSub, this sends LiteLLM SpendLogs Payloads to GCS PubSub.

Users can use this instead of sending their SpendLogs to their Postgres database.
"""

import asyncio
import json
import os
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from litellm.types.utils import StandardLoggingPayload

if TYPE_CHECKING:
    from litellm.proxy._types import SpendLogsPayload
else:
    SpendLogsPayload = Any

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)


class GcsPubSubLogger(CustomBatchLogger):
    def __init__(
        self,
        project_id: Optional[str] = None,
        topic_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Google Cloud Pub/Sub publisher

        Args:
            project_id (str): Google Cloud project ID
            topic_id (str): Pub/Sub topic ID
            credentials_path (str, optional): Path to Google Cloud credentials JSON file
        """
        from litellm.proxy.utils import _premium_user_check

        _premium_user_check()

        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )

        self.project_id = project_id or os.getenv("GCS_PUBSUB_PROJECT_ID")
        self.topic_id = topic_id or os.getenv("GCS_PUBSUB_TOPIC_ID")
        self.path_service_account_json = credentials_path or os.getenv(
            "GCS_PATH_SERVICE_ACCOUNT"
        )

        if not self.project_id or not self.topic_id:
            raise ValueError("Both project_id and topic_id must be provided")

        self.flush_lock = asyncio.Lock()
        super().__init__(**kwargs, flush_lock=self.flush_lock)
        asyncio.create_task(self.periodic_flush())
        self.log_queue: List[Union[SpendLogsPayload, StandardLoggingPayload]] = []

    async def construct_request_headers(self) -> Dict[str, str]:
        """Construct authorization headers using Vertex AI auth"""
        from litellm import vertex_chat_completion

        (
            _auth_header,
            vertex_project,
        ) = await vertex_chat_completion._ensure_access_token_async(
            credentials=self.path_service_account_json,
            project_id=self.project_id,
            custom_llm_provider="vertex_ai",
        )

        auth_header, _ = vertex_chat_completion._get_token_and_url(
            model="pub-sub",
            auth_header=_auth_header,
            vertex_credentials=self.path_service_account_json,
            vertex_project=vertex_project,
            vertex_location=None,
            gemini_api_key=None,
            stream=None,
            custom_llm_provider="vertex_ai",
            api_base=None,
        )

        headers = {
            "Authorization": f"Bearer {auth_header}",
            "Content-Type": "application/json",
        }
        return headers

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Async Log success events to GCS PubSub Topic

        - Creates a SpendLogsPayload
        - Adds to batch queue
        - Flushes based on CustomBatchLogger settings

        Raises:
            Raises a NON Blocking verbose_logger.exception if an error occurs
        """
        from litellm.proxy.spend_tracking.spend_tracking_utils import (
            get_logging_payload,
        )
        from litellm.proxy.utils import _premium_user_check

        _premium_user_check()

        try:
            verbose_logger.debug(
                "PubSub: Logging - Enters logging function for model %s", kwargs
            )
            standard_logging_payload = kwargs.get("standard_logging_object", None)

            # Backwards compatibility with old logging payload
            if litellm.gcs_pub_sub_use_v1 is True:
                spend_logs_payload = get_logging_payload(
                    kwargs=kwargs,
                    response_obj=response_obj,
                    start_time=start_time,
                    end_time=end_time,
                )
                self.log_queue.append(spend_logs_payload)
            else:
                # New logging payload, StandardLoggingPayload
                self.log_queue.append(standard_logging_payload)

            if len(self.log_queue) >= self.batch_size:
                await self.async_send_batch()

        except Exception as e:
            verbose_logger.exception(
                f"PubSub Layer Error - {str(e)}\n{traceback.format_exc()}"
            )
            pass

    async def async_send_batch(self):
        """
        Sends the batch of messages to Pub/Sub
        """
        try:
            if not self.log_queue:
                return

            verbose_logger.debug(
                f"PubSub - about to flush {len(self.log_queue)} events"
            )

            for message in self.log_queue:
                await self.publish_message(message)

        except Exception as e:
            verbose_logger.exception(
                f"PubSub Error sending batch - {str(e)}\n{traceback.format_exc()}"
            )
        finally:
            self.log_queue.clear()

    async def publish_message(
        self, message: Union[SpendLogsPayload, StandardLoggingPayload]
    ) -> Optional[Dict[str, Any]]:
        """
        Publish message to Google Cloud Pub/Sub using REST API

        Args:
            message: Message to publish (dict or string)

        Returns:
            dict: Published message response
        """
        try:
            headers = await self.construct_request_headers()

            # Prepare message data
            if isinstance(message, str):
                message_data = message
            else:
                message_data = json.dumps(message, default=str)

            # Base64 encode the message
            import base64

            encoded_message = base64.b64encode(message_data.encode("utf-8")).decode(
                "utf-8"
            )

            # Construct request body
            request_body = {"messages": [{"data": encoded_message}]}

            url = f"https://pubsub.googleapis.com/v1/projects/{self.project_id}/topics/{self.topic_id}:publish"

            response = await self.async_httpx_client.post(
                url=url, headers=headers, json=request_body
            )

            if response.status_code not in [200, 202]:
                verbose_logger.error("Pub/Sub publish error: %s", str(response.text))
                raise Exception(f"Failed to publish message: {response.text}")

            verbose_logger.debug("Pub/Sub response: %s", response.text)
            return response.json()

        except Exception as e:
            verbose_logger.error("Pub/Sub publish error: %s", str(e))
            return None
