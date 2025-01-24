import json
import os
from typing import Any, Dict, Optional, Union

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.utils import StandardLoggingPayload


class PubSub:
    def __init__(
        self,
        project_id: Optional[str] = None,
        topic_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
    ):
        """
        Initialize Google Cloud Pub/Sub publisher

        Args:
            project_id (str): Google Cloud project ID
            topic_id (str): Pub/Sub topic ID
            credentials_path (str, optional): Path to Google Cloud credentials JSON file
        """
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )

        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        self.topic_id = topic_id or os.getenv("GOOGLE_CLOUD_PUBSUB_TOPIC_ID")
        self.path_service_account_json = credentials_path or os.getenv(
            "GCS_PATH_SERVICE_ACCOUNT"
        )

        if not self.project_id or not self.topic_id:
            raise ValueError("Both project_id and topic_id must be provided")

    async def construct_request_headers(self) -> Dict[str, str]:
        """Construct authorization headers using Vertex AI auth"""
        from litellm import vertex_chat_completion

        _auth_header, vertex_project = (
            await vertex_chat_completion._ensure_access_token_async(
                credentials=self.path_service_account_json,
                project_id=None,
                custom_llm_provider="vertex_ai",
            )
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

    async def publish_message(
        self, message: Union[StandardLoggingPayload, Dict[str, Any], str]
    ) -> Dict[str, Any]:
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

            if response.status_code != 200:
                verbose_logger.error("Pub/Sub publish error: %s", str(response.text))
                raise Exception(f"Failed to publish message: {response.text}")

            verbose_logger.debug("Pub/Sub response: %s", response.text)
            return response.json()

        except Exception as e:
            verbose_logger.error("Pub/Sub publish error: %s", str(e))
            raise Exception(f"Failed to publish message to Pub/Sub: {str(e)}")
