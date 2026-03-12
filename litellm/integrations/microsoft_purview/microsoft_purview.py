"""
Microsoft Purview Integration - sends LLM prompts & responses to the Microsoft Graph
processContent API for compliance, DLP, and audit tracking.

Reference API: https://learn.microsoft.com/en-us/graph/api/userdatasecurityandgovernance-processcontent
"""

import asyncio
import os
import traceback
from collections import defaultdict
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone


import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.utils import StandardLoggingPayload
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps


class MicrosoftPurviewLogger(CustomBatchLogger):
    """
    Logger that sends LLM interactions to Microsoft Purview via the
    Microsoft Graph processContent API for compliance and audit.
    """

    def __init__(
        self,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        app_name: Optional[str] = None,
        app_version: Optional[str] = None,
        app_id: Optional[str] = None,
        default_user_id: Optional[str] = None,
        graph_api_version: str = "v1.0",
        log_prompts: bool = True,
        log_responses: bool = True,
        **kwargs,
    ):
        """
        Initialize Microsoft Purview logger using the Graph API
        """
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )

        self.tenant_id = (
            tenant_id
            or os.getenv("MICROSOFT_PURVIEW_TENANT_ID")
            or os.getenv("AZURE_TENANT_ID")
        )
        self.client_id = (
            client_id
            or os.getenv("MICROSOFT_PURVIEW_CLIENT_ID")
            or os.getenv("AZURE_CLIENT_ID")
        )
        self.client_secret = (
            client_secret
            or os.getenv("MICROSOFT_PURVIEW_CLIENT_SECRET")
            or os.getenv("AZURE_CLIENT_SECRET")
        )

        self.app_name = app_name or os.getenv(
            "MICROSOFT_PURVIEW_APP_NAME", "LiteLLM Proxy"
        )
        self.app_version = app_version or os.getenv(
            "MICROSOFT_PURVIEW_APP_VERSION", getattr(litellm, "_version", "0.0.0")
        )
        self.app_id = app_id or os.getenv("MICROSOFT_PURVIEW_APP_ID")

        self.default_user_id = default_user_id or os.getenv(
            "MICROSOFT_PURVIEW_DEFAULT_USER_ID", "lite-llm-unknown-user"
        )
        self.graph_api_version = graph_api_version

        # Boolean flags controls what is captured in logs
        self.log_prompts = log_prompts
        self.log_responses = log_responses

        if not self.tenant_id:
            raise ValueError(
                "MICROSOFT_PURVIEW_TENANT_ID is required to use Microsoft Purview integration"
            )
        if not self.client_id:
            raise ValueError(
                "MICROSOFT_PURVIEW_CLIENT_ID is required to use Microsoft Purview integration"
            )
        if not self.client_secret:
            raise ValueError(
                "MICROSOFT_PURVIEW_CLIENT_SECRET is required to use Microsoft Purview integration"
            )

        # OAuth2 scope for Microsoft Graph
        self.oauth_scope = "https://graph.microsoft.com/.default"
        self.oauth_token: Optional[str] = None
        self.oauth_token_expires_at: Optional[float] = None

        self.flush_lock = asyncio.Lock()
        super().__init__(**kwargs, flush_lock=self.flush_lock)
        asyncio.create_task(self.periodic_flush())
        self.log_queue: List[StandardLoggingPayload] = []

    async def _get_oauth_token(self) -> str:
        """
        Get OAuth2 Bearer token for Microsoft Graph
        """
        import time

        if (
            self.oauth_token
            and self.oauth_token_expires_at
            and time.time() < self.oauth_token_expires_at - 60
        ):  # Refresh 60 seconds before expiry
            return self.oauth_token

        assert self.tenant_id is not None, "tenant_id is required"
        assert self.client_id is not None, "client_id is required"
        assert self.client_secret is not None, "client_secret is required"

        token_url = (
            f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        )

        token_data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": self.oauth_scope,
            "grant_type": "client_credentials",
        }

        response = await self.async_httpx_client.post(
            url=token_url,
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            raise Exception(
                f"Failed to get OAuth2 token: {response.status_code} - {response.text}"
            )

        token_response = response.json()
        self.oauth_token = token_response.get("access_token")
        expires_in = token_response.get("expires_in", 3600)

        if not self.oauth_token:
            raise Exception("OAuth2 token response did not contain access_token")

        self.oauth_token_expires_at = time.time() + expires_in
        return self.oauth_token

    def _extract_user_id(self, payload: StandardLoggingPayload) -> str:
        """Get the user identity to map to the user-scoped Purview API"""
        metadata = payload.get("metadata", {}) or {}

        user_id = metadata.get("user_api_key_user_id")
        if user_id:
            return str(user_id)

        end_user = payload.get("end_user")
        if end_user:
            return str(end_user)

        # fallback if not available
        return self.default_user_id

    def _serialize_messages(self, messages: Any) -> str:
        """Serialize prompts to a string, limiting total size if necessary"""
        if isinstance(messages, str):
            text = messages
        elif isinstance(messages, list):
            try:
                # Try to extract just text for readability in Purview
                parts = []
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            parts.append(f"[{role}]: {content}")
                        else:
                            parts.append(f"[{role}]: {safe_dumps(content)}")
                    else:
                        parts.append(str(msg))
                text = "\n\n".join(parts)
            except Exception:
                text = safe_dumps(messages)
        else:
            text = safe_dumps(messages)

        return text

    def _extract_response_text(self, payload: StandardLoggingPayload) -> str:
        """Extract the model response"""
        response = payload.get("response", {})
        if not response:
            return ""

        if isinstance(response, str):
            return response

        try:
            choices = response.get("choices", [])
            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                content = message.get("content")
                if content:
                    return str(content)
        except Exception:
            pass

        return safe_dumps(response)

    def _format_time(self, timestamp: Any) -> str:
        """Format timestamp to ISO 8601 strictly"""
        try:
            if timestamp:
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            else:
                dt = datetime.now(timezone.utc)
            # Purview requires exactly yYYY-MM-DDThh:mm:ss format
            return dt.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    def _build_process_content_request(self, payload: StandardLoggingPayload) -> dict:
        """Assemble the Microsoft Graph API request body for processContent"""
        entries = []
        trace_id = payload.get("trace_id", "") or "purview-unknown-trace"

        # 1. Add User Prompts
        if self.log_prompts:
            messages = payload.get("messages", [])
            prompt_text = self._serialize_messages(messages)
            if prompt_text:
                entries.append(
                    {
                        "@odata.type": "microsoft.graph.processConversationMetadata",
                        "identifier": f"{trace_id}-prompt",
                        "content": {
                            "@odata.type": "microsoft.graph.textContent",
                            "data": prompt_text,
                        },
                        "name": "LLM Prompt",
                        "correlationId": trace_id,
                        "sequenceNumber": 0,
                        "isTruncated": False,
                        "createdDateTime": self._format_time(payload.get("startTime")),
                        "modifiedDateTime": self._format_time(payload.get("startTime")),
                    }
                )

        # 2. Add AI Response
        if self.log_responses:
            response_text = self._extract_response_text(payload)
            if response_text:
                entries.append(
                    {
                        "@odata.type": "microsoft.graph.processConversationMetadata",
                        "identifier": f"{trace_id}-response",
                        "content": {
                            "@odata.type": "microsoft.graph.textContent",
                            "data": response_text,
                        },
                        "name": "LLM Response",
                        "correlationId": trace_id,
                        "sequenceNumber": 1,
                        "isTruncated": False,
                        "createdDateTime": self._format_time(payload.get("endTime")),
                        "modifiedDateTime": self._format_time(payload.get("endTime")),
                    }
                )

        # If nothing to send based on configuration or empty payload
        if not entries:
            return {}

        req_body = {
            "contentToProcess": {
                "contentEntries": entries,
                "activityMetadata": {
                    "activity": "uploadText"  # Or "downloadText". "uploadText" represents generating intent + receiving content
                },
                "integratedAppMetadata": {
                    "name": self.app_name,
                    "version": self.app_version,
                },
            }
        }

        # Add protectedAppMetadata if an app_id was specified (needed for full mapping in Purview)
        if self.app_id:
            req_body["contentToProcess"]["protectedAppMetadata"] = {
                "name": self.app_name,
                "version": self.app_version,
                "applicationLocation": {
                    "@odata.type": "microsoft.graph.policyLocationApplication",
                    "value": self.app_id,
                },
            }

        return req_body

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Async log success events to Microsoft Purview API queue"""
        try:
            verbose_logger.debug(
                "Microsoft Purview: Queueing success log for model %s",
                kwargs.get("model"),
            )
            standard_logging_payload = kwargs.get("standard_logging_object", None)

            if standard_logging_payload is None:
                return

            self.log_queue.append(standard_logging_payload)

            if len(self.log_queue) >= self.batch_size:
                await self.async_send_batch()

        except Exception as e:
            verbose_logger.exception(
                f"Microsoft Purview Success Logging Error - {str(e)}\n{traceback.format_exc()}"
            )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Async log failure events to Microsoft Purview API queue"""
        try:
            verbose_logger.debug(
                "Microsoft Purview: Queueing failure log for model %s",
                kwargs.get("model"),
            )
            standard_logging_payload = kwargs.get("standard_logging_object", None)

            if standard_logging_payload is None:
                return

            self.log_queue.append(standard_logging_payload)

            if len(self.log_queue) >= self.batch_size:
                await self.async_send_batch()

        except Exception as e:
            verbose_logger.exception(
                f"Microsoft Purview Failure Logging Error - {str(e)}\n{traceback.format_exc()}"
            )

    async def async_send_batch(self):
        """
        Sends the batch of logs to Microsoft Graph Process Content API
        """
        try:
            if not self.log_queue:
                return

            verbose_logger.debug(
                "Microsoft Purview - about to flush %s events", len(self.log_queue)
            )

            # 1. Group payloads by user id (Since Graph API is per-user)
            groups: Dict[str, list] = defaultdict(list)
            for payload in self.log_queue:
                user_id = self._extract_user_id(payload)
                req_body = self._build_process_content_request(payload)
                if req_body:
                    groups[user_id].append(req_body)

            if not groups:
                self.log_queue.clear()
                return

            # 2. Get OAuth2 Token
            bearer_token = await self._get_oauth_token()
            headers = {
                "Authorization": f"Bearer {bearer_token}",
                "Content-Type": "application/json",
            }

            # 3. Fire requests concurrently
            # Although they belong to different users, we loop through. Graph API doesn't support batching processContent inside a single call currently.
            # We process them in parallel
            tasks = []
            for user_id, requests in groups.items():
                api_endpoint = f"https://graph.microsoft.com/{self.graph_api_version}/users/{user_id}/dataSecurityAndGovernance/processContent"

                for req_body in requests:
                    tasks.append(
                        self.async_httpx_client.post(
                            url=api_endpoint, json=req_body, headers=headers
                        )
                    )

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for index, response in enumerate(responses):
                if isinstance(response, Exception):
                    verbose_logger.error(
                        "Microsoft Purview Graph API encountered error: %s",
                        str(response),
                    )
                elif response.status_code not in [200, 202, 204]:
                    verbose_logger.error(
                        "Microsoft Purview Graph API error: status_code=%s, response=%s",
                        response.status_code,
                        response.text,
                    )

            verbose_logger.debug(
                "Microsoft Purview: Flushed %s processContent calls", len(tasks)
            )

        except Exception as e:
            verbose_logger.exception(
                f"Microsoft Purview Error sending batch API - {str(e)}\n{traceback.format_exc()}"
            )
        finally:
            self.log_queue.clear()
