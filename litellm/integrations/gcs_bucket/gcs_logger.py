"""
Production Logger with GCS Support for LiteLLM Proxy Server
Logs to separate GCS buckets for success/error events with custom folder structures
"""

from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.gcs_bucket.gcs_bucket_base import GCSBucketBase
from litellm._logging import verbose_logger
import litellm
import json
import time
import uuid
import os
from datetime import datetime
from typing import Optional
from urllib.parse import quote


class ProductionGCSLogger(CustomLogger):
    """Production logger with async GCS bucket support using custom folder structures"""

    def __init__(self):
        super().__init__()
        self.success_bucket_name = os.getenv("GCS_SUCCESS_BUCKET_NAME")
        self.error_bucket_name = os.getenv("GCS_ERROR_BUCKET_NAME")
        self.service_account_path = os.getenv("GCS_PATH_SERVICE_ACCOUNT")
        
        # Initialize GCS base for async operations
        self.gcs_base = GCSBucketBase(bucket_name=self.success_bucket_name)
        
        if not self.success_bucket_name or not self.error_bucket_name:
            verbose_logger.warning("⚠️  GCS bucket names not set. GCS logging disabled.")
        else:
            verbose_logger.info(f"✅ GCS initialized: {self.success_bucket_name}, {self.error_bucket_name}")

    async def _upload_to_gcs_async(self, data: dict, bucket_name: str, log_type: str):
        """Upload log data to GCS bucket using async I/O"""
        if not bucket_name:
            return

        try:
            timestamp = datetime.utcnow().strftime("%H-%M-%S")
            date = datetime.utcnow().strftime("%Y-%m-%d")
            correlation_id = data.get("correlation_id", str(uuid.uuid4()))

            if log_type == "success":
                # Success logs: date={date}/{timestamp}_{correlation_id}.json
                # Using hive-style partitioning for BigQuery cost optimization
                # User/dept/team info is in JSON for querying
                filename = f"{timestamp}_{correlation_id}.json"
                gcs_path = f"success/date={date}/{filename}"
            else:
                # Error logs: date={date}/{timestamp}_{correlation_id}.json
                # Using hive-style partitioning for BigQuery cost optimization
                filename = f"{timestamp}_{correlation_id}.json"
                gcs_path = f"failure/date={date}/{filename}"

            # Use async httpx to upload to GCS
            headers = await self.gcs_base.construct_request_headers(
                service_account_json=self.service_account_path,
                vertex_instance=None
            )
            
            # Upload using the GCS REST API
            # Note: No indent - BigQuery requires single-line JSON (NEWLINE_DELIMITED_JSON format)
            json_data = json.dumps(data, default=str)
            await self.gcs_base._log_json_data_on_gcs(
                headers=headers,
                bucket_name=bucket_name,
                object_name=gcs_path,
                logging_payload=json_data
            )

        except Exception as e:
            verbose_logger.exception(f"❌ GCS upload error: {e}")

    def log_pre_api_call(self, model, messages, kwargs):
        pass

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        pass

    def _get_session_id(self, kwargs, litellm_params, metadata) -> Optional[str]:
        """
        Extract session ID from request parameters.
        Priority: litellm_session_id > metadata.session_id
        """
        if litellm_params.get("litellm_session_id"):
            return str(litellm_params.get("litellm_session_id"))
        if metadata.get("session_id"):
            return str(metadata.get("session_id"))
        if kwargs.get("litellm_session_id"):
            return str(kwargs.get("litellm_session_id"))
        return None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Log successful requests for LLM training history"""
        try:
            correlation_id = getattr(response_obj, "id", None) or str(uuid.uuid4())
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get("metadata", {}) or litellm_params.get("litellm_metadata", {})
            # Extract date and session_id for queryability
            log_date = datetime.utcnow().strftime("%Y-%m-%d")
            session_id = self._get_session_id(kwargs, litellm_params, metadata)

            success_log = {
                "correlation_id": correlation_id,
                "timestamp": time.time(),
                "timestamp_iso": datetime.utcnow().isoformat(),
                "litellm_session_id": session_id,
                "type": "SUCCESS",
                "user": {
                    "email": metadata.get("user_api_key_user_email"),
                    "user_id": metadata.get("user_api_key_user_id"),
                    "team_alias": metadata.get("user_api_key_team_alias"),
                    "department": (metadata.get("user_api_key_metadata") or {}).get(
                        "department", "unknown"
                    ),
                },
                "model": {
                    "requested": kwargs.get("model"),
                    "used": getattr(response_obj, "model", None),
                    "deployment": metadata.get("deployment"),
                    "model_group": metadata.get("model_group"),
                    "mode": metadata.get("model_info", {}).get("mode"),
                },
                "conversation": {
                    "messages": kwargs.get("input", kwargs.get("messages", [])),
                    "temperature": kwargs.get("temperature"),
                    "max_tokens": kwargs.get("max_tokens"),
                    "top_p": kwargs.get("top_p"),
                    "frequency_penalty": kwargs.get("frequency_penalty"),
                    "presence_penalty": kwargs.get("presence_penalty"),
                    "tools": kwargs.get("tools"),
                    "tool_choice": kwargs.get("tool_choice"),
                },
                "response": {},
                "usage": {},
                "cost": 0,
                "timing": {
                    "start_time": str(start_time),
                    "end_time": str(end_time),
                    "duration_seconds": (
                        (end_time - start_time).total_seconds()
                        if start_time and end_time
                        else None
                    ),
                    "llm_api_duration_ms": metadata.get("llm_api_duration_ms"),
                },
                "headers": metadata.get("headers"),
            }

            if hasattr(response_obj, "choices") and response_obj.choices:
                choice = response_obj.choices[0]
                success_log["response"] = {
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "content": None,
                    "tool_calls": None,
                    "function_call": None,
                    "reasoning_content": None,
                }

                if hasattr(choice, "message"):
                    message = choice.message
                    success_log["response"]["content"] = getattr(
                        message, "content", None
                    )
                    success_log["response"]["reasoning_content"] = getattr(
                        message, "reasoning_content", None
                    )
                    success_log["response"]["tool_calls"] = getattr(
                        message, "tool_calls", None
                    )
                    success_log["response"]["function_call"] = getattr(
                        message, "function_call", None
                    )

            if hasattr(response_obj, "usage"):
                usage = response_obj.usage
                success_log["usage"] = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }

            try:
                success_log["cost"] = litellm.completion_cost(
                    completion_response=response_obj
                )
            except Exception:
                success_log["cost"] = 0

            if self.success_bucket_name:
                await self._upload_to_gcs_async(success_log, self.success_bucket_name, "success")

        except Exception as e:
            verbose_logger.exception(f"Error logging success: {e}")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Log failed requests for debugging"""
        try:
            correlation_id = getattr(response_obj, "id", None) or str(uuid.uuid4())
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get("metadata", {}) or litellm_params.get("litellm_metadata", {})
            # Extract date and session_id for queryability
            log_date = datetime.utcnow().strftime("%Y-%m-%d")
            session_id = self._get_session_id(kwargs, litellm_params, metadata)

            error_log = {
                "correlation_id": correlation_id,
                "timestamp": time.time(),
                "timestamp_iso": datetime.utcnow().isoformat(),
                "litellm_session_id": session_id,
                "type": "ERROR",
                "user": {
                    "email": metadata.get("user_api_key_user_email"),
                    "user_id": metadata.get("user_api_key_user_id"),
                    "team_alias": metadata.get("user_api_key_team_alias"),
                    "department": (metadata.get("user_api_key_metadata") or {}).get(
                        "department"
                    ),
                },
                "model": {
                    "requested": kwargs.get("model"),
                    "deployment": metadata.get("deployment"),
                    "model_group": metadata.get("model_group"),
                    "api_base": litellm_params.get("api_base"),
                    "provider": litellm_params.get("custom_llm_provider"),
                },
                "request": {
                    "messages_count": len(kwargs.get("messages", [])),
                    "first_message": json.dumps(
                        kwargs.get("messages", []), default=str
                    ),
                    "max_tokens": kwargs.get("max_tokens"),
                    "route": metadata.get("user_api_key_request_route"),
                },
                "error": {
                    "type": type(response_obj).__name__,
                    "message": str(response_obj),
                    "exception": str(kwargs.get("exception", "")),
                    "traceback": str(kwargs.get("traceback_exception", "")),
                },
                "timing": {
                    "start_time": str(start_time),
                    "end_time": str(end_time),
                    "duration_seconds": (
                        (end_time - start_time).total_seconds()
                        if start_time and end_time
                        else None
                    ),
                    "llm_api_duration_ms": metadata.get("llm_api_duration_ms"),
                },
            }

            if self.error_bucket_name:
                await self._upload_to_gcs_async(error_log, self.error_bucket_name, "error")

        except Exception as e:
            verbose_logger.exception(f"Error logging failure: {e}")


# Handler instance
logger_instance = ProductionGCSLogger()


if __name__ == "__main__":
    print("=" * 80)
    print("Production Logger with GCS Support")
    print("=" * 80)
    print("\n📝 Logs to:")
    print("   • GCS_SUCCESS_BUCKET_NAME (cloud)")
    print("   • GCS_ERROR_BUCKET_NAME (cloud)")
    print("\n🔧 Environment Variables:")
    print("   GCS_SUCCESS_BUCKET_NAME - Success logs bucket")
    print("   GCS_ERROR_BUCKET_NAME - Error logs bucket")
    print("   GCS_PATH_SERVICE_ACCOUNT - Service account JSON (optional)")
    print("\n📝 Config usage:")
    print("litellm_settings:")
    print("  callbacks: gcs_logger.logger_instance")
    print("=" * 80)
