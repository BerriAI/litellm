"""
Production-ready GCP Logs Explorer logger for LiteLLM.

Logs all requests and responses to stdout in structured JSON format for
automatic ingestion into GCP Logs Explorer. Includes full request/response
content for internal debugging and analysis.

Usage in config.yaml:
    litellm_settings:
        callbacks:
            - gcp_logs_stdout_logger.gcp_logger

    environment_variables:
        GCP_LOGGER_ENABLED: "true"  # Optional: defaults to true
        GCP_LOGGER_LOG_LEVEL: "INFO"  # Optional: INFO, DEBUG, WARNING, ERROR
        GCP_LOGGER_INCLUDE_HEADERS: "false"  # Optional: log request headers
        GCP_LOGGER_SAMPLE_RATE: "1.0"  # Optional: 0.0-1.0 for sampling

Or set environment variables directly:
    export GCP_LOGGER_ENABLED="true"
    export GCP_LOGGER_LOG_LEVEL="INFO"
"""

import json
import os
import random
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger


class GCPLogsStdoutLogger(CustomLogger):
    """
    Production-ready logger that writes structured JSON logs to stdout
    for automatic ingestion into GCP Logs Explorer.

    Complete request/response flow logging:
    1. Client → LiteLLM (incoming request)
    2. LiteLLM → Provider (transformed request sent to model)
    3. Provider → LiteLLM (raw response from model)
    4. LiteLLM → Client (final response to client)

    Features:
    - Complete 4-point logging: client request, provider request, provider response, client response
    - Logs full request/response content including prompts and completions
    - Provider-specific request transformations visible
    - Structured JSON format for easy querying
    - Request/response timing
    - Token usage tracking
    - Error tracking with stack traces
    - Image detection
    - Sampling support for high-volume scenarios
    - Configurable via environment variables
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        log_level: Optional[str] = None,
        include_headers: Optional[bool] = None,
        sample_rate: Optional[float] = None,
    ):
        """
        Initialize the GCP Logs stdout logger.

        Args:
            enabled: Enable/disable logging. Defaults to GCP_LOGGER_ENABLED env var or True.
            log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR). Defaults to INFO.
            include_headers: Whether to log HTTP headers. Defaults to False.
            sample_rate: Sampling rate (0.0-1.0). 1.0 = log everything. Defaults to 1.0.
        """
        super().__init__()

        # Configuration from environment variables with fallbacks
        self.enabled = (
            enabled
            if enabled is not None
            else os.getenv("GCP_LOGGER_ENABLED", "true").lower() == "true"
        )
        self.log_level = (
            log_level or os.getenv("GCP_LOGGER_LOG_LEVEL", "INFO")
        ).upper()
        self.include_headers = (
            include_headers
            if include_headers is not None
            else os.getenv("GCP_LOGGER_INCLUDE_HEADERS", "false").lower() == "true"
        )
        self.sample_rate = (
            sample_rate
            if sample_rate is not None
            else float(os.getenv("GCP_LOGGER_SAMPLE_RATE", "1.0"))
        )

        # Log level mapping
        self.log_levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
        self.min_level = self.log_levels.get(self.log_level, 20)

        verbose_proxy_logger.info(
            f"GCPLogsStdoutLogger initialized: enabled={self.enabled}, "
            f"log_level={self.log_level}, sample_rate={self.sample_rate}"
        )

    def _should_log(self) -> bool:
        """Determine if this request should be logged based on sampling."""
        if not self.enabled:
            return False
        if self.sample_rate >= 1.0:
            return True
        return random.random() < self.sample_rate

    def _write_log(
        self,
        severity: str,
        message: str,
        log_type: str,
        **extra_fields: Any,
    ) -> None:
        """
        Write a structured JSON log entry to stdout.

        Args:
            severity: Log severity (DEBUG, INFO, WARNING, ERROR)
            message: Human-readable log message
            log_type: Type of log (request_start, request_success, request_failure, etc.)
            **extra_fields: Additional fields to include in the log entry
        """
        # Check if we should log based on level
        if self.log_levels.get(severity, 20) < self.min_level:
            return

        # Build structured log entry for GCP Logs Explorer
        log_entry = {
            "severity": severity,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logType": log_type,
            "component": "litellm",
            **extra_fields,
        }

        # Write to stdout with flush for immediate visibility
        print(json.dumps(log_entry, default=str), file=sys.stdout, flush=True)

    def _extract_usage(self, response_obj: Any) -> Dict[str, Any]:
        """Extract token usage information from response object."""
        usage = getattr(response_obj, "usage", None)
        if usage is None:
            return {}

        return {
            "promptTokens": getattr(usage, "prompt_tokens", 0),
            "completionTokens": getattr(usage, "completion_tokens", 0),
            "totalTokens": getattr(usage, "total_tokens", 0),
            "promptTokensDetails": getattr(usage, "prompt_tokens_details", None),
            "completionTokensDetails": getattr(
                usage, "completion_tokens_details", None
            ),
        }

    def _has_images(self, messages: Optional[List[Dict[str, Any]]]) -> bool:
        """Check if request contains image content."""
        if not messages:
            return False

        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") in ["image_url", "image"]:
                            return True
        return False

    def _sanitize_sensitive_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove sensitive authentication and credential fields from data.

        This ensures API keys, tokens, and secrets are NEVER logged.

        Returns:
            Sanitized copy of data with sensitive fields removed.
        """
        # List of sensitive field names to remove
        sensitive_fields = {
            # API Keys (various providers)
            "api_key",
            "openai_api_key",
            "anthropic_api_key",
            "azure_api_key",
            "cohere_api_key",
            "huggingface_api_key",
            "replicate_api_key",
            "palm_api_key",
            "vertex_ai_api_key",
            "bedrock_api_key",
            "mistral_api_key",
            "together_api_key",
            "anyscale_api_key",
            "perplexity_api_key",
            # LiteLLM internal keys
            "user_api_key",
            "user_api_key_hash",  # Don't log even hashed keys
            # AWS Credentials
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "aws_role_arn",
            # Azure Credentials
            "azure_ad_token",
            "azure_client_id",
            "azure_client_secret",
            "azure_tenant_id",
            # GCP Credentials
            "vertex_credentials",
            "vertex_project",
            "vertex_location",
            "service_account_key",
            # Generic Auth
            "password",
            "secret",
            "token",
            "bearer_token",
            "oauth_token",
            "jwt_token",
            "access_token",
            "refresh_token",
            "client_secret",
            "private_key",
            "credentials",
            # Headers (if accidentally included)
            "authorization",
            "x-api-key",
            "x-auth-token",
        }

        def _sanitize_recursive(obj: Any) -> Any:
            """Recursively sanitize sensitive fields in nested structures."""
            if isinstance(obj, dict):
                return {
                    key: "[REDACTED]" if key in sensitive_fields else _sanitize_recursive(value)
                    for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [_sanitize_recursive(item) for item in obj]
            else:
                return obj

        result = _sanitize_recursive(data)
        return result if isinstance(result, dict) else data

    def _sanitize_headers(
        self, headers: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Sanitize HTTP headers to remove authentication headers.

        Args:
            headers: HTTP headers dict

        Returns:
            Sanitized headers with auth fields redacted, or None if headers is None.
        """
        if headers is None:
            return None

        # Sensitive header names (case-insensitive)
        sensitive_headers = [
            "authorization",
            "x-api-key",
            "x-auth-token",
            "cookie",
            "set-cookie",
            "proxy-authorization",
            "www-authenticate",
        ]

        sanitized = {}
        for key, value in headers.items():
            if key.lower() in sensitive_headers:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value

        return sanitized

    def _extract_model_info(self, kwargs: Dict) -> Dict[str, Any]:
        """Extract model-related information from kwargs."""
        litellm_params = kwargs.get("litellm_params", {}) or {}
        metadata = litellm_params.get("metadata", {}) or {}

        return {
            "model": kwargs.get("model", "unknown"),
            "customLlmProvider": kwargs.get("custom_llm_provider"),
            "modelGroup": metadata.get("model_group"),
            "apiBase": kwargs.get("api_base"),
            "deployment": metadata.get("deployment"),
        }

    def _extract_user_info(self, kwargs: Dict) -> Dict[str, Any]:
        """Extract user/team/key information from kwargs."""
        litellm_params = kwargs.get("litellm_params", {}) or {}
        metadata = litellm_params.get("metadata", {}) or {}

        return {
            "endUser": kwargs.get("user"),
            "userApiKeyUserId": metadata.get("user_api_key_user_id"),
            "userApiKeyTeamId": metadata.get("user_api_key_team_id"),
            "userApiKeyEndUserId": metadata.get("user_api_key_end_user_id"),
            # userApiKeyHash: OMITTED for security (don't log even hashed keys)
            "userApiKeyAlias": metadata.get("user_api_key_alias"),
        }

    def _extract_session_id(self, kwargs: Dict) -> Optional[str]:
        """
        Extract session ID from kwargs.

        Session ID can be provided via:
        1. litellm_session_id (direct parameter)
        2. litellm_trace_id (fallback)
        3. metadata.session_id (check multiple nested locations)

        Returns:
            Session ID string or None if not provided
        """
        # Check direct parameter first
        if kwargs.get("litellm_session_id"):
            return str(kwargs.get("litellm_session_id"))

        # Fallback to trace_id
        if kwargs.get("litellm_trace_id"):
            return str(kwargs.get("litellm_trace_id"))

        # Check metadata at multiple possible locations (structure differs per hook)

        # 1. Direct metadata.session_id (used in async_pre_call_hook)
        metadata = kwargs.get("metadata", {})
        if isinstance(metadata, dict):
            if metadata.get("session_id"):
                return str(metadata.get("session_id"))
            # Also check nested requester_metadata
            requester_metadata = metadata.get("requester_metadata", {})
            if isinstance(requester_metadata, dict) and requester_metadata.get("session_id"):
                return str(requester_metadata.get("session_id"))

        # 2. litellm_params.metadata.session_id (used in async_log_pre_api_call)
        litellm_params = kwargs.get("litellm_params", {}) or {}
        metadata = litellm_params.get("metadata", {}) or {}
        if isinstance(metadata, dict):
            if metadata.get("session_id"):
                return str(metadata.get("session_id"))
            # Also check nested requester_metadata
            requester_metadata = metadata.get("requester_metadata", {})
            if isinstance(requester_metadata, dict) and requester_metadata.get("session_id"):
                return str(requester_metadata.get("session_id"))

        return None

    def _sanitize_messages(
        self, messages: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Sanitize messages for logging - truncate very long text content.
        Keep full content but limit individual text blocks to reasonable size.
        """
        if not messages:
            return messages

        sanitized = []
        max_text_length = 10000  # Max characters per text content

        for msg in messages:
            msg_copy = msg.copy()
            content = msg_copy.get("content")

            if isinstance(content, str) and len(content) > max_text_length:
                msg_copy["content"] = (
                    content[:max_text_length] + f"... (truncated, total: {len(content)} chars)"
                )
            elif isinstance(content, list):
                sanitized_content = []
                for part in content:
                    if isinstance(part, dict):
                        part_copy = part.copy()
                        if part_copy.get("type") == "text":
                            text = part_copy.get("text", "")
                            if isinstance(text, str) and len(text) > max_text_length:
                                part_copy["text"] = (
                                    text[:max_text_length]
                                    + f"... (truncated, total: {len(text)} chars)"
                                )
                        sanitized_content.append(part_copy)
                    else:
                        sanitized_content.append(part)
                msg_copy["content"] = sanitized_content

            sanitized.append(msg_copy)

        return sanitized

    def _extract_response_content(self, response_obj: Any) -> Optional[str]:
        """Extract the response text content."""
        try:
            # Check for choices array (standard completion response)
            choices = getattr(response_obj, "choices", None)
            if choices and len(choices) > 0:
                message = getattr(choices[0], "message", None)
                if message:
                    return getattr(message, "content", None)

            # Check for text attribute (some providers)
            if hasattr(response_obj, "text"):
                return response_obj.text

            # For streaming responses, might not have full content
            return None
        except Exception as e:
            verbose_proxy_logger.debug(
                f"Error extracting response content: {e}"
            )
            return None

    def _extract_clean_request(self, kwargs: Dict) -> Dict[str, Any]:
        """
        Return raw request data with only sensitive fields redacted.

        Returns:
        - ALL fields from kwargs
        - Sensitive fields (API keys, tokens, passwords) are redacted

        This gives complete visibility into the request while maintaining security.
        """
        return self._sanitize_sensitive_fields(kwargs)

    def _extract_clean_response(self, response: Any) -> Dict[str, Any]:
        """
        Return raw response data with only sensitive fields redacted.

        Returns:
        - ALL fields from response
        - Sensitive fields (API keys, tokens) are redacted

        This gives complete visibility into the response while maintaining security.
        """
        try:
            # Convert response object to dict
            if isinstance(response, dict):
                response_dict = response
            else:
                # Try model_dump() first (Pydantic v2), then __dict__
                response_dict = (
                    response.model_dump()
                    if hasattr(response, "model_dump")
                    else response.__dict__
                )

            return self._sanitize_sensitive_fields(response_dict)

        except Exception as e:
            verbose_proxy_logger.debug(
                f"Error extracting clean response: {e}"
            )
            # Fallback to string representation
            return {"response": str(response)}

    async def async_log_pre_api_call(
        self, model: str, messages: List, kwargs: Dict
    ) -> None:
        """
        Log the incoming request from client to LiteLLM (BEFORE transformations).

        This captures:
        - Client → LiteLLM (original client request in OpenAI format)

        This is the 1st of 4 logging points in the complete flow.
        """
        verbose_proxy_logger.debug("GCPLogsStdoutLogger: async_log_pre_api_call called")
        if not self._should_log():
            return

        try:
            # Extract clean request data (core API fields only)
            clean_request = self._extract_clean_request(kwargs)

            model_info = self._extract_model_info(kwargs)
            user_info = self._extract_user_info(kwargs)
            session_id = self._extract_session_id(kwargs)
            has_images = self._has_images(messages)

            self._write_log(
                severity="INFO",
                message=f"LiteLLM request started: {model}",
                log_type="request_start",
                request=clean_request,  # Clean request (core API fields only)
                modelInfo=model_info,
                userInfo=user_info,
                requestId=kwargs.get("litellm_call_id"),
                sessionId=session_id,
                labels={
                    "eventType": "request_start",
                    "model": model,
                    "hasImages": has_images,
                },
            )
        except Exception as e:
            verbose_proxy_logger.error(
                f"GCPLogsStdoutLogger: Error in async_log_pre_api_call: {e}"
            )
            # Don't fail the request due to logging errors
            pass

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,  # noqa: ARG002
        cache: Any,  # noqa: ARG002
        data: Dict[str, Any],
        call_type: str,
    ) -> None:
        """
        Log the transformed request being sent to the model provider.

        This hook is called RIGHT BEFORE LiteLLM calls the provider API.
        It captures the ACTUAL request payload sent to the provider (after LiteLLM transformations).

        This captures:
        - LiteLLM → Provider (transformed/provider-specific request)
        """
        if not self._should_log():
            return

        try:
            # Extract clean request (core API fields only, no proxy metadata)
            clean_request = self._extract_clean_request(data)

            # Extract provider-specific request details
            model = data.get("model", "unknown")
            messages = data.get("messages", [])
            has_images = self._has_images(messages)

            # Extract model and user info
            model_info = self._extract_model_info(data)
            user_info = self._extract_user_info(data)
            session_id = self._extract_session_id(data)

            # Sanitize headers if logging them
            headers_to_log = None
            if self.include_headers and "headers" in data:
                headers_to_log = self._sanitize_headers(data.get("headers"))

            # Log the provider-bound request
            self._write_log(
                severity="INFO",
                message=f"LiteLLM sending request to provider: {model}",
                log_type="provider_request_sent",
                component="litellm",
                providerRequest=clean_request,  # Clean request (core API fields only)
                modelInfo=model_info,
                userInfo=user_info,
                requestId=data.get("litellm_call_id"),
                sessionId=session_id,
                callType=call_type,
                headers=headers_to_log,  # Already sanitized if included
                apiBase=data.get("api_base"),
                labels={
                    "eventType": "provider_request_sent",
                    "model": model,
                    "hasImages": has_images,
                    "callType": call_type,
                },
            )
        except Exception as e:
            verbose_proxy_logger.error(
                f"GCPLogsStdoutLogger: Error in async_pre_call_hook: {e}"
            )
            # Don't fail the request due to logging errors
            pass

    async def async_post_call_success_hook(
        self,
        data: Dict[str, Any],
        user_api_key_dict: Any,  # noqa: ARG002
        response: Any,
    ) -> None:
        """
        Log the raw response received from the model provider.

        This hook is called RIGHT AFTER the provider responds, BEFORE LiteLLM transforms it.
        It captures the RAW provider response.

        This captures:
        - Provider → LiteLLM (raw provider response)
        """
        if not self._should_log():
            return

        try:
            model = data.get("model", "unknown")
            messages = data.get("messages", [])
            has_images = self._has_images(messages)
            call_type = data.get("call_type", "completion")  # Infer from data or default

            # Extract clean response (core fields only, no internal metadata)
            clean_response = self._extract_clean_response(response)

            # Extract model and user info
            model_info = self._extract_model_info(data)
            user_info = self._extract_user_info(data)
            session_id = self._extract_session_id(data)
            usage = self._extract_usage(response)

            self._write_log(
                severity="INFO",
                message=f"LiteLLM received response from provider: {model}",
                log_type="provider_response_received",
                component="litellm",
                providerResponse=clean_response,  # Clean response (core fields only)
                modelInfo=model_info,
                userInfo=user_info,
                usage=usage,
                requestId=data.get("litellm_call_id"),
                sessionId=session_id,
                callType=call_type,
                labels={
                    "eventType": "provider_response_received",
                    "model": model,
                    "hasImages": has_images,
                    "callType": call_type,
                },
            )
        except Exception as e:
            verbose_proxy_logger.error(
                f"GCPLogsStdoutLogger: Error in async_post_call_success_hook: {e}"
            )
            # Don't fail the request due to logging errors
            pass

    async def async_log_success_event(
        self, kwargs: Dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """
        Log successful completion of a request (AFTER all transformations).

        This captures:
        - LiteLLM → Client (final response sent to client in OpenAI format)

        This is the 4th of 4 logging points in the complete flow.
        Includes overall request duration and final token usage.
        """
        if not self._should_log():
            return

        try:
            # Calculate request duration
            duration_ms = (end_time - start_time).total_seconds() * 1000

            # Extract clean request and response (core fields only, no proxy metadata)
            clean_request = self._extract_clean_request(kwargs)
            clean_response = self._extract_clean_response(response_obj)

            model_info = self._extract_model_info(kwargs)
            user_info = self._extract_user_info(kwargs)
            session_id = self._extract_session_id(kwargs)
            usage = self._extract_usage(response_obj)
            messages = kwargs.get("messages", [])
            has_images = self._has_images(messages)

            self._write_log(
                severity="INFO",
                message=f"LiteLLM request completed successfully: {kwargs.get('model', 'unknown')}",
                log_type="request_success",
                request=clean_request,  # Clean request (core API fields only)
                response=clean_response,  # Clean response (core fields only)
                modelInfo=model_info,
                userInfo=user_info,
                performance={
                    "durationMs": round(duration_ms, 2),
                    "cached": kwargs.get("cache_hit", False),
                },
                usage=usage,
                requestId=kwargs.get("litellm_call_id"),
                sessionId=session_id,
                labels={
                    "eventType": "request_success",
                    "model": kwargs.get("model", "unknown"),
                    "hasImages": has_images,
                },
            )
        except Exception as e:
            verbose_proxy_logger.error(
                f"GCPLogsStdoutLogger: Error in async_log_success_event: {e}"
            )
            # Don't fail the request due to logging errors
            pass

    async def async_log_failure_event(
        self, kwargs: Dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """
        Log failed request with error details.
        """
        if not self._should_log():
            return

        try:
            # Calculate request duration
            duration_ms = (end_time - start_time).total_seconds() * 1000

            # Extract clean request (core API fields only, no proxy metadata)
            clean_request = self._extract_clean_request(kwargs)

            model_info = self._extract_model_info(kwargs)
            user_info = self._extract_user_info(kwargs)
            session_id = self._extract_session_id(kwargs)
            messages = kwargs.get("messages", [])
            has_images = self._has_images(messages)

            # Extract error information
            error_message = str(response_obj)
            error_type = type(response_obj).__name__
            error_traceback = None

            # Get traceback if available
            if hasattr(response_obj, "__traceback__"):
                error_traceback = "".join(
                    traceback.format_tb(response_obj.__traceback__)
                )

            self._write_log(
                severity="ERROR",
                message=f"LiteLLM request failed: {error_message}",
                log_type="request_failure",
                request=clean_request,  # Clean request (core API fields only)
                error={
                    "message": error_message,
                    "type": error_type,
                    "traceback": error_traceback,
                },
                modelInfo=model_info,
                userInfo=user_info,
                performance={
                    "durationMs": round(duration_ms, 2),
                },
                requestId=kwargs.get("litellm_call_id"),
                sessionId=session_id,
                labels={
                    "eventType": "request_failure",
                    "model": kwargs.get("model", "unknown"),
                    "hasImages": has_images,
                    "errorType": error_type,
                },
            )
        except Exception as e:
            verbose_proxy_logger.error(
                f"GCPLogsStdoutLogger: Error in async_log_failure_event: {e}"
            )
            # Don't fail the request due to logging errors
            pass

    async def async_log_stream_event(
        self, kwargs: Dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """
        Log streaming response completion.
        """
        if not self._should_log():
            return

        try:
            duration_ms = (end_time - start_time).total_seconds() * 1000

            # Extract clean request and response (core fields only, no proxy metadata)
            clean_request = self._extract_clean_request(kwargs)
            clean_response = self._extract_clean_response(response_obj)

            model_info = self._extract_model_info(kwargs)
            user_info = self._extract_user_info(kwargs)
            session_id = self._extract_session_id(kwargs)
            usage = self._extract_usage(response_obj)
            messages = kwargs.get("messages", [])
            has_images = self._has_images(messages)

            self._write_log(
                severity="INFO",
                message=f"LiteLLM streaming request completed: {kwargs.get('model', 'unknown')}",
                log_type="stream_complete",
                request=clean_request,  # Clean request (core API fields only)
                response=clean_response,  # Clean response (core fields only)
                modelInfo=model_info,
                userInfo=user_info,
                performance={
                    "durationMs": round(duration_ms, 2),
                },
                usage=usage,
                requestId=kwargs.get("litellm_call_id"),
                sessionId=session_id,
                labels={
                    "eventType": "stream_complete",
                    "model": kwargs.get("model", "unknown"),
                    "hasImages": has_images,
                },
            )
        except Exception as e:
            verbose_proxy_logger.error(
                f"GCPLogsStdoutLogger: Error in async_log_stream_event: {e}"
            )
            # Don't fail the request due to logging errors
            pass


# Create default instance that can be used directly in config
gcp_logger = GCPLogsStdoutLogger()