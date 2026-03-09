"""
ScopeBlind Integration for LiteLLM

Device-level rate limiting for LLM proxy operators.
Verifies DPoP proofs (RFC 9449) to bind requests to unique devices
instead of IP addresses — survives proxy rotation, VPNs, and shared networks.

Requires:
    SCOPEBLIND_API_KEY  – your ScopeBlind API key (from scopeblind.com/t/<slug>)

Optional:
    SCOPEBLIND_ENDPOINT – verification endpoint (default: https://api.scopeblind.com)

Usage (proxy config.yaml):
    litellm_settings:
      callbacks: ["scopeblind"]

    environment_variables:
      SCOPEBLIND_API_KEY: "sb_..."

Or programmatically:
    import litellm
    litellm.callbacks = ["scopeblind"]

Docs: https://scopeblind.com/docs
"""

import os
import traceback
from typing import Any, Dict, List, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import (
    HTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)


def _get_utc_datetime():
    import datetime as dt
    from datetime import datetime

    if hasattr(dt, "UTC"):
        return datetime.now(dt.UTC)
    else:
        return datetime.utcnow()


class ScopeBlindLogger(CustomLogger):
    """
    ScopeBlind callback for device-level rate limiting.

    Extracts DPoP proof headers from incoming proxy requests,
    verifies them with the ScopeBlind API, and logs per-device
    usage data (model, cost, tokens) for abuse detection.
    """

    def __init__(self) -> None:
        super().__init__()
        self.validate_environment()
        self.async_http_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        self.sync_http_handler = HTTPHandler()
        self.scopeblind_endpoint = os.getenv(
            "SCOPEBLIND_ENDPOINT", "https://api.scopeblind.com"
        )
        self.scopeblind_api_key = os.getenv("SCOPEBLIND_API_KEY", "")

    def validate_environment(self):
        """Expects SCOPEBLIND_API_KEY in the environment."""
        missing_keys: List[str] = []
        if os.getenv("SCOPEBLIND_API_KEY", None) is None:
            missing_keys.append("SCOPEBLIND_API_KEY")

        if len(missing_keys) > 0:
            raise Exception(
                "ScopeBlind: Missing keys={} in environment.".format(missing_keys)
            )

    def _build_payload(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time,
        end_time,
        event_type: str,
    ) -> dict:
        """Build a ScopeBlind event payload from LiteLLM kwargs."""
        call_id = kwargs.get("litellm_call_id")
        model = kwargs.get("model")
        cost = kwargs.get("response_cost", None)
        user = kwargs.get("user", None)

        # Extract device identity from metadata
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata", {}) or {}
        headers = metadata.get("headers", {}) or {}

        # DPoP proof and device ID from incoming request headers
        dpop_proof = headers.get("x-scopeblind-dpop") or headers.get("dpop")
        device_id = headers.get("x-scopeblind-device-id") or headers.get(
            "x-device-identity"
        )

        # Token usage
        usage = {}
        if (
            isinstance(response_obj, litellm.ModelResponse)
            or isinstance(response_obj, litellm.EmbeddingResponse)
        ) and hasattr(response_obj, "usage"):
            usage = {
                "prompt_tokens": response_obj["usage"].get("prompt_tokens", 0),
                "completion_tokens": response_obj["usage"].get(
                    "completion_tokens", 0
                ),
                "total_tokens": response_obj["usage"].get("total_tokens", 0),
            }

        # If no user provided, try API key user ID
        if user is None:
            user = metadata.get("user_api_key_user_id")

        return {
            "event": event_type,
            "litellm_call_id": call_id,
            "model": model,
            "user": user,
            "cost": cost,
            "usage": usage,
            "dpop_proof": dpop_proof,
            "device_id": device_id,
            "timestamp": _get_utc_datetime().isoformat(),
            "start_time": str(start_time),
            "end_time": str(end_time),
        }

    async def _send_event(self, payload: dict) -> None:
        """Send an event to the ScopeBlind API."""
        try:
            url = f"{self.scopeblind_endpoint}/v1/proxy/events"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.scopeblind_api_key}",
            }
            await self.async_http_handler.post(
                url=url,
                json=payload,
                headers=headers,
            )
        except Exception as e:
            verbose_logger.debug(
                "ScopeBlind: Failed to send event: %s", str(e)
            )

    def _send_event_sync(self, payload: dict) -> None:
        """Send an event synchronously."""
        try:
            url = f"{self.scopeblind_endpoint}/v1/proxy/events"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.scopeblind_api_key}",
            }
            self.sync_http_handler.post(
                url=url,
                json=payload,
                headers=headers,
            )
        except Exception as e:
            verbose_logger.debug(
                "ScopeBlind: Failed to send event (sync): %s", str(e)
            )

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ):
        """Log successful LLM call with device identity."""
        try:
            payload = self._build_payload(
                kwargs, response_obj, start_time, end_time, "llm_call_success"
            )
            # Only send if we have device identity info
            if payload.get("dpop_proof") or payload.get("device_id"):
                await self._send_event(payload)
            else:
                verbose_logger.debug(
                    "ScopeBlind: No device identity headers found, skipping event"
                )
        except Exception as e:
            verbose_logger.debug(
                "ScopeBlind: Error logging success event: %s\n%s",
                str(e),
                traceback.format_exc(),
            )

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Log successful LLM call synchronously."""
        try:
            payload = self._build_payload(
                kwargs, response_obj, start_time, end_time, "llm_call_success"
            )
            if payload.get("dpop_proof") or payload.get("device_id"):
                self._send_event_sync(payload)
        except Exception as e:
            verbose_logger.debug(
                "ScopeBlind: Error logging success event (sync): %s", str(e)
            )

    async def async_log_failure_event(
        self, kwargs, response_obj, start_time, end_time
    ):
        """Log failed LLM call with device identity."""
        try:
            payload = self._build_payload(
                kwargs, response_obj, start_time, end_time, "llm_call_failure"
            )
            if payload.get("dpop_proof") or payload.get("device_id"):
                await self._send_event(payload)
        except Exception as e:
            verbose_logger.debug(
                "ScopeBlind: Error logging failure event: %s", str(e)
            )

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Log failed LLM call synchronously."""
        try:
            payload = self._build_payload(
                kwargs, response_obj, start_time, end_time, "llm_call_failure"
            )
            if payload.get("dpop_proof") or payload.get("device_id"):
                self._send_event_sync(payload)
        except Exception as e:
            verbose_logger.debug(
                "ScopeBlind: Error logging failure event (sync): %s", str(e)
            )
