"""Akto logging integration for LiteLLM.

Ingests LLM request/response traffic to Akto for monitoring and analysis.
Configure via success_callback/failure_callback: ["akto"].
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import (
    HTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)

HTTP_PROXY_PATH = "/api/http-proxy"
AKTO_CONNECTOR_NAME = "litellm"
SENSITIVE_HEADERS = {"authorization", "x-litellm-api-key", "x-api-key", "cookie"}


class AktoLogger(CustomLogger):
    """Logs LLM traffic to Akto for monitoring and analysis."""

    def __init__(self) -> None:
        super().__init__()
        self.validate_environment()
        self.async_http_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        self.sync_http_handler = HTTPHandler()

        self.akto_base_url = os.environ["AKTO_DATA_INGESTION_API_BASE"].rstrip("/")
        self.akto_api_key = os.environ["AKTO_API_KEY"]
        self.akto_account_id = os.environ.get("AKTO_ACCOUNT_ID", "1000000")
        self.akto_vxlan_id = os.environ.get("AKTO_VXLAN_ID", "0")

    def validate_environment(self) -> None:
        missing_keys = []
        if os.getenv("AKTO_DATA_INGESTION_API_BASE", None) is None:
            missing_keys.append("AKTO_DATA_INGESTION_API_BASE")
        if os.getenv("AKTO_API_KEY", None) is None:
            missing_keys.append("AKTO_API_KEY")
        if len(missing_keys) > 0:
            raise Exception("Missing keys={} in environment.".format(missing_keys))

    # ── Data extraction ──

    @staticmethod
    def extract_logging_data(kwargs: dict) -> dict:
        """Promote metadata and proxy_server_request from litellm_params to top level."""
        litellm_params = kwargs.get("litellm_params") or {}
        data = dict(kwargs)
        if (
            "proxy_server_request" not in data
            and "proxy_server_request" in litellm_params
        ):
            data["proxy_server_request"] = litellm_params["proxy_server_request"]
        if "metadata" not in data and "metadata" in litellm_params:
            data["metadata"] = litellm_params["metadata"]
        return data

    # ── Payload builders ──

    @staticmethod
    def resolve_metadata_value(data: Optional[dict], key: str) -> Optional[str]:
        """Look up a value from metadata or litellm_metadata."""
        if data is None:
            return None
        for k in ("litellm_metadata", "metadata"):
            container = data.get(k) or {}
            if isinstance(container, dict):
                val = container.get(key)
                if val is not None:
                    return str(val).strip()
        return None

    @staticmethod
    def extract_request_path(data: dict) -> str:
        """Get the API route, defaulting to /v1/chat/completions."""
        metadata = data.get("metadata") or {}
        if isinstance(metadata, dict):
            return metadata.get("user_api_key_request_route") or "/v1/chat/completions"
        return "/v1/chat/completions"

    @staticmethod
    def build_request_headers(data: dict) -> Dict[str, str]:
        """Build request headers from proxy headers, stripping sensitive values."""
        headers: Dict[str, str] = {"content-type": "application/json"}
        proxy_req = data.get("proxy_server_request")
        if isinstance(proxy_req, dict):
            for key, val in (proxy_req.get("headers") or {}).items():
                if key and val and str(key).lower() not in SENSITIVE_HEADERS:
                    headers[str(key).lower()] = str(val)
        if "host" not in headers:
            headers["host"] = "litellm.ai"
        return headers

    @staticmethod
    def build_tag_metadata(data: dict) -> Dict[str, str]:
        """Build tag dict with gen-ai marker, user_id, and team_id."""
        tag: Dict[str, str] = {"gen-ai": "Gen AI"}
        for meta_key, tag_key in [
            ("user_api_key_user_id", "user_id"),
            ("user_api_key_team_id", "team_id"),
        ]:
            val = AktoLogger.resolve_metadata_value(data, meta_key)
            if val:
                tag[tag_key] = val
        return tag

    @staticmethod
    def extract_client_ip(data: dict) -> str:
        """Extract client IP from proxy headers."""
        proxy_req = data.get("proxy_server_request")
        if isinstance(proxy_req, dict):
            headers = proxy_req.get("headers") or {}
            ip = headers.get("x-forwarded-for") or headers.get("x-real-ip") or ""
            if ip:
                return ip.split(",")[0].strip()
        return "0.0.0.0"

    def build_akto_payload(
        self, data: dict, *, status_code: int = 200, response_obj: Any = None
    ) -> Dict[str, Any]:
        """Build the MIRRORING payload for Akto's HTTP proxy endpoint."""
        request_body: Dict[str, Any] = {}
        if data.get("messages"):
            request_body["messages"] = data["messages"]
            request_body["model"] = data.get("model", "")
        if data.get("tools"):
            request_body["tools"] = data["tools"]

        response_payload = json.dumps({})
        response_headers: Dict[str, str] = {}
        if response_obj is not None and hasattr(response_obj, "model_dump"):
            response_payload = json.dumps(response_obj.model_dump())
            response_headers = {"content-type": "application/json"}

        tag = self.build_tag_metadata(data)

        return {
            "path": self.extract_request_path(data),
            "requestHeaders": json.dumps(self.build_request_headers(data)),
            "responseHeaders": json.dumps(response_headers),
            "method": "POST",
            "requestPayload": json.dumps(request_body),
            "responsePayload": response_payload,
            "ip": self.extract_client_ip(data),
            "destIp": "127.0.0.1",
            "time": str(int(datetime.now().timestamp() * 1000)),
            "statusCode": str(status_code),
            "type": "HTTP/1.1",
            "status": str(status_code),
            "akto_account_id": self.akto_account_id,
            "akto_vxlan_id": self.akto_vxlan_id,
            "is_pending": "false",
            "source": "MIRRORING",
            "direction": None,
            "process_id": None,
            "socket_id": None,
            "daemonset_id": None,
            "enabled_graph": None,
            "tag": json.dumps(tag),
            "metadata": json.dumps(tag),
            "contextSource": "AGENTIC",
        }

    # ── HTTP ──

    def request_kwargs(self, payload: dict) -> dict:
        """Build common HTTP request kwargs for Akto API."""
        return {
            "url": f"{self.akto_base_url}{HTTP_PROXY_PATH}",
            "data": json.dumps(payload),
            "params": {"akto_connector": AKTO_CONNECTOR_NAME, "ingest_data": "true"},
            "headers": {
                "content-type": "application/json",
                "Authorization": self.akto_api_key,
            },
        }

    async def async_health_check(self) -> dict:
        """Ping Akto's health check endpoint."""
        try:
            response = await self.async_http_handler.get(
                url=f"{self.akto_base_url}/",
                headers={"Authorization": self.akto_api_key},
                timeout=5,
            )
            if response.status_code == 200:
                return {"status": "healthy", "error_message": None}
            return {
                "status": "unhealthy",
                "error_message": f"Akto returned status {response.status_code}",
            }
        except Exception:
            return {"status": "unhealthy", "error_message": "Akto health check failed"}

    # ── Logging callbacks ──

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            data = self.extract_logging_data(kwargs)
            payload = self.build_akto_payload(data, response_obj=response_obj)
            self.sync_http_handler.post(**self.request_kwargs(payload))
        except Exception as e:
            verbose_logger.error("Akto logging error: %s", e)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            data = self.extract_logging_data(kwargs)
            payload = self.build_akto_payload(data, response_obj=response_obj)
            await self.async_http_handler.post(**self.request_kwargs(payload))
        except Exception as e:
            verbose_logger.error("Akto logging error: %s", e)

    @staticmethod
    def get_failure_status_code(kwargs: dict) -> int:
        """Return the appropriate status code for a failed request."""
        exc = kwargs.get("exception")
        if (
            exc is not None
            and hasattr(exc, "status_code")
            and isinstance(exc.status_code, int)
        ):
            return exc.status_code
        return 500

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            data = self.extract_logging_data(kwargs)
            status = self.get_failure_status_code(kwargs)
            payload = self.build_akto_payload(data, status_code=status)
            self.sync_http_handler.post(**self.request_kwargs(payload))
        except Exception as e:
            verbose_logger.error("Akto logging error (failure): %s", e)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            data = self.extract_logging_data(kwargs)
            status = self.get_failure_status_code(kwargs)
            payload = self.build_akto_payload(data, status_code=status)
            await self.async_http_handler.post(**self.request_kwargs(payload))
        except Exception as e:
            verbose_logger.error("Akto logging error (failure): %s", e)
