"""Akto guardrail integration for LiteLLM proxy.

Modes:
  - pre_call: Validates request, blocks if flagged.
  - logging_only: Non-blocking ingestion of request+response.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple, Type

from fastapi import HTTPException
import httpx

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

HTTP_PROXY_PATH = "/api/http-proxy"
AKTO_CONNECTOR_NAME = "litellm"
DEFAULT_GUARDRAIL_TIMEOUT = 5


class AktoGuardrail(CustomGuardrail):
    """Validates and ingests LLM traffic via the Akto API."""

    @staticmethod
    def get_config_model() -> Type["GuardrailConfigModel"]:
        from litellm.types.proxy.guardrails.guardrail_hooks.akto import AktoConfigModel

        return AktoConfigModel

    def __init__(
        self,
        akto_base_url: Optional[str] = None,
        akto_api_key: Optional[str] = None,
        akto_account_id: Optional[str] = None,
        akto_vxlan_id: Optional[str] = None,
        unreachable_fallback: Literal["fail_closed", "fail_open"] = "fail_closed",
        guardrail_timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.background_tasks: set = set()

        self.akto_base_url = (
            akto_base_url or os.environ.get("AKTO_GUARDRAIL_API_BASE", "")
        ).rstrip("/")
        if not self.akto_base_url:
            raise ValueError(
                "akto_base_url is required. Set AKTO_GUARDRAIL_API_BASE or pass it in litellm_params."
            )

        self.akto_api_key = akto_api_key or os.environ.get("AKTO_API_KEY", "")
        if not self.akto_api_key:
            raise ValueError(
                "akto_api_key is required. Set AKTO_API_KEY or pass it in litellm_params."
            )

        self.unreachable_fallback: Literal[
            "fail_closed", "fail_open"
        ] = unreachable_fallback
        self.guardrail_timeout = guardrail_timeout or DEFAULT_GUARDRAIL_TIMEOUT
        self.akto_account_id = akto_account_id or os.environ.get(
            "AKTO_ACCOUNT_ID", "1000000"
        )
        self.akto_vxlan_id = akto_vxlan_id or os.environ.get("AKTO_VXLAN_ID", "0")

        kwargs["supported_event_hooks"] = [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.logging_only,
        ]
        super().__init__(**kwargs)

    # ── Helpers ──

    def has_hook(self, name: str) -> bool:
        """Check if this instance handles the given hook (event_hook can be str or list)."""
        hook = self.event_hook
        return hook == name or (isinstance(hook, list) and name in hook)

    def schedule(self, coro) -> None:
        """Schedule a coroutine as a background task with strong reference."""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

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
        """Build request headers from proxy headers."""
        headers: Dict[str, str] = {"content-type": "application/json"}
        proxy_req = data.get("proxy_server_request")
        if isinstance(proxy_req, dict):
            for key, val in (proxy_req.get("headers") or {}).items():
                if key and val:
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
            val = AktoGuardrail.resolve_metadata_value(data, meta_key)
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

    async def send_to_akto(
        self, *, guardrails: bool, ingest_data: bool, payload: dict
    ) -> httpx.Response:
        """POST payload to Akto API."""
        params: Dict[str, str] = {"akto_connector": AKTO_CONNECTOR_NAME}
        if guardrails:
            params["guardrails"] = "true"
        if ingest_data:
            params["ingest_data"] = "true"
        return await self.async_handler.post(
            url=f"{self.akto_base_url}{HTTP_PROXY_PATH}",
            data=json.dumps(payload),
            params=params,
            headers={
                "content-type": "application/json",
                "Authorization": self.akto_api_key,
            },
            timeout=self.guardrail_timeout,
        )

    async def fire_and_forget(
        self, *, guardrails: bool, ingest_data: bool, payload: dict
    ) -> None:
        """Send without blocking. Errors are logged."""
        try:
            resp = await self.send_to_akto(
                guardrails=guardrails, ingest_data=ingest_data, payload=payload
            )
            if resp.status_code != 200:
                verbose_proxy_logger.error(
                    "Akto fire-and-forget HTTP %d", resp.status_code
                )
        except Exception as e:
            verbose_proxy_logger.error("Akto fire-and-forget error: %s", e)

    # ── Response parsing ──

    @staticmethod
    def parse_guardrail_response(response: httpx.Response) -> Tuple[bool, str]:
        """Parse Akto response. Returns (allowed, reason). Raises on non-200 or bad JSON."""
        if response.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Akto returned status {response.status_code}",
                request=response.request,
                response=response,
            )
        try:
            result = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            raise httpx.RequestError(
                "Akto returned non-JSON body",
                request=response.request,
            ) from e

        if not isinstance(result, dict):
            return True, ""
        data = result.get("data") or {}
        if not isinstance(data, dict):
            return True, ""
        gr = data.get("guardrailsResult") or {}
        if not isinstance(gr, dict):
            return True, ""
        return bool(gr.get("Allowed", True)), str(gr.get("Reason", ""))

    # ── Main hooks ──

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj=None,
    ) -> GenericGuardrailAPIInputs:
        """Pre_call: validate request against Akto guardrails, block if flagged."""
        if not self.has_hook("pre_call") or input_type != "request":
            return inputs

        payload = self.build_akto_payload(request_data)
        try:
            response = await self.send_to_akto(
                guardrails=True, ingest_data=False, payload=payload
            )
            allowed, reason = self.parse_guardrail_response(response)
        except HTTPException:
            raise
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            if self.unreachable_fallback == "fail_open":
                verbose_proxy_logger.critical("Akto unreachable (fail-open): %s", e)
                return inputs
            raise HTTPException(
                status_code=503, detail="Akto guardrail service unreachable"
            )

        if not allowed:
            blocked = self.build_akto_payload(request_data, status_code=403)
            blocked["responsePayload"] = json.dumps(
                {"x-blocked-by": "Akto Proxy", "reason": reason}
            )
            blocked["responseHeaders"] = json.dumps(
                {"content-type": "application/json"}
            )
            self.schedule(
                self.fire_and_forget(
                    guardrails=False, ingest_data=True, payload=blocked
                )
            )
            raise HTTPException(
                status_code=403, detail=reason or "Blocked by Akto Guardrails"
            )

        return inputs

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Logging_only: non-blocking ingestion of request+response to Akto."""
        if not self.has_hook("logging_only"):
            return
        try:
            litellm_params = kwargs.get("litellm_params") or {}
            data = dict(kwargs)
            if (
                "proxy_server_request" not in data
                and "proxy_server_request" in litellm_params
            ):
                data["proxy_server_request"] = litellm_params["proxy_server_request"]
            if "metadata" not in data and "metadata" in litellm_params:
                data["metadata"] = litellm_params["metadata"]
            payload = self.build_akto_payload(data, response_obj=response_obj)
            self.schedule(
                self.fire_and_forget(
                    guardrails=False, ingest_data=True, payload=payload
                )
            )
        except Exception as e:
            verbose_proxy_logger.error("Akto logging error: %s", e)
