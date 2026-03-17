"""Akto guardrail integration for LiteLLM proxy.

Uses a two-config-entry pattern:
  - akto-validate (pre_call): Checks request against Akto guardrails, blocks if flagged.
  - akto-ingest (post_call): Sends request+response to Akto for data ingestion.

For monitor-only mode, enable only akto-ingest without akto-validate.
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
    """LiteLLM guardrail hook that validates and ingests LLM traffic via the Akto API."""

    # Maps event_hook to the input_type it should handle; mismatches are no-ops
    HOOK_TO_INPUT = {"pre_call": "request", "post_call": "response"}

    @staticmethod
    def get_config_model() -> Type["GuardrailConfigModel"]:
        """Return the Pydantic config model for YAML-based initialization."""
        from litellm.types.proxy.guardrails.guardrail_hooks.akto import (
            AktoConfigModel,
        )

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
        """Initialize the Akto guardrail.

        Args:
            akto_base_url: Akto API base URL. Falls back to AKTO_GUARDRAIL_API_BASE env var.
            akto_api_key: Akto API key. Falls back to AKTO_API_KEY env var.
            akto_account_id: Akto account ID. Falls back to AKTO_ACCOUNT_ID env var, then "1000000".
            akto_vxlan_id: Akto VXLAN ID. Falls back to AKTO_VXLAN_ID env var, then "0".
            unreachable_fallback: Behavior when Akto is unreachable — block or allow.
            guardrail_timeout: HTTP timeout in seconds for Akto API calls.
        """
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )
        self.background_tasks: set = set()

        self.akto_base_url = (akto_base_url or os.environ.get("AKTO_GUARDRAIL_API_BASE", "")).rstrip("/")
        if not self.akto_base_url:
            raise ValueError("akto_base_url is required. Set AKTO_GUARDRAIL_API_BASE or pass it in litellm_params.")

        self.akto_api_key = akto_api_key or os.environ.get("AKTO_API_KEY", "")
        if not self.akto_api_key:
            raise ValueError("akto_api_key is required. Set AKTO_API_KEY or pass it in litellm_params.")

        self.unreachable_fallback: Literal["fail_closed", "fail_open"] = unreachable_fallback
        self.guardrail_timeout = guardrail_timeout or DEFAULT_GUARDRAIL_TIMEOUT
        self.akto_account_id = akto_account_id or os.environ.get("AKTO_ACCOUNT_ID", "1000000")
        self.akto_vxlan_id = akto_vxlan_id or os.environ.get("AKTO_VXLAN_ID", "0")

        kwargs["supported_event_hooks"] = [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ]
        super().__init__(**kwargs)

        verbose_proxy_logger.debug(
            "Akto guardrail initialized: base_url=%s fallback=%s",
            self.akto_base_url,
            self.unreachable_fallback,
        )

    @staticmethod
    def resolve_metadata_value(request_data: Optional[dict], key: str) -> Optional[str]:
        """Look up a metadata value from litellm_metadata or metadata dicts."""
        if request_data is None:
            return None
        for dict_key in ("litellm_metadata", "metadata"):
            container = request_data.get(dict_key) or {}
            if isinstance(container, dict) and container:
                value = container.get(key)
                if value is not None:
                    return str(value).strip()
        return None

    @staticmethod
    def extract_request_path(request_data: dict) -> str:
        """Extract the API route from request metadata, defaulting to /v1/chat/completions."""
        metadata = request_data.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        route = metadata.get("user_api_key_request_route")
        return route if route else "/v1/chat/completions"

    def prepare_headers(self) -> Dict[str, str]:
        """Build HTTP headers for the Akto API call."""
        return {
            "content-type": "application/json",
            "Authorization": self.akto_api_key,
        }

    @staticmethod
    def build_query_params(*, guardrails: bool, ingest_data: bool) -> Dict[str, str]:
        """Build query params that control Akto backend behavior (guardrail check and/or data ingestion)."""
        params: Dict[str, str] = {"akto_connector": AKTO_CONNECTOR_NAME}
        if guardrails:
            params["guardrails"] = "true"
        if ingest_data:
            params["ingest_data"] = "true"
        return params

    @staticmethod
    def build_request_headers(request_data: dict) -> Dict[str, str]:
        """Build the requestHeaders field from proxy request headers."""
        headers: Dict[str, str] = {"content-type": "application/json"}
        proxy_req = request_data.get("proxy_server_request", {})
        if not isinstance(proxy_req, dict):
            return headers
        proxy_req_headers = proxy_req.get("headers")
        if isinstance(proxy_req_headers, dict):
            for key, val in proxy_req_headers.items():
                if key and val:
                    headers[str(key).lower()] = str(val)
        return headers

    @staticmethod
    def build_request_body(
        inputs: GenericGuardrailAPIInputs,
        request_data: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Build the LLM request body from guardrail inputs (messages, model, tools)."""
        model = inputs.get("model", "") or ""
        body: Dict[str, Any] = {"model": model}

        structured = inputs.get("structured_messages")
        if structured:
            body["messages"] = structured
        elif request_data is not None and request_data.get("messages"):
            body["messages"] = request_data["messages"]
            if request_data.get("model"):
                body["model"] = request_data["model"]
        else:
            texts = inputs.get("texts", [])
            body["messages"] = [{"role": "user", "content": t} for t in texts] if texts else []

        tools = inputs.get("tools")
        if tools:
            body["tools"] = tools
        elif request_data is not None and request_data.get("tools"):
            body["tools"] = request_data["tools"]

        tool_calls = inputs.get("tool_calls")
        if tool_calls:
            body["tool_calls"] = tool_calls

        return body

    @staticmethod
    def build_response_body(
        inputs: GenericGuardrailAPIInputs,
        request_data: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Build the LLM response body, preferring the actual model response if available."""
        model_response = request_data.get("response") if request_data else None
        if model_response is not None and hasattr(model_response, "model_dump"):
            return model_response.model_dump()

        texts = inputs.get("texts", [])
        if texts:
            return {"choices": [{"message": {"content": t, "role": "assistant"}} for t in texts]}
        return {}

    @staticmethod
    def build_tag_metadata(request_data: dict) -> Dict[str, str]:
        """Build tag/metadata dict with user_id and team_id for Akto tracking."""
        tag: Dict[str, str] = {"gen-ai": "Gen AI"}
        user_id = AktoGuardrail.resolve_metadata_value(request_data, "user_api_key_user_id")
        team_id = AktoGuardrail.resolve_metadata_value(request_data, "user_api_key_team_id")
        if user_id:
            tag["user_id"] = user_id
        if team_id:
            tag["team_id"] = team_id
        return tag

    def build_akto_payload(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        *,
        status_code: int = 200,
        include_response: bool = False,
    ) -> Dict[str, Any]:
        """Build the flat MIRRORING payload sent to Akto's HTTP proxy endpoint.

        All body fields use double-encoding: json.dumps({"body": json.dumps(actual_body)})
        to match the canonical CLI hook format.
        """
        request_path = self.extract_request_path(request_data)
        request_headers = self.build_request_headers(request_data)
        request_body = self.build_request_body(inputs, request_data)
        tag = self.build_tag_metadata(request_data)

        response_payload = json.dumps({})  # Empty body wrapper when no response yet
        response_headers: Dict[str, str] = {}
        if include_response:
            response_body = self.build_response_body(inputs, request_data)
            response_payload = json.dumps({"body": json.dumps(response_body)})  # Double-encoded
            response_headers = {"content-type": "application/json"}

        # Extract client IP from proxy headers
        ip = ""
        proxy_req = request_data.get("proxy_server_request", {})
        proxy_headers = proxy_req.get("headers", {}) if isinstance(proxy_req, dict) else {}
        if isinstance(proxy_headers, dict):
            ip = proxy_headers.get("x-forwarded-for") or proxy_headers.get("x-real-ip") or ""
            if "," in ip:
                ip = ip.split(",")[0].strip()

        return {
            "path": request_path,
            "requestHeaders": json.dumps(request_headers),
            "responseHeaders": json.dumps(response_headers),
            "method": "POST",
            "requestPayload": json.dumps({"body": json.dumps(request_body)}),  # Double-encoded
            "responsePayload": response_payload,
            "ip": ip,
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

    async def send_request(
        self,
        *,
        guardrails: bool,
        ingest_data: bool,
        payload: dict,
    ) -> httpx.Response:
        """Send an HTTP POST to the Akto API endpoint."""
        endpoint = f"{self.akto_base_url}{HTTP_PROXY_PATH}"
        params = self.build_query_params(guardrails=guardrails, ingest_data=ingest_data)
        headers = self.prepare_headers()
        return await self.async_handler.post(
            url=endpoint,
            json=payload,
            params=params,
            headers=headers,
            timeout=self.guardrail_timeout,
        )

    @staticmethod
    def handle_guardrail_response(response: httpx.Response) -> Tuple[bool, str]:
        """Parse the Akto guardrail response. Returns (allowed, reason)."""
        if response.status_code != 200:
            verbose_proxy_logger.error("Akto returned HTTP %d", response.status_code)
            raise httpx.HTTPStatusError(
                f"Akto returned unexpected status {response.status_code}",
                request=response.request,
                response=response,
            )
        try:
            result = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            response_text = getattr(response, "text", "")
            verbose_proxy_logger.error(
                "Akto returned non-JSON body for status 200: %r",
                response_text[:200],
            )
            raise httpx.RequestError(
                "Akto returned non-JSON body",
                request=getattr(response, "request", None),
            ) from e
        if not isinstance(result, dict):
            return True, ""
        data = result.get("data") or {}
        if not isinstance(data, dict):
            return True, ""
        guardrails_result = data.get("guardrailsResult") or {}
        if not isinstance(guardrails_result, dict):
            return True, ""
        return (
            bool(guardrails_result.get("Allowed", True)),
            str(guardrails_result.get("Reason", "")),
        )

    def handle_unreachable(
        self,
        inputs: GenericGuardrailAPIInputs,
        error: Exception,
    ) -> GenericGuardrailAPIInputs:
        """Handle Akto being unreachable based on fail_open/fail_closed config."""
        if self.unreachable_fallback == "fail_open":
            verbose_proxy_logger.critical(
                "Akto unreachable (fail-open): %s",
                str(error),
                exc_info=error,
            )
            return inputs

        verbose_proxy_logger.error("Akto unreachable (fail-closed): %s", str(error))
        raise HTTPException(
            status_code=503,
            detail="Akto guardrail service unreachable",
        )

    async def fire_and_forget_request(
        self,
        *,
        guardrails: bool,
        ingest_data: bool,
        payload: dict,
    ) -> None:
        """Send a request without awaiting it in the caller. Errors are logged, not raised."""
        try:
            response = await self.send_request(
                guardrails=guardrails,
                ingest_data=ingest_data,
                payload=payload,
            )
            if response.status_code != 200:
                verbose_proxy_logger.error(
                    "Akto fire-and-forget returned HTTP %d",
                    response.status_code,
                )
        except Exception as e:
            verbose_proxy_logger.error("Akto fire-and-forget error: %s", str(e))

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj=None,
    ) -> GenericGuardrailAPIInputs:
        """Main entry point called by LiteLLM's guardrail framework.

        Pre_call (input_type="request"):
          - Awaits guardrail check. If blocked, fires off ingest with 403 marker and raises.
        Post_call (input_type="response"):
          - Fire-and-forget combined guardrail + ingest call.
        """
        # Skip if this hook doesn't handle the current input_type
        expected = self.HOOK_TO_INPUT.get(str(self.event_hook))
        if expected and expected != input_type:
            return inputs

        if input_type == "request":
            # Pre_call: awaited guardrail check (no ingestion)
            payload = self.build_akto_payload(inputs, request_data, include_response=False)
            try:
                response = await self.send_request(
                    guardrails=True,
                    ingest_data=False,
                    payload=payload,
                )
                allowed, reason = self.handle_guardrail_response(response)
            except HTTPException:
                raise
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                return self.handle_unreachable(
                    inputs=inputs,
                    error=e,
                )

            if not allowed:
                # Build a blocked marker payload with 403 status and reason
                blocked_payload = self.build_akto_payload(
                    inputs,
                    request_data,
                    include_response=False,
                    status_code=403,
                )
                blocked_payload["responsePayload"] = json.dumps(
                    {
                        "body": json.dumps({"x-blocked-by": "Akto Proxy", "reason": reason}),
                    }
                )
                blocked_payload["responseHeaders"] = json.dumps(
                    {"content-type": "application/json"},
                )
                # Fire-and-forget ingest of the blocked request, then raise 403
                task = asyncio.create_task(
                    self.fire_and_forget_request(
                        guardrails=False,
                        ingest_data=True,
                        payload=blocked_payload,
                    )
                )
                self.background_tasks.add(task)
                task.add_done_callback(self.background_tasks.discard)
                raise HTTPException(
                    status_code=403,
                    detail=reason or "Blocked by Akto Guardrails",
                )

        elif input_type == "response":
            # Post_call: fire-and-forget combined guardrail + ingest
            payload = self.build_akto_payload(inputs, request_data, include_response=True)
            task = asyncio.create_task(
                self.fire_and_forget_request(
                    guardrails=True,
                    ingest_data=True,
                    payload=payload,
                )
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

        return inputs
