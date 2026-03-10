# +-------------------------------------------------------------+
#
#           Akto Guardrail Integration for LiteLLM
#
# +-------------------------------------------------------------+
#
# Integrates Akto's guardrail and data-ingestion capabilities
# with LiteLLM's unified guardrail framework.
#
# Two modes of operation controlled by `on_flagged`:
#
#   on_flagged="block" (blocking):
#     Pre-call:  POST ?akto_connector&guardrails=true
#       → Allowed=true  → proceed to LLM → post-call ingest
#       → Allowed=false → ingest blocked details → raise error
#     Post-call: POST ?akto_connector&ingest_data=true
#       → Sends request+response for observability
#
#   on_flagged="monitor" (non-blocking):
#     Post-call only: POST ?akto_connector&guardrails=true&ingest_data=true
#       → Single call after LLM response (log-only, does not block)
#
# Reference: https://www.akto.io/

import json
import os
from datetime import datetime
from fastapi import HTTPException
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple, Type

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import Timeout
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs, GuardrailStatus

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

GUARDRAIL_NAME = "akto"
HTTP_PROXY_PATH = "/api/http-proxy"
AKTO_CONNECTOR_NAME = "litellm"
DEFAULT_GUARDRAIL_TIMEOUT = 5


class AktoGuardrail(CustomGuardrail):
    """
    Akto Guardrail integration for LiteLLM.

    Sends requests to the Akto Data Ingestion Service's HTTP proxy endpoint
    for guardrail validation and data ingestion.

    Two modes controlled by `on_flagged`:
      on_flagged="block"   (blocking):     pre-call guardrail check + post-call ingest
      on_flagged="monitor" (non-blocking): single post-call call with guardrails+ingest

    Config example (config.yaml):
        guardrails:
          - guardrail_name: "akto-guard"
            litellm_params:
              guardrail: akto
              mode: [pre_call, post_call]                 # block mode
              akto_base_url: os.environ/AKTO_DATA_INGESTION_URL
              akto_api_key: os.environ/AKTO_API_KEY             # required
              on_flagged: block                           # or "monitor"
              akto_account_id: os.environ/AKTO_ACCOUNT_ID  # optional
              akto_vxlan_id: os.environ/AKTO_VXLAN_ID      # optional
              unreachable_fallback: fail_open              # optional, default fail_closed
              guardrail_timeout: 10                        # optional, default 5s
    """

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.akto import (
            AktoConfigModel,
        )

        return AktoConfigModel

    def __init__(
        self,
        akto_base_url: Optional[str] = None,
        akto_api_key: Optional[str] = None,
        on_flagged: Optional[Literal["block", "monitor"]] = None,
        akto_account_id: Optional[str] = None,
        akto_vxlan_id: Optional[str] = None,
        unreachable_fallback: Literal["fail_closed", "fail_open"] = "fail_closed",
        guardrail_timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        self.akto_base_url = (
            akto_base_url or os.environ.get("AKTO_DATA_INGESTION_URL", "")
        ).rstrip("/")
        if not self.akto_base_url:
            raise ValueError(
                "akto_base_url is required for Akto guardrail. "
                "Set AKTO_DATA_INGESTION_URL environment variable or pass it in litellm_params."
            )

        self.akto_api_key = (
            akto_api_key or os.environ.get("AKTO_API_KEY", "")
        )
        if not self.akto_api_key:
            raise ValueError(
                "akto_api_key is required for Akto guardrail. "
                "Set AKTO_API_KEY environment variable or pass akto_api_key in litellm_params."
            )
        self.akto_account_id = akto_account_id or os.environ.get(
            "AKTO_ACCOUNT_ID", "1000000"
        )
        self.akto_vxlan_id = akto_vxlan_id or os.environ.get("AKTO_VXLAN_ID", "0")

        # Resolve on_flagged: explicit param > env var > default (block)
        if on_flagged:
            self.on_flagged = on_flagged
        else:
            env_on_flagged = os.environ.get("AKTO_ON_FLAGGED", "block").lower()
            self.on_flagged = env_on_flagged if env_on_flagged in ("block", "monitor") else "block"

        # Sync mode internal flag: True if we are blocking (run pre_call + post_call)
        self.sync_mode = (self.on_flagged == "block")

        self.unreachable_fallback: Literal["fail_closed", "fail_open"] = (
            unreachable_fallback
        )

        self.guardrail_timeout = guardrail_timeout or DEFAULT_GUARDRAIL_TIMEOUT

        # Declare all event hooks that Akto's apply_guardrail can handle.
        # Do NOT override event_hook here — it comes from the config's "mode" field.
        # The proxy's pre_call_hook/post_call dispatch checks should_run_guardrail(event_type)
        # which compares against event_hook. If we override it, those dispatchers skip us.
        kwargs["supported_event_hooks"] = [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.during_call,
        ]

        super().__init__(**kwargs)

        verbose_proxy_logger.debug(
            "Akto guardrail initialized: akto_base_url=%s sync_mode=%s unreachable_fallback=%s",
            self.akto_base_url,
            self.sync_mode,
            self.unreachable_fallback,
        )

    # ------------------------------------------------------------------ #
    #  Metadata resolution
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_metadata_value(
        request_data: Optional[dict], key: str
    ) -> Optional[str]:
        """
        Resolve metadata value from request_data, checking both metadata locations.

        During pre-call: metadata is at request_data["metadata"][key]
        During post-call: metadata is at request_data["litellm_metadata"][key]
        """
        if request_data is None:
            return None

        litellm_metadata = request_data.get("litellm_metadata", {})
        if litellm_metadata:
            value = litellm_metadata.get(key)
            if value is not None:
                return str(value).strip()

        metadata = request_data.get("metadata", {})
        if metadata:
            value = metadata.get(key)
            if value is not None:
                return str(value).strip()

        return None

    # ------------------------------------------------------------------ #
    #  Request path extraction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_request_path(request_data: dict) -> str:
        """Extract the original request path from request metadata."""
        fallback = "/v1/chat/completions"
        try:
            metadata = request_data.get("metadata", {})
            route = metadata.get("user_api_key_request_route")
            if route:
                return route
        except Exception:
            pass
        return fallback

    # ------------------------------------------------------------------ #
    #  Header construction
    # ------------------------------------------------------------------ #

    def _prepare_headers(self) -> Dict[str, str]:
        """Build request headers for the Akto HTTP proxy endpoint."""
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.akto_api_key}",
        }
        return headers

    # ------------------------------------------------------------------ #
    #  Query params construction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_query_params(
        *, guardrails: bool, ingest_data: bool
    ) -> Dict[str, str]:
        """Build query parameters for the Akto HTTP proxy endpoint."""
        params: Dict[str, str] = {"akto_connector": AKTO_CONNECTOR_NAME}
        if guardrails:
            params["guardrails"] = "true"
        if ingest_data:
            params["ingest_data"] = "true"
        return params

    # ------------------------------------------------------------------ #
    #  Payload construction — Akto data-ingestion format
    # ------------------------------------------------------------------ #

    def _build_request_headers(self, request_data: dict) -> Dict[str, str]:
        """
        Extract and build request headers from proxy context.

        Returns a dict of sanitized request headers to send to Akto.
        """
        headers: Dict[str, str] = {"content-type": "application/json"}

        proxy_req = request_data.get("proxy_server_request", {})
        if proxy_req.get("headers") and isinstance(proxy_req["headers"], dict):
            for key, val in proxy_req["headers"].items():
                if key and val:
                    headers[str(key).lower()] = str(val)

        return headers

    def _build_request_body(
        self,
        inputs: GenericGuardrailAPIInputs,
    ) -> Dict[str, Any]:
        """
        Build the request body from guardrail inputs.

        Reconstructs the LLM request payload shape.
        """
        model = inputs.get("model", "") or ""
        body: Dict[str, Any] = {"model": model}

        # Prefer structured_messages if available (preserves role info)
        structured = inputs.get("structured_messages")
        if structured:
            body["messages"] = structured
        else:
            texts = inputs.get("texts", [])
            if texts:
                body["messages"] = [{"role": "user", "content": t} for t in texts]
            else:
                body["messages"] = []

        tools = inputs.get("tools")
        if tools:
            body["tools"] = tools

        tool_calls = inputs.get("tool_calls")
        if tool_calls:
            body["tool_calls"] = tool_calls

        return body

    def _build_response_body(
        self,
        inputs: GenericGuardrailAPIInputs,
    ) -> Dict[str, Any]:
        """Build a response body from post-call guardrail inputs."""
        texts = inputs.get("texts", [])
        if texts:
            return {
                "choices": [
                    {"message": {"content": t, "role": "assistant"}} for t in texts
                ]
            }
        return {}

    def _build_tag_metadata(self, request_data: dict) -> Dict[str, str]:
        """
        Build tags and metadata for Akto ingestion.

        Includes gen-ai marker and any LiteLLM auth context.
        """
        tag: Dict[str, str] = {"gen-ai": "Gen AI"}

        user_id = self._resolve_metadata_value(request_data, "user_api_key_user_id")
        team_id = self._resolve_metadata_value(request_data, "user_api_key_team_id")
        if user_id:
            tag["user_id"] = user_id
        if team_id:
            tag["team_id"] = team_id

        return tag

    def _build_akto_payload(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        *,
        status_code: int = 200,
        include_response: bool = False,
        response_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build the full payload in Akto data-ingestion format.

        This matches the exact schema expected by the Akto Data Ingestion Service:
        {
            "path", "requestHeaders", "responseHeaders",
            "method", "requestPayload", "responsePayload",
            "ip", "destIp", "time", "statusCode", "status",
            "akto_account_id", "akto_vxlan_id", "is_pending",
            "source", "tag", "metadata", "contextSource"
        }
        """
        request_path = self._extract_request_path(request_data)
        request_headers = self._build_request_headers(request_data)
        request_body = self._build_request_body(inputs)
        tag = self._build_tag_metadata(request_data)

        request_payload = json.dumps(request_body)

        response_payload = ""
        response_headers: Dict[str, str] = {}
        if response_override is not None:
            response_payload = json.dumps(response_override)
            response_headers = {"content-type": "application/json"}
        elif include_response:
            response_body = self._build_response_body(inputs)
            response_payload = json.dumps(response_body)
            response_headers = {"content-type": "application/json"}

        # Use actual client IP from proxy headers, fall back to empty string
        ip = ""
        proxy_req = request_data.get("proxy_server_request", {})
        proxy_headers = proxy_req.get("headers", {}) if isinstance(proxy_req, dict) else {}
        if isinstance(proxy_headers, dict):
            ip = (
                proxy_headers.get("x-forwarded-for")
                or proxy_headers.get("x-real-ip")
                or ""
            )
            # X-Forwarded-For may contain multiple IPs; use the first (client) one
            if "," in ip:
                ip = ip.split(",")[0].strip()

        return {
            "path": request_path,
            "requestHeaders": json.dumps(request_headers),
            "responseHeaders": json.dumps(response_headers),
            "method": "POST",
            "requestPayload": request_payload,
            "responsePayload": response_payload,
            "ip": ip,
            "destIp": "127.0.0.1",
            "time": str(int(datetime.now().timestamp() * 1000)),
            "statusCode": str(status_code),
            "type": None,
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

    # ------------------------------------------------------------------ #
    #  HTTP send
    # ------------------------------------------------------------------ #

    async def _send_request(
        self,
        *,
        guardrails: bool,
        ingest_data: bool,
        payload: dict,
    ) -> httpx.Response:
        """POST to the Akto HTTP proxy endpoint."""
        endpoint = f"{self.akto_base_url}{HTTP_PROXY_PATH}"
        params = self._build_query_params(
            guardrails=guardrails, ingest_data=ingest_data
        )
        headers = self._prepare_headers()

        return await self.async_handler.post(
            url=endpoint,
            json=payload,
            params=params,
            headers=headers,
            timeout=self.guardrail_timeout,
        )

    # ------------------------------------------------------------------ #
    #  Response handling
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_guardrails_result(result: Any) -> Tuple[bool, str]:
        """
        Parse the guardrails result from an Akto HTTP proxy response.

        Returns:
            (allowed: bool, reason: str)
        """
        if not isinstance(result, dict):
            return True, ""
        guardrails_result = (
            result.get("data", {}).get("guardrailsResult", {}) or {}
        )
        return guardrails_result.get("Allowed", True), guardrails_result.get(
            "Reason", ""
        )

    def _handle_guardrail_response(
        self, response: httpx.Response
    ) -> Tuple[bool, str]:
        """
        Handle the HTTP response from Akto guardrail API.

        Returns:
            (allowed: bool, reason: str)
        """
        if response.status_code != 200:
            verbose_proxy_logger.error(
                "Akto guardrail returned HTTP %d", response.status_code
            )
            return True, ""

        response_json = response.json()
        verbose_proxy_logger.debug("Akto guardrail response: %s", response_json)
        return self._parse_guardrails_result(response_json)

    # ------------------------------------------------------------------ #
    #  Fail-open / fail-closed
    # ------------------------------------------------------------------ #

    def _handle_unreachable(
        self,
        inputs: GenericGuardrailAPIInputs,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"],
        error: Exception,
        http_status_code: Optional[int] = None,
    ) -> GenericGuardrailAPIInputs:
        """Handle unreachable guardrail endpoint based on fail-open/fail-closed config."""
        if self.unreachable_fallback == "fail_open":
            status_suffix = (
                f" http_status_code={http_status_code}" if http_status_code else ""
            )
            verbose_proxy_logger.critical(
                "Akto guardrail unreachable (fail-open). Proceeding without guardrail.%s "
                "guardrail_name=%s akto_base_url=%s input_type=%s",
                status_suffix,
                getattr(self, "guardrail_name", None),
                self.akto_base_url,
                input_type,
                exc_info=error,
            )
            return dict(inputs)  # type: ignore[return-value]

        verbose_proxy_logger.error("Akto guardrail unreachable (fail-closed): %s", str(error))
        raise HTTPException(
            status_code=503,
            detail=f"Akto guardrail service unreachable: {str(error)}",
        )

    # ------------------------------------------------------------------ #
    #  Observability
    # ------------------------------------------------------------------ #

    def _add_guardrail_observability(
        self,
        request_data: dict,
        start_time: datetime,
        guardrail_status: GuardrailStatus,
        guardrail_json_response: Any,
    ) -> None:
        """Add standard logging guardrail information to the request data."""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_provider="akto",
            guardrail_json_response=guardrail_json_response,
            request_data=request_data,
            guardrail_status=guardrail_status,
            start_time=start_time.timestamp(),
            end_time=end_time.timestamp(),
            duration=duration,
        )

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Apply Akto guardrail to the given inputs.

        SYNC mode (blocking):
          request:  guardrails check → block if Allowed=false + ingest blocked details
          response: ingest request+response for observability

        ASYNC mode (non-blocking):
          response only: single call with guardrails+ingest_data (log-only)
        """
        start_time = datetime.now()
        guardrail_status: GuardrailStatus = "success"
        guardrail_json_response: Any = {}

        try:
            if self.sync_mode:
                if input_type == "request":
                    # Sync pre-call: POST ?akto_connector&guardrails=true
                    payload = self._build_akto_payload(
                        inputs, request_data, include_response=False
                    )
                    
                    try:
                        response = await self._send_request(
                            guardrails=True, ingest_data=False, payload=payload
                        )
                        allowed, reason = self._handle_guardrail_response(response)

                        if not allowed:
                            # Ingest the blocked request for observability
                            await self._ingest_blocked_request(
                                inputs=inputs,
                                request_data=request_data,
                                reason=reason,
                            )
                            raise HTTPException(
                                status_code=400,
                                detail=reason or "Blocked by Akto Guardrails",
                            )
                    except HTTPException:
                        raise
                    except (Timeout, httpx.RequestError) as e:
                        return self._handle_unreachable(
                            inputs=inputs,
                            input_type=input_type,
                            logging_obj=logging_obj,
                            error=e,
                        )
                    except httpx.HTTPStatusError as e:
                        status_code = getattr(
                            getattr(e, "response", None), "status_code", None
                        )
                        return self._handle_unreachable(
                            inputs=inputs,
                            input_type=input_type,
                            logging_obj=logging_obj,
                            error=e,
                            http_status_code=status_code,
                        )

                else:
                    # Sync post-call: POST ?akto_connector&ingest_data=true
                    payload = self._build_akto_payload(
                        inputs, request_data, include_response=True
                    )
                    try:
                        response = await self._send_request(
                            guardrails=False, ingest_data=True, payload=payload
                        )
                        if response.status_code != 200:
                            verbose_proxy_logger.error(
                                "Akto guardrail: ingestion returned HTTP %d", response.status_code
                            )
                    except Exception as e:
                        verbose_proxy_logger.error(
                            "Akto guardrail: post-call ingestion error: %s", str(e)
                        )
            else:
                # Async mode: only response
                if input_type == "response":
                    payload = self._build_akto_payload(
                        inputs, request_data, include_response=True
                    )
                    try:
                        response = await self._send_request(
                            guardrails=True, ingest_data=True, payload=payload
                        )
                        if response.status_code == 200:
                            allowed, reason = self._handle_guardrail_response(response)
                            if not allowed:
                                verbose_proxy_logger.info(
                                    "Akto guardrail: response flagged (async mode, logged only): %s",
                                    reason,
                                )
                    except Exception as e:
                        verbose_proxy_logger.error(
                            "Akto guardrail: async post-call error: %s", str(e)
                        )

            return inputs

        except HTTPException:
            guardrail_status = "guardrail_intervened"
            raise
        except Exception as e:
            guardrail_status = "guardrail_failed_to_respond"
            guardrail_json_response = str(e)
            verbose_proxy_logger.error("Akto guardrail request error: %s", str(e))
            raise
        finally:
            self._add_guardrail_observability(
                request_data=request_data,
                start_time=start_time,
                guardrail_status=guardrail_status,
                guardrail_json_response=guardrail_json_response,
            )

    # ------------------------------------------------------------------ #
    #  Ingest blocked request
    # ------------------------------------------------------------------ #

    async def _ingest_blocked_request(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        reason: str,
    ) -> None:
        """Ingest a blocked request into Akto for observability."""
        blocked_response = {"x-blocked-by": "Akto Guardrails", "reason": reason}
        payload = self._build_akto_payload(
            inputs,
            request_data,
            status_code=403,
            include_response=False,
            response_override=blocked_response,
        )

        try:
            await self._send_request(
                guardrails=False, ingest_data=True, payload=payload
            )
        except Exception as e:
            verbose_proxy_logger.error(
                "Akto guardrail: failed to ingest blocked request: %s", str(e)
            )
