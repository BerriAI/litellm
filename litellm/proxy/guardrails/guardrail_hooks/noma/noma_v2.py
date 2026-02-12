# +-------------------------------------------------------------+
#
#      Noma Security V2 Guardrail Integration for LiteLLM
#
# +-------------------------------------------------------------+

import asyncio
import enum
import json
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Type, cast
from urllib.parse import urljoin

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy.guardrails.guardrail_hooks.noma.noma import NomaBlockedMessage
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


_DEFAULT_API_BASE = "https://api.noma.security/"
_AIDR_SCAN_ENDPOINT = "/ai-dr/v2/litellm/guardrails/scan"
_DEFAULT_APPLICATION_ID = "litellm"
_DEFAULT_TOKEN_TTL_SECONDS = 300
_TOKEN_REFRESH_BUFFER_SECONDS = 30
_INTERVENED_INPUT_FIELDS = ("texts", "images", "tools")


class _Action(str, enum.Enum):
    BLOCKED = "BLOCKED"
    NONE = "NONE"
    GUARDRAIL_INTERVENED = "GUARDRAIL_INTERVENED"


class NomaV2Guardrail(CustomGuardrail):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        application_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        token_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        self.api_key = api_key or os.environ.get("NOMA_API_KEY")
        self.api_base = (
            api_base or os.environ.get("NOMA_API_BASE") or _DEFAULT_API_BASE
        ).rstrip("/")
        self.application_id = application_id or os.environ.get("NOMA_APPLICATION_ID")

        self.client_id = client_id or os.environ.get("NOMA_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("NOMA_CLIENT_SECRET")
        self.token_url = token_url or os.environ.get("NOMA_TOKEN_URL")

        self._oauth_access_token: Optional[str] = None
        self._oauth_access_token_expiry_epoch: float = 0.0
        self._oauth_lock = asyncio.Lock()

        if not self.api_key and not (self.client_id and self.client_secret):
            raise ValueError(
                "Noma v2 guardrail requires either api_key or client_id+client_secret"
            )

        if "supported_event_hooks" not in kwargs:
            from litellm.types.guardrails import GuardrailEventHooks

            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.during_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.pre_mcp_call,
                GuardrailEventHooks.during_mcp_call,
            ]

        super().__init__(**kwargs)

    def _resolve_token_url(self) -> str:
        if self.token_url:
            return self.token_url
        return f"{self.api_base}/auth"

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.noma import (
            NomaV2GuardrailConfigModel,
        )

        return NomaV2GuardrailConfigModel

    async def _get_authorization_header(self) -> str:
        if self.api_key:
            return f"Bearer {self.api_key}"
        token = await self._get_oauth_access_token()
        return f"Bearer {token}"

    async def _get_oauth_access_token(self) -> str:
        now = time.time()
        if (
            self._oauth_access_token
            and now
            < self._oauth_access_token_expiry_epoch - _TOKEN_REFRESH_BUFFER_SECONDS
        ):
            return self._oauth_access_token

        async with self._oauth_lock:
            now = time.time()
            if (
                self._oauth_access_token
                and now
                < self._oauth_access_token_expiry_epoch - _TOKEN_REFRESH_BUFFER_SECONDS
            ):
                return self._oauth_access_token

            token_url = self._resolve_token_url()
            verbose_proxy_logger.debug(
                "Noma v2 OAuth token request: url=%s client_id_present=%s",
                token_url,
                bool(self.client_id),
            )
            response = await self.async_handler.post(
                url=token_url,
                headers={"Content-Type": "application/json"},
                json={"clientId": self.client_id, "secret": self.client_secret},
            )
            verbose_proxy_logger.debug(
                "Noma v2 OAuth token response: status_code=%s",
                response.status_code,
            )
            response.raise_for_status()
            token_response = response.json()

            access_token = token_response.get("accessToken")
            if not isinstance(access_token, str) or not access_token:
                raise ValueError("Noma OAuth response missing accessToken")

            expires_in = token_response.get("expiresIn") or token_response.get(
                "expires_in"
            )
            if not isinstance(expires_in, int) or expires_in <= 0:
                expires_in = _DEFAULT_TOKEN_TTL_SECONDS

            self._oauth_access_token = access_token
            self._oauth_access_token_expiry_epoch = time.time() + expires_in
            verbose_proxy_logger.debug(
                "Noma v2 OAuth token cached: expires_in=%s", expires_in
            )
            return access_token

    def _resolve_action_from_response(
        self,
        response_json: dict,
    ) -> _Action:
        action = response_json.get("action")
        if isinstance(action, str):
            try:
                return _Action(action)
            except ValueError:
                pass

        raise ValueError("Noma v2 response missing valid action")

    def _build_noma_context(
        self,
        request_data: dict,
        logging_obj: Optional["LiteLLMLoggingObj"],
        dynamic_params: dict,
    ) -> dict:
        metadata = request_data.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        litellm_metadata = request_data.get("litellm_metadata")
        if not isinstance(litellm_metadata, dict):
            litellm_metadata = {}
        headers = metadata.get("headers", {})
        if not isinstance(headers, dict):
            headers = {}

        key_alias = litellm_metadata.get("user_api_key_alias") or metadata.get(
            "user_api_key_alias"
        )
        user_id = (
            litellm_metadata.get("user_api_key_user_email")
            or litellm_metadata.get("user_api_key_user_id")
            or metadata.get("user_api_key_user_email")
            or metadata.get("user_api_key_user_id")
        )

        call_id = None
        if logging_obj is not None:
            call_id = getattr(logging_obj, "litellm_call_id", None)
        if call_id is None:
            call_id = request_data.get("litellm_call_id")

        request_id = None
        if logging_obj is not None:
            request_id = getattr(logging_obj, "litellm_trace_id", None)

        application_id = (
            dynamic_params.get("application_id")
            or headers.get("x-noma-application-id")
            or self.application_id
            or key_alias
            or _DEFAULT_APPLICATION_ID
        )

        return {
            "applicationId": application_id,
            "ipAddress": metadata.get("requester_ip_address"),
            "userId": user_id,
            "sessionId": call_id,
            "requestId": request_id,
        }

    def _build_scan_payload(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"],
        dynamic_params: dict,
    ) -> dict:
        payload_request_data = dict(request_data)
        if logging_obj is not None:
            payload_request_data["litellm_logging_obj"] = getattr(
                logging_obj, "model_call_details", logging_obj
            )

        payload: Dict[str, Any] = {
            "inputs": inputs,
            "request_data": payload_request_data,
            "input_type": input_type,
            "dynamic_params": dynamic_params,
        }
        payload["x-noma-context"] = self._build_noma_context(
            request_data=request_data,
            logging_obj=logging_obj,
            dynamic_params=dynamic_params,
        )
        return payload

    @staticmethod
    def _sanitize_payload_for_transport(payload: dict) -> dict:
        def _default(obj: Any) -> Any:
            if hasattr(obj, "model_dump"):
                try:
                    return obj.model_dump()
                except Exception:
                    pass
            return str(obj)

        try:
            json_str = json.dumps(payload, default=_default)
        except (ValueError, TypeError):
            json_str = safe_dumps(payload)

        safe_payload = safe_json_loads(json_str, default={})
        if isinstance(safe_payload, dict):
            return safe_payload
        return {}

    async def _call_noma_scan(
        self,
        payload: dict,
        request_data: dict,
        logging_obj: Optional["LiteLLMLoggingObj"],
    ) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": await self._get_authorization_header(),
        }

        call_id = None
        if logging_obj is not None:
            call_id = getattr(logging_obj, "litellm_call_id", None)
        if call_id is None:
            call_id = request_data.get("litellm_call_id")
        if call_id:
            headers["X-Noma-Request-ID"] = call_id

        endpoint = urljoin(f"{self.api_base}/", _AIDR_SCAN_ENDPOINT.lstrip("/"))
        headers_for_log = {
            key: ("<redacted>" if key.lower() == "authorization" else value)
            for key, value in headers.items()
        }
        sanitized_payload = self._sanitize_payload_for_transport(payload)
        verbose_proxy_logger.debug(
            "Noma v2 AIDR request: endpoint=%s headers=%s payload=%s",
            endpoint,
            safe_dumps(headers_for_log),
            safe_dumps(sanitized_payload),
        )
        response = await self.async_handler.post(
            url=endpoint,
            headers=headers,
            json=sanitized_payload,
        )
        verbose_proxy_logger.debug(
            "Noma v2 AIDR response: status_code=%s body=%s",
            response.status_code,
            response.text,
        )
        if response.status_code >= 400:
            verbose_proxy_logger.error(
                "Noma v2 AIDR request failed: status_code=%s body=%s",
                response.status_code,
                response.text,
            )
        response.raise_for_status()
        response_json = response.json()
        verbose_proxy_logger.debug(
            "Noma v2 AIDR response parsed: %s",
            json.dumps(response_json, default=str),
        )
        return response_json

    def _apply_action(
        self,
        inputs: GenericGuardrailAPIInputs,
        response_json: dict,
        action: _Action,
    ) -> GenericGuardrailAPIInputs:
        if action == _Action.BLOCKED:
            raise NomaBlockedMessage(response_json)

        if action == _Action.GUARDRAIL_INTERVENED:
            updated_inputs = cast(GenericGuardrailAPIInputs, dict(inputs))
            texts = response_json.get("texts")
            if isinstance(texts, list):
                updated_inputs["texts"] = texts

            images = response_json.get("images")
            if isinstance(images, list):
                updated_inputs["images"] = images

            tools = response_json.get("tools")
            if isinstance(tools, list):
                updated_inputs["tools"] = tools
            return updated_inputs

        return inputs

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        start_time = datetime.now()
        dynamic_params = self.get_guardrail_dynamic_request_body_params(request_data)

        try:
            payload = self._build_scan_payload(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                logging_obj=logging_obj,
                dynamic_params=dynamic_params,
            )

            response_json = await self._call_noma_scan(
                payload=payload, request_data=request_data, logging_obj=logging_obj
            )
            action = self._resolve_action_from_response(response_json=response_json)
            verbose_proxy_logger.debug(
                "Noma v2 guardrail decision: input_type=%s action=%s",
                input_type,
                action.value,
            )
            processed_inputs = self._apply_action(
                inputs=inputs,
                response_json=response_json,
                action=action,
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider="noma",
                guardrail_json_response=response_json,
                request_data=request_data,
                guardrail_status="success" if action == _Action.NONE else "guardrail_intervened",  # type: ignore[arg-type]
                start_time=start_time.timestamp(),
                end_time=end_time.timestamp(),
                duration=duration,
            )
            return processed_inputs

        except NomaBlockedMessage:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider="noma",
                guardrail_json_response={"error": "blocked"},
                request_data=request_data,
                guardrail_status="guardrail_intervened",
                start_time=start_time.timestamp(),
                end_time=end_time.timestamp(),
                duration=duration,
            )
            raise
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider="noma",
                guardrail_json_response=str(e),
                request_data=request_data,
                guardrail_status="guardrail_failed_to_respond",
                start_time=start_time.timestamp(),
                end_time=end_time.timestamp(),
                duration=duration,
            )
            verbose_proxy_logger.error("Noma v2 guardrail failed: %s", str(e))
            return inputs
