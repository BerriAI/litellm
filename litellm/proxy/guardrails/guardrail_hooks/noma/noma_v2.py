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
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Type, cast
from urllib.parse import urljoin

from litellm._logging import verbose_proxy_logger
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
)
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy.guardrails.guardrail_hooks.noma.noma import NomaBlockedMessage
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


_REQUEST_ROLE = "user"
_RESPONSE_ROLE = "assistant"
_DEFAULT_API_BASE = "https://api.noma.security/"
_AIDR_SCAN_ENDPOINT = "/ai-dr/v2/prompt/scan"
_DEFAULT_APPLICATION_ID = "litellm"
_DEFAULT_TOKEN_TTL_SECONDS = 300
_TOKEN_REFRESH_BUFFER_SECONDS = 30


class _Action(str, enum.Enum):
    ALLOW = "allow"
    MASK = "mask"
    BLOCK = "block"


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
        self._responses_transform_handler = LiteLLMResponsesTransformationHandler()

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

    def _extract_anonymized_content_for_role(
        self, response_json: dict, role: str
    ) -> List[str]:
        anonymized_contents: List[str] = []
        scan_result = response_json.get("scanResult")
        if not isinstance(scan_result, list):
            return anonymized_contents

        for item in scan_result:
            if not isinstance(item, dict) or item.get("role") != role:
                continue
            results = item.get("results", {})
            if not isinstance(results, dict):
                continue
            anonymized = results.get("anonymizedContent", {})
            if not isinstance(anonymized, dict):
                continue
            anonymized_text = anonymized.get("anonymized")
            if isinstance(anonymized_text, str) and anonymized_text:
                anonymized_contents.append(anonymized_text)
        return anonymized_contents

    def _extract_text_roles(
        self,
        inputs: GenericGuardrailAPIInputs,
        input_type: Literal["request", "response"],
    ) -> List[str]:
        default_role = _RESPONSE_ROLE if input_type == "response" else _REQUEST_ROLE
        structured_messages = inputs.get("structured_messages") or []
        if not structured_messages:
            return [default_role] * len(inputs.get("texts", []))

        roles: List[str] = []
        for message in structured_messages:
            if not isinstance(message, dict):
                continue
            role_value = message.get("role")
            role: str = role_value if isinstance(role_value, str) else default_role
            content = message.get("content")
            if isinstance(content, str):
                roles.append(role)
            elif isinstance(content, list):
                for content_item in content:
                    if isinstance(content_item, dict) and isinstance(
                        content_item.get("text"), str
                    ):
                        roles.append(role)
        return roles

    def _apply_anonymization_to_texts(
        self,
        inputs: GenericGuardrailAPIInputs,
        input_type: Literal["request", "response"],
        response_json: dict,
    ) -> GenericGuardrailAPIInputs:
        texts = inputs.get("texts", [])
        if not texts:
            raise NomaBlockedMessage(response_json.get("scanResult", response_json))

        target_role = _RESPONSE_ROLE if input_type == "response" else _REQUEST_ROLE
        anonymized_values = self._extract_anonymized_content_for_role(
            response_json=response_json, role=target_role
        )
        if not anonymized_values:
            raise NomaBlockedMessage(response_json.get("scanResult", response_json))

        text_roles = self._extract_text_roles(inputs=inputs, input_type=input_type)
        updated_texts = list(texts)
        anonymized_index = 0

        for idx, _text in enumerate(updated_texts):
            role_for_idx = text_roles[idx] if idx < len(text_roles) else target_role
            if role_for_idx != target_role:
                continue
            replacement_idx = min(anonymized_index, len(anonymized_values) - 1)
            updated_texts[idx] = anonymized_values[replacement_idx]
            anonymized_index += 1

        if anonymized_index == 0:
            # Best-effort fallback: no role-matched entry was found, so apply to the last text.
            updated_texts[-1] = anonymized_values[0]

        updated_inputs = cast(GenericGuardrailAPIInputs, dict(inputs))
        updated_inputs["texts"] = updated_texts
        return updated_inputs

    def _resolve_action_from_response(
        self,
        response_json: dict,
    ) -> _Action:
        aggregated_action = response_json.get("aggregatedAction")
        if isinstance(aggregated_action, str):
            try:
                return _Action(aggregated_action.lower())
            except ValueError:
                pass

        raise ValueError("Noma response missing aggregatedAction")

    @staticmethod
    def _status_for_action(action: _Action) -> str:
        if action == _Action.ALLOW:
            return "success"
        return "guardrail_intervened"

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

    def _build_input_items_from_structured_messages(
        self, structured_messages: List[Any]
    ) -> List[dict]:
        (
            input_items,
            instructions,
        ) = self._responses_transform_handler.convert_chat_completion_messages_to_responses_api(
            structured_messages
        )
        if instructions:
            input_items.insert(
                0,
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": instructions}],
                },
            )
        return input_items

    def _build_input_items_from_generic_inputs(
        self,
        inputs: GenericGuardrailAPIInputs,
        input_type: Literal["request", "response"],
    ) -> List[dict]:
        role = _RESPONSE_ROLE if input_type == "response" else _REQUEST_ROLE
        content_items: List[dict] = []

        for text in inputs.get("texts", []) or []:
            content_items.append({"type": "input_text", "text": text})

        for image in inputs.get("images", []) or []:
            content_items.append({"type": "input_image", "image_url": image})

        tool_calls = inputs.get("tool_calls") or []
        for tool_call in tool_calls:
            content_items.append(
                {
                    "type": "input_text",
                    "text": json.dumps(tool_call, default=str),
                }
            )

        if not content_items:
            return []

        return [{"type": "message", "role": role, "content": content_items}]

    def _build_scan_payload(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"],
        dynamic_params: dict,
    ) -> dict:
        structured_messages = inputs.get("structured_messages")
        if isinstance(structured_messages, list) and structured_messages:
            input_items = self._build_input_items_from_structured_messages(
                structured_messages=structured_messages
            )
        else:
            input_items = self._build_input_items_from_generic_inputs(
                inputs=inputs, input_type=input_type
            )

        payload: Dict[str, Any] = {"input": input_items}
        if inputs.get("model"):
            payload["model"] = inputs.get("model")

        payload["x-noma-context"] = self._build_noma_context(
            request_data=request_data,
            logging_obj=logging_obj,
            dynamic_params=dynamic_params,
        )
        return payload

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
        verbose_proxy_logger.debug(
            "Noma v2 AIDR request: endpoint=%s headers=%s payload=%s",
            endpoint,
            json.dumps(headers_for_log, default=str),
            json.dumps(payload, default=str),
        )
        response = await self.async_handler.post(
            url=endpoint,
            headers=headers,
            json=payload,
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
        input_type: Literal["request", "response"],
        response_json: dict,
        action: _Action,
    ) -> GenericGuardrailAPIInputs:
        if action == _Action.BLOCK:
            raise NomaBlockedMessage(response_json.get("scanResult", response_json))

        if action == _Action.MASK:
            return self._apply_anonymization_to_texts(
                inputs=inputs,
                input_type=input_type,
                response_json=response_json,
            )

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
                input_type=input_type,
                response_json=response_json,
                action=action,
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider="noma",
                guardrail_json_response=response_json,
                request_data=request_data,
                guardrail_status=self._status_for_action(action),  # type: ignore[arg-type]
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
