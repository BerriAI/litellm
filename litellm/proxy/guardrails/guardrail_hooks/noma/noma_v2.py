# +-------------------------------------------------------------+
#
#      Noma Security V2 Guardrail Integration for LiteLLM
#
# +-------------------------------------------------------------+

import enum
import json
import os
from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Optional, Type, cast
from urllib.parse import urlparse

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail, log_guardrail_information
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy.guardrails.guardrail_hooks.noma.noma import NomaBlockedMessage
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs, GuardrailStatus

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


_DEFAULT_API_BASE = "https://api.noma.security/"
_AIDR_SCAN_ENDPOINT = "/litellm/guardrail"
_INTERVENED_INPUT_FIELDS = ("texts", "images", "tools", "tool_calls")
_DEFAULT_API_BASE_HOSTNAME = urlparse(_DEFAULT_API_BASE).hostname


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
        monitor_mode: Optional[bool] = None,
        block_failures: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)

        self.api_key = api_key or os.environ.get("NOMA_API_KEY")
        self.api_base = (api_base or os.environ.get("NOMA_API_BASE") or _DEFAULT_API_BASE).rstrip("/")
        self.application_id = application_id or os.environ.get("NOMA_APPLICATION_ID")
        if monitor_mode is None:
            self.monitor_mode = os.environ.get("NOMA_MONITOR_MODE", "false").lower() == "true"
        else:
            self.monitor_mode = monitor_mode

        if block_failures is None:
            self.block_failures = os.environ.get("NOMA_BLOCK_FAILURES", "true").lower() == "true"
        else:
            self.block_failures = block_failures

        if self._requires_api_key(api_base=self.api_base) and not self.api_key:
            raise ValueError("Noma v2 guardrail requires api_key when using Noma SaaS endpoint")

        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.during_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.pre_mcp_call,
                GuardrailEventHooks.during_mcp_call,
            ]

        super().__init__(**kwargs)

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.noma import (
            NomaV2GuardrailConfigModel,
        )

        return NomaV2GuardrailConfigModel

    def _get_authorization_header(self) -> str:
        if not self.api_key:
            return ""
        return f"Bearer {self.api_key}"

    @staticmethod
    def _requires_api_key(api_base: str) -> bool:
        parsed = urlparse(api_base)
        return parsed.hostname == _DEFAULT_API_BASE_HOSTNAME

    @staticmethod
    def _get_non_empty_str(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        return stripped or None

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

    def _build_scan_payload(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"],
        application_id: Optional[str],
    ) -> dict:
        payload_request_data = deepcopy(request_data)
        if logging_obj is not None:
            payload_request_data["litellm_logging_obj"] = getattr(logging_obj, "model_call_details", None)

        payload: dict[str, Any] = {
            "inputs": inputs,
            "request_data": payload_request_data,
            "input_type": input_type,
            "monitor_mode": self.monitor_mode,
        }
        if application_id:
            payload["application_id"] = application_id
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
        if safe_payload == {} and payload:
            verbose_proxy_logger.warning(
                "Noma v2 guardrail: payload serialization failed, falling back to empty payload"
            )

        if isinstance(safe_payload, dict):
            return safe_payload

        verbose_proxy_logger.warning(
            "Noma v2 guardrail: payload sanitization produced non-dict output (type=%s), falling back to empty payload",
            type(safe_payload).__name__,
        )
        return {}

    async def _call_noma_scan(
        self,
        payload: dict,
    ) -> dict:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        authorization_header = self._get_authorization_header()
        if authorization_header:
            headers["Authorization"] = authorization_header

        endpoint = f"{self.api_base}{_AIDR_SCAN_ENDPOINT}"
        sanitized_payload = self._sanitize_payload_for_transport(payload)
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
        response.raise_for_status()
        response_json = response.json()
        verbose_proxy_logger.debug(
            "Noma v2 AIDR response parsed: %s",
            json.dumps(response_json, default=str),
        )
        return response_json

    def _add_guardrail_observability(
        self,
        request_data: dict,
        start_time: datetime,
        guardrail_status: GuardrailStatus,
        guardrail_json_response: Any,
    ) -> None:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_provider="noma_v2",
            guardrail_json_response=guardrail_json_response,
            request_data=request_data,
            guardrail_status=guardrail_status,
            start_time=start_time.timestamp(),
            end_time=end_time.timestamp(),
            duration=duration,
        )

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
            for field in _INTERVENED_INPUT_FIELDS:
                value = response_json.get(field)
                if isinstance(value, list):
                    updated_inputs[field] = value  # type: ignore[literal-required]
            return updated_inputs

        return inputs

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        start_time = datetime.now()
        guardrail_status: GuardrailStatus = "success"
        guardrail_json_response: Any = {}
        dynamic_params = self.get_guardrail_dynamic_request_body_params(request_data)
        if not isinstance(dynamic_params, dict):
            dynamic_params = {}
        response_json: Optional[dict] = None

        # Per-request dynamic params can override configured application context.
        application_id = self._get_non_empty_str(dynamic_params.get("application_id"))

        if application_id is None:
            application_id = self._get_non_empty_str(self.application_id)

        try:
            payload = self._build_scan_payload(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                logging_obj=logging_obj,
                application_id=application_id,
            )

            response_json = await self._call_noma_scan(payload=payload)
            if self.monitor_mode:
                action = _Action.NONE
            else:
                action = self._resolve_action_from_response(response_json=response_json)
            guardrail_json_response = response_json
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

            guardrail_status = "success" if action == _Action.NONE else "guardrail_intervened"
            return processed_inputs

        except NomaBlockedMessage as e:
            guardrail_status = "guardrail_intervened"
            guardrail_json_response = (
                response_json if isinstance(response_json, dict) else getattr(e, "detail", {"error": "blocked"})
            )
            raise
        except Exception as e:
            guardrail_status = "guardrail_failed_to_respond"
            guardrail_json_response = str(e)
            verbose_proxy_logger.error("Noma v2 guardrail failed: %s", str(e))
            if self.block_failures:
                raise
            return inputs
        finally:
            self._add_guardrail_observability(
                request_data=request_data,
                start_time=start_time,
                guardrail_status=guardrail_status,
                guardrail_json_response=guardrail_json_response,
            )
