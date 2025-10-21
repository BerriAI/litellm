"""Gray Swan Cygnal guardrail integration."""

import os
from typing import Any, Dict, Literal, Optional, Union

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import LLMResponseTypes


class GraySwanGuardrailMissingSecrets(Exception):
    """Raised when the Gray Swan API key is missing."""


class GraySwanGuardrailAPIError(Exception):
    """Raised when the Gray Swan API returns an error."""


class GraySwanGuardrail(CustomGuardrail):
    """
    Guardrail that calls Gray Swan's Cygnal monitoring endpoint.

    see: https://docs.grayswan.ai/cygnal/monitor-requests
    """

    SUPPORTED_ON_FLAGGED_ACTIONS = {"block", "monitor"}
    DEFAULT_ON_FLAGGED_ACTION = "monitor"
    BASE_API_URL = "https://api.grayswan.ai"
    MONITOR_PATH = "/cygnal/monitor"
    SUPPORTED_REASONING_MODES = {"off", "hybrid", "thinking"}

    def __init__(
        self,
        guardrail_name: Optional[str] = "grayswan",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        on_flagged_action: Optional[str] = None,
        violation_threshold: Optional[float] = None,
        reasoning_mode: Optional[str] = None,
        categories: Optional[Dict[str, str]] = None,
        policy_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        api_key_value = api_key or os.getenv("GRAYSWAN_API_KEY")
        if not api_key_value:
            raise GraySwanGuardrailMissingSecrets(
                "Gray Swan API key missing. Set `GRAYSWAN_API_KEY` or pass `api_key`."
            )
        self.api_key: str = api_key_value

        base = api_base or os.getenv("GRAYSWAN_API_BASE") or self.BASE_API_URL
        self.api_base = base.rstrip("/")
        self.monitor_url = f"{self.api_base}{self.MONITOR_PATH}"

        action = on_flagged_action
        if action and action.lower() in self.SUPPORTED_ON_FLAGGED_ACTIONS:
            self.on_flagged_action = action.lower()
        else:
            if action:
                verbose_proxy_logger.warning(
                    "Gray Swan Guardrail: Unsupported on_flagged_action '%s', defaulting to '%s'.",
                    action,
                    self.DEFAULT_ON_FLAGGED_ACTION,
                )
            self.on_flagged_action = self.DEFAULT_ON_FLAGGED_ACTION

        self.violation_threshold = self._resolve_threshold(violation_threshold)
        self.reasoning_mode = self._resolve_reasoning_mode(reasoning_mode)
        self.categories = categories
        self.policy_id = policy_id

        supported_event_hooks = [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
        ]

        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=supported_event_hooks,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Guardrail hook entry points
    # ------------------------------------------------------------------

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
            "mcp_call",
            "anthropic_messages",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        if (
            self.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.pre_call
            )
            is not True
        ):
            return data

        verbose_proxy_logger.debug("Gray Swan Guardrail: pre-call hook triggered")

        messages = data.get("messages")
        if not messages:
            verbose_proxy_logger.debug("Gray Swan Guardrail: No messages in data")
            return data

        dynamic_body = self.get_guardrail_dynamic_request_body_params(data) or {}

        payload = self._prepare_payload(messages, dynamic_body)
        if payload is None:
            verbose_proxy_logger.debug(
                "Gray Swan Guardrail: no content to scan; skipping request"
            )
            return data

        await self.run_grayswan_guardrail(payload)
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return data

    @log_guardrail_information
    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "responses",
            "mcp_call",
            "anthropic_messages",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        if (
            self.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.during_call
            )
            is not True
        ):
            return data

        verbose_proxy_logger.debug("GraySwan Guardrail: during-call hook triggered")

        messages = data.get("messages")
        if not messages:
            verbose_proxy_logger.debug("Gray Swan Guardrail: No messages in data")
            return data

        dynamic_body = self.get_guardrail_dynamic_request_body_params(data) or {}

        payload = self._prepare_payload(messages, dynamic_body)
        if payload is None:
            verbose_proxy_logger.debug(
                "Gray Swan Guardrail: no content to scan; skipping request"
            )
            return data

        await self.run_grayswan_guardrail(payload)
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return data

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: LLMResponseTypes,
    ) -> LLMResponseTypes:
        if (
            self.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.post_call
            )
            is not True
        ):
            return response

        verbose_proxy_logger.debug("GraySwan Guardrail: post-call hook triggered")

        response_dict = response.model_dump() if hasattr(response, "model_dump") else {}
        response_messages = [
            msg if isinstance(msg, dict) else msg.model_dump()
            for choice in response_dict.get("choices", [])
            if isinstance(choice, dict)
            for msg in [choice.get("message")]
            if msg is not None
        ]

        if not response_messages:
            verbose_proxy_logger.debug(
                "Gray Swan Guardrail: no response messages detected; skipping post-call scan"
            )
            return response

        dynamic_body = self.get_guardrail_dynamic_request_body_params(data) or {}

        payload = self._prepare_payload(response_messages, dynamic_body)
        if payload is None:
            verbose_proxy_logger.debug(
                "Gray Swan Guardrail: no content to scan; skipping request"
            )
            return response

        await self.run_grayswan_guardrail(payload)
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return response

    # ------------------------------------------------------------------
    # Core GraySwan interaction
    # ------------------------------------------------------------------

    async def run_grayswan_guardrail(self, payload: dict):
        headers = self._prepare_headers()

        try:
            response = await self.async_handler.post(
                url=self.monitor_url,
                headers=headers,
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()
            verbose_proxy_logger.debug(
                "Gray Swan Guardrail: monitor response %s", safe_dumps(result)
            )
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - depends on HTTP client behaviour
            verbose_proxy_logger.exception(
                "Gray Swan Guardrail: API request failed: %s", exc
            )
            raise GraySwanGuardrailAPIError(str(exc)) from exc

        self._process_grayswan_response(result)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "grayswan-api-key": self.api_key,
        }

    def _prepare_payload(
        self, messages: list[dict], dynamic_body: dict
    ) -> Optional[Dict[str, Any]]:
        payload: Dict[str, Any] = {}
        payload["messages"] = messages

        categories = dynamic_body.get("categories") or self.categories
        if categories:
            payload["categories"] = categories

        policy_id = dynamic_body.get("policy_id") or self.policy_id
        if policy_id:
            payload["policy_id"] = policy_id

        reasoning_mode = dynamic_body.get("reasoning_mode") or self.reasoning_mode
        if reasoning_mode:
            payload["reasoning_mode"] = reasoning_mode

        return payload

    def _process_grayswan_response(self, response_json: Dict[str, Any]) -> None:
        violation_score = float(response_json.get("violation", 0.0) or 0.0)
        violated_rules = response_json.get("violated_rules", [])
        mutation_detected = response_json.get("mutation")
        ipi_detected = response_json.get("ipi")

        flagged = violation_score >= self.violation_threshold
        if not flagged:
            verbose_proxy_logger.debug(
                "Gray Swan Guardrail: request passed (score=%s, rules=%s)",
                violation_score,
                violated_rules,
            )
            return

        verbose_proxy_logger.warning(
            "Gray Swan Guardrail: violation score %.3f exceeds threshold %.3f",
            violation_score,
            self.violation_threshold,
        )

        if self.on_flagged_action == "block":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Blocked by Gray Swan Guardrail",
                    "violation": violation_score,
                    "violated_rules": violated_rules,
                    "mutation": mutation_detected,
                    "ipi": ipi_detected,
                },
            )

    def _resolve_threshold(self, threshold: Optional[float]) -> float:
        if threshold is not None:
            return min(max(threshold, 0.0), 1.0)
        return 0.5

    def _resolve_reasoning_mode(self, candidate: Optional[str]) -> Optional[str]:
        if candidate is None:
            return None
        normalised = candidate.strip().lower()
        if normalised in self.SUPPORTED_REASONING_MODES:
            return normalised
        verbose_proxy_logger.warning(
            "Gray Swan Guardrail: ignoring unsupported reasoning_mode '%s'",
            candidate,
        )
        return None

    @staticmethod
    def get_config_model():
        from litellm.types.proxy.guardrails.guardrail_hooks.grayswan import (
            GraySwanGuardrailConfigModel,
        )

        return GraySwanGuardrailConfigModel
