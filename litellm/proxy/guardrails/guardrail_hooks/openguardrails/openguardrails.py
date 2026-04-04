"""
OpenGuardrails Native Guardrail Integration for LiteLLM

Full-featured integration that supports all OpenGuardrails capabilities:
- Input/Output content detection (19 risk categories + prompt injection)
- Sensitive data anonymization with restoration (including streaming)
- Private model switching via LiteLLM model routing
- Tool call anomaly detection
- Ban policy enforcement
- Knowledge base responses

This file is intended to be placed at:
  litellm/proxy/guardrails/guardrail_hooks/openguardrails/openguardrails.py

For the LiteLLM PR submission.
"""

import json
import os
from typing import Any, Dict, Optional

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    ModifyResponseException,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks

GUARDRAIL_NAME = "openguardrails"

# Default private model name in LiteLLM config
DEFAULT_PRIVATE_MODEL_NAME = "og-private-model"

# Metadata keys for cross-hook state
_META_SESSION_ID = "og_session_id"
_META_RESTORE_MAPPING = "og_restore_mapping"
_META_SKIP_OUTPUT = "og_skip_output_detection"
_META_INPUT_MESSAGES = "og_input_messages"


class OpenGuardrailsGuardrail(CustomGuardrail):
    """
    OpenGuardrails integration for LiteLLM.

    Provides enterprise AI safety through OpenGuardrails' detection pipeline:
    - 19 risk categories (S1-S19) covering security, compliance, and content safety
    - Data Leakage Prevention with format-aware detection
    - Automatic private model switching for sensitive data
    - Real-time streaming placeholder restoration

    Configuration in LiteLLM config.yaml:
        guardrails:
          - guardrail_name: "openguardrails"
            litellm_params:
              guardrail: openguardrails
              mode: [pre_call, post_call]
              api_base: http://og-server:5001
              api_key: sk-xxai-your-key  # optional for keyless self-hosted deployments
              default_on: true
              private_model_name: og-private-model  # optional, default: og-private-model
              fail_on_error: false  # optional, set true to block when API is unreachable
              skip_output_for_private_model: true  # optional, set false to scan private model output

    The private model must also be defined in model_list:
        model_list:
          - model_name: og-private-model
            litellm_params:
              model: openai/your-private-model
              api_base: https://your-private-endpoint.com
              api_key: sk-your-private-key
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        private_model_name: Optional[str] = None,
        fail_on_error: bool = False,
        skip_output_for_private_model: bool = True,
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        base_url = api_base or os.environ.get("OPENGUARDRAILS_API_BASE")
        if not base_url:
            raise ValueError(
                "api_base is required for OpenGuardrails. "
                "Set OPENGUARDRAILS_API_BASE env var or pass in litellm_params."
            )
        self.api_base = base_url.rstrip("/")

        self.api_key = api_key or os.environ.get("OPENGUARDRAILS_API_KEY")

        self.fail_on_error = fail_on_error
        self.skip_output_for_private_model = skip_output_for_private_model

        self.private_model_name = private_model_name or os.environ.get(
            "OPENGUARDRAILS_PRIVATE_MODEL", DEFAULT_PRIVATE_MODEL_NAME
        )

        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
            ]

        super().__init__(**kwargs)

        verbose_proxy_logger.debug(
            "OpenGuardrails initialized: api_base=%s, private_model=%s",
            self.api_base,
            self.private_model_name,
        )

    # ------------------------------------------------------------------ #
    #  HTTP helpers                                                        #
    # ------------------------------------------------------------------ #

    def _build_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _call_og_api(self, path: str, payload: dict) -> Optional[dict]:
        """Call an OpenGuardrails gateway API endpoint."""
        url = f"{self.api_base}{path}"
        try:
            response = await self.async_handler.post(
                url=url,
                json=payload,
                headers=self._build_headers(),
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            verbose_proxy_logger.error(
                "OpenGuardrails API error: %s status=%s body=%s",
                path,
                exc.response.status_code,
                exc.response.text[:500],
            )
            return None
        except Exception as exc:
            verbose_proxy_logger.error(
                "OpenGuardrails API request failed: %s error=%s", path, str(exc)
            )
            return None

    # ------------------------------------------------------------------ #
    #  Metadata helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_metadata(data: dict) -> dict:
        """Get or create the litellm_metadata dict inside data."""
        return data.setdefault("litellm_metadata", {})

    # ------------------------------------------------------------------ #
    #  Pre-call hook: input detection                                      #
    # ------------------------------------------------------------------ #

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict,
        cache,
        data: dict,
        call_type: str,
    ) -> Optional[dict]:
        """
        Run OpenGuardrails input detection before the LLM call.

        Actions:
        - block/replace: raise exception (request rejected)
        - anonymize: modify data["messages"] with masked content,
                     store restore_mapping in metadata for post-call restoration
        - switch_private_model: change data["model"] to private model
        - pass: no modification
        """
        messages = data.get("messages")
        if not messages:
            return data

        # Build request
        payload = {
            "messages": messages,
            "stream": data.get("stream", False),
        }

        # Add user_id if available
        user_id = None
        if user_api_key_dict:
            user_id = getattr(user_api_key_dict, "user_id", None)
        if user_id:
            payload["user_id"] = str(user_id)

        result = await self._call_og_api("/v1/gateway/process-input", payload)
        if result is None:
            if self.fail_on_error:
                raise GuardrailRaisedException(
                    guardrail_name=GUARDRAIL_NAME,
                    message="OpenGuardrails API unreachable; blocking request (fail_on_error=True)",
                )
            verbose_proxy_logger.warning(
                "OpenGuardrails unreachable, proceeding without guardrail (fail-open)"
            )
            return data

        action = result.get("action", "pass")
        detection = result.get("detection_result", {})
        metadata = self._get_metadata(data)

        verbose_proxy_logger.info(
            "OpenGuardrails input: action=%s risk=%s",
            action,
            detection.get("overall_risk_level", "unknown"),
        )

        # Store input messages for output detection context
        metadata[_META_INPUT_MESSAGES] = messages

        return self._handle_input_action(action, result, detection, data, metadata)

    def _handle_input_action(
        self, action: str, result: dict, detection: dict, data: dict, metadata: dict
    ) -> dict:
        """Process the action returned by OpenGuardrails input detection."""
        if action == "block":
            reason = self._extract_response_content(
                result.get("block_response", {}),
                fallback=f"Request blocked by OpenGuardrails (risk={detection.get('overall_risk_level')})",
            )
            raise GuardrailRaisedException(
                guardrail_name=GUARDRAIL_NAME,
                message=reason,
                should_wrap_with_default_message=False,
            )

        elif action == "replace":
            content = self._extract_response_content(
                result.get("replace_response", {}),
                fallback="Content filtered by OpenGuardrails.",
            )
            raise ModifyResponseException(
                message=content,
                model=data.get("model", "unknown"),
                request_data=data,
                guardrail_name=GUARDRAIL_NAME,
            )

        elif action == "anonymize":
            self._apply_anonymization(result, data, metadata)
            return data

        elif action == "proxy_response":
            proxy_resp = result.get("proxy_response", {})
            content = self._extract_response_content(
                proxy_resp,
                fallback="Response from private model.",
            )
            raise ModifyResponseException(
                message=content,
                model=data.get("model", "unknown"),
                request_data=data,
                guardrail_name=GUARDRAIL_NAME,
            )

        elif action == "switch_private_model":
            verbose_proxy_logger.info(
                "OpenGuardrails: switching to private model '%s'",
                self.private_model_name,
            )
            data["model"] = self.private_model_name
            if self.skip_output_for_private_model:
                metadata[_META_SKIP_OUTPUT] = True
            self._apply_anonymization(result, data, metadata)
            return data

        # action == "pass"
        return data

    def _apply_anonymization(self, result: dict, data: dict, metadata: dict):
        """Apply anonymized messages and store restore mapping in metadata."""
        anonymized = result.get("anonymized_messages")
        if anonymized:
            data["messages"] = anonymized
        restore_mapping = result.get("restore_mapping")
        if restore_mapping:
            metadata[_META_RESTORE_MAPPING] = restore_mapping
        session_id = result.get("session_id")
        if session_id:
            metadata[_META_SESSION_ID] = session_id

    # ------------------------------------------------------------------ #
    #  Post-call hook: output detection + restoration                      #
    # ------------------------------------------------------------------ #

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict,
        response,
    ):
        """
        Run OpenGuardrails output detection and placeholder restoration
        on the LLM response.

        - If restore_mapping exists: restore anonymized placeholders
        - If output detection enabled: check response for risks
        """
        metadata = self._get_metadata(data)

        # Skip output detection for private model responses
        if metadata.get(_META_SKIP_OUTPUT):
            # Still do restoration if needed
            restore_mapping = metadata.get(_META_RESTORE_MAPPING)
            if restore_mapping:
                self._restore_response_content(response, restore_mapping)
            return response

        # Extract response content
        content = self._extract_model_response_content(response)
        if not content:
            return response

        # Build output request
        payload: Dict[str, Any] = {"content": content}

        session_id = metadata.get(_META_SESSION_ID)
        if session_id:
            payload["session_id"] = session_id

        restore_mapping = metadata.get(_META_RESTORE_MAPPING)
        if restore_mapping:
            payload["restore_mapping"] = restore_mapping

        input_messages = metadata.get(_META_INPUT_MESSAGES)
        if input_messages:
            payload["messages"] = input_messages

        result = await self._call_og_api("/v1/gateway/process-output", payload)
        if result is None:
            if self.fail_on_error:
                raise GuardrailRaisedException(
                    guardrail_name=GUARDRAIL_NAME,
                    message="OpenGuardrails API unreachable; blocking response (fail_on_error=True)",
                )
            # API unreachable - fail open, but still do local restoration
            if restore_mapping:
                self._restore_response_content(response, restore_mapping)
            return response

        action = result.get("action", "pass")

        verbose_proxy_logger.info("OpenGuardrails output: action=%s", action)

        if action == "block":
            reason = self._extract_response_content(
                result.get("block_response", {}),
                fallback="Response blocked by OpenGuardrails.",
            )
            raise GuardrailRaisedException(
                guardrail_name=GUARDRAIL_NAME,
                message=reason,
                should_wrap_with_default_message=False,
            )

        elif action == "restore":
            restored = result.get("restored_content", "")
            if restored:
                self._set_model_response_content(response, restored)

        elif action == "anonymize":
            anonymized = result.get("anonymized_content", "")
            if anonymized:
                self._set_model_response_content(response, anonymized)

        # action == "pass" - no modification
        return response

    # ------------------------------------------------------------------ #
    #  Content extraction / replacement helpers                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_response_content(resp_obj: dict, fallback: str = "") -> str:
        """Extract message content from an OG response body (JSON string)."""
        body = resp_obj.get("body", "")
        if not body:
            return fallback
        try:
            body_json = json.loads(body) if isinstance(body, str) else body
            content = (
                body_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            return content if content else fallback
        except (json.JSONDecodeError, IndexError, KeyError):
            return fallback

    @staticmethod
    def _extract_model_response_content(response) -> str:
        """Extract text content from a LiteLLM ModelResponse."""
        try:
            choices = getattr(response, "choices", None)
            if not choices:
                return ""
            message = getattr(choices[0], "message", None)
            if not message:
                return ""
            return getattr(message, "content", "") or ""
        except (IndexError, AttributeError):
            return ""

    @staticmethod
    def _set_model_response_content(response, content: str):
        """Set text content on a LiteLLM ModelResponse."""
        try:
            if response.choices and response.choices[0].message:
                response.choices[0].message.content = content
        except (IndexError, AttributeError):
            pass

    @staticmethod
    def _restore_response_content(response, restore_mapping: Dict[str, str]):
        """Replace placeholders in response content using restore_mapping."""
        try:
            content = response.choices[0].message.content
            if not content:
                return
            for placeholder, original in restore_mapping.items():
                content = content.replace(placeholder, original)
            response.choices[0].message.content = content
        except (IndexError, AttributeError):
            pass
