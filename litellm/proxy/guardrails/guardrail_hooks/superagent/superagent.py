"""LiteLLM SuperAgent guardrail.

Runs user prompts through the SuperAgent guard endpoint to determine whether
the request should ``pass`` or ``block`` and surface violation details.
"""

from __future__ import annotations

import json
import re
import os
from datetime import datetime
from typing import Any, Dict, Literal, Optional, Union

import httpx

from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail, log_guardrail_information
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


DEFAULT_SUPERAGENT_API_BASE = "https://app.superagent.sh/api/guard"


class SuperAgentGuardrail(CustomGuardrail):
    """Simple guardrail that delegates moderation to a SuperAgent server."""

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        mock_decision: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # Allow tests to inject a mock decision without making an HTTP call.
        self.mock_decision = mock_decision

        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.during_call,
            ]

        super().__init__(**kwargs)

        self.guardrail_provider = "superagent"
        self.api_base = (
            api_base
            or os.getenv("SUPERAGENT_API_BASE")
            or DEFAULT_SUPERAGENT_API_BASE
        )
        self.api_key = api_key or os.getenv("SUPERAGENT_API_KEY")

        self._client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

    async def _call_superagent(self, prompt: str) -> Dict[str, Any]:
        if self.mock_decision is not None:
            return {
                "classification": self.mock_decision,
                "violation_types": [],
                "cwe_codes": [],
            }

        if not self.api_key:
            raise ValueError(
                "Missing SuperAgent API key. Set SUPERAGENT_API_KEY or provide api_key."
            )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {"prompt": prompt}

        url = self.api_base
        if not url:
            raise ValueError("SuperAgent API base URL is not configured")

        try:
            response = await self._client.post(
                url,
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(10.0),
            )
        except Exception as exc:  # pragma: no cover - network errors bubble up
            verbose_proxy_logger.error(
                "SuperAgent guardrail request failed: %s", exc
            )
            raise

        if response.status_code >= 400:
            raise ValueError(
                f"SuperAgent request failed with status {response.status_code}: {response.text}"
            )

        try:
            parsed: Dict[str, Any] = response.json()
        except ValueError:
            raise ValueError(
                "SuperAgent response was not valid JSON text"
            )

        decision_payload = self._extract_decision_payload(parsed)

        if not isinstance(decision_payload, dict):
            raise ValueError(
                f"Unexpected classifier payload type: {type(decision_payload)}"
            )

        classification = decision_payload.get("classification")
        if not isinstance(classification, str):
            raise ValueError(
                f"Classifier payload missing 'classification': {decision_payload!r}"
            )

        decision_payload["classification"] = classification.strip().lower()

        violation_types = decision_payload.get("violation_types")
        if violation_types is None:
            decision_payload["violation_types"] = []
        elif not isinstance(violation_types, list):
            raise ValueError("'violation_types' must be a list if provided")

        cwe_codes = decision_payload.get("cwe_codes")
        if cwe_codes is None:
            decision_payload["cwe_codes"] = []
        elif not isinstance(cwe_codes, list):
            raise ValueError("'cwe_codes' must be a list if provided")

        return decision_payload

    def _extract_decision_payload(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Handle both direct JSON payloads and nested Fireworks responses."""

        direct_payload = self._maybe_extract_direct_payload(parsed)
        if direct_payload is not None:
            return direct_payload

        content = self._extract_content_from_choices(parsed)
        if not content:
            raise ValueError(
                f"SuperAgent response missing classifier content: {parsed!r}"
            )

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return self._parse_classifier_content(content)

    def _maybe_extract_direct_payload(
        self, response_body: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        candidates: list[Dict[str, Any]] = []
        if isinstance(response_body, dict):
            candidates.append(response_body)
            for key in ("data", "result"):
                nested = response_body.get(key)
                if isinstance(nested, dict):
                    candidates.append(nested)

        for candidate in candidates:
            if self._is_classifier_payload(candidate):
                return candidate

        return None

    def _extract_content_from_choices(self, parsed: Dict[str, Any]) -> Optional[str]:
        choices = parsed.get("choices") if isinstance(parsed, dict) else None
        if not isinstance(choices, list) or not choices:
            return None

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return None

        message = first_choice.get("message", {})
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content

        return None

    def _is_classifier_payload(self, payload: Dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False
        return "classification" in payload

    def _extract_user_prompt(self, data: dict) -> Optional[str]:
        messages = data.get("messages")
        if not isinstance(messages, list):
            return None

        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content
        return None

    async def _evaluate_guardrail(
        self,
        data: dict,
        event_type: GuardrailEventHooks,
    ) -> Optional[dict]:
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        prompt = self._extract_user_prompt(data)
        if prompt is None:
            verbose_proxy_logger.debug(
                "SuperAgent guardrail: no user prompt found to inspect."
            )
            return data

        start_time = datetime.now()
        guardrail_status: Literal["success", "guardrail_intervened", "guardrail_failed_to_respond", "not_run"] = "success"
        decision_payload: Union[str, Dict[str, Any]] = "pass"

        try:
            decision_payload = await self._call_superagent(prompt)
            decision = decision_payload.get("classification", "pass")
            verbose_proxy_logger.debug(
                "SuperAgent guardrail decision for %s: %s",
                self.guardrail_name,
                decision,
            )

            if decision == "block":
                guardrail_status = "guardrail_intervened"
                error_detail = {
                    "error": "Blocked by SuperAgent guardrail",
                    "guardrail_name": self.guardrail_name,
                    "guardrail_response": decision_payload,
                }
                raise HTTPException(status_code=400, detail=error_detail)

            add_guardrail_to_applied_guardrails_header(
                request_data=data, guardrail_name=self.guardrail_name
            )
            return data
        except HTTPException:
            raise
        except Exception as exc:
            guardrail_status = "guardrail_failed_to_respond"
            decision_payload = {"error": str(exc)}
            raise
        finally:
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider=self.guardrail_provider,
                guardrail_json_response=decision_payload,
                request_data=data,
                guardrail_status=guardrail_status,
                start_time=start_time.timestamp(),
                end_time=datetime.now().timestamp(),
                duration=(datetime.now() - start_time).total_seconds(),
            )

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,  # noqa: ARG002 - part of hook interface
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
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        event_type = GuardrailEventHooks.pre_call
        return await self._evaluate_guardrail(data=data, event_type=event_type)

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
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        event_type = GuardrailEventHooks.during_call
        return await self._evaluate_guardrail(data=data, event_type=event_type)

    async def async_post_call_success_hook(  # type: ignore[override]
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: LLMResponseTypes,
    ) -> LLMResponseTypes:
        # Minimal implementation does not modify responses post-call.
        return response

    def _parse_classifier_content(self, content: str) -> Dict[str, Any]:
        """Attempt to recover JSON payloads that include markdown fences or trailing text."""

        cleaned = content.strip()

        # handle markdown fenced blocks ```json ... ```
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[len("json") :].strip()

        # If there are multiple JSON objects, read the first one
        decoder = json.JSONDecoder()
        try:
            obj, idx = decoder.raw_decode(cleaned)
        except json.JSONDecodeError as exc:
            # Try removing any trailing non-json characters heuristically
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            raise ValueError(
                f"Unable to decode SuperAgent classifier JSON: {exc}"
            ) from exc

        remaining = cleaned[idx:].strip()
        if remaining.startswith("```"):
            remaining = remaining.strip("`").strip()

        if remaining:
            verbose_proxy_logger.debug(
                "SuperAgent classifier content had trailing text after JSON: %s",
                remaining,
            )

        if not isinstance(obj, dict):
            raise ValueError(
                f"Unexpected classifier payload type: {type(obj)}"
            )

        return obj
