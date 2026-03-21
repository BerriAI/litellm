"""
PromptGuard guardrail integration for LiteLLM.

Calls the PromptGuard Guard API to scan messages for prompt injection,
PII, topic violations, and entity blocklist matches before and after
LLM calls.
"""

import os
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Type

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )
    from litellm.types.proxy.guardrails.guardrail_hooks.base import (
        GuardrailConfigModel,
    )

_DEFAULT_API_BASE = "https://api.promptguard.co"
_GUARD_ENDPOINT = "/api/v1/guard"


class PromptGuardMissingCredentials(Exception):
    pass


class PromptGuardGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )
        self.api_key = api_key or os.environ.get(
            "PROMPTGUARD_API_KEY",
        )
        self.api_base = (
            api_base or os.environ.get("PROMPTGUARD_API_BASE") or _DEFAULT_API_BASE
        ).rstrip("/")

        if not self.api_key:
            raise PromptGuardMissingCredentials(
                "PromptGuard API key is required. "
                "Set PROMPTGUARD_API_KEY in the environment "
                "or pass api_key in the guardrail config."
            )

        super().__init__(**kwargs)

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.promptguard import (
            PromptGuardConfigModel,
        )

        return PromptGuardConfigModel

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        texts = inputs.get("texts", [])
        structured_messages = inputs.get("structured_messages", [])
        model = inputs.get("model")

        if structured_messages:
            messages = list(structured_messages)
        elif texts:
            messages = [{"role": "user", "content": text} for text in texts]
        else:
            return inputs

        direction = "input" if input_type == "request" else "output"

        payload: dict[str, Any] = {
            "messages": messages,
            "direction": direction,
        }
        if model:
            payload["model"] = model

        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        endpoint = f"{self.api_base}{_GUARD_ENDPOINT}"

        verbose_proxy_logger.debug(
            "PromptGuard guardrail: calling %s direction=%s messages=%d",
            endpoint,
            direction,
            len(messages),
        )

        response = await self.async_handler.post(
            url=endpoint,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        result = response.json()

        verbose_proxy_logger.debug(
            "PromptGuard guardrail: decision=%s confidence=%s threat_type=%s",
            result.get("decision"),
            result.get("confidence"),
            result.get("threat_type"),
        )

        decision = result.get("decision", "allow")

        if decision == "block":
            threat_type = result.get("threat_type", "unknown")
            event_id = result.get("event_id", "")
            confidence = result.get("confidence", 0.0)
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=(
                    f"Blocked by PromptGuard: {threat_type} "
                    f"(confidence={confidence}, event_id={event_id})"
                ),
            )

        if decision == "redact":
            redacted_messages = result.get("redacted_messages")
            if redacted_messages:
                redacted_texts = self._extract_texts_from_messages(
                    redacted_messages,
                )
                if redacted_texts:
                    inputs["texts"] = redacted_texts

        return inputs

    @staticmethod
    def _extract_texts_from_messages(messages: list) -> List[str]:
        """Extract text content strings from a list of chat messages."""
        texts: List[str] = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text")
                        if text:
                            texts.append(text)
        return texts
