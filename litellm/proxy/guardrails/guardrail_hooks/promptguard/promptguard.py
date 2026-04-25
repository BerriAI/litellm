"""
PromptGuard guardrail integration for LiteLLM.

Calls the PromptGuard Guard API to scan messages for prompt
injection, PII, topic violations, and entity blocklist matches
before and after LLM calls.
"""

import os
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Type,
)

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
from litellm.types.guardrails import GuardrailEventHooks
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
        block_on_error: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key or os.environ.get(
            "PROMPTGUARD_API_KEY",
        )
        if not self.api_key:
            raise PromptGuardMissingCredentials(
                "PromptGuard API key is required. "
                "Set PROMPTGUARD_API_KEY in the "
                "environment or pass api_key in "
                "the guardrail config."
            )

        self.api_base = (
            api_base or os.environ.get("PROMPTGUARD_API_BASE") or _DEFAULT_API_BASE
        ).rstrip("/")

        if block_on_error is None:
            env = os.environ.get("PROMPTGUARD_BLOCK_ON_ERROR", "true")
            self.block_on_error = env.lower() in (
                "true",
                "1",
                "yes",
            )
        else:
            self.block_on_error = block_on_error

        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )

        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
            ]

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
        images = inputs.get("images", [])
        structured_messages = inputs.get("structured_messages", [])
        model = inputs.get("model")

        if structured_messages:
            messages = list(structured_messages)
        elif texts:
            messages = [{"role": "user", "content": text} for text in texts]
        else:
            return inputs

        direction = "input" if input_type == "request" else "output"

        payload: Dict[str, Any] = {
            "messages": messages,
            "direction": direction,
        }
        if model:
            payload["model"] = model
        if images:
            payload["images"] = images

        endpoint = f"{self.api_base}{_GUARD_ENDPOINT}"

        verbose_proxy_logger.debug(
            "PromptGuard: %s direction=%s msgs=%d imgs=%d",
            endpoint,
            direction,
            len(messages),
            len(images),
        )

        try:
            response = await self.async_handler.post(
                url=endpoint,
                headers={
                    "X-API-Key": self.api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=10.0,
            )
            response.raise_for_status()
            result = response.json()
        except Exception as exc:
            verbose_proxy_logger.error("PromptGuard API error: %s", str(exc))
            if self.block_on_error:
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=f"PromptGuard API unreachable (block_on_error=True): {exc}",
                ) from exc
            return inputs

        verbose_proxy_logger.debug(
            "PromptGuard: decision=%s threat=%s",
            result.get("decision"),
            result.get("threat_type"),
        )

        decision = result.get("decision") or "allow"

        if decision == "block":
            threat_type = result.get("threat_type", "unknown")
            event_id = result.get("event_id", "")
            confidence = result.get("confidence", 0.0)
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=(
                    f"Blocked by PromptGuard: "
                    f"{threat_type} "
                    f"(confidence={confidence}, "
                    f"event_id={event_id})"
                ),
            )

        if decision == "redact":
            redacted = result.get("redacted_messages")
            if redacted:
                if structured_messages:
                    inputs["structured_messages"] = redacted
                if "texts" in inputs:
                    extracted = self._extract_texts_from_messages(
                        redacted,
                    )
                    if extracted:
                        inputs["texts"] = extracted

        return inputs

    @staticmethod
    def _extract_texts_from_messages(messages: list) -> List[str]:
        """Extract text content from user-role messages only.

        Only user messages are extracted to avoid injecting system or
        assistant content into the ``texts`` list, which should mirror
        the original user-provided input.
        """
        texts: List[str] = []
        for message in messages:
            if message.get("role") != "user":
                continue
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
