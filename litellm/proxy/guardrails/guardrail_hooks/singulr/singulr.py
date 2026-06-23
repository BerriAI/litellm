
"""
Singulr guardrail integration for LiteLLM.

Calls the Singulr SDK Guard API to scan messages.
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

_DEFAULT_API_BASE = "http://localhost:8000"
_GUARD_ENDPOINT = "/api/v1/ai-platform/controller/singulr-guardrails-litellm"


class SingulrMissingCredentials(Exception):
    pass


class SingulrGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        enforcement_entity_id: Optional[str] = None,
        guardrail_id: Optional[str] = None,
        block_on_error: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key or os.environ.get("SINGULR_API_KEY")

        self.api_base = (
            api_base or os.environ.get("SINGULR_API_BASE") or _DEFAULT_API_BASE
        ).rstrip("/")

        self.enforcement_entity_id = enforcement_entity_id or os.environ.get(
            "SINGULR_ENFORCEMENT_ENTITY_ID"
        )
        self.guardrail_id = guardrail_id or os.environ.get(
            "SINGULR_GUARDRAIL_ID"
        )

        if block_on_error is None:
            env = os.environ.get("SINGULR_BLOCK_ON_ERROR", "true")
            self.block_on_error = env.lower() in ("true", "1", "yes")
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
        from litellm.types.proxy.guardrails.guardrail_hooks.singulr import (
            SingulrGuardrailConfigModel,
        )

        return SingulrGuardrailConfigModel

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

        if structured_messages:
            prompt = self._extract_prompt_from_messages(list(structured_messages))
        elif texts:
            prompt = "\n".join(texts)
        else:
            return inputs

        if not prompt:
            return inputs

        payload: Dict[str, Any] = {
            "prompt": prompt,
        }

        endpoint = f"{self.api_base}{_GUARD_ENDPOINT}"

        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.enforcement_entity_id:
            headers["X-Singulr-Enforcement-Entity-Id"] = self.enforcement_entity_id

        if self.guardrail_id:
            headers["X-Singulr-Guardrail-Id"] = self.guardrail_id

        verbose_proxy_logger.debug(
            "Singulr: %s",
            endpoint,
        )

        try:
            response = await self.async_handler.post(
                url=endpoint,
                headers=headers,
                json=payload,
                timeout=10.0,
            )
            response.raise_for_status()
            result = response.json()
        except Exception as exc:
            verbose_proxy_logger.error("Singulr API error: %s", str(exc))
            if self.block_on_error:
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=f"Singulr API unreachable (block_on_error=True): {exc}",
                ) from exc
            return inputs

        should_block = result.get("should_block", False)

        verbose_proxy_logger.debug(
            "Singulr: should_block=%s blocking_due_to=%s",
            should_block,
            result.get("blocking_due_to"),
        )

        if should_block:
            blocking_due_to = result.get("blocking_due_to", "unknown")
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"Blocked by Singulr: {blocking_due_to}",
            )

        return inputs

    @staticmethod
    def _extract_prompt_from_messages(messages: list) -> str:
        """Extract text content from messages to build a single prompt."""
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
        return "\n".join(texts)