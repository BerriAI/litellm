import os
import uuid
from typing import (
    TYPE_CHECKING,
    Any,  # noqa: TID251  # **kwargs forwards verbatim to CustomGuardrail.__init__; see ruff-strict.toml
    Final,
    Literal,
)

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

GUARDRAIL_PATH: Final = "/v1/guardrails/litellm"

ACTION_NONE: Final = "NONE"
ACTION_BLOCKED: Final = "BLOCKED"
ACTION_INTERVENED: Final = "GUARDRAIL_INTERVENED"

DEFAULT_BLOCKED_MESSAGE: Final = "Blocked by your organization's content policy."

SUPPORTED_EVENT_HOOKS: Final = (GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call)


class AiriaGuardrail(CustomGuardrail):
    """Evaluates prompts and responses against your Airia guardrail policy."""

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        supported_event_hooks: list[GuardrailEventHooks] | None = None,  # mutable-ok: matches CustomGuardrail.__init__
        **kwargs: Any,  # kwargs-ok: forwarded verbatim to CustomGuardrail.__init__, whose param list is wide and evolving
    ) -> None:
        resolved_timeout: Final = timeout or float(os.getenv("AIRIA_TIMEOUT", "10"))

        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
            params={"timeout": httpx.Timeout(timeout=resolved_timeout, connect=5.0)},  # mutable-ok: one-shot params
        )

        self.api_base = (api_base or os.getenv("AIRIA_GATEWAY_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("AIRIA_API_KEY")
        self.streaming_transform_mode: Final[Literal["block_only", "incremental_diff"]] = "incremental_diff"
        self.streaming_end_of_stream_only: Final = True

        if not self.api_base:
            raise ValueError("AiriaGuardrail requires api_base, or the AIRIA_GATEWAY_URL environment variable.")
        if not self.api_key:
            raise ValueError("AiriaGuardrail requires api_key, or the AIRIA_API_KEY environment variable.")

        self.optional_params = kwargs
        super().__init__(
            supported_event_hooks=supported_event_hooks or [*SUPPORTED_EVENT_HOOKS],  # mutable-ok: base needs a list
            **kwargs,
        )
        verbose_proxy_logger.info("AiriaGuardrail initialized with gateway: %s", self.api_base)

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],  # mutable-ok: overrides CustomGuardrail.apply_guardrail's plain-dict contract
        input_type: Literal["request", "response"],
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> GenericGuardrailAPIInputs:
        call_id: Final = (
            logging_obj.litellm_call_id
            if logging_obj
            else (request_data.get("litellm_call_id") if request_data else None)
        ) or str(uuid.uuid4())

        try:
            response: Final = await self.async_handler.post(
                f"{self.api_base}{GUARDRAIL_PATH}",
                json={  # mutable-ok: one-shot HTTP request body, never mutated after construction
                    "input_type": input_type,
                    "texts": inputs.get("texts") or (),
                    "structured_messages": inputs.get("structured_messages") or (),
                    "tools": inputs.get("tools") or (),
                    "tool_calls": inputs.get("tool_calls") or (),
                    "model": inputs.get("model"),
                    "litellm_call_id": call_id,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},  # mutable-ok: one-shot HTTP headers
            )
            response.raise_for_status()
            body: Final = response.json()
        except Exception as error:
            verbose_proxy_logger.error(
                "Airia guardrail could not evaluate the request (litellm_call_id=%s, input_type=%s): %s",
                call_id,
                input_type,
                error,
            )
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"Airia guardrail could not evaluate the request: {error}",
                blocked_content=False,
            ) from error

        action: Final = body.get("action")

        if action == ACTION_BLOCKED:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=body.get("blocked_reason") or DEFAULT_BLOCKED_MESSAGE,
                blocked_content=True,
            )

        if action == ACTION_INTERVENED:
            return self._rewritten(body, inputs)

        if action != ACTION_NONE:
            raise self._blocked()

        return inputs

    def _rewritten(
        self,
        body: dict[str, object],  # mutable-ok: response.json() returns a plain dict
        inputs: GenericGuardrailAPIInputs,
    ) -> GenericGuardrailAPIInputs:
        texts: Final = body.get("texts")
        structured_messages: Final = body.get("structured_messages")
        if texts is None and structured_messages is None:
            raise self._blocked()
        if not (texts is None or isinstance(texts, list)):
            raise self._blocked()
        if not (structured_messages is None or isinstance(structured_messages, list)):
            raise self._blocked()

        rewritten: Final[GenericGuardrailAPIInputs] = {**inputs}  # mutable-ok: fresh copy; caller's object untouched
        if isinstance(texts, list):
            rewritten["texts"] = texts
        if isinstance(structured_messages, list):
            rewritten["structured_messages"] = structured_messages
        return rewritten

    def _blocked(self) -> GuardrailRaisedException:
        return GuardrailRaisedException(
            guardrail_name=self.guardrail_name,
            message=DEFAULT_BLOCKED_MESSAGE,
            blocked_content=True,
        )

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.airia import (
            AiriaGuardrailConfigModel,
        )

        return AiriaGuardrailConfigModel
