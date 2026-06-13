"""Bastion Prompt Protection guardrail for the LiteLLM proxy.

Local, ONNX-based prompt-injection / jailbreak detection (~5 ms warm on CPU; no
network calls). The detection engine ships in the optional
``bastion-prompt-protection`` package and is imported lazily, so litellm has no
hard dependency on it. Install it where you run the proxy::

    pip install bastion-prompt-protection

The free ``tiny`` model is AGPL-3.0; a commercial multilingual model (and an
AGPL exemption) is available at https://bastionsoft.com.
"""

import asyncio
from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    Union,
)

from fastapi import HTTPException

from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from bastion_prompt_protection import Guard, GuardResult

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


DEFAULT_VIOLATION_MESSAGE = (
    "I can't help with that request: it was flagged as a potential "
    "prompt-injection attempt and blocked."
)


class BastionGuardrail(CustomGuardrail):
    """Screen text for prompt injection / jailbreak using Bastion Prompt Protection."""

    def __init__(
        self,
        guardrail_name: Optional[str] = None,
        preset: str = "tiny",
        threshold: Optional[float] = None,
        violation_message: str = DEFAULT_VIOLATION_MESSAGE,
        event_hook: Optional[Union[str, list[str]]] = None,
        default_on: bool = False,
    ) -> None:
        _event_hook: Optional[Union[GuardrailEventHooks, list[GuardrailEventHooks]]] = (
            None
        )
        if event_hook is not None:
            if isinstance(event_hook, list):
                _event_hook = [
                    GuardrailEventHooks(h) if isinstance(h, str) else h
                    for h in event_hook
                ]
            else:
                _event_hook = GuardrailEventHooks(event_hook)
        super().__init__(
            guardrail_name=guardrail_name or "bastion",
            supported_event_hooks=[
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.during_call,
            ],
            event_hook=_event_hook or [GuardrailEventHooks.pre_call],
            default_on=default_on,
        )
        self.preset = preset
        self.threshold = threshold
        self.violation_message = violation_message
        self._guard: Optional["Guard"] = None  # lazily constructed (loads the model)

    def _get_guard(self) -> "Guard":
        if self._guard is None:
            try:
                from bastion_prompt_protection import Guard
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "The 'bastion-prompt-protection' package is required for the "
                    "Bastion guardrail. Install it with: "
                    "pip install bastion-prompt-protection"
                ) from e
            self._guard = Guard(preset=self.preset)
        return self._guard

    def _is_attack(self, result: "GuardResult") -> bool:
        if self.threshold is not None:
            return bool(result.risk >= self.threshold)
        return bool(result.is_attack)

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        texts = inputs.get("texts", [])
        if not texts:
            return inputs

        guard = self._get_guard()
        for text in texts:
            if not text:
                continue
            # Bastion inference is synchronous and CPU-bound (~5 ms); offload to a
            # thread so it never blocks the proxy's event loop under concurrency.
            result = await asyncio.to_thread(guard.protect, text)
            if self._is_attack(result):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": self.violation_message,
                        "bastion_guardrail": {
                            "risk": float(result.risk),
                            "stage": result.stage_reached,
                            "input_type": input_type,
                        },
                    },
                )
        return inputs
