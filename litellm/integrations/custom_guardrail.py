from typing import List, Literal, Optional

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.guardrails import GuardrailEventHooks


class CustomGuardrail(CustomLogger):

    def __init__(
        self,
        guardrail_name: Optional[str] = None,
        supported_event_hooks: Optional[List[GuardrailEventHooks]] = None,
        event_hook: Optional[GuardrailEventHooks] = None,
        **kwargs,
    ):
        self.guardrail_name = guardrail_name
        self.supported_event_hooks = supported_event_hooks
        self.event_hook: Optional[GuardrailEventHooks] = event_hook

        if supported_event_hooks:
            ## validate event_hook is in supported_event_hooks
            if event_hook and event_hook not in supported_event_hooks:
                raise ValueError(
                    f"Event hook {event_hook} is not in the supported event hooks {supported_event_hooks}"
                )
        super().__init__(**kwargs)

    def should_run_guardrail(self, data, event_type: GuardrailEventHooks) -> bool:
        metadata = data.get("metadata") or {}
        requested_guardrails = metadata.get("guardrails") or []
        verbose_logger.debug(
            "inside should_run_guardrail for guardrail=%s event_type= %s guardrail_supported_event_hooks= %s requested_guardrails= %s",
            self.guardrail_name,
            event_type,
            self.event_hook,
            requested_guardrails,
        )

        if (
            self.event_hook
            and self.guardrail_name not in requested_guardrails
            and event_type.value != "logging_only"
        ):
            return False

        if self.event_hook and self.event_hook != event_type.value:
            return False

        return True
