from typing import Final

import litellm
from litellm.types.guardrails import Guardrail, GuardrailEventHooks, LitellmParams, SupportedGuardrailIntegrations

from .ismalicious import IsMaliciousGuardrail


def initialize_guardrail(litellm_params: LitellmParams, guardrail: Guardrail) -> IsMaliciousGuardrail:
    modes: Final = [litellm_params.mode] if isinstance(litellm_params.mode, str) else litellm_params.mode
    if not isinstance(modes, list):
        raise ValueError("IsMalicious requires explicit MCP modes")
    callback: Final = IsMaliciousGuardrail(
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        guardrail_name=guardrail["guardrail_name"],
        event_hook=[GuardrailEventHooks(mode) for mode in modes],
        default_on=litellm_params.default_on is True,
    )
    litellm.logging_callback_manager.add_litellm_callback(callback)  # pyright: ignore[reportUnknownMemberType]  # manager exposes an untyped callable union
    return callback


guardrail_initializer_registry: Final = {SupportedGuardrailIntegrations.ISMALICIOUS.value: initialize_guardrail}
guardrail_class_registry: Final = {SupportedGuardrailIntegrations.ISMALICIOUS.value: IsMaliciousGuardrail}
