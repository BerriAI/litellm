from typing import TYPE_CHECKING

import litellm
from litellm.types.guardrails import SupportedGuardrailIntegrations
from litellm.types.proxy.guardrails.guardrail_hooks.straiker import (
    StraikerGuardrailConfigModel,
)

from .straiker import StraikerGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    config_values = {
        field: value
        for field in StraikerGuardrailConfigModel.model_fields
        if field != "optional_params"
        for value in [getattr(litellm_params, field, None)]
        if value is not None
    }
    config = StraikerGuardrailConfigModel.model_validate(config_values)
    _callback = StraikerGuardrail(
        **config.model_dump(exclude={"optional_params"}),
        guardrail_name=guardrail.get("guardrail_name", "straiker"),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )

    litellm.logging_callback_manager.add_litellm_callback(_callback)
    return _callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.STRAIKER.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.STRAIKER.value: StraikerGuardrail,
}
