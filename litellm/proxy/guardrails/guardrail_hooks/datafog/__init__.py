from typing import TYPE_CHECKING, Optional

import litellm
from litellm.proxy.guardrails.guardrail_hooks.datafog.datafog import DataFogGuardrail
from litellm.types.guardrails import SupportedGuardrailIntegrations

if TYPE_CHECKING:
    from litellm import Router
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
    llm_router: Optional["Router"] = None,
):
    """
    Initialize the DataFog PII Guardrail.

    Args:
        litellm_params: Guardrail configuration parameters
        guardrail: Guardrail metadata

    Returns:
        Initialized DataFogGuardrail instance
    """
    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("DataFog: guardrail_name is required")

    optional_params = getattr(litellm_params, "optional_params", None)

    def _param(name: str):
        value = getattr(litellm_params, name, None)
        if value is None and optional_params is not None:
            value = getattr(optional_params, name, None)
        return value

    datafog_guardrail = DataFogGuardrail(
        guardrail_name=guardrail_name,
        datafog_action=getattr(litellm_params, "datafog_action", None),
        datafog_entity_types=_param("datafog_entity_types"),
        datafog_locales=_param("datafog_locales"),
        datafog_fail_policy=_param("datafog_fail_policy"),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on or False,
    )

    litellm.logging_callback_manager.add_litellm_callback(datafog_guardrail)

    return datafog_guardrail


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.DATAFOG.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.DATAFOG.value: DataFogGuardrail,
}
