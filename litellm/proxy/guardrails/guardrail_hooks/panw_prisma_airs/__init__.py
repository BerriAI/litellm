from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .panw_prisma_airs import PanwPrismaAirsHandler

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    guardrail_name: Final = guardrail.get("guardrail_name")

    # Note: api_key and profile_name can be None - handler will use env vars or API key's linked profile
    if not guardrail_name:
        raise ValueError("PANW Prisma AIRS: guardrail_name is required")

    litellm_params_dump: Final = litellm_params.model_dump(exclude_unset=True)
    litellm_params_dump["guardrail_name"] = guardrail_name
    litellm_params_dump["event_hook"] = litellm_params.mode
    litellm_params_dump["default_on"] = litellm_params.default_on or False

    _panw_callback: Final = PanwPrismaAirsHandler(**litellm_params_dump)
    litellm.logging_callback_manager.add_litellm_callback(_panw_callback)

    return _panw_callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.PANW_PRISMA_AIRS.value: initialize_guardrail,
}


guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.PANW_PRISMA_AIRS.value: PanwPrismaAirsHandler,
}
