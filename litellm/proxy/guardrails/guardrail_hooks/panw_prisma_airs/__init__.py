from typing import TYPE_CHECKING, Optional, cast

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .panw_prisma_airs import PanwPrismaAirsHandler

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    guardrail_name = guardrail.get("guardrail_name")
    profile_name = cast(Optional[str], getattr(litellm_params, "profile_name", None))
    
    # Note: api_key can be None here - handler will fallback to PANW_PRISMA_AIRS_API_KEY env var
    if not profile_name:
        raise ValueError("PANW Prisma AIRS: profile_name is required")
    if not guardrail_name:
        raise ValueError("PANW Prisma AIRS: guardrail_name is required")

    _panw_callback = PanwPrismaAirsHandler(
        **{
            **litellm_params.model_dump(),
            "guardrail_name": guardrail_name,
            "event_hook": litellm_params.mode,
            "default_on": litellm_params.default_on or False,
        }
    )
    litellm.logging_callback_manager.add_litellm_callback(_panw_callback)

    return _panw_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.PANW_PRISMA_AIRS.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.PANW_PRISMA_AIRS.value: PanwPrismaAirsHandler,
}
