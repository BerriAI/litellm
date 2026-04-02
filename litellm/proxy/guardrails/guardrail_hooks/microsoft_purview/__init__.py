from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .purview_dlp import MicrosoftPurviewDLPGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm
    from litellm.types.guardrails import GuardrailEventHooks

    tenant_id = litellm_params.get("tenant_id")
    client_id = litellm_params.get("client_id")

    # client_secret can be passed via the standard api_key field or as
    # a dedicated client_secret parameter.
    client_secret = litellm_params.api_key or litellm_params.get("client_secret")

    if not tenant_id:
        raise ValueError("Microsoft Purview: tenant_id is required")
    if not client_id:
        raise ValueError("Microsoft Purview: client_id is required")
    if not client_secret:
        raise ValueError("Microsoft Purview: client_secret (or api_key) is required")

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Microsoft Purview: guardrail_name is required")

    mode = litellm_params.mode
    logging_only = False
    if isinstance(mode, str) and mode == GuardrailEventHooks.logging_only.value:
        logging_only = True

    purview_guardrail = MicrosoftPurviewDLPGuardrail(
        guardrail_name=guardrail_name,
        tenant_id=str(tenant_id),
        client_id=str(client_id),
        client_secret=str(client_secret),
        purview_app_name=str(litellm_params.get("purview_app_name") or "LiteLLM"),
        user_id_field=str(litellm_params.get("user_id_field") or "user_id"),
        logging_only=logging_only,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )

    litellm.logging_callback_manager.add_litellm_callback(purview_guardrail)
    return purview_guardrail


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.MICROSOFT_PURVIEW.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.MICROSOFT_PURVIEW.value: MicrosoftPurviewDLPGuardrail,
}
