# litellm/proxy/guardrails/guardrail_registry.py

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .guardrail_initializers import (
    initialize_aim,
    initialize_aporia,
    initialize_bedrock,
    initialize_guardrails_ai,
    initialize_hide_secrets,
    initialize_lakera,
    initialize_presidio,
)

guardrail_registry = {
    SupportedGuardrailIntegrations.APORIA.value: initialize_aporia,
    SupportedGuardrailIntegrations.BEDROCK.value: initialize_bedrock,
    SupportedGuardrailIntegrations.LAKERA.value: initialize_lakera,
    SupportedGuardrailIntegrations.AIM.value: initialize_aim,
    SupportedGuardrailIntegrations.PRESIDIO.value: initialize_presidio,
    SupportedGuardrailIntegrations.HIDE_SECRETS.value: initialize_hide_secrets,
    SupportedGuardrailIntegrations.GURDRAILS_AI.value: initialize_guardrails_ai,
}
