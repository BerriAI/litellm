# litellm/proxy/guardrails/guardrail_initializers.py
import litellm
from litellm.types.guardrails import *


def initialize_aporia(litellm_params, guardrail):
    from litellm.proxy.guardrails.guardrail_hooks.aporia_ai import AporiaGuardrail

    _aporia_callback = AporiaGuardrail(
        api_base=litellm_params["api_base"],
        api_key=litellm_params["api_key"],
        guardrail_name=guardrail["guardrail_name"],
        event_hook=litellm_params["mode"],
        default_on=litellm_params["default_on"],
    )
    litellm.logging_callback_manager.add_litellm_callback(_aporia_callback)


def initialize_bedrock(litellm_params, guardrail):
    from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
        BedrockGuardrail,
    )

    _bedrock_callback = BedrockGuardrail(
        guardrail_name=guardrail["guardrail_name"],
        event_hook=litellm_params["mode"],
        guardrailIdentifier=litellm_params["guardrailIdentifier"],
        guardrailVersion=litellm_params["guardrailVersion"],
        default_on=litellm_params["default_on"],
    )
    litellm.logging_callback_manager.add_litellm_callback(_bedrock_callback)


def initialize_lakera(litellm_params, guardrail):
    from litellm.proxy.guardrails.guardrail_hooks.lakera_ai import lakeraAI_Moderation

    _lakera_callback = lakeraAI_Moderation(
        api_base=litellm_params["api_base"],
        api_key=litellm_params["api_key"],
        guardrail_name=guardrail["guardrail_name"],
        event_hook=litellm_params["mode"],
        category_thresholds=litellm_params.get("category_thresholds"),
        default_on=litellm_params["default_on"],
    )
    litellm.logging_callback_manager.add_litellm_callback(_lakera_callback)


def initialize_aim(litellm_params, guardrail):
    from litellm.proxy.guardrails.guardrail_hooks.aim import AimGuardrail

    _aim_callback = AimGuardrail(
        api_base=litellm_params["api_base"],
        api_key=litellm_params["api_key"],
        guardrail_name=guardrail["guardrail_name"],
        event_hook=litellm_params["mode"],
        default_on=litellm_params["default_on"],
    )
    litellm.logging_callback_manager.add_litellm_callback(_aim_callback)


def initialize_presidio(litellm_params, guardrail):
    from litellm.proxy.guardrails.guardrail_hooks.presidio import (
        _OPTIONAL_PresidioPIIMasking,
    )

    _presidio_callback = _OPTIONAL_PresidioPIIMasking(
        guardrail_name=guardrail["guardrail_name"],
        event_hook=litellm_params["mode"],
        output_parse_pii=litellm_params["output_parse_pii"],
        presidio_ad_hoc_recognizers=litellm_params["presidio_ad_hoc_recognizers"],
        mock_redacted_text=litellm_params.get("mock_redacted_text") or None,
        default_on=litellm_params["default_on"],
    )
    litellm.logging_callback_manager.add_litellm_callback(_presidio_callback)

    if litellm_params["output_parse_pii"]:
        _success_callback = _OPTIONAL_PresidioPIIMasking(
            output_parse_pii=True,
            guardrail_name=guardrail["guardrail_name"],
            event_hook=GuardrailEventHooks.post_call.value,
            presidio_ad_hoc_recognizers=litellm_params["presidio_ad_hoc_recognizers"],
            default_on=litellm_params["default_on"],
        )
        litellm.logging_callback_manager.add_litellm_callback(_success_callback)


def initialize_hide_secrets(litellm_params, guardrail):
    from enterprise.enterprise_hooks.secret_detection import _ENTERPRISE_SecretDetection

    _secret_detection_object = _ENTERPRISE_SecretDetection(
        detect_secrets_config=litellm_params.get("detect_secrets_config"),
        event_hook=litellm_params["mode"],
        guardrail_name=guardrail["guardrail_name"],
        default_on=litellm_params["default_on"],
    )
    litellm.logging_callback_manager.add_litellm_callback(_secret_detection_object)


def initialize_guardrails_ai(litellm_params, guardrail):
    from litellm.proxy.guardrails.guardrail_hooks.guardrails_ai import GuardrailsAI

    _guard_name = litellm_params.get("guard_name")
    if not _guard_name:
        raise Exception(
            "GuardrailsAIException - Please pass the Guardrails AI guard name via 'litellm_params::guard_name'"
        )

    _guardrails_ai_callback = GuardrailsAI(
        api_base=litellm_params.get("api_base"),
        guard_name=_guard_name,
        guardrail_name=SupportedGuardrailIntegrations.GURDRAILS_AI.value,
        default_on=litellm_params["default_on"],
    )
    litellm.logging_callback_manager.add_litellm_callback(_guardrails_ai_callback)
