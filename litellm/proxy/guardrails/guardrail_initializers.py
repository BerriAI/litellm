# litellm/proxy/guardrails/guardrail_initializers.py
import litellm
from litellm.proxy._types import CommonProxyErrors
from litellm.types.guardrails import *


def initialize_aporia(
    litellm_params: LitellmParams,
    guardrail: Guardrail,
):
    from litellm.proxy.guardrails.guardrail_hooks.aporia_ai import AporiaGuardrail

    _aporia_callback = AporiaGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_aporia_callback)
    return _aporia_callback


def initialize_bedrock(litellm_params: LitellmParams, guardrail: Guardrail):
    from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
        BedrockGuardrail,
    )

    _bedrock_callback = BedrockGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        guardrailIdentifier=litellm_params.guardrailIdentifier,
        guardrailVersion=litellm_params.guardrailVersion,
        default_on=litellm_params.default_on,
        mask_request_content=litellm_params.mask_request_content,
        mask_response_content=litellm_params.mask_response_content,
        aws_region_name=litellm_params.aws_region_name,
        aws_access_key_id=litellm_params.aws_access_key_id,
        aws_secret_access_key=litellm_params.aws_secret_access_key,
        aws_session_token=litellm_params.aws_session_token,
        aws_session_name=litellm_params.aws_session_name,
        aws_profile_name=litellm_params.aws_profile_name,
        aws_role_name=litellm_params.aws_role_name,
        aws_web_identity_token=litellm_params.aws_web_identity_token,
        aws_sts_endpoint=litellm_params.aws_sts_endpoint,
        aws_bedrock_runtime_endpoint=litellm_params.aws_bedrock_runtime_endpoint,
    )
    litellm.logging_callback_manager.add_litellm_callback(_bedrock_callback)
    return _bedrock_callback


def initialize_lakera(litellm_params: LitellmParams, guardrail: Guardrail):
    from litellm.proxy.guardrails.guardrail_hooks.lakera_ai import lakeraAI_Moderation

    _lakera_callback = lakeraAI_Moderation(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        category_thresholds=litellm_params.category_thresholds,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_lakera_callback)
    return _lakera_callback


def initialize_lakera_v2(litellm_params: LitellmParams, guardrail: Guardrail):
    from litellm.proxy.guardrails.guardrail_hooks.lakera_ai_v2 import LakeraAIGuardrail

    _lakera_v2_callback = LakeraAIGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        project_id=litellm_params.project_id,
        payload=litellm_params.payload,
        breakdown=litellm_params.breakdown,
        metadata=litellm_params.metadata,
        dev_info=litellm_params.dev_info,
    )
    litellm.logging_callback_manager.add_litellm_callback(_lakera_v2_callback)
    return _lakera_v2_callback


def initialize_aim(litellm_params: LitellmParams, guardrail: Guardrail):
    from litellm.proxy.guardrails.guardrail_hooks.aim import AimGuardrail

    _aim_callback = AimGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_aim_callback)

    return _aim_callback


def initialize_presidio(litellm_params: LitellmParams, guardrail: Guardrail):
    from litellm.proxy.guardrails.guardrail_hooks.presidio import (
        _OPTIONAL_PresidioPIIMasking,
    )

    _presidio_callback = _OPTIONAL_PresidioPIIMasking(
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        output_parse_pii=litellm_params.output_parse_pii,
        presidio_ad_hoc_recognizers=litellm_params.presidio_ad_hoc_recognizers,
        mock_redacted_text=litellm_params.mock_redacted_text,
        default_on=litellm_params.default_on,
        pii_entities_config=litellm_params.pii_entities_config,
        presidio_analyzer_api_base=litellm_params.presidio_analyzer_api_base,
        presidio_anonymizer_api_base=litellm_params.presidio_anonymizer_api_base,
        presidio_language=litellm_params.presidio_language,
    )
    litellm.logging_callback_manager.add_litellm_callback(_presidio_callback)

    if litellm_params.output_parse_pii:
        _success_callback = _OPTIONAL_PresidioPIIMasking(
            output_parse_pii=True,
            guardrail_name=guardrail.get("guardrail_name", ""),
            event_hook=GuardrailEventHooks.post_call.value,
            presidio_ad_hoc_recognizers=litellm_params.presidio_ad_hoc_recognizers,
            default_on=litellm_params.default_on,
            presidio_analyzer_api_base=litellm_params.presidio_analyzer_api_base,
            presidio_anonymizer_api_base=litellm_params.presidio_anonymizer_api_base,
            presidio_language=litellm_params.presidio_language,
        )
        litellm.logging_callback_manager.add_litellm_callback(_success_callback)

    return _presidio_callback


def initialize_hide_secrets(litellm_params: LitellmParams, guardrail: Guardrail):
    try:
        from litellm_enterprise.enterprise_callbacks.secret_detection import (
            _ENTERPRISE_SecretDetection,
        )
    except ImportError:
        raise Exception(
            "Trying to use Secret Detection"
            + CommonProxyErrors.missing_enterprise_package.value
        )

    _secret_detection_object = _ENTERPRISE_SecretDetection(
        detect_secrets_config=litellm_params.detect_secrets_config,
        event_hook=litellm_params.mode,
        guardrail_name=guardrail.get("guardrail_name", ""),
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_secret_detection_object)
    return _secret_detection_object


def initialize_guardrails_ai(litellm_params, guardrail):
    from litellm.proxy.guardrails.guardrail_hooks.guardrails_ai import GuardrailsAI

    _guard_name = litellm_params.guard_name
    if not _guard_name:
        raise Exception(
            "GuardrailsAIException - Please pass the Guardrails AI guard name via 'litellm_params::guard_name'"
        )

    _guardrails_ai_callback = GuardrailsAI(
        api_base=litellm_params.api_base,
        guard_name=_guard_name,
        guardrail_name=SupportedGuardrailIntegrations.GURDRAILS_AI.value,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_guardrails_ai_callback)

    return _guardrails_ai_callback


def initialize_pangea(litellm_params, guardrail):
    from litellm.proxy.guardrails.guardrail_hooks.pangea import PangeaHandler

    _pangea_callback = PangeaHandler(
        guardrail_name=guardrail["guardrail_name"],
        pangea_input_recipe=litellm_params.pangea_input_recipe,
        pangea_output_recipe=litellm_params.pangea_output_recipe,
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_pangea_callback)

    return _pangea_callback


def initialize_lasso(
    litellm_params: LitellmParams,
    guardrail: Guardrail,
):
    from litellm.proxy.guardrails.guardrail_hooks.lasso import LassoGuardrail

    _lasso_callback = LassoGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        lasso_api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        user_id=litellm_params.lasso_user_id,
        conversation_id=litellm_params.lasso_conversation_id,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_lasso_callback)

    return _lasso_callback


def initialize_panw_prisma_airs(litellm_params, guardrail):
    from litellm.proxy.guardrails.guardrail_hooks.panw_prisma_airs import (
        PanwPrismaAirsHandler,
    )
    
    if not litellm_params.api_key:
        raise ValueError("PANW Prisma AIRS: api_key is required")
    if not litellm_params.profile_name:
        raise ValueError("PANW Prisma AIRS: profile_name is required")

    _panw_callback = PanwPrismaAirsHandler(
        guardrail_name=guardrail.get("guardrail_name", "panw_prisma_airs"),  # Use .get() with default
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base or "https://service.api.aisecurity.paloaltonetworks.com/v1/scan/sync/request",
        profile_name=litellm_params.profile_name,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_panw_callback)

    return _panw_callback
