"""
Aliyun AI Security Guardrail Integration for LiteLLM
阿里云AI安全护栏集成
This module provides integration with Aliyun's AI Security Guardrail service for:
- ContentModeration 内容合规检测
- PromptAttack 提示词攻击检测
- SensitiveData 敏感内容检测
- ModelHallucination 模型幻觉
- MaliciousUrl 恶意URL检测
...
Documentation: https://help.aliyun.com/document_detail/2873209.html
"""

from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .aliyun_ai_guardrail import AliyunAIGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> AliyunAIGuardrail:
    """
    Initialize an Aliyun AI Guardrail instance.
    Credentials are configured in config.yaml (litellm_params) and support
    os.environ/ references:
    - access_key_id: Aliyun Access Key ID
    - access_key_secret: Aliyun Access Key Secret
    Args:
        litellm_params: The LiteLLM parameters for the guardrail
        guardrail: The guardrail configuration
    Returns:
        AliyunAIGuardrail instance
    """
    import litellm
    from litellm.secret_managers.main import get_secret_str

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Aliyun AI Guardrail: guardrail_name is required")

    # Get optional parameters from config
    level = getattr(litellm_params, "level", None)
    max_text_length = getattr(litellm_params, "max_text_length", None)
    stream_window_size = getattr(litellm_params, "stream_window_size", None)
    stream_slide_step = getattr(litellm_params, "stream_slide_step", None)
    stream_first_check_step = getattr(litellm_params, "stream_first_check_step", None)
    region_id = getattr(litellm_params, "region_id", None)
    service_input = getattr(litellm_params, "service_input", None)
    service_output = getattr(litellm_params, "service_output", None)

    # Get credentials from config. These custom fields are not auto-resolved by
    # guardrail_registry.py (only api_key/api_base are), so resolve os.environ/
    # references manually here.
    access_key_id = getattr(litellm_params, "access_key_id", None)
    access_key_secret = getattr(litellm_params, "access_key_secret", None)
    if isinstance(access_key_id, str) and access_key_id.startswith("os.environ/"):
        access_key_id = get_secret_str(access_key_id)
    if isinstance(access_key_secret, str) and access_key_secret.startswith("os.environ/"):
        access_key_secret = get_secret_str(access_key_secret)

    # Create guardrail instance
    aliyun_guardrail = AliyunAIGuardrail(
        guardrail_name=guardrail_name,
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        level=level,
        max_text_length=max_text_length,
        stream_window_size=stream_window_size,
        stream_slide_step=stream_slide_step,
        stream_first_check_step=stream_first_check_step,
        region_id=region_id,
        service_input=service_input,
        service_output=service_output,
        default_on=litellm_params.default_on,
        event_hook=litellm_params.mode,
    )

    # Add to callback manager
    litellm.logging_callback_manager.add_litellm_callback(aliyun_guardrail)

    return aliyun_guardrail


# Registry for guardrail initializers
guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.ALIYUN_AI_GUARDRAIL.value: initialize_guardrail,
}

# Registry for guardrail classes
guardrail_class_registry = {
    SupportedGuardrailIntegrations.ALIYUN_AI_GUARDRAIL.value: AliyunAIGuardrail,
}
