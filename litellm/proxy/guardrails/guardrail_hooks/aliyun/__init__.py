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

from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .aliyun_ai_guardrail import AliyunAIGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def _resolve_os_environ_reference(value: str | None) -> str | None:
    """Resolve an ``os.environ/`` reference.

    guardrail_registry.py only auto-resolves api_key/api_base, so the Aliyun
    credential fields have to be resolved here.
    """
    from litellm.secret_managers.main import get_secret_str

    if isinstance(value, str) and value.startswith("os.environ/"):
        return get_secret_str(value)
    return value


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

    guardrail_name: Final = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Aliyun AI Guardrail: guardrail_name is required")

    level: Final = getattr(litellm_params, "level", None)
    max_text_length: Final = getattr(litellm_params, "max_text_length", None)
    stream_window_size: Final = getattr(litellm_params, "stream_window_size", None)
    stream_slide_step: Final = getattr(litellm_params, "stream_slide_step", None)
    stream_first_check_step: Final = getattr(litellm_params, "stream_first_check_step", None)
    region_id: Final = getattr(litellm_params, "region_id", None)
    service_input: Final = getattr(litellm_params, "service_input", None)
    service_output: Final = getattr(litellm_params, "service_output", None)
    service_mcp: Final = getattr(litellm_params, "service_mcp", None)

    # These custom credential fields are not auto-resolved by guardrail_registry.py
    # (only api_key/api_base are), so os.environ/ references are resolved here.
    access_key_id: Final = _resolve_os_environ_reference(getattr(litellm_params, "access_key_id", None))
    access_key_secret: Final = _resolve_os_environ_reference(getattr(litellm_params, "access_key_secret", None))

    aliyun_guardrail: Final = AliyunAIGuardrail(
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
        service_mcp=service_mcp,
        default_on=litellm_params.default_on,
        event_hook=litellm_params.mode,
    )

    litellm.logging_callback_manager.add_litellm_callback(aliyun_guardrail)

    return aliyun_guardrail


# Registry for guardrail initializers.
# Plain dicts: guardrail_registry.py gates discovery on `isinstance(registry, dict)`,
# which a MappingProxyType would fail, silently skipping this guardrail's registration.
guardrail_initializer_registry: Final = {  # mutable-ok: loader requires a real dict
    SupportedGuardrailIntegrations.ALIYUN_AI_GUARDRAIL.value: initialize_guardrail,
}

# Registry for guardrail classes
guardrail_class_registry: Final = {  # mutable-ok: loader requires a real dict
    SupportedGuardrailIntegrations.ALIYUN_AI_GUARDRAIL.value: AliyunAIGuardrail,
}
